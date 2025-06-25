#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include "esp_timer.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_check.h"
#include "anim_player.h"
#include "anim_vfs.h"
#include "anim_dec.h"

#include "gfx_obj.h"
#include "gfx_types.h"
#include "gfx_sw_blend.h"
#include "gfx_font_internal.h"

static const char *TAG = "anim_player";

#define NEED_DELETE     BIT0
#define DELETE_DONE     BIT1
#define WAIT_FLUSH_DONE BIT2
#define WAIT_STOP       BIT3
#define WAIT_STOP_DONE  BIT4
#define PLAYER_START    BIT5
#define PLAYER_STOP     BIT6

#define FPS_TO_MS(fps) (1000 / (fps))  // Convert FPS to milliseconds

typedef struct {
    EventGroupHandle_t event_group;
} anim_player_events_t;

typedef struct {
    uint32_t start;
    uint32_t end;
    anim_vfs_handle_t file_desc;
} anim_player_info_t;

typedef struct anim_player_child_t {
    int type;
    void *src;
    struct anim_player_child_t *next;  // Pointer to next child in the list
} anim_player_child_t;

typedef struct {
    struct {
        unsigned char swap: 1;
        unsigned char mirror: 1;
    } flags;
} anim_player_display_t;

typedef struct {
    anim_player_child_t *child_list;
    ft_lib_handle_t font_lib;
    SemaphoreHandle_t lock_mutex;  /*!< Recursive mutex for protecting rendering operations */

    /*!< Frame buffer management */
    uint16_t *frame_buf1;          /*!< Primary frame buffer */
    uint16_t *frame_buf2;          /*!< Secondary frame buffer */
    size_t buf_size;               /*!< Current buffer size */
    bool buffers_allocated;        /*!< Whether buffers are allocated */
    uint8_t mirror_offset;         /*!< Mirror buffer offset for positioning */
    gfx_color_t default_color;        /*!< Default background color for frame buffers */
} anim_player_gfx_t;

typedef struct {
    int run_start;
    int run_end;
    bool repeat;
    int fps;
} anim_player_params_t;

typedef struct {
    anim_player_info_t info;
    anim_player_params_t run_cfg;
    anim_flush_cb_t flush_cb;
    anim_update_cb_t update_cb;
    anim_player_events_t events;
    anim_player_display_t display;
    anim_player_gfx_t gfx;
    void *user_data;
} anim_player_context_t;

typedef struct {
    player_action_t action;
    anim_player_params_t config;
    int64_t last_frame_time;
} anim_player_run_ctx_t;

/**
 * @brief Allocate or reallocate frame buffers if needed
 * @param ctx Player context
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return esp_err_t ESP_OK on success, otherwise error code
 */
static esp_err_t ensure_frame_buffers(anim_player_context_t *ctx, int width, int height)
{
    size_t required_size;
    if (ctx->display.flags.mirror) {
        required_size = (width + width + ctx->gfx.mirror_offset) * height * sizeof(uint16_t);
    } else {
        required_size = width * height * sizeof(uint16_t);
    }

    if (ctx->gfx.buffers_allocated && ctx->gfx.buf_size >= required_size) {
        return ESP_OK; // Buffers already allocated and sufficient
    }

    if (ctx->gfx.frame_buf1) {
        free(ctx->gfx.frame_buf1);
        ctx->gfx.frame_buf1 = NULL;
    }
    if (ctx->gfx.frame_buf2) {
        free(ctx->gfx.frame_buf2);
        ctx->gfx.frame_buf2 = NULL;
    }

    ctx->gfx.frame_buf1 = (uint16_t *)heap_caps_malloc(required_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ctx->gfx.frame_buf1) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer 1");
        return ESP_ERR_NO_MEM;
    }

    ctx->gfx.frame_buf2 = (uint16_t *)heap_caps_malloc(required_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!ctx->gfx.frame_buf2) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer 2");
        free(ctx->gfx.frame_buf1);
        ctx->gfx.frame_buf1 = NULL;
        return ESP_ERR_NO_MEM;
    }

    ctx->gfx.buf_size = required_size;
    ctx->gfx.buffers_allocated = true;
    ESP_LOGI(TAG, "new buffers, size: %zu bytes (mirror: %s, w: %d, h: %d)",
             required_size, ctx->display.flags.mirror ? "enabled" : "disabled", width, height);

    return ESP_OK;
}

/**
 * @brief Free frame buffers
 * @param ctx Player context
 */
static void free_frame_buffers(anim_player_context_t *ctx)
{
    if (ctx->gfx.frame_buf1) {
        free(ctx->gfx.frame_buf1);
        ctx->gfx.frame_buf1 = NULL;
    }
    if (ctx->gfx.frame_buf2) {
        free(ctx->gfx.frame_buf2);
        ctx->gfx.frame_buf2 = NULL;
    }
    ctx->gfx.buf_size = 0;
    ctx->gfx.buffers_allocated = false;
}

/**
 * @brief Update mirror buffer allocation based on mirror flag
 * @param ctx Player context
 * @return esp_err_t ESP_OK on success, otherwise error code
 */
static esp_err_t update_mirror_buffer(anim_player_context_t *ctx)
{
    if (!ctx->gfx.buffers_allocated) {
        return ESP_OK; // No buffers allocated yet
    }

    ctx->gfx.buffers_allocated = false;
    ESP_LOGD(TAG, "Marked buffers for reallocation due to mirror flag change");

    return ESP_OK;
}

ft_lib_handle_t anim_player_get_font_lib(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        return NULL;
    }
    return ctx->gfx.font_lib;
}

void anim_player_blend_child(anim_player_context_t *ctx, int x1, int y1, int x2, int y2, const void *dest_buf)
{
    if (ctx->gfx.child_list == NULL) {
        return;
    }

    // Lock the recursive render mutex to prevent external operations during rendering
    if (ctx->gfx.lock_mutex && xSemaphoreTakeRecursive(ctx->gfx.lock_mutex, portMAX_DELAY) == pdTRUE) {
        anim_player_child_t *current = ctx->gfx.child_list;
        while (current != NULL) {
            gfx_obj_t *obj = (gfx_obj_t *)current->src;

            if (obj->type == GFX_OBJ_TYPE_LABEL) {
                gfx_draw_label(obj, x1, y1, x2, y2, dest_buf);
            } else if (obj->type == GFX_OBJ_TYPE_IMAGE) {
                gfx_draw_img(obj, x1, y1, x2, y2, dest_buf);
            }

            current = current->next;
        }

        // Release the recursive mutex after rendering is complete
        xSemaphoreGiveRecursive(ctx->gfx.lock_mutex);
    }
}

static esp_err_t anim_player_parse(const uint8_t *data, size_t data_len, image_header_t *header, anim_player_context_t *ctx)
{
    int width = header->width;
    int height = header->height;
    int split_height = header->split_height;
    int splits = header->splits;
    uint16_t color_depth = 0;

    bool mirror_enabled = ctx->display.flags.mirror;
    uint8_t mirror_offset = ctx->gfx.mirror_offset;
    uint16_t default_color = ctx->gfx.default_color.full;
    size_t buf_size = ctx->gfx.buf_size;

    uint16_t *offsets = (uint16_t *)malloc(splits * sizeof(uint16_t));
    if (offsets == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for offsets");
        return ESP_FAIL;
    }

    anim_dec_calculate_offsets(header, offsets);

    esp_err_t buf_ret = ensure_frame_buffers(ctx, width, split_height);
    if (buf_ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to ensure frame buffers");
        free(offsets);
        return buf_ret;
    }

    uint8_t *decode_buffer = NULL;
    if (header->bit_depth == 4) {
        decode_buffer = (uint8_t *)malloc(width * (split_height + (split_height % 2)) / 2);
    } else if (header->bit_depth == 8) {
        decode_buffer = (uint8_t *)heap_caps_malloc(width * split_height, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (decode_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for decode buffer");
        free(offsets);
        return ESP_FAIL;
    }

    if (header->bit_depth == 4) {
        color_depth = 16;
    } else if (header->bit_depth == 8) {
        color_depth = 256;
    }

    uint32_t *palette_cache = (uint32_t *)heap_caps_malloc(color_depth * sizeof(uint32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (palette_cache == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for palette_cache");
        free(decode_buffer);
        free(offsets);
        return ESP_FAIL;
    }

    for (int i = 0; i < color_depth; i++) {
        palette_cache[i] = 0xFFFFFFFF;
    }

    uint16_t *buf_act = NULL;

    for (int split = 0; split < splits; split++) {
        const uint8_t *compressed_data = data + offsets[split];
        int compressed_len = header->split_lengths[split];

        buf_act = (buf_act == NULL || buf_act == ctx->gfx.frame_buf2) ? ctx->gfx.frame_buf1 : ctx->gfx.frame_buf2;

        // Initialize buffer with default color instead of 0
        for (size_t i = 0; i < buf_size / sizeof(uint16_t); i++) {
            buf_act[i] = default_color;
        }

        esp_err_t decode_result = ESP_FAIL;
        int valid_height;

        if (split == splits - 1) {
            valid_height = height - split * split_height;
        } else {
            valid_height = split_height;
        }
        ESP_LOGD(TAG, "split:%d(%d), height:%d(%d), compressed_len:%d", split, splits, split_height, valid_height, compressed_len);

        if (compressed_data[0] == ENCODING_TYPE_RLE) {
            decode_result = anim_dec_rte_decode(compressed_data + 1, compressed_len - 1,
                                                decode_buffer, width * split_height);
        } else if (compressed_data[0] == ENCODING_TYPE_HUFFMAN) {
            uint8_t *huffman_buffer = malloc(width * split_height);
            if (huffman_buffer == NULL) {
                ESP_LOGE(TAG, "Failed to allocate memory for Huffman buffer");
                continue;
            }

            size_t huffman_decoded_len = 0;
            anim_dec_huffman_decode(compressed_data, compressed_len, huffman_buffer, &huffman_decoded_len);
            decode_result = ESP_OK;
            if (decode_result == ESP_OK) {
                decode_result = anim_dec_rte_decode(huffman_buffer, huffman_decoded_len,
                                                    decode_buffer, width * split_height);
            }
            free(huffman_buffer);
        } else {
            ESP_LOGE(TAG, "Unknown encoding type: %02X", compressed_data[0]);
            continue;
        }

        if (decode_result != ESP_OK) {
            ESP_LOGE(TAG, "Failed to decode split %d", split);
            continue;
        }

        if (header->bit_depth == 4) {
            // Calculate stride once
            int stride = mirror_enabled ? (width + width + mirror_offset) : width;

            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < width; x += 2) {
                    uint8_t packed_gray = decode_buffer[y * (width / 2) + (x / 2)];
                    uint8_t index1 = (packed_gray & 0xF0) >> 4;
                    uint8_t index2 = (packed_gray & 0x0F);

                    if (palette_cache[index1] == 0xFFFFFFFF) {
                        uint16_t color = anim_dec_parse_palette(header, index1, ctx->display.flags.swap);
                        palette_cache[index1] = color;
                    }

                    uint16_t color1 = (uint16_t)palette_cache[index1];
                    buf_act[y * stride + x] = color1;

                    // Sync write to mirror position if mirror is enabled
                    if (mirror_enabled) {
                        int mirror_x = width + mirror_offset + width - 1 - x;
                        buf_act[y * stride + mirror_x] = color1;
                    }

                    if (x + 1 < width) {
                        if (palette_cache[index2] == 0xFFFFFFFF) {
                            uint16_t color = anim_dec_parse_palette(header, index2, ctx->display.flags.swap);
                            palette_cache[index2] = color;
                        }

                        uint16_t color2 = (uint16_t)palette_cache[index2];
                        buf_act[y * stride + x + 1] = color2;

                        // Sync write to mirror position if mirror is enabled
                        if (mirror_enabled) {
                            int mirror_x = width + mirror_offset + width - 1 - (x + 1);
                            buf_act[y * stride + mirror_x] = color2;
                        }
                    }
                }
            }

        } else if (header->bit_depth == 8) {
            // Calculate stride once
            int stride = mirror_enabled ? (width + width + mirror_offset) : width;

            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < width; x++) {
                    uint8_t index = decode_buffer[y * width + x];
                    if (palette_cache[index] == 0xFFFFFFFF) {
                        uint16_t color = anim_dec_parse_palette(header, index, ctx->display.flags.swap);
                        palette_cache[index] = color;
                    }

                    uint16_t color_val = (uint16_t)palette_cache[index];
                    buf_act[y * stride + x] = color_val;

                    // Sync write to mirror position if mirror is enabled
                    if (mirror_enabled) {
                        int mirror_x = width + mirror_offset + width - 1 - x;
                        buf_act[y * stride + mirror_x] = color_val;
                    }
                }
            }
        } else {
            ESP_LOGE(TAG, "Unsupported bit depth: %d", header->bit_depth);
            continue;
        }

        if (ctx->flush_cb) {
            xEventGroupClearBits(ctx->events.event_group, WAIT_FLUSH_DONE);

            if (mirror_enabled) {
                int total_width = width + width + mirror_offset;
                anim_player_blend_child(ctx, 0, split * split_height, total_width, split * split_height + valid_height, buf_act);
                ctx->flush_cb(ctx, 0, split * split_height, total_width, split * split_height + valid_height, buf_act);
            } else {
                anim_player_blend_child(ctx, 0, split * split_height, width, split * split_height + valid_height, buf_act);
                ctx->flush_cb(ctx, 0, split * split_height, width, split * split_height + valid_height, buf_act);
            }

            xEventGroupWaitBits(ctx->events.event_group, WAIT_FLUSH_DONE, pdTRUE, pdFALSE, pdMS_TO_TICKS(20));
        }
    }

    free(palette_cache);
    free(offsets);
    free(decode_buffer);
    anim_dec_free_header(header);

    return ESP_OK;
}

static void anim_player_task(void *arg)
{
    image_header_t header;
    anim_player_context_t *ctx = (anim_player_context_t *)arg;
    anim_player_run_ctx_t run_ctx;

    run_ctx.action = PLAYER_ACTION_STOP;
    run_ctx.config.run_start = ctx->run_cfg.run_start;
    run_ctx.config.run_end = ctx->run_cfg.run_end;
    run_ctx.config.repeat = ctx->run_cfg.repeat;
    run_ctx.config.fps = ctx->run_cfg.fps;
    run_ctx.last_frame_time = esp_timer_get_time();

    while (1) {
        EventBits_t bits = xEventGroupWaitBits(ctx->events.event_group,
                                               NEED_DELETE | WAIT_STOP | PLAYER_START | PLAYER_STOP,
                                               pdTRUE, pdFALSE, pdMS_TO_TICKS(10));

        if (bits & NEED_DELETE) {
            ESP_LOGW(TAG, "Player deleted");
            xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
            vTaskDeleteWithCaps(NULL);
        }

        if (bits & WAIT_STOP) {
            xEventGroupSetBits(ctx->events.event_group, WAIT_STOP_DONE);
        }

        // Check for player action events
        if (bits & PLAYER_START) {
            run_ctx.action = PLAYER_ACTION_START;
            run_ctx.config.run_start = ctx->run_cfg.run_start;
            run_ctx.config.run_end = ctx->run_cfg.run_end;
            run_ctx.config.repeat = ctx->run_cfg.repeat;
            run_ctx.config.fps = ctx->run_cfg.fps;
            ESP_LOGD(TAG, "Player updated [START]: %d -> %d, repeat:%d, fps:%d",
                     run_ctx.config.run_start, run_ctx.config.run_end, run_ctx.config.repeat, run_ctx.config.fps);
        }

        if (bits & PLAYER_STOP) {
            run_ctx.action = PLAYER_ACTION_STOP;
            run_ctx.config.repeat = false;
            ESP_LOGD(TAG, "Player updated [STOP]");
        }

        if (run_ctx.action == PLAYER_ACTION_STOP) {
            continue;
        }

        do {
            for (int i = run_ctx.config.run_start; (i <= run_ctx.config.run_end) && (run_ctx.action != PLAYER_ACTION_STOP); i++) {
                // Frame rate control
                int64_t elapsed = esp_timer_get_time() - run_ctx.last_frame_time;
                elapsed = elapsed / 1000;
                if (elapsed < FPS_TO_MS(run_ctx.config.fps)) {
                    vTaskDelay(pdMS_TO_TICKS(FPS_TO_MS(run_ctx.config.fps) - elapsed));
                } else {
                    vTaskDelay(pdMS_TO_TICKS(1));
                }
                run_ctx.last_frame_time = esp_timer_get_time();

                bits = xEventGroupWaitBits(ctx->events.event_group,
                                           NEED_DELETE | WAIT_STOP | PLAYER_START | PLAYER_STOP,
                                           pdTRUE, pdFALSE, pdMS_TO_TICKS(0));
                if (bits & NEED_DELETE) {
                    ESP_LOGW(TAG, "Playing deleted");
                    xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
                    vTaskDelete(NULL);
                }
                if (bits & WAIT_STOP) {
                    xEventGroupSetBits(ctx->events.event_group, WAIT_STOP_DONE);
                }
                if (bits & PLAYER_START) {
                    run_ctx.action = PLAYER_ACTION_START;
                    run_ctx.config.run_start = ctx->run_cfg.run_start;
                    run_ctx.config.run_end = ctx->run_cfg.run_end;
                    run_ctx.config.repeat = ctx->run_cfg.repeat;
                    run_ctx.config.fps = ctx->run_cfg.fps;
                    ESP_LOGD(TAG, "Playing updated [START]: %d -> %d, repeat:%d, fps:%d",
                             run_ctx.config.run_start, run_ctx.config.run_end, run_ctx.config.repeat, run_ctx.config.fps);
                    break;
                }
                if (bits & PLAYER_STOP) {
                    run_ctx.action = PLAYER_ACTION_STOP;
                    run_ctx.config.repeat = false;
                    ESP_LOGD(TAG, "Playing updated [STOP]");
                    break;
                }

                const void *frame_data = anim_vfs_get_frame_data(ctx->info.file_desc, i);
                size_t frame_size = anim_vfs_get_frame_size(ctx->info.file_desc, i);

                image_format_t format = anim_dec_parse_header(frame_data, frame_size, &header);

                if (format == IMAGE_FORMAT_INVALID) {
                    ESP_LOGE(TAG, "Invalid frame format");
                    continue;
                } else if (format == IMAGE_FORMAT_REDIRECT) {
                    ESP_LOGE(TAG, "Invalid redirect frame");
                    continue;
                } else if (format == IMAGE_FORMAT_SBMP) {
                    anim_player_parse(frame_data, frame_size, &header, ctx);
                    if (ctx->update_cb) {
                        ctx->update_cb(ctx, PLAYER_EVENT_ONE_FRAME_DONE);
                    }
                }
            }
            if (ctx->update_cb) {
                ctx->update_cb(ctx, PLAYER_EVENT_ALL_FRAME_DONE);
            }
        } while (run_ctx.config.repeat);

        run_ctx.action = PLAYER_ACTION_STOP;

        if (ctx->update_cb) {
            ctx->update_cb(ctx, PLAYER_EVENT_IDLE);
        }
    }
}

bool anim_player_flush_ready(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        return false;
    }

    if (xPortInIsrContext()) {
        BaseType_t pxHigherPriorityTaskWoken = pdFALSE;
        bool result = xEventGroupSetBitsFromISR(ctx->events.event_group, WAIT_FLUSH_DONE, &pxHigherPriorityTaskWoken);
        if (pxHigherPriorityTaskWoken == pdTRUE) {
            portYIELD_FROM_ISR();
        }
        return result;
    } else {
        return xEventGroupSetBits(ctx->events.event_group, WAIT_FLUSH_DONE);
    }
}

void anim_player_update(anim_player_handle_t handle, player_action_t event)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

    EventBits_t event_bit = (event == PLAYER_ACTION_START) ? PLAYER_START : PLAYER_STOP;

    xEventGroupSetBits(ctx->events.event_group, event_bit);
    ESP_LOGD(TAG, "update event: %s", event == PLAYER_ACTION_START ? "START" : "STOP");
}

esp_err_t anim_player_set_src_data(anim_player_handle_t handle, const void *src_data, size_t src_len)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_FAIL;
    }

    anim_vfs_handle_t new_desc;
    anim_vfs_init(src_data, src_len, &new_desc);
    if (new_desc == NULL) {
        ESP_LOGE(TAG, "Failed to initialize asset parser");
        return ESP_FAIL;
    }

    anim_player_update(handle, PLAYER_ACTION_STOP);
    xEventGroupSetBits(ctx->events.event_group, WAIT_STOP);
    xEventGroupWaitBits(ctx->events.event_group, WAIT_STOP_DONE, pdTRUE, pdFALSE, portMAX_DELAY);

    //delete old file_desc
    if (ctx->info.file_desc) {
        anim_vfs_deinit(ctx->info.file_desc);
        ctx->info.file_desc = NULL;
    }

    ctx->info.file_desc = new_desc;
    ctx->info.start = 0;
    ctx->info.end = anim_vfs_get_total_frames(new_desc) - 1;

    //default segment
    ctx->run_cfg.run_start = ctx->info.start;
    ctx->run_cfg.run_end = ctx->info.end;
    ctx->run_cfg.repeat = true;
    ctx->run_cfg.fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;

    return ESP_OK;
}

void anim_player_get_segment(anim_player_handle_t handle, uint32_t *start, uint32_t *end)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

    *start = ctx->info.start;
    *end = ctx->info.end;
}

void anim_player_set_segment(anim_player_handle_t handle, uint32_t start, uint32_t end, uint32_t fps, bool repeat)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

    if (end > ctx->info.end || (start > end)) {
        ESP_LOGE(TAG, "Invalid segment");
        return;
    }

    ctx->run_cfg.run_start = start;
    ctx->run_cfg.run_end = end;
    ctx->run_cfg.repeat = repeat;
    ctx->run_cfg.fps = fps;
    ESP_LOGD(TAG, "set segment: %" PRIu32 " -> %" PRIu32 ", repeat:%d, fps:%" PRIu32 "", start, end, repeat, fps);
}

void *anim_player_get_user_data(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return NULL;
    }

    return ctx->user_data;
}

anim_player_handle_t anim_player_init(const anim_player_config_t *config)
{
    if (!config) {
        ESP_LOGE(TAG, "Invalid configuration");
        return NULL;
    }

    anim_player_context_t *player = malloc(sizeof(anim_player_context_t));
    if (!player) {
        ESP_LOGE(TAG, "Failed to allocate player context");
        return NULL;
    }

    player->info.file_desc = NULL;
    player->info.start = 0;
    player->info.end = 0;
    player->run_cfg.run_start = 0;
    player->run_cfg.run_end = 0;
    player->run_cfg.repeat = false;
    player->run_cfg.fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;
    player->flush_cb = config->flush_cb;
    player->update_cb = config->update_cb;
    player->user_data = config->user_data;

    player->display.flags.mirror = config->flags.mirror;
    player->display.flags.swap = config->flags.swap;

    player->events.event_group = xEventGroupCreate();
    player->gfx.child_list = NULL;

    // Initialize buffer management
    player->gfx.frame_buf1 = NULL;
    player->gfx.frame_buf2 = NULL;
    player->gfx.buf_size = 0;
    player->gfx.buffers_allocated = false;
    player->gfx.mirror_offset = 60;
    player->gfx.default_color.full = 0x0000;

    // Create recursive render mutex for protecting rendering operations
    player->gfx.lock_mutex = xSemaphoreCreateRecursiveMutex();
    if (player->gfx.lock_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create recursive render mutex");
        vEventGroupDelete(player->events.event_group);
        free(player);
        return NULL;
    }

    // Initialize font library for this player instance
    esp_err_t font_ret = gfx_ft_lib_create(&player->gfx.font_lib);
    if (font_ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create font library");
        vEventGroupDelete(player->events.event_group);
        free(player);
        return NULL;
    }

    // Set default task configuration if not specified
    const uint32_t caps = config->task.task_stack_caps ? config->task.task_stack_caps : MALLOC_CAP_DEFAULT; // caps cannot be zero
    if (config->task.task_affinity < 0) {
        xTaskCreateWithCaps(anim_player_task, "Anim Player", config->task.task_stack, player, config->task.task_priority, NULL, caps);
    } else {
        xTaskCreatePinnedToCoreWithCaps(anim_player_task, "Anim Player", config->task.task_stack, player, config->task.task_priority, NULL, config->task.task_affinity, caps);
    }

    return (anim_player_handle_t)player;
}

void anim_player_deinit(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

    // Free all child nodes
    anim_player_child_t *current = ctx->gfx.child_list;
    while (current != NULL) {
        anim_player_child_t *next = current->next;
        free(current);
        current = next;
    }
    ctx->gfx.child_list = NULL;

    // Send event to stop the task
    if (ctx->events.event_group) {
        xEventGroupSetBits(ctx->events.event_group, NEED_DELETE);
        xEventGroupWaitBits(ctx->events.event_group, DELETE_DONE, pdTRUE, pdFALSE, portMAX_DELAY);
    }

    // Delete event group
    if (ctx->events.event_group) {
        vEventGroupDelete(ctx->events.event_group);
        ctx->events.event_group = NULL;
    }

    if (ctx->info.file_desc) {
        anim_vfs_deinit(ctx->info.file_desc);
        ctx->info.file_desc = NULL;
    }

    if (ctx->gfx.font_lib) {
        gfx_ft_lib_cleanup(ctx->gfx.font_lib);
        ctx->gfx.font_lib = NULL;
    }

    free_frame_buffers(ctx);

    if (ctx->gfx.lock_mutex) {
        vSemaphoreDelete(ctx->gfx.lock_mutex);
        ctx->gfx.lock_mutex = NULL;
    }

    free(ctx);
}

esp_err_t anim_player_add_child(anim_player_handle_t handle, int type, void *src)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_FAIL;
    }

    // Create new child node
    anim_player_child_t *new_child = (anim_player_child_t *)malloc(sizeof(anim_player_child_t));
    if (new_child == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for child");
        return ESP_ERR_NO_MEM;
    }

    // Initialize child node
    new_child->type = type;
    new_child->src = src;
    new_child->next = NULL;

    // Add to the end of the list
    if (ctx->gfx.child_list == NULL) {
        ctx->gfx.child_list = new_child;
    } else {
        anim_player_child_t *current = ctx->gfx.child_list;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_child;
    }

    ESP_LOGI(TAG, "Added child(%p): type=%d, src=%p", new_child, new_child->type, new_child->src);
    return ESP_OK;
}

esp_err_t gfx_player_lock(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_ERR_INVALID_ARG;
    }

    if (ctx->gfx.lock_mutex == NULL) {
        ESP_LOGE(TAG, "Recursive render mutex not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (xSemaphoreTakeRecursive(ctx->gfx.lock_mutex, portMAX_DELAY) != pdTRUE) {
        ESP_LOGE(TAG, "Failed to acquire recursive render mutex");
        return ESP_ERR_TIMEOUT;
    }

    return ESP_OK;
}

esp_err_t gfx_player_unlock(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_ERR_INVALID_ARG;
    }

    if (ctx->gfx.lock_mutex == NULL) {
        ESP_LOGE(TAG, "Recursive render mutex not initialized");
        return ESP_ERR_INVALID_STATE;
    }

    if (xSemaphoreGiveRecursive(ctx->gfx.lock_mutex) != pdTRUE) {
        ESP_LOGE(TAG, "Failed to release recursive render mutex");
        return ESP_ERR_INVALID_STATE;
    }

    return ESP_OK;
}

esp_err_t anim_player_set_mirror_config(anim_player_handle_t handle, bool mirror, uint8_t offset)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_ERR_INVALID_ARG;
    }

    // Set mirror flag
    if (ctx->display.flags.mirror != mirror) {
        ctx->display.flags.mirror = mirror;

        esp_err_t ret = update_mirror_buffer(ctx);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to update mirror buffer");
            return ret;
        }
    }

    // Set mirror offset
    ctx->gfx.mirror_offset = offset;

    ESP_LOGD(TAG, "Mirror config set: mirror=%s, offset=%d", mirror ? "true" : "false", offset);
    return ESP_OK;
}

esp_err_t anim_player_set_default_color(anim_player_handle_t handle, gfx_color_t color)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return ESP_ERR_INVALID_ARG;
    }

    // Set default background color
    ctx->gfx.default_color = color;

    ESP_LOGI(TAG, "Default color set: 0x%04X", color.full);
    return ESP_OK;
}
