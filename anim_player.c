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
#include "object.h"

#include "ft_blend.h"
#include "ft_label.h"

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

static const char *TAG = "anim_player";

#define NEED_DELETE     BIT0
#define DELETE_DONE     BIT1
#define WAIT_FLUSH_DONE BIT2
#define WAIT_STOP       BIT3
#define WAIT_STOP_DONE  BIT4

#define FPS_TO_MS(fps) (1000 / (fps))  // Convert FPS to milliseconds

typedef struct {
    player_action_t action;
} anim_player_event_t;

typedef struct {
    EventGroupHandle_t event_group;
    QueueHandle_t event_queue;
} anim_player_events_t;

typedef struct {
    uint32_t start;
    uint32_t end;
    anim_vfs_handle_t file_desc;
} anim_player_info_t;

typedef struct child_t {
    int type;
    void *src;
    // size_t len;
    // uint16_t x1;
    // uint16_t y1;
    struct child_t *next;  // Pointer to next child in the list
} child_t;

typedef struct {
    anim_player_info_t info;
    int run_start;
    int run_end;
    bool repeat;
    int fps;
    anim_flush_cb_t flush_cb;
    anim_update_cb_t update_cb;
    void *user_data;
    anim_player_events_t events;
    TaskHandle_t handle_task;

    uint16_t screen_w;
    uint16_t screen_h;
    struct {
        unsigned char swap: 1;
    } flags;
    child_t *child_list;  // Head of the child list
} anim_player_context_t;

typedef struct {
    player_action_t action;
    int run_start;
    int run_end;
    bool repeat;
    int fps;
    int64_t last_frame_time;
} anim_player_run_ctx_t;

typedef struct {
    uint32_t magic: 8;          /**< Magic number. Must be LV_IMAGE_HEADER_MAGIC*/
    uint32_t cf : 8;            /**< Color format: See `lv_color_format_t`*/
    uint32_t flags: 16;         /**< Image flags, see `lv_image_flags_t`*/

    uint32_t w: 16;
    uint32_t h: 16;
    uint32_t stride: 16;        /**< Number of bytes in a row*/
    uint32_t reserved_2: 16;    /**< Reserved to be used later*/
} gfx_image_header_t;

typedef struct {
    gfx_image_header_t header;   /**< A header describing the basics of the image*/
    uint32_t data_size;         /**< Size of the image in bytes*/
    const uint8_t * data;       /**< Pointer to the data of the image*/
    const void * reserved;      /**< A reserved field to make it has same size as lv_draw_buf_t*/
    const void * reserved_2;    /**< A reserved field to make it has same size as lv_draw_buf_t*/
} gfx_image_dsc_t;

void anim_player_blend_child(anim_player_context_t *ctx, int x1, int y1, int x2, int y2, const void *dest_buf)
{

    child_t *current = ctx->child_list;
    if (current != NULL) {
        int index = 0;
        while (current != NULL) {
            gfx_obj_t *obj = (gfx_obj_t *)current->src;
            if (obj->type == CHILD_TYPE_LABEL) {
                // ESP_LOGW(TAG, "out:%03d,%03d, %03d,%03d", x1, y1, x2, y2);
                // ESP_LOGI(TAG, "obj:%03d,%03d, %03d,%03d", obj->x, obj->y, (obj->x + obj->width), (obj->y + obj->height));

                label_area_t clip_area;
                clip_area.x1 = MAX(x1, obj->x);
                clip_area.y1 = MAX(y1, obj->y);
                clip_area.x2 = MIN(x2, obj->x + obj->width);
                clip_area.y2 = MIN(y2, obj->y + obj->height);

                if (clip_area.x1 < clip_area.x2 && clip_area.y1 < clip_area.y2) {
                    // ESP_LOGE(TAG, "clip:%03d,%03d, %03d,%03d", clip_area.x1, clip_area.y1, clip_area.x2, clip_area.y2);
                } else {
                    // ESP_LOGW(TAG, "no clip");
                    current = current->next;
                    continue;
                }

                ft_sw_draw_label(obj->src); //no use now

                blend_color_t *dest = (blend_color_t *)dest_buf + (clip_area.y1 - y1) * (x2 - x1) + (clip_area.x1);

                label_coord_t dest_stride = (x2 - x1);
                label_coord_t mask_height = (clip_area.y1 - obj->y);
                label_coord_t mask_stride = obj->width;

                // ESP_LOGI(TAG, "mask_stride:%d", mask_stride);
                // ESP_LOGI(TAG, "dest_stride:%d", dest_stride);
                // ESP_LOGI(TAG, "mask_height:%d", mask_height);

                ft_label_render_mask(obj->src, dest, dest_stride, mask_height, &clip_area);

            } else if (obj->type == CHILD_TYPE_IMAGE) {
                gfx_image_dsc_t *img = (gfx_image_dsc_t *)obj->src;
                gfx_image_header_t *header = &img->header;
                // ESP_LOGI(TAG, "header->w: %d", header->w);
                // ESP_LOGI(TAG, "header->h: %d", header->h);

                // ESP_LOGI(TAG, "flush:%03d,%03d, %03d,%03d", x1, y1, x2, y2);
                // ESP_LOGI(TAG, "obj:%03d,%03d, %03d,%03d", obj->x, obj->y, obj->x + header->w, obj->y + header->h);

                label_area_t clip_area;
                clip_area.x1 = MAX(x1, obj->x);
                clip_area.y1 = MAX(y1, obj->y);
                clip_area.x2 = MIN(x2, obj->x + header->w);
                clip_area.y2 = MIN(y2, obj->y + header->h);

                if (clip_area.x1 < clip_area.x2 && clip_area.y1 < clip_area.y2) {
                    // ESP_LOGW(TAG, "clip:%03d,%03d, %03d,%03d", clip_area.x1, clip_area.y1, clip_area.x2, clip_area.y2);
                } else {
                    // ESP_LOGW(TAG, "no clip");
                    current = current->next;
                    continue;
                }

                // ESP_LOGW(TAG, "clip:%03d,%03d, %03d,%03d", clip_area.x1, clip_area.y1, clip_area.x2, clip_area.y2);

                blend_color_t *src = (blend_color_t *)img->data + (clip_area.y1 - obj->y) * header->w;
                label_opa_t *mask = (label_opa_t *)(img->data + header->w * header->h * 2 + (clip_area.y1 - obj->y) * header->w);

                blend_color_t *dest = (blend_color_t *)dest_buf + (clip_area.y1 - y1) * (x2 - x1) + (clip_area.x1 - x1);
                src = (blend_color_t *)img->data + (clip_area.y1 - obj->y) * header->w;

                // ESP_LOGI(TAG, "src:%p, mask:%p, dest:%p", src, mask, dest);
                // ESP_LOGI(TAG, "dest:%p, stride:%d", dest, x2 - x1);
                // ESP_LOGI(TAG, "src:%p, mask:%p, stride:%d, offset:%d", src, mask, header->w, (int)((char *)mask - (char *)src));
                // printf("\r\n");

                blend_sw_img_draw(
                    (blend_color_t *)dest,
                    x2 - x1,
                    src,
                    header->w,
                    mask,
                    header->w,
                    &clip_area,
                    255
                );
            }
            current = current->next;
        }
    }
}

static esp_err_t anim_player_parse(const uint8_t *data, size_t data_len, image_header_t *header, anim_player_context_t *ctx)
{
    // Allocate memory for split offsets
    uint16_t *offsets = (uint16_t *)malloc(header->splits * sizeof(uint16_t));
    if (offsets == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for offsets");
        return ESP_FAIL;
    }

    anim_dec_calculate_offsets(header, offsets);

    // Allocate frame buffer
    // void *buf1 = malloc(header->width * header->split_height * sizeof(uint16_t));
    uint16_t *buf1 = (uint16_t *)heap_caps_malloc(header->width * header->split_height * sizeof(uint16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf1 == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for frame buffer");
        free(offsets);
        return ESP_FAIL;
    }

    uint16_t *buf2 = (uint16_t *)heap_caps_malloc(header->width * header->split_height * sizeof(uint16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (buf2 == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for frame buffer");
        free(offsets);
        free(buf1);
        return ESP_FAIL;
    }

    // Allocate decode buffer
    uint8_t *decode_buffer = NULL;
    if (header->bit_depth == 4) {
        decode_buffer = (uint8_t *)malloc(header->width * (header->split_height + (header->split_height % 2)) / 2);
    } else if (header->bit_depth == 8) {
        // decode_buffer = (uint8_t *)malloc(header->width * header->split_height);
        decode_buffer = (uint8_t *)heap_caps_malloc(header->width * header->split_height, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (decode_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for decode buffer");
        free(offsets);
        free(buf1);
        free(buf2);
        return ESP_FAIL;
    }

    uint16_t color_depth = 0;

    if (header->bit_depth == 4) {
        color_depth = 16;
    } else if (header->bit_depth == 8) {
        color_depth = 256;
    }

    // uint32_t *palette_cache = (uint32_t *)malloc(color_depth * sizeof(uint32_t));
    uint32_t *palette_cache = (uint32_t *)heap_caps_malloc(color_depth * sizeof(uint32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (palette_cache == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for palette_cache");
        free(buf1);
        free(buf2);
        free(decode_buffer);
        free(offsets);
        return ESP_FAIL;
    }

    for (int i = 0; i < color_depth; i++) {
        palette_cache[i] = 0xFFFFFFFF;
    }

    // Process each split
    for (int split = 0; split < header->splits; split++) {
        const uint8_t *compressed_data = data + offsets[split];
        int compressed_len = header->split_lengths[split];

        uint16_t *buf = (split % 2 == 0) ? buf1 : buf2;

        esp_err_t decode_result = ESP_FAIL;
        int valid_height;

        if (split == header->splits - 1) {
            valid_height = header->height - split * header->split_height;
        } else {
            valid_height = header->split_height;
        }
        ESP_LOGD(TAG, "split:%d(%d), height:%d(%d), compressed_len:%d", split, header->splits, header->split_height, valid_height, compressed_len);

        // Check encoding type from first byte
        if (compressed_data[0] == ENCODING_TYPE_RLE) {
            decode_result = anim_dec_rte_decode(compressed_data + 1, compressed_len - 1,
                                                decode_buffer, header->width * header->split_height);
        } else if (compressed_data[0] == ENCODING_TYPE_HUFFMAN) {
            uint8_t *huffman_buffer = malloc(header->width * header->split_height);
            if (huffman_buffer == NULL) {
                ESP_LOGE(TAG, "Failed to allocate memory for Huffman buffer");
                continue;
            }

            size_t huffman_decoded_len = 0;
            anim_dec_huffman_decode(compressed_data, compressed_len, huffman_buffer, &huffman_decoded_len);
            decode_result = ESP_OK;
            if (decode_result == ESP_OK) {
                decode_result = anim_dec_rte_decode(huffman_buffer, huffman_decoded_len,
                                                    decode_buffer, header->width * header->split_height);
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

        // ESP_LOGI(TAG, "header->bit_depth: %d", header->bit_depth);
        if (header->bit_depth == 4) {
            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < header->width; x += 2) {
                    uint8_t packed_gray = decode_buffer[y * (header->width / 2) + (x / 2)];
                    uint8_t index1 = (packed_gray & 0xF0) >> 4;
                    uint8_t index2 = (packed_gray & 0x0F);

                    if (palette_cache[index1] == 0xFFFFFFFF) {
                        uint16_t color = anim_dec_parse_palette(header, index1, ctx->flags.swap);
                        palette_cache[index1] = color;
                    }
                    buf[y * header->width + x] = (uint16_t)palette_cache[index1];

                    if (x + 1 < header->width) {
                        if (palette_cache[index2] == 0xFFFFFFFF) {
                            uint16_t color = anim_dec_parse_palette(header, index2, ctx->flags.swap);
                            palette_cache[index2] = color;
                        }
                        buf[y * header->width + x + 1] = (uint16_t)palette_cache[index2];
                    }
                }
            }

        } else if (header->bit_depth == 8) {
            int64_t start_time = esp_timer_get_time();
            for (int y = 0; y < valid_height; y++) {
                for (int x = 0; x < header->width; x++) {
                    uint8_t index = decode_buffer[y * header->width + x];

#if 1
                    if (palette_cache[index] == 0xFFFFFFFF) {
                        uint16_t color = anim_dec_parse_palette(header, index, ctx->flags.swap);
                        palette_cache[index] = color;
                    }
                    // Copy the color value directly
                    buf[y * header->width + x] = (uint16_t)palette_cache[index];
#else
                    buf[y * header->width + x] = anim_dec_parse_palette(header, index, ctx->flags.swap);
#endif
                }
            }
            int64_t end_time = esp_timer_get_time();
            // ESP_LOGI(TAG, "8-bit mode: %" PRIu64 " ms", (end_time - start_time) / 1000);
        } else {
            ESP_LOGE(TAG, "Unsupported bit depth: %d", header->bit_depth);
            continue;
        }

        // Flush decoded data
        xEventGroupClearBits(ctx->events.event_group, WAIT_FLUSH_DONE);
        if (ctx->flush_cb) {
            anim_player_blend_child(ctx, 0, split * header->split_height, header->width, split * header->split_height + valid_height, buf);
            ctx->flush_cb(ctx, 0, split * header->split_height, header->width, split * header->split_height + valid_height, buf);
        }
        xEventGroupWaitBits(ctx->events.event_group, WAIT_FLUSH_DONE, pdTRUE, pdFALSE, pdMS_TO_TICKS(20));
    }

    // Cleanup
    free(palette_cache);
    free(offsets);
    free(buf1);
    free(buf2);
    free(decode_buffer);
    anim_dec_free_header(header);

    return ESP_OK;
}

static void anim_player_task(void *arg)
{
    image_header_t header;
    anim_player_context_t *ctx = (anim_player_context_t *)arg;
    anim_player_run_ctx_t run_ctx;

    anim_player_event_t player_event;

    run_ctx.action = PLAYER_ACTION_STOP;
    run_ctx.run_start = ctx->run_start;
    run_ctx.run_end = ctx->run_end;
    run_ctx.repeat = ctx->repeat;
    run_ctx.fps = ctx->fps;
    run_ctx.last_frame_time = esp_timer_get_time();

    while (1) {
        EventBits_t bits = xEventGroupWaitBits(ctx->events.event_group,
                                               NEED_DELETE | WAIT_STOP,
                                               pdTRUE, pdFALSE, pdMS_TO_TICKS(10));

        if (bits & NEED_DELETE) {
            ESP_LOGW(TAG, "Player deleted");
            xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
            vTaskDeleteWithCaps(NULL);
        }

        if (bits & WAIT_STOP) {
            xEventGroupSetBits(ctx->events.event_group, WAIT_STOP_DONE);
        }

        // Check for new events in queue
        if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
            run_ctx.action = player_event.action;
            run_ctx.run_start = ctx->run_start;
            run_ctx.run_end = ctx->run_end;
            run_ctx.repeat = ctx->repeat;
            run_ctx.fps = ctx->fps;
            ESP_LOGD(TAG, "Player updated [%s]: %d -> %d, repeat:%d, fps:%d",
                     run_ctx.action == PLAYER_ACTION_START ? "START" : "STOP",
                     run_ctx.run_start, run_ctx.run_end, run_ctx.repeat, run_ctx.fps);
        }

        if (run_ctx.action == PLAYER_ACTION_STOP) {
            continue;
        }

        // Process animation frames
        do {
            for (int i = run_ctx.run_start; (i <= run_ctx.run_end) && (run_ctx.action != PLAYER_ACTION_STOP); i++) {
                // Frame rate control
                int64_t elapsed = esp_timer_get_time() - run_ctx.last_frame_time;
                elapsed = elapsed / 1000;
                if (elapsed < FPS_TO_MS(run_ctx.fps)) {
                    vTaskDelay(pdMS_TO_TICKS(FPS_TO_MS(run_ctx.fps) - elapsed));
                    ESP_LOGD(TAG, "delay: %d ms", (int)(FPS_TO_MS(run_ctx.fps) - elapsed));
                    // ESP_LOGW(TAG, "%d, delay: %d ms, fps: %d, MS: %d ms, elapsed: %d ms", i, (int)(FPS_TO_MS(run_ctx.fps) - elapsed), run_ctx.fps, FPS_TO_MS(run_ctx.fps), (int)elapsed);
                } else {
                    vTaskDelay(pdMS_TO_TICKS(1));
                }
                run_ctx.last_frame_time = esp_timer_get_time();

                // Check for new events or delete request
                bits = xEventGroupWaitBits(ctx->events.event_group,
                                           NEED_DELETE | WAIT_STOP,
                                           pdTRUE, pdFALSE, pdMS_TO_TICKS(0));
                if (bits & NEED_DELETE) {
                    ESP_LOGW(TAG, "Playing deleted");
                    xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
                    vTaskDelete(NULL);
                }
                if (bits & WAIT_STOP) {
                    xEventGroupSetBits(ctx->events.event_group, WAIT_STOP_DONE);
                }

                if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
                    run_ctx.action = player_event.action;
                    run_ctx.run_start = ctx->run_start;
                    run_ctx.run_end = ctx->run_end;
                    run_ctx.fps = ctx->fps;
                    if (run_ctx.action == PLAYER_ACTION_STOP) {
                        run_ctx.repeat = false;
                    } else {
                        run_ctx.repeat = ctx->repeat;
                    }

                    ESP_LOGD(TAG, "Playing updated [%s]: %d -> %d, repeat:%d, fps:%d",
                             run_ctx.action == PLAYER_ACTION_START ? "START" : "STOP",
                             run_ctx.run_start, run_ctx.run_end, run_ctx.repeat, run_ctx.fps);
                    break;
                }

                const void *frame_data = anim_vfs_get_frame_data(ctx->info.file_desc, i);
                size_t frame_size = anim_vfs_get_frame_size(ctx->info.file_desc, i);

                image_format_t format = anim_dec_parse_header(frame_data, frame_size, &header);

                ctx->screen_w = header.width;
                ctx->screen_h = header.height;

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
        } while (run_ctx.repeat);

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

    anim_player_event_t player_event = {
        .action = event,
    };

    if (xQueueSend(ctx->events.event_queue, &player_event, pdMS_TO_TICKS(10)) != pdTRUE) {
        ESP_LOGE(TAG, "Failed to send event to queue");
    }
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
    ctx->run_start = ctx->info.start;
    ctx->run_end = ctx->info.end;
    ctx->repeat = true;
    ctx->fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;

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

    ctx->run_start = start;
    ctx->run_end = end;
    ctx->repeat = repeat;
    ctx->fps = fps;
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
    player->run_start = 0;
    player->run_end = 0;
    player->repeat = false;
    player->fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;
    player->flush_cb = config->flush_cb;
    player->update_cb = config->update_cb;
    player->user_data = config->user_data;
    player->flags.swap = config->flags.swap;
    player->events.event_group = xEventGroupCreate();
    player->events.event_queue = xQueueCreate(5, sizeof(anim_player_event_t));
    player->child_list = NULL;

    // Set default task configuration if not specified
    const uint32_t caps = config->task.task_stack_caps ? config->task.task_stack_caps : MALLOC_CAP_DEFAULT; // caps cannot be zero
    if (config->task.task_affinity < 0) {
        xTaskCreateWithCaps(anim_player_task, "Anim Player", config->task.task_stack, player, config->task.task_priority, &player->handle_task, caps);
    } else {
        xTaskCreatePinnedToCoreWithCaps(anim_player_task, "Anim Player", config->task.task_stack, player, config->task.task_priority, &player->handle_task, config->task.task_affinity, caps);
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
    child_t *current = ctx->child_list;
    while (current != NULL) {
        child_t *next = current->next;
        free(current);
        current = next;
    }
    ctx->child_list = NULL;

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

    // Delete event queue
    if (ctx->events.event_queue) {
        vQueueDelete(ctx->events.event_queue);
        ctx->events.event_queue = NULL;
    }

    if (ctx->info.file_desc) {
        anim_vfs_deinit(ctx->info.file_desc);
        ctx->info.file_desc = NULL;
    }

    // Free player context
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
    child_t *new_child = (child_t *)malloc(sizeof(child_t));
    if (new_child == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for child");
        return ESP_ERR_NO_MEM;
    }

    // Initialize child node
    new_child->type = type;
    new_child->src = src;
    new_child->next = NULL;

    // Add to the end of the list
    if (ctx->child_list == NULL) {
        ctx->child_list = new_child;
    } else {
        child_t *current = ctx->child_list;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_child;
    }

    ESP_LOGI(TAG, "Added child(%p): type=%d, src=%p", new_child, new_child->type, new_child->src);
    return ESP_OK;
}
