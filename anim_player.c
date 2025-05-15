#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include "esp_err.h"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_mmap_assets.h"
#include "anim_player.h"

static const char *TAG = "anim_player";

#define NEED_DELETE     BIT0
#define DELETE_DONE     BIT1
#define WAIT_FLUSH_DONE BIT2

#define FPS_TO_MS(fps) (1000 / (fps))  // Convert FPS to milliseconds

typedef struct {
    player_event_t action;
    int start_index;
    int end_index;
    bool repeat;
    int fps;
} anim_player_event_t;

typedef struct {
    EventGroupHandle_t event_group;
    QueueHandle_t event_queue;
} anim_player_events_t;

// Image format types
typedef enum {
    IMAGE_FORMAT_SBMP = 0,  // Split BMP format
    IMAGE_FORMAT_REDIRECT = 1,  // Redirect format
    IMAGE_FORMAT_INVALID = 2
} image_format_t;

// Image header structure
typedef struct {
    char format[3];        // Format identifier (e.g., "_S")
    char version[6];       // Version string
    uint8_t bit_depth;     // Bit depth (4 or 8)
    uint16_t width;        // Image width
    uint16_t height;       // Image height
    uint16_t splits;       // Number of splits
    uint16_t split_height; // Height of each split
    uint16_t *split_lengths; // Data length of each split
    uint16_t data_offset;  // Offset to data segment
    uint8_t *palette;      // Color palette (dynamically allocated)
    int num_colors;        // Number of colors in palette
} image_header_t;

// Animation player context
typedef struct {
    player_event_t action;
    int start_index;
    int end_index;
    bool repeat;
    int fps;            // Frame rate (frames per second)
    mmap_assets_handle_t assets_handle;
    flush_cb_t flush_callback;
    anim_player_events_t events;
    TaskHandle_t handle_task;
    uint32_t last_frame_time;
} anim_player_context_t;

static image_format_t anim_decoder_parse_header(const uint8_t *data, size_t data_len, image_header_t *header)
{
    // Initialize header fields
    memset(header, 0, sizeof(image_header_t));

    // Read format identifier
    memcpy(header->format, data, 2);
    header->format[2] = '\0';

    if (strncmp(header->format, "_S", 2) == 0) {
        // Parse format
        memcpy(header->version, data + 3, 6);

        // Read bit depth
        header->bit_depth = data[9];

        // Validate bit depth
        if (header->bit_depth != 4 && header->bit_depth != 8) {
            ESP_LOGE(TAG, "Invalid bit depth: %d", header->bit_depth);
            return IMAGE_FORMAT_INVALID;
        }

        header->width = *(uint16_t *)(data + 10);
        header->height = *(uint16_t *)(data + 12);
        header->splits = *(uint16_t *)(data + 14);
        header->split_height = *(uint16_t *)(data + 16);

        // Allocate and read split lengths
        header->split_lengths = (uint16_t *)malloc(header->splits * sizeof(uint16_t));
        if (header->split_lengths == NULL) {
            ESP_LOGE(TAG, "Failed to allocate memory for split lengths");
            return IMAGE_FORMAT_INVALID;
        }

        for (int i = 0; i < header->splits; i++) {
            header->split_lengths[i] = *(uint16_t *)(data + 18 + i * 2);
        }

        // Calculate number of colors based on bit depth
        header->num_colors = 1 << header->bit_depth;

        // Allocate and read color palette
        header->palette = (uint8_t *)malloc(header->num_colors * 4);
        if (header->palette == NULL) {
            ESP_LOGE(TAG, "Failed to allocate memory for palette");
            free(header->split_lengths);
            header->split_lengths = NULL;
            return IMAGE_FORMAT_INVALID;
        }

        // Read palette data
        memcpy(header->palette, data + 18 + header->splits * 2, header->num_colors * 4);

        header->data_offset = 18 + header->splits * 2 + header->num_colors * 4;
        return IMAGE_FORMAT_SBMP;

    } else if (strncmp(header->format, "_R", 2) == 0) {
        // Parse redirect format
        uint8_t file_length = *(uint8_t *)(data + 2);

        // For redirect format, we'll use the palette field to store the filename
        header->palette = (uint8_t *)malloc(file_length + 1);
        if (header->palette == NULL) {
            ESP_LOGE(TAG, "Failed to allocate memory for redirect filename");
            return IMAGE_FORMAT_INVALID;
        }

        // Copy filename to palette buffer
        memcpy(header->palette, data + 3, file_length);
        header->palette[file_length] = '\0';  // Ensure null termination
        header->num_colors = file_length + 1;

        return IMAGE_FORMAT_REDIRECT;

    } else {
        ESP_LOGE(TAG, "Invalid format: %s", header->format);
        printf("%02X %02X %02X\r\n", header->format[0], header->format[1], header->format[2]);
        return IMAGE_FORMAT_INVALID;
    }
}

static void anim_decoder_calculate_offsets(const image_header_t *header, uint16_t *offsets)
{
    offsets[0] = header->data_offset;
    for (int i = 1; i < header->splits; i++) {
        offsets[i] = offsets[i - 1] + header->split_lengths[i - 1];
    }
}

static void anim_decoder_free_header(image_header_t *header)
{
    if (header->split_lengths != NULL) {
        free(header->split_lengths);
        header->split_lengths = NULL;
    }
    if (header->palette != NULL) {
        free(header->palette);
        header->palette = NULL;
    }
}

static esp_err_t anim_decoder_rle_decode(const uint8_t *input, size_t input_len, uint8_t *output, size_t output_len)
{
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (in_pos + 1 <= input_len) {
        uint8_t count = input[in_pos++];
        uint8_t value = input[in_pos++];

        if (out_pos + count > output_len) {
            ESP_LOGE(TAG, "Output buffer overflow");
            return ESP_FAIL;
        }

        for (uint8_t i = 0; i < count; i++) {
            output[out_pos++] = value;
        }
    }

    return ESP_OK;
}

static inline uint16_t anim_decoder_rgb888_to_rgb565(uint32_t color)
{
    return (((color >> 16) & 0xF8) << 8) | (((color >> 8) & 0xFC) << 3) | ((color & 0xF8) >> 3);
}

static uint32_t get_color_from_palette(const image_header_t *header, uint8_t index)
{
    const uint8_t *color = &header->palette[index * 4];
    return (color[0] << 16) | (color[1] << 8) | color[2];
}

static esp_err_t anim_decoder_parse_image(const uint8_t *data, size_t data_len, image_header_t *header, anim_player_context_t *ctx)
{
    // Allocate memory for split offsets
    uint16_t *offsets = (uint16_t *)malloc(header->splits * sizeof(uint16_t));
    if (offsets == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for offsets");
        return ESP_FAIL;
    }

    anim_decoder_calculate_offsets(header, offsets);

    // Allocate frame buffer
    void *frame_buffer = malloc(header->width * header->split_height * sizeof(uint16_t));
    if (frame_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for frame buffer");
        free(offsets);
        return ESP_FAIL;
    }

    // Allocate decode buffer
    uint8_t *decode_buffer  = NULL;
    if (header->bit_depth == 4) {
        decode_buffer = (uint8_t *)malloc(header->width * header->split_height / 2);
    } else if (header->bit_depth == 8) {
        decode_buffer = (uint8_t *)malloc(header->width * header->split_height);
    }
    if (decode_buffer == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for decode buffer");
        free(offsets);
        free(frame_buffer);
        return ESP_FAIL;
    }

    uint16_t *pixels = (uint16_t *)frame_buffer;

    // Process each split
    for (int split = 0; split < header->splits; split++) {
        const uint8_t *compressed_data = data + offsets[split];
        int compressed_len = header->split_lengths[split];

        // Decode compressed data
        if (anim_decoder_rle_decode(compressed_data, compressed_len, decode_buffer, header->width * header->split_height) != ESP_OK) {
            ESP_LOGE(TAG, "Failed to decode split %d", split);
            continue;
        }

        // Convert to RGB565 based on bit depth
        if (header->bit_depth == 4) {
            // 4-bit mode: each byte contains two pixels
            for (int y = 0; y < header->split_height; y++) {
                for (int x = 0; x < header->width; x += 2) {
                    uint8_t packed_gray = decode_buffer[y * (header->width / 2) + (x / 2)];
                    uint8_t index1 = (packed_gray & 0xF0) >> 4;
                    uint8_t index2 = (packed_gray & 0x0F);

                    uint32_t color1 = get_color_from_palette(header, index1);
                    uint32_t color2 = get_color_from_palette(header, index2);

                    pixels[y * header->width + x] = __builtin_bswap16(anim_decoder_rgb888_to_rgb565(color1));
                    if (x + 1 < header->width) {
                        pixels[y * header->width + x + 1] = __builtin_bswap16(anim_decoder_rgb888_to_rgb565(color2));
                    }
                }
            }
        } else if (header->bit_depth == 8) {
            // 8-bit mode: each byte is one pixel
            for (int y = 0; y < header->split_height; y++) {
                for (int x = 0; x < header->width; x++) {
                    uint8_t index = decode_buffer[y * header->width + x];
                    uint32_t color = get_color_from_palette(header, index);
                    pixels[y * header->width + x] = __builtin_bswap16(anim_decoder_rgb888_to_rgb565(color));
                }
            }
        } else {
            ESP_LOGE(TAG, "Unsupported bit depth: %d", header->bit_depth);
            continue;
        }

        // Flush decoded data
        xEventGroupClearBits(ctx->events.event_group, WAIT_FLUSH_DONE);
        ctx->flush_callback(0, split * header->split_height, header->width, (split + 1) * header->split_height, pixels);
        xEventGroupWaitBits(ctx->events.event_group, WAIT_FLUSH_DONE, pdTRUE, pdFALSE, pdMS_TO_TICKS(20));
    }

    // Cleanup
    free(offsets);
    free(frame_buffer);
    free(decode_buffer);
    anim_decoder_free_header(header);

    return ESP_OK;
}

static int16_t anim_decoder_find_asset(mmap_assets_handle_t handle, const char *name)
{
    for (int i = 0; i < mmap_assets_get_stored_files(handle); i++) {
        const void *asset_name = mmap_assets_get_name(handle, i);
        if (strcmp(asset_name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void anim_player_task(void *arg)
{
    image_header_t header;
    anim_player_context_t *ctx = (anim_player_context_t *)arg;
    ctx->last_frame_time = xTaskGetTickCount();
    anim_player_event_t player_event;

    while (1) {
        EventBits_t bits = xEventGroupWaitBits(ctx->events.event_group,
                                             NEED_DELETE,
                                             pdTRUE, pdFALSE, pdMS_TO_TICKS(10));

        if (bits & NEED_DELETE) {
            ESP_LOGW(TAG, "Player deleted");
            xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
            vTaskDelete(NULL);
        }

        // Check for new events in queue
        if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
            ctx->action = player_event.action;
            ctx->start_index = player_event.start_index;
            ctx->end_index = player_event.end_index;
            ctx->repeat = player_event.repeat;
            ctx->fps = player_event.fps;
            ESP_LOGI(TAG, "Player updated: %d -> %d, repeat:%d, fps:%d",
                     ctx->start_index, ctx->end_index, ctx->repeat, ctx->fps);
        }

        if (ctx->action == PLAYER_ACTION_STOP) {
            continue;
        }

        // Process animation frames
        do {
            for (int i = ctx->start_index; (i <= ctx->end_index) && (ctx->action != PLAYER_ACTION_STOP); i++) {
                // Frame rate control
                uint32_t current_time = xTaskGetTickCount();
                uint32_t elapsed = current_time - ctx->last_frame_time;
                if (elapsed < pdMS_TO_TICKS(FPS_TO_MS(ctx->fps))) {
                    vTaskDelay(pdMS_TO_TICKS(FPS_TO_MS(ctx->fps)) - elapsed);
                }
                ctx->last_frame_time = xTaskGetTickCount();

                const void *frame_data = mmap_assets_get_mem(ctx->assets_handle, i);
                size_t frame_size = mmap_assets_get_size(ctx->assets_handle, i);

                image_format_t format = anim_decoder_parse_header(frame_data, frame_size, &header);

                if (format == IMAGE_FORMAT_INVALID) {
                    ESP_LOGE(TAG, "Invalid frame format");
                    continue;
                } else if (format == IMAGE_FORMAT_REDIRECT) {
                    // Use the palette buffer as the filename
                    const char *name = (const char *)header.palette;
                    int16_t new_index = anim_decoder_find_asset(ctx->assets_handle, name);
                    free(header.palette);

                    if(new_index < mmap_assets_get_stored_files(ctx->assets_handle) && new_index >= 0){
                        frame_data = mmap_assets_get_mem(ctx->assets_handle, new_index);
                        frame_size = mmap_assets_get_size(ctx->assets_handle, new_index);
                        format = anim_decoder_parse_header(frame_data, frame_size, &header);

                        if (format == IMAGE_FORMAT_SBMP) {
                            anim_decoder_parse_image(frame_data, frame_size, &header, ctx);
                        }
                    }
                    continue;
                } else if (format == IMAGE_FORMAT_SBMP) {
                    anim_decoder_parse_image(frame_data, frame_size, &header, ctx);
                }

                // Check for new events or delete request
                bits = xEventGroupWaitBits(ctx->events.event_group,
                                         NEED_DELETE,
                                         pdTRUE, pdFALSE, pdMS_TO_TICKS(0));
                if (bits & NEED_DELETE) {
                    ESP_LOGW(TAG, "Player deleted");
                    xEventGroupSetBits(ctx->events.event_group, DELETE_DONE);
                    vTaskDelete(NULL);
                }

                if (xQueueReceive(ctx->events.event_queue, &player_event, 0) == pdTRUE) {
                    ctx->action = player_event.action;
                    ctx->start_index = player_event.start_index;
                    ctx->end_index = player_event.end_index;
                    ctx->repeat = player_event.repeat;
                    ctx->fps = player_event.fps;
                    ESP_LOGI(TAG, "Player updated: %d -> %d, repeat:%d, fps:%d",
                             ctx->start_index, ctx->end_index, ctx->repeat, ctx->fps);
                    break;
                }
            }
        } while (ctx->repeat);

        ctx->action = PLAYER_ACTION_STOP;
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

void anim_player_update(anim_player_handle_t handle, player_event_t event, int start_index, int end_index, bool repeat, int fps)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

    anim_player_event_t player_event = {
        .action = event,
        .start_index = start_index,
        .end_index = end_index,
        .repeat = repeat,
        .fps = fps
    };

    if (xQueueSend(ctx->events.event_queue, &player_event, pdMS_TO_TICKS(10)) != pdTRUE) {
        ESP_LOGE(TAG, "Failed to send event to queue");
    }
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

    mmap_assets_handle_t assets_handle;
    const mmap_assets_config_t asset_config = {
        .partition_label = config->partition_label,
        .max_files = config->max_files,
        .checksum = config->checksum,
        .flags = {.mmap_enable = true, .full_check = true}
    };

    esp_err_t ret = mmap_assets_new(&asset_config, &assets_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize assets");
        free(player);
        return NULL;
    }

    player->assets_handle = assets_handle;
    player->action = PLAYER_ACTION_STOP;
    player->start_index = 0;
    player->end_index = 0;
    player->repeat = false;
    player->fps = CONFIG_ANIM_PLAYER_DEFAULT_FPS;
    player->flush_callback = config->flush_cb;
    player->events.event_group = xEventGroupCreate();
    player->events.event_queue = xQueueCreate(5, sizeof(anim_player_event_t));

    xTaskCreatePinnedToCore(anim_player_task, "Anim Player", 4 * 1024, player, 5, &player->handle_task, 0);

    return (anim_player_handle_t)player;
}

void anim_player_deinit(anim_player_handle_t handle)
{
    anim_player_context_t *ctx = (anim_player_context_t *)handle;
    if (ctx == NULL) {
        ESP_LOGE(TAG, "Invalid player context");
        return;
    }

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

    // Free assets
    if (ctx->assets_handle) {
        mmap_assets_del(ctx->assets_handle);
        ctx->assets_handle = NULL;
    }

    // Free player context
    free(ctx);
}