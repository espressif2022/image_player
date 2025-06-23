#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "unity.h"
#include "unity_test_utils.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "bsp/esp-bsp.h"
#include "esp_timer.h"
#include "esp_err.h"
#include "esp_check.h"

#include "anim_player.h"
#include "mmap_generate_test_4bit.h"
#include "mmap_generate_test_8bit.h"
#include "mmap_generate_spiffs_assets.h"

#include "gfx_types.h"
#include "gfx_obj.h"

static const char *TAG = "player";

#define TEST_MEMORY_LEAK_THRESHOLD  (500)

static size_t before_free_8bit;
static size_t before_free_32bit;

static anim_player_handle_t handle = NULL;
static esp_lcd_panel_io_handle_t io_handle = NULL;
static esp_lcd_panel_handle_t panel_handle = NULL;

void setUp(void)
{
    before_free_8bit = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    before_free_32bit = heap_caps_get_free_size(MALLOC_CAP_32BIT);
}

void tearDown(void)
{
    size_t after_free_8bit = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t after_free_32bit = heap_caps_get_free_size(MALLOC_CAP_32BIT);
    unity_utils_check_leak(before_free_8bit, after_free_8bit, "8BIT", TEST_MEMORY_LEAK_THRESHOLD);
    unity_utils_check_leak(before_free_32bit, after_free_32bit, "32BIT", TEST_MEMORY_LEAK_THRESHOLD);
}

static bool flush_io_ready(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_io_event_data_t *edata, void *user_ctx)
{
    anim_player_handle_t handle = (anim_player_handle_t)user_ctx;
    anim_player_flush_ready(handle);
    return true;
}

static void flush_callback(anim_player_handle_t handle, int x1, int y1, int x2, int y2, const void *data)
{
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t)anim_player_get_user_data(handle);
    // if(y1 == 0) {
        // ESP_LOGI(TAG, "Flush: (%03d,%03d) (%03d,%03d)", x1, y1, x2, y2);
    // }
    esp_lcd_panel_draw_bitmap(panel, x1, y1, x2, y2, data);
    anim_player_flush_ready(handle);
    // ESP_LOGI(TAG, "Flush done");
}

extern const lv_image_dsc_t icon1;
extern const lv_image_dsc_t icon2;
extern const lv_image_dsc_t icon3;
extern const lv_image_dsc_t icon4;
extern const lv_image_dsc_t icon5;
extern const lv_image_dsc_t icon5_new;

static void update_callback(anim_player_handle_t handle, player_event_t event)
{
    static uint32_t start_time = 0;
    static int total_frames = 0;

    switch (event) {
    case PLAYER_EVENT_IDLE:
        ESP_LOGI(TAG, "Event: IDLE");
        break;
    case PLAYER_EVENT_ONE_FRAME_DONE:
        // ESP_LOGI(TAG, "Event: ONE_FRAME_DONE");
        if (start_time == 0) {
            start_time = esp_timer_get_time();
        }
        total_frames++;
        break;
    case PLAYER_EVENT_ALL_FRAME_DONE:
        uint32_t end_time = esp_timer_get_time();
        float duration_sec = (end_time - start_time) / 1000000.0f;
        float fps = (total_frames - 1) / duration_sec;
        ESP_LOGI(TAG, "Event: ALL_FRAME_DONE - FPS: %.2f (Frames: %d, Duration: %.2fs)",
                 fps, total_frames, duration_sec);
        // Reset counters for next playback
        start_time = 0;
        total_frames = 0;
        break;
    default:
        ESP_LOGI(TAG, "Event: UNKNOWN");
        break;
    }
}

static void test_anim_player_common(const char *partition_label, uint32_t max_files, uint32_t checksum, uint32_t delay_ms)
{
    mmap_assets_handle_t assets_handle = NULL;
    const mmap_assets_config_t asset_config = {
        .partition_label = partition_label,
        .max_files = max_files,
        .checksum = checksum,
        .flags = {.mmap_enable = true, .full_check = true}
    };

    esp_err_t ret = mmap_assets_new(&asset_config, &assets_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize assets");
        return;
    }

    const bsp_display_config_t bsp_disp_cfg = {
        .max_transfer_sz = (240 * 10) * sizeof(uint16_t),
    };
    bsp_display_new(&bsp_disp_cfg, &panel_handle, &io_handle);

    esp_lcd_panel_disp_on_off(panel_handle, true);
    bsp_display_brightness_init();
    bsp_display_backlight_on();

    anim_player_config_t config = {
        .flush_cb = flush_callback,
        .update_cb = update_callback,
        .user_data = panel_handle,
        .flags = {
            .swap = true,
        },
        .task = ANIM_PLAYER_INIT_CONFIG()
    };
    // config.task.task_stack_caps = MALLOC_CAP_INTERNAL;
    config.task.task_stack_caps = MALLOC_CAP_DEFAULT;
    config.task.task_affinity = 1;
    config.task.task_priority = 7;
    config.task.task_stack = 20*1024;

    mmap_assets_handle_t assets_font = NULL;
    const mmap_assets_config_t asset_config_font = {
        .partition_label = "assets",
        .max_files = MMAP_SPIFFS_ASSETS_FILES,
        .checksum = MMAP_SPIFFS_ASSETS_CHECKSUM,
        .flags = {.mmap_enable = true, .full_check = true}
    };

    ret = mmap_assets_new(&asset_config_font, &assets_font);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize assets");
        return;
    }

    handle = anim_player_init(&config);

    gfx_label_cfg_t font_config;

    font_config.name = "DejaVuSans.ttf";
    font_config.mem = mmap_assets_get_mem(assets_font, MMAP_SPIFFS_ASSETS_DEJAVUSANS_TTF);
    font_config.mem_size = mmap_assets_get_size(assets_font, MMAP_SPIFFS_ASSETS_DEJAVUSANS_TTF);

    gfx_obj_t *label1 = gfx_label_create(handle, &font_config);
    gfx_obj_set_pos(label1, 10, 170);
    gfx_obj_set_size(label1, 200, 50);

    gfx_label_set_color(label1, GFX_COLOR_HEX(0xFF0000));
    gfx_label_set_opa(label1, 0xFF);
    gfx_label_set_font_size(label1, 20);
    gfx_label_set_text(label1, "ABCD");

    gfx_obj_t *image1 = gfx_img_create(handle);
    gfx_obj_set_pos(image1, 20, 100);
    gfx_img_set_src(image1, (void *)&icon5_new);

    const esp_lcd_panel_io_callbacks_t cbs = {
        .on_color_trans_done = flush_io_ready,
    };
    esp_lcd_panel_io_register_event_callbacks(io_handle, &cbs, handle);

    uint32_t start, end;
    const void *src_data;
    size_t src_len;

    for (int i = 0; i < mmap_assets_get_stored_files(assets_handle); i++) {

        i = MMAP_TEST_8BIT_OUTPUT_AAF;

        src_data = mmap_assets_get_mem(assets_handle, i);
        src_len = mmap_assets_get_size(assets_handle, i);

        ESP_LOGW(TAG, "set src, %s", mmap_assets_get_name(assets_handle, i));
        anim_player_set_src_data(handle, src_data, src_len);
        anim_player_get_segment(handle, &start, &end);
        // anim_player_set_segment(handle, start, end, 50, true);
        anim_player_set_segment(handle, start, end, 5, true);
        ESP_LOGW(TAG, "start:%" PRIu32 ", end:%" PRIu32 "", start, end);

        anim_player_update(handle, PLAYER_ACTION_START);
        vTaskDelay(pdMS_TO_TICKS(1000 * delay_ms));

        anim_player_update(handle, PLAYER_ACTION_STOP);
        vTaskDelay(pdMS_TO_TICKS(1000 * delay_ms));
    }

    ESP_LOGI(TAG, "test done");

    if (handle) {
        anim_player_deinit(handle);
        handle = NULL;
    }

    if (assets_handle) {
        mmap_assets_del(assets_handle);
        assets_handle = NULL;
    }

    if (panel_handle) {
        esp_lcd_panel_del(panel_handle);
    }
    if (io_handle) {
        esp_lcd_panel_io_del(io_handle);
    }
    spi_bus_free(BSP_LCD_SPI_NUM);

    vTaskDelay(pdMS_TO_TICKS(1000));
}

TEST_CASE("test anim player init and deinit", "[anim_player][4bit]")
{
    test_anim_player_common("assets_4bit", MMAP_TEST_4BIT_FILES, MMAP_TEST_4BIT_CHECKSUM, 5);
}

TEST_CASE("test anim player init and deinit", "[anim_player][8bit]")
{
    test_anim_player_common("assets_8bit", MMAP_TEST_8BIT_FILES, MMAP_TEST_8BIT_CHECKSUM, 5);
}

void app_main(void)
{
    printf("Animation player test\n");
    // unity_run_menu();

    test_anim_player_common("assets_8bit", MMAP_TEST_8BIT_FILES, MMAP_TEST_8BIT_CHECKSUM, 5000);
}