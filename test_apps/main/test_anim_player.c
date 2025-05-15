#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "unity.h"
#include "unity_test_utils.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "bsp/esp-bsp.h"

#include "anim_player.h"
#include "mmap_generate_test_4bit.h"
#include "mmap_generate_test_8bit.h"

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

static void flush_callback(int x, int y, int width, int height, const void *data)
{
    ESP_LOGI(TAG, "Flush: (%03d,%03d) (%03d,%03d)", x, y, width, height);
    esp_lcd_panel_draw_bitmap(panel_handle, x, y, width, height, data);
    anim_player_flush_ready(handle);
}

static void test_anim_player_common(const char *partition_label, uint32_t max_files, uint32_t checksum,
                                    uint32_t start_file, uint32_t end_file, uint32_t delay_ms)
{
    const bsp_display_config_t bsp_disp_cfg = {
        .max_transfer_sz = (240 * 10) * sizeof(uint16_t),
    };
    bsp_display_new(&bsp_disp_cfg, &panel_handle, &io_handle);

    esp_lcd_panel_disp_on_off(panel_handle, true);
    bsp_display_brightness_init();
    bsp_display_backlight_on();

    anim_player_config_t config = {
        .partition_label = partition_label,
        .max_files = max_files,
        .checksum = checksum,
        .flush_cb = flush_callback
    };

    handle = anim_player_init(&config);
    TEST_ASSERT_NOT_NULL(handle);

    // Test update
    anim_player_update(handle, PLAYER_ACTION_START, start_file, end_file, true, 25);
    vTaskDelay(pdMS_TO_TICKS(1000 * delay_ms));

    // Test stop
    anim_player_update(handle, PLAYER_ACTION_STOP, 0, 0, false, 0);
    vTaskDelay(pdMS_TO_TICKS(1000 * 2));

    // Cleanup
    anim_player_deinit(handle);

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
    test_anim_player_common("assets_4bit", MMAP_TEST_4BIT_FILES, MMAP_TEST_4BIT_CHECKSUM,
                            MMAP_TEST_4BIT_B0001_SBMP, MMAP_TEST_4BIT_B0007_SBMP, 2);
}

TEST_CASE("test anim player init and deinit", "[anim_player][8bit]")
{
    test_anim_player_common("assets_8bit", MMAP_TEST_8BIT_FILES, MMAP_TEST_8BIT_CHECKSUM,
                            MMAP_TEST_8BIT_CONNECTING_0000_SBMP, MMAP_TEST_8BIT_CONNECTING_0009_SBMP, 20 * 1000);
}

void app_main(void)
{
    printf("Animation player test\n");
    unity_run_menu();
}