/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gfx.h"
#include "esp_log.h"
#include "gfx_font_internal.h"
#include "anim_player.h"

static const char *TAG = "gfx";

/*********************
 *      DEFINES
 *********************/

/**********************
 *      TYPEDEFS
 **********************/

/**********************
 *  STATIC PROTOTYPES
 **********************/

/**********************
 *   GLOBAL FUNCTIONS
 **********************/

esp_err_t gfx_init(void)
{
    ESP_LOGI(TAG, "GFX framework initialized");
    return ESP_OK;
}

esp_err_t gfx_deinit(void)
{
    ESP_LOGI(TAG, "GFX framework deinitialized");
    return ESP_OK;
}