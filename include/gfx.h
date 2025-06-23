/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file gfx.h
 * @brief Graphics Framework (GFX) - Main header file
 * 
 * This header file includes all the public APIs for the GFX framework.
 * The framework provides:
 * - Object system for images and labels
 * - Drawing functions for rendering to buffers
 * - Color utilities and type definitions
 * - Software blending capabilities
 */

#include "gfx_types.h"
#include "gfx_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************
 *      DEFINES
 *********************/

/**********************
 *      TYPEDEFS
 **********************/

/**********************
 * GLOBAL PROTOTYPES
 **********************/

/*=====================
 * Initialization
 *====================*/

/**
 * @brief Initialize the GFX framework
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_init(void);

/**
 * @brief Deinitialize the GFX framework
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_deinit(void);

#ifdef __cplusplus
}
#endif 