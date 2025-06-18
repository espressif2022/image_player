/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t     label_opa_t;   /*!< Type for label opacity, range from 0 (fully transparent) to 255 (fully opaque) */
typedef int16_t     label_coord_t; /*!< Type for label coordinates, supports negative values for positioning */
typedef uint32_t    label_color_t; /*!< Type for label color, typically in 0xRRGGBB format */

/**
 * @brief Macro to cast a color value to label_color_t type
 */
#define FT_COLOR_HEX(color) ((label_color_t)color)

/**
 * @brief config for initializing font labels
 */
typedef struct {
    const char * name;  /* The name of the font file */
    const void * mem;   /* The pointer to the font file */
    size_t mem_size;    /* The size of the memory */
} ft_label_cfg_t;

/**
 * @brief config for blending area, used for rendering text
 */
typedef struct {
    uint8_t * buf_area; /* Buffer area for blending */
    uint16_t  width;    /* Width of the blending area */
    uint16_t  height;   /* Height of the blending area */
} ft_blend_area_t;

/**
 * @brief Type of font handle
 */
typedef void *ft_font_handle_t;

/**
 * @brief Type of library handle
 */
typedef void *ft_lib_handle_t;

/**
 * @brief Create a new font library
 *
 * This function initializes the FreeType library and creates a font manager structure.
 *
 * @param ret_lib Pointer to the handle of the created font library
 *
 * @return
 *      - ESP_OK: Success
 *      - ESP_ERR_NO_MEM: Not enough memory
 *      - ESP_FAIL: FreeType library initialization failed
 */
esp_err_t ft_library_create(ft_lib_handle_t *ret_lib);

/**
 * @brief Clean up the font library
 *
 * This function cleans up the font manager structure, releases all fonts, and deinitializes the FreeType library.
 *
 * @param lib_handle Handle of the font library to clean up
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_library_cleanup(ft_lib_handle_t lib_handle);

/**
 * @brief Create a new font label
 *
 * This function creates a new font label using the provided font file information and configuration.
 *
 * @param lib_handle Handle of the font library
 * @param cfg Pointer to the configuration structure containing font file information
 * @param ret_handle Pointer to the handle of the created font label
 *
 * @return
 *      - ESP_OK: Success
 *      - ESP_ERR_NO_MEM: Not enough memory
 *      - ESP_FAIL: Font creation failed
 */
esp_err_t ft_label_new_font(ft_lib_handle_t lib_handle, const ft_label_cfg_t *cfg, ft_font_handle_t *ret_handle);

/**
 * @brief Delete a font label
 *
 * @param handle Font label handle
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_del_font(ft_font_handle_t handle);

/**
 * @brief Set the font size
 *
 * @param handle Font label handle
 * @param font_size Font size in pixels
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_font_size(ft_font_handle_t handle, uint8_t font_size);

/**
 * @brief Set the font opacity
 *
 * This function sets the opacity of the font label. The opacity value should be between 0 and 255,
 * where 0 is fully transparent and 255 is fully opaque.
 *
 * @param handle Font label handle
 * @param opa Opacity value (0-255)
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_opa(ft_font_handle_t handle, label_opa_t opa);

/**
 * @brief Set the font color
 *
 * @param handle Font label handle
 * @param color Color value
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_color(ft_font_handle_t handle, label_color_t color);

/**
 * @brief Set the X coordinate of the font label
 *
 * @param handle Font label handle
 * @param x X coordinate
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_x(ft_font_handle_t handle, label_coord_t x);

/**
 * @brief Set the Y coordinate of the font label
 *
 * @param handle Font label handle
 * @param y Y coordinate
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_y(ft_font_handle_t handle, label_coord_t y);

/**
 * @brief Set the position of the font label
 *
 * @param handle Font label handle
 * @param x X coordinate
 * @param y Y coordinate
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_pos(ft_font_handle_t handle, label_coord_t x, label_coord_t y);

/**
 * @brief Set the width of the font label
 *
 * @param handle Font label handle
 * @param w Width
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_width(ft_font_handle_t handle, int16_t w);

/**
 * @brief Set the height of the font label
 *
 * @param handle Font label handle
 * @param h Height
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_height(ft_font_handle_t handle, int16_t h);

/**
 * @brief Set the size of the font label
 *
 * @param handle Font label handle
 * @param w Width
 * @param h Height
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_size(ft_font_handle_t handle, int16_t w, int16_t h);

/**
 * @brief Set the text content of the font label
 *
 * @param handle Font label handle
 * @param text Text content
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_set_text(ft_font_handle_t handle, const char * text);

/**
 * @brief Set the text for a font label using formatted input.
 *
 * @param handle Font handle, must be valid.
 * @param fmt Format string for the text.
 * @param ... Additional arguments for the format string.
 * @return esp_err_t ESP_OK on success, ESP_ERR_INVALID_ARG if handle or fmt is invalid,
 *         ESP_ERR_NO_MEM if memory allocation fails.
 */
esp_err_t ft_label_set_text_fmt(ft_font_handle_t handle, const char * fmt, ...);

/**
 * @brief Render the text of the font label
 *
 * @param handle Font label handle
 * @param blend_cfg Blending area configuration
 *
 * @return
 *      - ESP_OK: Success
 *      - Other error codes
 */
esp_err_t ft_label_render_text(ft_font_handle_t handle, ft_blend_area_t *blend_cfg);

#ifdef __cplusplus
}
#endif
