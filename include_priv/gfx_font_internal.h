/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "esp_err.h"
#include "gfx_types.h"
#include "gfx_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef void *ft_font_handle_t;
typedef void *ft_lib_handle_t;

typedef struct {
    void *face;           /*!< FreeType font face object */
    uint16_t font_size;   /*!< Size of the font in points */
    gfx_opa_t opa;      /*!< Opacity of the label */
    gfx_color_t color;  /*!< Color of the label */

    char *text;           /*!< Text content of the label */
    gfx_coord_t x;      /*!< X coordinate of the label's position */
    gfx_coord_t y;      /*!< Y coordinate of the label's position */
    uint16_t width;       /*!< Width of the label */
    uint16_t height;      /*!< Height of the label */
    uint8_t *mask;
} gfx_label_property_t;

typedef struct face_entry {
    void *face;
    const void *mem;
    struct face_entry *next;
} ft_face_entry_t;

typedef struct {
    ft_face_entry_t *ft_face_head;
    void *ft_library;
} ft_library_t;

// Internal function declarations
esp_err_t gfx_ft_lib_create(ft_lib_handle_t *ret_lib);
esp_err_t gfx_ft_lib_cleanup(ft_lib_handle_t lib_handle);

esp_err_t gfx_sw_draw_label(gfx_obj_t * obj);

/**
 * @brief Create a new font
 * @param handle Animation player handle
 * @param cfg Font configuration
 * @param ret_handle Pointer to store the font handle
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_new_font(anim_player_handle_t handle, const gfx_label_cfg_t *cfg, ft_font_handle_t *ret_handle);

#ifdef __cplusplus
}
#endif