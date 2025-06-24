/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gfx_types.h"
#include "anim_player.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************
 *      DEFINES
 *********************/

/* Font handle type - hides internal FreeType implementation */
typedef void *gfx_font_t;

/* Label configuration structure */
typedef struct {
    const char * name;      /**< The name of the font file */
    const void * mem;       /**< The pointer to the font file */
    size_t mem_size;        /**< The size of the memory */
} gfx_label_cfg_t;

/**********************
 * GLOBAL PROTOTYPES
 **********************/

/*=====================
 * Object creation
 *====================*/

/**
 * @brief Create an image object
 * @param handle Animation player handle
 * @return Pointer to the created image object
 */
gfx_obj_t * gfx_img_create(anim_player_handle_t handle);

/**
 * @brief Create a label object
 * @param handle Animation player handle
 * @param cfg Font configuration
 * @return Pointer to the created label object

 */
gfx_obj_t * gfx_label_create(anim_player_handle_t handle);

/**
 * @brief Create a new font
 * @param handle Animation player handle
 * @param cfg Font configuration
 * @param ret_font Pointer to store the font handle
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_new_font(anim_player_handle_t handle, const gfx_label_cfg_t *cfg, gfx_font_t *ret_font);

/*=====================
 * Setter functions
 *====================*/

/**
 * @brief Set the source data for an image object
 * @param obj Pointer to the image object
 * @param src Pointer to the image source data
 * @return Pointer to the object
 */
gfx_obj_t * gfx_img_set_src(gfx_obj_t *obj, void *src);

/**
 * @brief Set the text for a label object
 * @param obj Pointer to the label object
 * @param text Text string to display
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_set_text(gfx_obj_t *obj, const char *text);

/**
 * @brief Set the text for a label object
 * @param obj Pointer to the label object
 * @param fmt Format string
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_set_text_fmt(gfx_obj_t * obj, const char * fmt, ...);

/*=====================
 * Label property setters
 *====================*/

/**
 * @brief Set the color for a label object
 * @param obj Pointer to the label object
 * @param color Color value
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_set_color(gfx_obj_t *obj, gfx_color_t color);

/**
 * @brief Set the opacity for a label object
 * @param obj Pointer to the label object
 * @param opa Opacity value (0-255)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_set_opa(gfx_obj_t *obj, gfx_opa_t opa);

/**
 * @brief Set the font size for a label object
 * @param obj Pointer to the label object
 * @param font_size Font size in points
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t gfx_label_set_font_size(gfx_obj_t *obj, uint8_t font_size);

/**
 * @brief Set the font for a label object
 * @param obj Pointer to the label object
 * @param font Font handle
 */
esp_err_t gfx_label_set_font(gfx_obj_t *obj, gfx_font_t font);

/**
 * @brief Set the position of an object
 * @param obj Pointer to the object
 * @param x X coordinate
 * @param y Y coordinate
 */
void gfx_obj_set_pos(gfx_obj_t *obj, uint16_t x, uint16_t y);

/**
 * @brief Set the size of an object
 * @param obj Pointer to the object
 * @param w Width
 * @param h Height
 */
void gfx_obj_set_size(gfx_obj_t *obj, uint16_t w, uint16_t h);

/*=====================
 * Getter functions
 *====================*/

/**
 * @brief Get the position of an object
 * @param obj Pointer to the object
 * @param x Pointer to store X coordinate
 * @param y Pointer to store Y coordinate
 */
void gfx_obj_get_pos(gfx_obj_t *obj, uint16_t *x, uint16_t *y);

/**
 * @brief Get the size of an object
 * @param obj Pointer to the object
 * @param w Pointer to store width
 * @param h Pointer to store height
 */
void gfx_obj_get_size(gfx_obj_t *obj, uint16_t *w, uint16_t *h);

/*=====================
 * Other functions
 *====================*/

/**
 * @brief Delete an object
 * @param obj Pointer to the object to delete
 */
void gfx_obj_delete(gfx_obj_t *obj);

#ifdef __cplusplus
}
#endif 