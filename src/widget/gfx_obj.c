/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>
#include "esp_timer.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_check.h"
#include "anim_player.h"
#include "gfx_obj.h"
#include "gfx_types.h"

#include "gfx_font_internal.h"

static const char *TAG = "gfx_obj";

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

/*=====================
 * Object creation
 *====================*/

gfx_obj_t * gfx_img_create(anim_player_handle_t handle)
{
    gfx_obj_t *obj = (gfx_obj_t *)malloc(sizeof(gfx_obj_t));
    if (obj == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for image object");
        return NULL;
    }
    
    memset(obj, 0, sizeof(gfx_obj_t));
    obj->type = GFX_OBJ_TYPE_IMAGE;
    anim_player_add_child(handle, GFX_OBJ_TYPE_IMAGE, obj);
    ESP_LOGD(TAG, "Created image object");
    return obj;
}

gfx_obj_t * gfx_label_create(anim_player_handle_t handle, const gfx_label_cfg_t *font_cfg)
{
    gfx_obj_t *obj = (gfx_obj_t *)malloc(sizeof(gfx_obj_t));
    if (obj == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for label object");
        return NULL;
    }
    
    memset(obj, 0, sizeof(gfx_obj_t));
    obj->type = GFX_OBJ_TYPE_LABEL;
    
    // If font configuration is provided, create font
    if (font_cfg != NULL) {
        ft_font_handle_t font_handle;
        esp_err_t ret = gfx_label_new_font(handle, font_cfg, &font_handle);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to create font for label");
            free(obj);
            return NULL;
        }
        // Store font handle in object's src field
        obj->src = font_handle;
    }
    
    anim_player_add_child(handle, GFX_OBJ_TYPE_LABEL, obj);
    ESP_LOGD(TAG, "Created label object");
    return obj;
}

/*=====================
 * Setter functions
 *====================*/

gfx_obj_t * gfx_img_set_src(gfx_obj_t *obj, void *src)
{
    if (obj == NULL) {
        ESP_LOGE(TAG, "Object is NULL");
        return NULL;
    }
    
    if (obj->type != GFX_OBJ_TYPE_IMAGE) {
        ESP_LOGE(TAG, "Object is not an image type");
        return NULL;
    }
    
    obj->src = src;
    
    // Update object size based on image data
    if (src != NULL) {
        gfx_image_dsc_t *img = (gfx_image_dsc_t *)src;
        obj->width = img->header.w;
        obj->height = img->header.h;
    }
    
    ESP_LOGD(TAG, "Set image source, size: %dx%d", obj->width, obj->height);
    return obj;
}

void gfx_obj_set_pos(gfx_obj_t *obj, uint16_t x, uint16_t y)
{
    if (obj == NULL) {
        ESP_LOGE(TAG, "Object is NULL");
        return;
    }
    
    obj->x = x;
    obj->y = y;
    
    ESP_LOGD(TAG, "Set object position: (%d, %d)", x, y);
}

extern esp_err_t gfx_label_set_size(gfx_obj_t * obj, int16_t w, int16_t h);

void gfx_obj_set_size(gfx_obj_t *obj, uint16_t w, uint16_t h)
{
    if (obj == NULL) {
        ESP_LOGE(TAG, "Object is NULL");
        return;
    }
    
    obj->width = w;
    obj->height = h;

    if (obj->type == GFX_OBJ_TYPE_LABEL) {
        gfx_label_set_size(obj, w, h);
    }
    
    ESP_LOGD(TAG, "Set object size: %dx%d", w, h);
}

/*=====================
 * Getter functions
 *====================*/

void gfx_obj_get_pos(gfx_obj_t *obj, uint16_t *x, uint16_t *y)
{
    if (obj == NULL || x == NULL || y == NULL) {
        ESP_LOGE(TAG, "Invalid parameters");
        return;
    }
    
    *x = obj->x;
    *y = obj->y;
}

void gfx_obj_get_size(gfx_obj_t *obj, uint16_t *w, uint16_t *h)
{
    if (obj == NULL || w == NULL || h == NULL) {
        ESP_LOGE(TAG, "Invalid parameters");
        return;
    }
    
    *w = obj->width;
    *h = obj->height;
}

/*=====================
 * Other functions
 *====================*/

void gfx_obj_delete(gfx_obj_t *obj)
{
    if (obj == NULL) {
        ESP_LOGE(TAG, "Object is NULL");
        return;
    }
    
    ESP_LOGD(TAG, "Deleting object type: %d", obj->type);
    free(obj);
} 