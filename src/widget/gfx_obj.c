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

gfx_obj_t * gfx_label_create(anim_player_handle_t handle)
{
    gfx_obj_t *obj = (gfx_obj_t *)malloc(sizeof(gfx_obj_t));
    if (obj == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for label object");
        return NULL;
    }
    
    memset(obj, 0, sizeof(gfx_obj_t));
    obj->type = GFX_OBJ_TYPE_LABEL;
    
    gfx_label_property_t *label = (gfx_label_property_t *)malloc(sizeof(gfx_label_property_t));
    if (label == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory for label object");
        free(obj);
        return NULL;
    }
    memset(label, 0, sizeof(gfx_label_property_t));
    
    // Apply default font configuration
    gfx_font_t default_font;
    uint16_t default_size;
    gfx_color_t default_color;
    gfx_opa_t default_opa;
    
    // Get default font configuration from internal function
    gfx_get_default_font_config(&default_font, &default_size, &default_color, &default_opa);
    
    label->font_size = default_size;
    label->color = default_color;
    label->opa = default_opa;
    
    // Set default font automatically
    if (default_font) {
        label->face = (void *)default_font;
    }
    
    obj->src = label;
    
    anim_player_add_child(handle, GFX_OBJ_TYPE_LABEL, obj);
    ESP_LOGD(TAG, "Created label object with default font config");
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

void gfx_obj_set_size(gfx_obj_t *obj, uint16_t w, uint16_t h)
{
    if (obj == NULL) {
        ESP_LOGE(TAG, "Object is NULL");
        return;
    }
    
    obj->width = w;
    obj->height = h;
    
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
    if (obj->type == GFX_OBJ_TYPE_LABEL) {
        gfx_label_property_t *label = (gfx_label_property_t *)obj->src;
        if (label) {
            if (label->text) {
                free(label->text);
            }
            if (label->mask) {
                free(label->mask);
            }
            free(label);
        }
    }
    free(obj);
} 