/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "esp_log.h"
#include "esp_err.h"
#include "ft_blend.h"
#include "anim_player.h"
#include "object.h"

#define TAG "gfx_obj"

typedef struct {
    uint8_t *data;
    uint32_t data_size;
    uint16_t width;
    uint16_t height;
} gfx_image_t;

ft_lib_handle_t ft_lib = NULL;

void gfx_font_init(void)
{
    ft_library_create(&ft_lib);
}

gfx_obj_t * gfx_label_create(ft_label_cfg_t *config, anim_player_handle_t handle)
{
    ESP_LOGI(TAG, "Create label font:%p", config->mem);

    ft_font_handle_t font = NULL;
    ft_label_new_font(ft_lib, config, &font);

    gfx_obj_t *obj = malloc(sizeof(gfx_obj_t));
    if (!obj) {
        return NULL;
    }

    obj->type = CHILD_TYPE_LABEL;
    obj->src = font;

    anim_player_add_child(handle, CHILD_TYPE_LABEL, obj);

    return obj;
}

gfx_obj_t * gfx_image_create(anim_player_handle_t handle)
{
    gfx_obj_t *obj = malloc(sizeof(gfx_obj_t));
    if (!obj) {
        return NULL;
    }

    obj->type = CHILD_TYPE_IMAGE;
    obj->src = NULL;

    anim_player_add_child(handle, CHILD_TYPE_IMAGE, obj);

    return obj;
}

gfx_obj_t * gfx_image_set_src(gfx_obj_t *obj, void *src)
{
    if (obj) {
        obj->src = src;
    }
    return obj;
}

void gfx_obj_set_pos(gfx_obj_t *obj, uint16_t x, uint16_t y)
{
    if (obj) {
        obj->x = x;
        obj->y = y;
        if (obj->type == CHILD_TYPE_LABEL) {
            ft_label_set_pos(obj->src, x, y);
        }
    }
}

void gfx_obj_set_size(gfx_obj_t *obj, uint16_t w, uint16_t h)
{
    if (obj) {
        if (obj->type == CHILD_TYPE_LABEL) {
            obj->width = w;
            obj->height = h;
            ft_label_set_size(obj->src, w, h);
        } else if (obj->type == CHILD_TYPE_IMAGE) {
            ESP_LOGE(TAG, "Image size is not supported");
        }
    }
}

void gfx_obj_delete(gfx_obj_t *obj)
{
    if (obj) {
        free(obj);
    }
}