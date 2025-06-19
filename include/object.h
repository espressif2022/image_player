
/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "anim_player.h"
#include "ft_label.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    void *src;
    int type;
    uint16_t x;
    uint16_t y;
    uint16_t width;
    uint16_t height;
} gfx_obj_t;

void gfx_font_init(void);

gfx_obj_t * gfx_label_create(ft_label_cfg_t *config, anim_player_handle_t handle);

gfx_obj_t * gfx_image_create(anim_player_handle_t handle);
gfx_obj_t * gfx_image_set_src(gfx_obj_t *obj, void *src);

void gfx_obj_set_pos(gfx_obj_t *obj, uint16_t x, uint16_t y);

void gfx_obj_set_size(gfx_obj_t *obj, uint16_t w, uint16_t h);

#ifdef __cplusplus
}
#endif