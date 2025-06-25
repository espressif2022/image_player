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
#include "gfx_sw_blend.h"
#include "gfx_obj.h"
#include "gfx_draw.h"
#include "gfx_comm.h"

static const char *TAG = "gfx_draw_img";

/*********************
 *      DEFINES
 *********************/

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/**********************
 *      TYPEDEFS
 **********************/

/**********************
 *  STATIC PROTOTYPES
 **********************/

/**********************
 *   GLOBAL FUNCTIONS
 **********************/

void gfx_draw_img(gfx_obj_t *obj, int x1, int y1, int x2, int y2, const void *dest_buf)
{
    if (obj == NULL || obj->src == NULL) {
        ESP_LOGW(TAG, "Invalid object or source");
        return;
    }

    if (obj->type != GFX_OBJ_TYPE_IMAGE) {
        ESP_LOGW(TAG, "Object is not an image type");
        return;
    }

    gfx_image_dsc_t *image_desc = (gfx_image_dsc_t *)obj->src;
    gfx_image_header_t *image_header = &image_desc->header;

    // Check color format - only support RGB565A8 format
    if (image_header->cf != GFX_COLOR_FORMAT_RGB565A8) {
        ESP_LOGW(TAG, "Unsupported color format: 0x%02X, only RGB565A8 (0x%02X) is supported",
                 image_header->cf, GFX_COLOR_FORMAT_RGB565A8);
        return;
    }

    gfx_area_t clip_region;
    clip_region.x1 = MAX(x1, obj->x);
    clip_region.y1 = MAX(y1, obj->y);
    clip_region.x2 = MIN(x2, obj->x + image_header->w);
    clip_region.y2 = MIN(y2, obj->y + image_header->h);

    // Check if there's any overlap
    if (clip_region.x1 >= clip_region.x2 || clip_region.y1 >= clip_region.y2) {
        return;
    }

    gfx_coord_t dest_buffer_stride = (x2 - x1);
    gfx_coord_t source_buffer_stride = image_header->w;

    gfx_color_t *source_pixels = (gfx_color_t *)image_desc->data + (clip_region.y1 - obj->y) * source_buffer_stride;
    gfx_opa_t *alpha_mask = (gfx_opa_t *)(image_desc->data + source_buffer_stride * image_header->h * 2 + (clip_region.y1 - obj->y) * source_buffer_stride);
    gfx_color_t *dest_pixels = (gfx_color_t *)dest_buf + (clip_region.y1 - y1) * dest_buffer_stride + (clip_region.x1 - x1);

    gfx_sw_blend_img_draw(
        dest_pixels,
        dest_buffer_stride,
        source_pixels,
        source_buffer_stride,
        alpha_mask,
        source_buffer_stride,
        &clip_region,
        255
    );
}