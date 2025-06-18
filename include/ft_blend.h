/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ft_label.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure representing a rectangular area with coordinates.
 */
typedef struct {
    label_coord_t x1; /*!< X coordinate of the top-left corner */
    label_coord_t y1; /*!< Y coordinate of the top-left corner */
    label_coord_t x2; /*!< X coordinate of the bottom-right corner */
    label_coord_t y2; /*!< Y coordinate of the bottom-right corner */
} label_area_t;

/**
 * @brief Union representing a color with different bit-field layouts.
 */
typedef union {
    uint16_t full; /*!< Full 16-bit color value */
} blend_color_t;

/**
 * @brief Convert a 32-bit hexadecimal color to blend_color_t.
 *
 * @param c The 32-bit hexadecimal color to convert.
 * @return Converted color in blend_color_t type.
 */
blend_color_t blend_color_hex(uint32_t c);

/**
 * @brief Draw a blended color onto a destination buffer.
 *
 * @param dest_buf Pointer to the destination buffer where the color will be drawn.
 * @param dest_stride Stride (width) of the destination buffer.
 * @param color The color to draw in blend_color_t type.
 * @param opa The opacity of the color to draw (0-255).
 * @param mask Pointer to the mask buffer, if any.
 * @param clip_area Pointer to the clipping area, which limits the area to draw.
 * @param mask_stride Stride (width) of the mask buffer.
 */
void blend_sw_draw(blend_color_t *dest_buf, label_coord_t dest_stride, blend_color_t color, label_opa_t opa,
                   const label_opa_t *mask, label_area_t *clip_area, label_coord_t mask_stride);


void blend_sw_img_draw(blend_color_t *dest_buf, label_coord_t dest_stride,
                      const blend_color_t *src_buf, label_coord_t src_stride,
                      const label_opa_t *mask, label_coord_t mask_stride,
                      label_area_t *clip_area, label_opa_t opa);

#ifdef __cplusplus
}
#endif
