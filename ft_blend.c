/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "esp_log.h"
#include "esp_err.h"
#include "ft_blend.h"

#define OPA_MAX      253  /*Opacities above this will fully cover*/
#define OPA_TRANSP   0
#define OPA_COVER    0xFF

#define FILL_NORMAL_MASK_PX(color)                              \
    if(*mask == OPA_COVER) *dest_buf = color;                \
    else *dest_buf = blend_color_mix(color, *dest_buf, *mask);     \
    mask++;                                                     \
    dest_buf++;

static inline blend_color_t blend_color_mix(blend_color_t c1, blend_color_t c2, uint8_t mix)
{
    blend_color_t ret;

#if CONFIG_LABEL_COLOR_16_SWAP == 1
    c1.full = c1.full << 8 | c1.full >> 8;
    c2.full = c2.full << 8 | c2.full >> 8;
#endif
    /*Source: https://stackoverflow.com/a/50012418/1999969*/
    mix = (uint32_t)((uint32_t)mix + 4) >> 3;
    uint32_t bg = (uint32_t)((uint32_t)c2.full | ((uint32_t)c2.full << 16)) &
                  0x7E0F81F; /*0b00000111111000001111100000011111*/
    uint32_t fg = (uint32_t)((uint32_t)c1.full | ((uint32_t)c1.full << 16)) & 0x7E0F81F;
    uint32_t result = ((((fg - bg) * mix) >> 5) + bg) & 0x7E0F81F;
    ret.full = (uint16_t)((result >> 16) | result);
#if CONFIG_LABEL_COLOR_16_SWAP == 1
    ret.full = ret.full << 8 | ret.full >> 8;
#endif

    return ret;
}

blend_color_t blend_color_hex(uint32_t c)
{
    blend_color_t r;
#if CONFIG_LABEL_COLOR_16_SWAP == 0
    r.full = (uint16_t)(((c & 0xF80000) >> 8) | ((c & 0xFC00) >> 5) | ((c & 0xFF) >> 3));
#else
    /* We want: rrrr rrrr GGGg gggg bbbb bbbb => gggb bbbb rrrr rGGG */
    r.full = (uint16_t)(((c & 0xF80000) >> 16) | ((c & 0xFC00) >> 13) | ((c & 0x1C00) << 3) | ((c & 0xF8) << 5));
#endif
    return r;
}

void blend_sw_draw(blend_color_t *dest_buf, label_coord_t dest_stride,
                   blend_color_t color, label_opa_t opa,
                   const label_opa_t *mask, label_area_t *clip_area, label_coord_t mask_stride)
{
    int32_t w = clip_area->x2 - clip_area->x1;
    int32_t h = clip_area->y2 - clip_area->y1;

    int32_t x, y;
    uint32_t c32 = color.full + ((uint32_t)color.full << 16);

    /*Only the mask matters*/
    if (opa >= OPA_MAX) {
        int32_t x_end4 = w - 4;

        for (y = 0; y < h; y++) {
            for (x = 0; x < w && ((unsigned int)(mask) & 0x3); x++) {
                FILL_NORMAL_MASK_PX(color)
            }

            for (; x <= x_end4; x += 4) {
                uint32_t mask32 = *((uint32_t *)mask);
                if (mask32 == 0xFFFFFFFF) {
                    if ((unsigned int)dest_buf & 0x3) {/*dest_buf is not 4-byte aligned*/
                        *(dest_buf + 0) = color;
                        uint32_t * d = (uint32_t *)(dest_buf + 1);
                        *d = c32;
                        *(dest_buf + 3) = color;
                    } else {
                        uint32_t * d = (uint32_t *)dest_buf;
                        *d = c32;
                        *(d + 1) = c32;
                    }
                    dest_buf += 4;
                    mask += 4;
                } else if (mask32) {
                    FILL_NORMAL_MASK_PX(color)
                    FILL_NORMAL_MASK_PX(color)
                    FILL_NORMAL_MASK_PX(color)
                    FILL_NORMAL_MASK_PX(color)
                } else {
                    mask += 4;
                    dest_buf += 4;
                }
            }

            for (; x < w ; x++) {
                FILL_NORMAL_MASK_PX(color)
            }
            dest_buf += (dest_stride - w);
            mask += (mask_stride - w);
        }
    } else { /*With opacity*/
        /*Buffer the result color to avoid recalculating the same color*/
        blend_color_t last_dest_color;
        blend_color_t last_res_color;
        label_opa_t last_mask = OPA_TRANSP;
        last_dest_color.full = dest_buf[0].full;
        last_res_color.full = dest_buf[0].full;
        label_opa_t opa_tmp = OPA_TRANSP;

        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                if (*mask) {
                    if (*mask != last_mask) opa_tmp = *mask == OPA_COVER ? opa :
                                                          (uint32_t)((uint32_t)(*mask) * opa) >> 8;
                    if (*mask != last_mask || last_dest_color.full != dest_buf[x].full) {
                        if (opa_tmp == OPA_COVER) {
                            last_res_color = color;
                        } else {
                            last_res_color = blend_color_mix(color, dest_buf[x], opa_tmp);
                        }
                        last_mask = *mask;
                        last_dest_color.full = dest_buf[x].full;
                    }
                    dest_buf[x] = last_res_color;
                }
                mask++;
            }
            dest_buf += dest_stride;
            mask += (mask_stride - w);
        }
    }
}

void blend_sw_img_draw(blend_color_t *dest_buf, label_coord_t dest_stride,
                       const blend_color_t *src_buf, label_coord_t src_stride,
                       const label_opa_t *mask, label_coord_t mask_stride,
                       label_area_t *clip_area, label_opa_t opa)
{
    int32_t w = clip_area->x2 - clip_area->x1;
    int32_t h = clip_area->y2 - clip_area->y1;

    int32_t x, y;
    blend_color_t last_dest_color;
    blend_color_t last_res_color;
    blend_color_t last_src_color;
    label_opa_t last_mask = OPA_TRANSP;
    last_dest_color.full = dest_buf[0].full;
    last_res_color.full = dest_buf[0].full;
    last_src_color.full = src_buf[0].full;
    label_opa_t opa_tmp = OPA_TRANSP;

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            if (mask == NULL || *mask) {
                if (mask && *mask != last_mask) {
                    opa_tmp = (*mask == OPA_COVER) ? opa : (uint32_t)((uint32_t)(*mask) * opa) >> 8;
                }

                if (mask == NULL || *mask != last_mask || last_dest_color.full != dest_buf[x].full || last_src_color.full != src_buf[x].full) {
                    if (opa_tmp == OPA_COVER) {
                        last_res_color = src_buf[x];
                    } else {
                        last_res_color = blend_color_mix(src_buf[x], dest_buf[x], opa_tmp);
                    }
                    if (mask) {
                        last_mask = *mask;
                    }
                    last_dest_color.full = dest_buf[x].full;
                    last_src_color.full = src_buf[x].full;
                } else {
                    // ESP_LOGI("blend", "1:%d, 2:%d, 3:%d", *mask != last_mask, last_dest_color.full != dest_buf[x].full, last_src_color.full != src_buf[x].full);
                }
                dest_buf[x] = last_res_color;
            }
            if (mask) {
                mask++;
            }
        }
        dest_buf += dest_stride;
        src_buf += src_stride;
        if (mask) {
            mask += (mask_stride - w);
        }
    }
}
