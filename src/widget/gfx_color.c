/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gfx_types.h"

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

gfx_color_t gfx_color_hex(uint32_t c)
{
    gfx_color_t r;
#if CONFIG_LABEL_COLOR_16_SWAP == 0
    r.full = (uint16_t)(((c & 0xF80000) >> 8) | ((c & 0xFC00) >> 5) | ((c & 0xFF) >> 3));
#else
    /* We want: rrrr rrrr GGGg gggg bbbb bbbb => gggb bbbb rrrr rGGG */
    r.full = (uint16_t)(((c & 0xF80000) >> 16) | ((c & 0xFC00) >> 13) | ((c & 0x1C00) << 3) | ((c & 0xF8) << 5));
#endif
    return r;
}