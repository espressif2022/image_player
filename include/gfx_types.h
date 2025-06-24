/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "esp_err.h"
#include "stdbool.h"

#ifdef __cplusplus
extern "C" {
#endif

/*********************
 *      DEFINES
 *********************/

/* Magic numbers for image headers */
#define GFX_IMAGE_HEADER_MAGIC    0x19

/* Object types */
#define GFX_OBJ_TYPE_IMAGE        0x01
#define GFX_OBJ_TYPE_LABEL        0x02

/**********************
 *      TYPEDEFS
 **********************/

/* Basic types */
typedef uint8_t     gfx_opa_t;      /**< Opacity (0-255) */
typedef int16_t     gfx_coord_t;    /**< Coordinate type */

/* Color type with full member for compatibility */
typedef union {
    uint16_t full;                  /**< Full 16-bit color value */
} gfx_color_t;

/* Color format enumeration */
typedef enum {
    GFX_COLOR_FORMAT_RGB565A8 = 0x14,
} gfx_color_format_t;

/* Image flags */
typedef enum {
    GFX_IMAGE_FLAG_NONE = 0x00,
    GFX_IMAGE_FLAG_ALPHA = 0x01,
} gfx_image_flags_t;

/* Area structure */
typedef struct {
    gfx_coord_t x1;
    gfx_coord_t y1;
    gfx_coord_t x2;
    gfx_coord_t y2;
} gfx_area_t;

/* Image header structure */
typedef struct {
    uint32_t magic: 8;          /**< Magic number. Must be GFX_IMAGE_HEADER_MAGIC */
    uint32_t cf : 8;            /**< Color format: See `gfx_color_format_t` */
    uint32_t flags: 16;         /**< Image flags, see `gfx_image_flags_t` */

    uint32_t w: 16;             /**< Width of the image */
    uint32_t h: 16;             /**< Height of the image */
    uint32_t stride: 16;        /**< Number of bytes in a row */
    uint32_t reserved: 16;      /**< Reserved for future use */
} gfx_image_header_t;

/* Image descriptor structure */
typedef struct {
    gfx_image_header_t header;   /**< A header describing the basics of the image */
    uint32_t data_size;         /**< Size of the image in bytes */
    const uint8_t * data;       /**< Pointer to the data of the image */
    const void * reserved;      /**< Reserved field for future use */
    const void * reserved_2;    /**< Reserved field for future use */
} gfx_image_dsc_t;

/* Graphics object structure */
typedef struct gfx_obj {
    void *src;                  /**< Source data (image, label, etc.) */
    int type;                   /**< Object type */
    uint16_t x;                 /**< X position */
    uint16_t y;                 /**< Y position */
    uint16_t width;             /**< Object width */
    uint16_t height;            /**< Object height */
    bool is_visible;            /**< Object visibility */
    bool is_dirty;              /**< Object dirty flag */
} gfx_obj_t;

/**********************
 * GLOBAL PROTOTYPES
 **********************/

/**
 * @brief Convert a 32-bit hexadecimal color to gfx_color_t
 * @param c The 32-bit hexadecimal color to convert
 * @return Converted color in gfx_color_t type
 */
gfx_color_t gfx_color_hex(uint32_t c);


/**********************
 *      MACROS
 **********************/

#define GFX_COLOR_HEX(color) ((gfx_color_t)gfx_color_hex(color))

#ifdef __cplusplus
}
#endif 