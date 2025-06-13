/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/queue.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_err.h"
#include "esp_check.h"

#include "ft2build.h"
#include "ft_blend.h"

#include FT_FREETYPE_H

static const char *TAG = "FT_label";

typedef struct {
    FT_Face face;           /*!< FreeType font face object */
    uint16_t font_size;     /*!< Size of the font in points */
    label_opa_t opa;        /*!< Opacity of the label */
    blend_color_t color;    /*!< Color of the label */

    char *text;             /*!< Text content of the label */
    label_coord_t x;        /*!< X coordinate of the label's position */
    label_coord_t y;        /*!< Y coordinate of the label's position */
    uint16_t width;         /*!< Width of the label */
    uint16_t height;        /*!< Height of the label */
} ft_label_property_t;

typedef struct face_entry {
    FT_Face face;
    const void *mem;
    SLIST_ENTRY(face_entry) entries;
} ft_face_entry_t;

typedef struct {
    SLIST_HEAD(face_list, face_entry) ft_face_head;
    FT_Library ft_library;
} ft_library_t;

static const ft_label_property_t ft_property_default = {
    .face = NULL,
    .font_size = 20,
    .opa = 0xFF,
    .color = {
        .full = 0xFFFF,
    },
    .text = NULL,
    .x = 0,
    .y = 0,
    .width = 100,
    .height = 50
};

esp_err_t ft_library_create(ft_lib_handle_t *ret_lib)
{
    ESP_RETURN_ON_FALSE(ret_lib, ESP_ERR_INVALID_ARG, TAG, "invalid arguments");

    FT_Error error;
    esp_err_t ret = ESP_OK;

    ft_library_t *lib = (ft_library_t *)calloc(1, sizeof(ft_library_t));
    ESP_RETURN_ON_FALSE(lib, ESP_ERR_NO_MEM, TAG, "no mem for FT library");
    SLIST_INIT(&lib->ft_face_head);

    error = FT_Init_FreeType(&lib->ft_library);
    ESP_GOTO_ON_FALSE(!error, ESP_ERR_INVALID_STATE, err, TAG, "error initializing FT library");
    *ret_lib = lib;

    // ESP_LOGI(TAG, "FT library create success, version: %d.%d.%d",
    //          FREETYPE_LABEL_VER_MAJOR, FREETYPE_LABEL_VER_MINOR, FREETYPE_LABEL_VER_PATCH);

    return ret;

err:
    if (lib) {
        free(lib);
    }
    return ret;
}

esp_err_t ft_library_cleanup(ft_lib_handle_t lib_handle)
{
    ESP_RETURN_ON_FALSE(lib_handle, ESP_ERR_INVALID_ARG, TAG, "invalid library");

    ft_library_t *lib = (ft_library_t *)lib_handle;
    while (!SLIST_EMPTY(&lib->ft_face_head)) {
        ft_face_entry_t *entry = SLIST_FIRST(&lib->ft_face_head);
        SLIST_REMOVE_HEAD(&lib->ft_face_head, entries);
        FT_Done_Face(entry->face);
        free(entry);
    }
    FT_Done_FreeType(lib->ft_library);
    free(lib);

    return ESP_OK;
}

esp_err_t ft_label_new_font(ft_lib_handle_t lib_handle, const ft_label_cfg_t *cfg, ft_font_handle_t *ret_handle)
{
    ESP_RETURN_ON_FALSE(lib_handle && cfg && ret_handle, ESP_ERR_INVALID_ARG, TAG, "invalid arguments");
    ESP_RETURN_ON_FALSE(cfg->mem && cfg->mem_size, ESP_ERR_INVALID_ARG, TAG, "invalid memory input");

    FT_Face face = NULL;
    FT_Error error;

    ft_library_t *lib = (ft_library_t *)lib_handle;
    ft_face_entry_t *entry;
    SLIST_FOREACH(entry, &lib->ft_face_head, entries) {
        if (entry->mem == cfg->mem) {
            face = entry->face;
            break;
        }
    }

    if (!face) {
        error = FT_New_Memory_Face(lib->ft_library, cfg->mem, cfg->mem_size, 0, &face);
        ESP_RETURN_ON_FALSE(!error, ESP_ERR_INVALID_ARG, TAG, "error loading font");

        ft_face_entry_t *new_face_entry = (ft_face_entry_t *)calloc(1, sizeof(ft_face_entry_t));
        ESP_RETURN_ON_FALSE(new_face_entry, ESP_ERR_NO_MEM, TAG, "no mem for ft_face_entry");

        new_face_entry->face = face;
        new_face_entry->mem = cfg->mem;
        SLIST_INSERT_HEAD(&lib->ft_face_head, new_face_entry, entries);
    }

    ft_label_property_t *ft_info = (ft_label_property_t *)calloc(1, sizeof(ft_label_property_t));
    ESP_RETURN_ON_FALSE(ft_info, ESP_ERR_NO_MEM, TAG, "no mem for ft_info");

    memcpy(ft_info, &ft_property_default, sizeof(ft_label_property_t));
    ft_info->face = face;

    ESP_LOGI(TAG, "new font(%s):@%p", cfg->name, ft_info);
    *ret_handle = ft_info;

    return ESP_OK;
}

esp_err_t ft_label_del_font(ft_font_handle_t handle)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    if (font_info->text) {
        free(font_info->text);
    }
    if (font_info) {
        free(font_info);
    }

    return ESP_OK;
}

esp_err_t ft_label_set_text(ft_font_handle_t handle, const char *text)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;

    if (text == NULL) {
        text = font_info->text;
    }

    if (font_info->text == text) {
        font_info->text = realloc(font_info->text, strlen(font_info->text) + 1);
        assert(font_info->text);
        if (font_info->text == NULL) {
            return ESP_FAIL;
        }
    } else {
        if (font_info->text != NULL) {
            free(font_info->text);
            font_info->text = NULL;
        }

        size_t len = strlen(text) + 1;

        font_info->text = malloc(len);
        assert(font_info->text);
        if (font_info->text == NULL) {
            return ESP_FAIL;
        }
        strcpy(font_info->text, text);
    }

    return ESP_OK;
}

esp_err_t ft_label_set_text_fmt(ft_font_handle_t handle, const char * fmt, ...)
{
    ESP_RETURN_ON_FALSE(handle && fmt, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;

    if (font_info->text != NULL) {
        free(font_info->text);
        font_info->text = NULL;
    }

    va_list args;
    va_start(args, fmt);

    /*Allocate space for the new text by using trick from C99 standard section 7.19.6.12*/
    va_list args_copy;
    va_copy(args_copy, args);
    uint32_t len = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);

    font_info->text = malloc(len + 1);
    if (font_info->text == NULL) {
        va_end(args);
        return ESP_ERR_NO_MEM;
    }
    font_info->text[len] = '\0'; /*Ensure NULL termination*/

    vsnprintf(font_info->text, len + 1, fmt, args);
    va_end(args);

    return ESP_OK;
}

esp_err_t ft_label_set_font_size(ft_font_handle_t handle, uint8_t font_size)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");
    ESP_RETURN_ON_FALSE(font_size > 0, ESP_ERR_INVALID_ARG, TAG, "invalid font size");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->font_size = font_size;
    ESP_LOGD(TAG, "set font size: %d", font_info->font_size);

    return ESP_OK;
}

esp_err_t ft_label_set_opa(ft_font_handle_t handle, label_opa_t opa)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->opa = opa;
    ESP_LOGD(TAG, "set font opa: %d", font_info->opa);

    return ESP_OK;
}

esp_err_t ft_label_set_color(ft_font_handle_t handle, label_color_t color)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->color = blend_color_hex(color);
    ESP_LOGD(TAG, "set font color: %02x", font_info->color.full);

    return ESP_OK;
}

esp_err_t ft_label_set_x(ft_font_handle_t handle, label_coord_t x)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->x = x;
    ESP_LOGD(TAG, "set font x: %d", font_info->x);

    return ESP_OK;
}

esp_err_t ft_label_set_y(ft_font_handle_t handle, label_coord_t y)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->y = y;
    ESP_LOGD(TAG, "set font y: %d", font_info->y);

    return ESP_OK;
}

esp_err_t ft_label_set_pos(ft_font_handle_t handle, label_coord_t x, label_coord_t y)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->x = x;
    font_info->y = y;
    ESP_LOGD(TAG, "set font pos: %d, %d", x, y);

    return ESP_OK;
}

esp_err_t ft_label_set_width(ft_font_handle_t handle, int16_t w)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->width = w;
    ESP_LOGD(TAG, "set font width: %d", font_info->width);

    return ESP_OK;
}

esp_err_t ft_label_set_height(ft_font_handle_t handle, int16_t h)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->height = h;
    ESP_LOGD(TAG, "set font height: %d", font_info->height);

    return ESP_OK;
}

esp_err_t ft_label_set_size(ft_font_handle_t handle, int16_t w, int16_t h)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    font_info->height = h;
    font_info->width = w;
    ESP_LOGD(TAG, "set font size: w:%d, h:%d", w, h);

    return ESP_OK;
}

esp_err_t ft_label_render_text(ft_font_handle_t handle, ft_blend_area_t *blend_area)
{
    ESP_RETURN_ON_FALSE(handle, ESP_ERR_INVALID_ARG, TAG, "invalid handle");
    ESP_RETURN_ON_FALSE(blend_area, ESP_ERR_INVALID_ARG, TAG, "invalid blend_area");
    ESP_RETURN_ON_FALSE(blend_area->buf_area, ESP_ERR_INVALID_ARG, TAG, "invalid blend_area->buf_area");

    esp_err_t ret = ESP_OK;
    FT_Error error;

    ft_label_property_t *font_info = (ft_label_property_t *)handle;
    ESP_RETURN_ON_FALSE(font_info->text, ESP_ERR_INVALID_ARG, TAG, "Text is NULL");

    int buf_area_w = blend_area->width;
    int buf_area_h = blend_area->height;
    blend_color_t *dest = (blend_color_t *)blend_area->buf_area;

    label_coord_t mask_stride = font_info->width;
    label_coord_t dest_stride = buf_area_w;

    label_opa_t *mask_buf = (label_opa_t *)malloc(font_info->width * font_info->height);
    ESP_RETURN_ON_FALSE(mask_buf, ESP_ERR_NO_MEM, TAG, "no mem for mask_buf");
    label_opa_t *mask = (label_opa_t *)mask_buf;
    memset(mask, 0x00, font_info->height * font_info->width);

    FT_Face face = font_info->face;
    error = FT_Set_Pixel_Sizes(face, 0, font_info->font_size);
    ESP_GOTO_ON_FALSE(!error, ESP_ERR_INVALID_STATE, err, TAG, "error setting font size");

    label_area_t clip_area;
    clip_area.x1 = font_info->x >= 0 ? 0 : (0 - font_info->x);
    clip_area.x2 = font_info->x + font_info->width <= buf_area_w ? font_info->width : (buf_area_w - font_info->x);
    clip_area.y1 = font_info->y >= 0 ? 0 : (0 - font_info->y);
    clip_area.y2 = font_info->y + font_info->height <= buf_area_h ? font_info->height : (buf_area_h - font_info->y);

    ESP_LOGD(TAG, "clip:col:%d->%d in [%d], row:%d->%d in [%d], position:[%d,%d]",
             clip_area.x1, clip_area.x2, font_info->width,
             clip_area.y1, clip_area.y2, font_info->height,
             font_info->x, font_info->y);

    int x = 0;
    int y = 0;

    const char *p = font_info->text;

    while (*p) {
        FT_UInt glyph_index;
        uint8_t c = (uint8_t) * p;
        int bytes_in_char = 1;
        if (c < 0x80) {
            glyph_index = FT_Get_Char_Index(face, c);
        } else if ((c & 0xE0) == 0xC0) {
            bytes_in_char = 2;
            if (*(p + 1) == 0) {
                break;
            }
            glyph_index = FT_Get_Char_Index(face, ((c & 0x1F) << 6) | (*(p + 1) & 0x3F));
        } else if ((c & 0xF0) == 0xE0) {
            bytes_in_char = 3;
            if (*(p + 1) == 0 || *(p + 2) == 0) {
                break;
            }
            glyph_index = FT_Get_Char_Index(face, ((c & 0x0F) << 12) | ((*(p + 1) & 0x3F) << 6) | (*(p + 2) & 0x3F));
        } else if ((c & 0xF8) == 0xF0) {
            bytes_in_char = 4;
            if (*(p + 1) == 0 || *(p + 2) == 0 || *(p + 3) == 0) {
                break;
            }
            glyph_index = FT_Get_Char_Index(face, ((c & 0x07) << 18) | ((*(p + 1) & 0x3F) << 12) | ((*(p + 2) & 0x3F) << 6) | (*(p + 3) & 0x3F));
        } else {
            glyph_index = 0;
        }
        p += bytes_in_char;

        /* load glyph image into the slot (erase previous one) */
        error = FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT);
        ESP_GOTO_ON_FALSE(!error, ESP_ERR_NOT_FOUND, err, TAG, "error loading glyph");

        /* convert to a bitmap */
        error = FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
        ESP_GOTO_ON_FALSE(!error, ESP_ERR_INVALID_STATE, err, TAG, "error rendering glyph");

        /* copy the glyph bitmap into the overall bitmap */
        FT_GlyphSlot slot = face->glyph;

        int line_height = (face->size->metrics.height >> 6);
        int base_line = -(face->size->metrics.descender >> 6);

        int ofs_x = slot->bitmap_left;
        int ofs_y = line_height - base_line - slot->bitmap_top;

        for (int32_t iy = 0; iy < slot->bitmap.rows; iy++) {
            for (int32_t ix = 0; ix < slot->bitmap.width; ix++) {
                int32_t res_x = ix + x + ofs_x;
                int32_t res_y = iy + y + ofs_y;
                if (res_x > clip_area.x2 || res_x < clip_area.x1 ||
                        res_y > clip_area.y2 || res_y < clip_area.y1) {
                    continue;
                }
                uint8_t value = slot->bitmap.buffer[ix + iy * slot->bitmap.width];
                *(mask_buf + (res_y + 0) * font_info->width + (res_x + 0)) = value;
            }
        }

        /* increment horizontal position */
        x += slot->advance.x >> 6;
        if (x >= (clip_area.x2 - clip_area.x1)) {
            break;
        }
    }

    dest += dest_stride * (font_info->y > 0 ? font_info->y : 0) + (font_info->x > 0 ? font_info->x : 0);
    mask += mask_stride * (clip_area.y1 > 0 ? clip_area.y1 : 0) + (clip_area.x1 > 0 ? clip_area.x1 : 0);

    blend_sw_draw(dest, dest_stride, font_info->color, font_info->opa, mask, &clip_area, mask_stride);

err:
    if (mask_buf) {
        free(mask_buf);
    }
    return ret;
}
