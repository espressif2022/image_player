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

#include "gfx_font_internal.h"
#include "anim_player.h"
#include "gfx_types.h"
#include "gfx_obj.h"
#include "gfx_draw.h"
#include "gfx_comm.h"
#include "gfx_sw_blend.h"

static const char *TAG = "FT_label";

// Default font configuration (internal use)
static gfx_font_t s_default_font = NULL;
static uint16_t s_default_font_size = 20;
static gfx_color_t s_default_font_color = {.full = 0xFFFF}; // White
static gfx_opa_t s_default_font_opa = 0xFF;

// Internal function to get default font configuration
void gfx_get_default_font_config(gfx_font_t *font, uint16_t *size, gfx_color_t *color, gfx_opa_t *opa)
{
    if (font) *font = s_default_font;
    if (size) *size = s_default_font_size;
    if (color) *color = s_default_font_color;
    if (opa) *opa = s_default_font_opa;
}

esp_err_t gfx_ft_lib_create(ft_lib_handle_t *ret_lib)
{
    ESP_RETURN_ON_FALSE(ret_lib, ESP_ERR_INVALID_ARG, TAG, "invalid arguments");

    FT_Error error;
    esp_err_t ret = ESP_OK;

    ft_library_t *lib = (ft_library_t *)calloc(1, sizeof(ft_library_t));
    ESP_RETURN_ON_FALSE(lib, ESP_ERR_NO_MEM, TAG, "no mem for FT library");
    
    // Initialize the linked list manually since we removed SLIST macros
    lib->ft_face_head = NULL;

    error = FT_Init_FreeType((FT_Library *)&lib->ft_library);
    ESP_GOTO_ON_FALSE(!error, ESP_ERR_INVALID_STATE, err, TAG, "error initializing FT library");
    *ret_lib = lib;

    return ret;

err:
    if (lib) {
        free(lib);
    }
    return ret;
}

esp_err_t gfx_ft_lib_cleanup(ft_lib_handle_t lib_handle)
{
    ESP_RETURN_ON_FALSE(lib_handle, ESP_ERR_INVALID_ARG, TAG, "invalid library");

    ft_library_t *lib = (ft_library_t *)lib_handle;
    
    // Clean up the linked list manually
    ft_face_entry_t *entry = lib->ft_face_head;
    while (entry != NULL) {
        ft_face_entry_t *next = entry->next;
        FT_Done_Face((FT_Face)entry->face);
        free(entry);
        entry = next;
    }
    
    FT_Done_FreeType((FT_Library)lib->ft_library);
    free(lib);

    return ESP_OK;
}

extern ft_lib_handle_t anim_player_get_font_lib(anim_player_handle_t handle);

esp_err_t gfx_label_new_font(anim_player_handle_t handle, const gfx_label_cfg_t *cfg, gfx_font_t *ret_font)
{
    ESP_RETURN_ON_FALSE(handle && cfg && ret_font, ESP_ERR_INVALID_ARG, TAG, "invalid arguments");
    ESP_RETURN_ON_FALSE(cfg->mem && cfg->mem_size, ESP_ERR_INVALID_ARG, TAG, "invalid memory input");

    FT_Face face = NULL;
    FT_Error error;

    ft_library_t *lib = anim_player_get_font_lib(handle);
    ESP_RETURN_ON_FALSE(lib, ESP_ERR_INVALID_STATE, TAG, "font library is NULL");
    
    ft_face_entry_t *entry;
    
    // Search for existing font
    entry = lib->ft_face_head;
    while (entry != NULL) {
        if (entry->mem == cfg->mem) {
            face = (FT_Face)entry->face;
            break;
        }
        entry = entry->next;
    }

    if (!face) {
        error = FT_New_Memory_Face((FT_Library)lib->ft_library, cfg->mem, cfg->mem_size, 0, &face);
        ESP_RETURN_ON_FALSE(!error, ESP_ERR_INVALID_ARG, TAG, "error loading font");

        ft_face_entry_t *new_face_entry = (ft_face_entry_t *)calloc(1, sizeof(ft_face_entry_t));
        ESP_RETURN_ON_FALSE(new_face_entry, ESP_ERR_NO_MEM, TAG, "no mem for ft_face_entry");

        new_face_entry->face = face;
        new_face_entry->mem = cfg->mem;
        new_face_entry->next = lib->ft_face_head;
        lib->ft_face_head = new_face_entry;
    }

    gfx_font_t font_handle = (gfx_font_t)face;
    
    // Set first font as default font automatically
    if (s_default_font == NULL) {
        s_default_font = font_handle;
        ESP_LOGI(TAG, "Set first font as default: %s", cfg->name);
    }

    ESP_LOGI(TAG, "new font(%s):@%p", cfg->name, face);
    *ret_font = font_handle;

    return ESP_OK;
}

esp_err_t gfx_label_set_font(gfx_obj_t *obj, gfx_font_t font)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");
    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    font_info->face = (void *)font;
    return ESP_OK;
}

esp_err_t gfx_label_set_text(gfx_obj_t * obj, const char *text)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;

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

    obj->is_dirty = true;

    return ESP_OK;
}

esp_err_t gfx_label_set_text_fmt(gfx_obj_t * obj, const char * fmt, ...)
{
    ESP_RETURN_ON_FALSE(obj && fmt, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;

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

    obj->is_dirty = true;

    return ESP_OK;
}

esp_err_t gfx_label_set_font_size(gfx_obj_t * obj, uint8_t font_size)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");
    ESP_RETURN_ON_FALSE(font_size > 0, ESP_ERR_INVALID_ARG, TAG, "invalid font size");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    font_info->font_size = font_size;
    obj->is_dirty = true;
    ESP_LOGD(TAG, "set font size: %d", font_info->font_size);

    return ESP_OK;
}

esp_err_t gfx_label_set_opa(gfx_obj_t * obj, gfx_opa_t opa)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    font_info->opa = opa;
    ESP_LOGD(TAG, "set font opa: %d", font_info->opa);

    return ESP_OK;
}

esp_err_t gfx_label_set_color(gfx_obj_t * obj, gfx_color_t color)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    font_info->color = color;
    ESP_LOGD(TAG, "set font color: %02x", font_info->color.full);

    return ESP_OK;
}

esp_err_t gfx_get_glphy_dsc(gfx_obj_t * obj)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    esp_err_t ret = ESP_OK;
    FT_Error error;

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    ESP_RETURN_ON_FALSE(font_info->text, ESP_ERR_INVALID_ARG, TAG, "Text is NULL");

    if (font_info->mask && !obj->is_dirty) {
        // ESP_LOGI(TAG, "mask already rendered");
        return ESP_OK;
    }

    if (font_info->mask) {
        free(font_info->mask);
        font_info->mask = NULL;
    }

    gfx_opa_t *mask_buf = (gfx_opa_t *)malloc(obj->width * obj->height);
    ESP_RETURN_ON_FALSE(mask_buf, ESP_ERR_NO_MEM, TAG, "no mem for mask_buf");
    gfx_opa_t *mask = (gfx_opa_t *)mask_buf;
    memset(mask, 0x00, obj->height * obj->width);

    FT_Face face = (FT_Face)font_info->face;
    error = FT_Set_Pixel_Sizes(face, 0, font_info->font_size);
    ESP_GOTO_ON_FALSE(!error, ESP_ERR_INVALID_STATE, err, TAG, "error setting font size");

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
                if (res_x >= obj->width || res_y >= obj->height) {
                    continue;
                }
                uint8_t value = slot->bitmap.buffer[ix + iy * slot->bitmap.width];
                *(mask_buf + (res_y + 0) * obj->width + (res_x + 0)) = value;
            }
        }

        /* increment horizontal position */
        x += slot->advance.x >> 6;
        if (x >= obj->width) {
            break;
        }
    }

    font_info->mask = mask;

    /* output the resulting bitmap to console */
    // for (int iy = 0; iy < obj->height; iy++) {
    //     for (int ix = 0; ix < obj->width; ix++) {
    //         int val = mask_buf[iy * obj->width + ix];
    //         if (val > 127) {
    //             putchar('#');
    //         } else if (val > 64) {
    //             putchar('+');
    //         } else if (val > 32) {
    //             putchar('.');
    //         } else {
    //             putchar(' ');
    //         }
    //     }
    //     putchar('\n');
    // }

    obj->is_dirty = false;
err:
    return ret;
}

/**
 * @brief Blend label object to destination buffer
 *
 * @param obj Graphics object containing label data
 * @param x1 Left boundary of destination area
 * @param y1 Top boundary of destination area
 * @param x2 Right boundary of destination area
 * @param y2 Bottom boundary of destination area
 * @param dest_buf Destination buffer for blending
 */
esp_err_t gfx_draw_label(gfx_obj_t *obj, int x1, int y1, int x2, int y2, const void *dest_buf)
{
    ESP_RETURN_ON_FALSE(obj, ESP_ERR_INVALID_ARG, TAG, "invalid handle");

    gfx_label_property_t *font_info = (gfx_label_property_t *)obj->src;
    ESP_RETURN_ON_FALSE(font_info->text, ESP_ERR_INVALID_ARG, TAG, "Text is NULL");

    gfx_area_t clip_region;
    clip_region.x1 = MAX(x1, obj->x);
    clip_region.y1 = MAX(y1, obj->y);
    clip_region.x2 = MIN(x2, obj->x + obj->width);
    clip_region.y2 = MIN(y2, obj->y + obj->height);

    // Check if there's any overlap
    if (clip_region.x1 >= clip_region.x2 || clip_region.y1 >= clip_region.y2) {
        return ESP_ERR_INVALID_STATE;
    }
    // ESP_LOGI(TAG, "clip: (%d,%d),(%d,%d)", clip_region.x1, clip_region.y1, clip_region.x2, clip_region.y2);

    // ESP_LOGI(TAG, "draw label");
    gfx_get_glphy_dsc(obj);

    gfx_color_t *dest_pixels = (gfx_color_t *)dest_buf + (clip_region.y1 - y1) * (x2 - x1) + (clip_region.x1 - x1);
    gfx_coord_t dest_buffer_stride = (x2 - x1);
    gfx_coord_t mask_offset_y = (clip_region.y1 - obj->y);

    gfx_opa_t *mask = font_info->mask;
    gfx_coord_t mask_stride = obj->width;
    mask += mask_offset_y * mask_stride;

    gfx_sw_blend_draw(dest_pixels, dest_buffer_stride, font_info->color, font_info->opa, mask, &clip_region, mask_stride);

    return ESP_OK;
}
