#!/usr/bin/env python3
"""
PNG to RGB565A8 C file converter
Converts PNG images to RGB565A8 format with optional byte swapping
RGB565 and Alpha data are stored separately
"""

import argparse
import os
import sys
from PIL import Image
import re

def rgb888_to_rgb565(r, g, b):
    """Convert RGB888 to RGB565"""
    r = (r >> 3) & 0x1F
    g = (g >> 2) & 0x3F
    b = (b >> 3) & 0x1F
    return (r << 11) | (g << 5) | b

def rgb565_to_bytes(rgb565, swap16=False):
    """Convert RGB565 to bytes, optionally swapping byte order"""
    high_byte = (rgb565 >> 8) & 0xFF
    low_byte = rgb565 & 0xFF
    
    if swap16:
        return [low_byte, high_byte]
    else:
        return [high_byte, low_byte]

def format_array(data, indent=4, per_line=130):
    """Format data as C array with proper indentation and line breaks"""
    lines = []
    for i in range(0, len(data), per_line):
        line = ', '.join(f'0x{b:02x}' for b in data[i:i + per_line])
        lines.append(' ' * indent + line + ',')
    return '\n'.join(lines)

def generate_c_file(image_path, output_path, var_name, swap16=False):
    """Generate C file from PNG image"""
    
    # Open and convert image
    try:
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return False
    
    width, height = img.size
    pixels = list(img.getdata())
    
    # Convert to RGB565A8 format - separate RGB565 and Alpha data
    rgb565_data = []
    alpha_data = []
    
    for pixel in pixels:
        r, g, b, a = pixel
        
        # Convert RGB to RGB565
        rgb565 = rgb888_to_rgb565(r, g, b)
        
        # Add RGB565 bytes (2 bytes) to RGB565 array
        rgb565_bytes = rgb565_to_bytes(rgb565, swap16)
        rgb565_data.extend(rgb565_bytes)
        
        # Add Alpha byte (1 byte) to Alpha array
        alpha_data.append(a)
    
    # Combine RGB565 and Alpha data: RGB565 first, then Alpha
    rgb565a8_data = rgb565_data + alpha_data
    
    # Generate C file content
    c_content = f"""#include "gfx_draw.h"

const uint8_t {var_name}_map[] = {{
{format_array(rgb565a8_data)}
}};

const gfx_image_dsc_t {var_name} = {{
    .header.cf = GFX_COLOR_FORMAT_RGB565A8,
    .header.magic = GFX_IMAGE_HEADER_MAGIC,
    .header.w = {width},
    .header.h = {height},
    .data_size = {len(rgb565a8_data)},
    .data = {var_name}_map,
}};
"""
    
    # Write to file
    try:
        with open(output_path, 'w') as f:
            f.write(c_content)
        print(f"Successfully generated {output_path}")
        print(f"Image size: {width}x{height}")
        print(f"Data size: {len(rgb565a8_data)} bytes")
        print(f"RGB565 data: {len(rgb565_data)} bytes ({width * height * 2} bytes)")
        print(f"Alpha data: {len(alpha_data)} bytes ({width * height} bytes)")
        print(f"Swap16: {'enabled' if swap16 else 'disabled'}")
        return True
    except Exception as e:
        print(f"Error writing file {output_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert PNG to RGB565A8 C file')
    parser.add_argument('input', help='Input PNG file path')
    parser.add_argument('-o', '--output', help='Output C file path (default: input_name.c)')
    parser.add_argument('-n', '--name', help='Variable name (default: derived from filename)')
    parser.add_argument('--swap16', action='store_true', help='Enable byte swapping for RGB565')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return 1
    
    if not args.input.lower().endswith('.png'):
        print("Warning: Input file doesn't have .png extension")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}.c"
    
    # Determine variable name
    if args.name:
        var_name = args.name
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        # Convert to valid C identifier
        var_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
        if var_name[0].isdigit():
            var_name = 'img_' + var_name
    
    # Generate C file
    if generate_c_file(args.input, output_path, var_name, args.swap16):
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main()) 