#!/usr/bin/env python3
"""
Example usage of PNG to RGB565A8 converter
Demonstrates how to use the converter with different options
"""

from PIL import Image, ImageDraw
import os
import sys

def create_sample_image(filename, size=(16, 16)):
    """Create a sample image with different colors and transparency"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw different colored squares with varying transparency
    colors = [
        ((255, 0, 0, 255), (0, 0, 0, 0)),      # Red square, transparent center
        ((0, 255, 0, 128), (0, 0, 255, 255)),  # Semi-transparent green, blue center
        ((255, 255, 0, 200), (128, 128, 128, 100)),  # Yellow, gray center
        ((255, 0, 255, 180), (0, 255, 255, 150)),    # Magenta, cyan center
    ]
    
    for i, (outer_color, inner_color) in enumerate(colors):
        x = (i % 2) * (size[0] // 2)
        y = (i // 2) * (size[1] // 2)
        
        # Draw outer square
        draw.rectangle([x, y, x + size[0]//2 - 1, y + size[1]//2 - 1], 
                      fill=outer_color)
        
        # Draw inner square
        inner_x = x + 2
        inner_y = y + 2
        draw.rectangle([inner_x, inner_y, 
                       inner_x + size[0]//2 - 5, inner_y + size[1]//2 - 5], 
                      fill=inner_color)
    
    img.save(filename)
    print(f"Created sample image: {filename}")
    return filename

def main():
    print("PNG to RGB565A8 Converter - Example Usage")
    print("=" * 50)
    
    # Create sample image
    sample_png = "sample_image.png"
    create_sample_image(sample_png, (16, 16))
    
    # Import the converter
    try:
        from png_to_rgb565a8 import generate_c_file
    except ImportError:
        print("Error: Could not import png_to_rgb565a8 module")
        print("Make sure png_to_rgb565a8.py is in the same directory")
        return 1
    
    print("\nConverting sample image...")
    
    # Example 1: Basic conversion
    print("\n1. Basic conversion:")
    if generate_c_file(sample_png, "sample_basic.c", "sample_basic", swap16=False):
        print("✓ Basic conversion successful")
    
    # Example 2: With swap16
    print("\n2. With swap16:")
    if generate_c_file(sample_png, "sample_swap.c", "sample_swap", swap16=True):
        print("✓ Swap16 conversion successful")
    
    # Example 3: Custom variable name
    print("\n3. Custom variable name:")
    if generate_c_file(sample_png, "sample_custom.c", "my_custom_image", swap16=False):
        print("✓ Custom name conversion successful")
    
    # Show file sizes
    print("\nGenerated files:")
    for filename in ["sample_basic.c", "sample_swap.c", "sample_custom.c"]:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  {filename}: {size} bytes")
    
    # Clean up
    if os.path.exists(sample_png):
        os.remove(sample_png)
        print(f"\nCleaned up: {sample_png}")
    
    print("\nExample completed!")
    print("\nYou can now use the generated .c files in your LVGL project.")
    print("The data format is: RGB565 data first, then Alpha data.")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 