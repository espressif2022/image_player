#!/usr/bin/env python3
"""
Test script for PNG to RGB565A8 converter
Creates a simple test image and converts it
"""

from PIL import Image, ImageDraw
import os
import sys

def create_test_image(filename, size=(32, 32)):
    """Create a simple test image with gradients and transparency"""
    img = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a gradient background
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(255 * x / size[0])
            g = int(255 * y / size[1])
            b = 128
            a = int(255 * (x + y) / (size[0] + size[1]))
            draw.point((x, y), fill=(r, g, b, a))
    
    # Draw a circle
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                 fill=(255, 255, 255, 200))
    
    img.save(filename)
    print(f"Created test image: {filename}")
    return filename

def main():
    # Create test image
    test_png = "test_image.png"
    create_test_image(test_png, (64, 64))
    
    # Test conversion
    print("\nTesting PNG to RGB565A8 conversion...")
    
    # Import the converter
    try:
        from png_to_rgb565a8 import generate_c_file
    except ImportError:
        print("Error: Could not import png_to_rgb565a8 module")
        return 1
    
    # Test without swap16
    print("\n1. Testing without swap16:")
    if generate_c_file(test_png, "test_output.c", "test_image", swap16=False):
        print("✓ Conversion successful")
    else:
        print("✗ Conversion failed")
    
    # Test with swap16
    print("\n2. Testing with swap16:")
    if generate_c_file(test_png, "test_output_swap.c", "test_image_swap", swap16=True):
        print("✓ Conversion successful")
    else:
        print("✗ Conversion failed")
    
    # Clean up test files
    if os.path.exists(test_png):
        os.remove(test_png)
        print(f"\nCleaned up: {test_png}")
    
    print("\nTest completed!")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 