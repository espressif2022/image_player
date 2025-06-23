#!/usr/bin/env python3
"""
Verify the format of generated RGB565A8 C files
Checks that RGB565 and Alpha data are properly separated
"""

import re
import sys

def parse_c_file(filename):
    """Parse a generated C file and extract data"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    
    # Extract array data
    array_match = re.search(r'uint8_t \w+_map\[\] = \{([^}]*)\};', content, re.DOTALL)
    if not array_match:
        print(f"Error: Could not find array data in {filename}")
        return None
    
    # Extract width and height
    width_match = re.search(r'\.header\.w = (\d+)', content)
    height_match = re.search(r'\.header\.h = (\d+)', content)
    
    if not width_match or not height_match:
        print(f"Error: Could not find width/height in {filename}")
        return None
    
    width = int(width_match.group(1))
    height = int(height_match.group(1))
    
    # Parse array data
    data_str = array_match.group(1)
    data_str = re.sub(r'/\*.*?\*/', '', data_str)  # Remove comments
    data_str = re.sub(r'//.*$', '', data_str, flags=re.MULTILINE)  # Remove line comments
    
    # Extract hex values
    hex_values = []
    for match in re.finditer(r'0x([0-9a-fA-F]{2})', data_str):
        hex_values.append(int(match.group(1), 16))
    
    return {
        'width': width,
        'height': height,
        'data': hex_values,
        'filename': filename
    }

def verify_format(parsed_data):
    """Verify the data format is correct"""
    if not parsed_data:
        return False
    
    width = parsed_data['width']
    height = parsed_data['height']
    data = parsed_data['data']
    filename = parsed_data['filename']
    
    total_pixels = width * height
    expected_rgb565_bytes = total_pixels * 2
    expected_alpha_bytes = total_pixels
    expected_total_bytes = expected_rgb565_bytes + expected_alpha_bytes
    
    print(f"\nVerifying {filename}:")
    print(f"  Image size: {width}x{height} = {total_pixels} pixels")
    print(f"  Expected RGB565 bytes: {expected_rgb565_bytes}")
    print(f"  Expected Alpha bytes: {expected_alpha_bytes}")
    print(f"  Expected total bytes: {expected_total_bytes}")
    print(f"  Actual total bytes: {len(data)}")
    
    if len(data) != expected_total_bytes:
        print(f"  ❌ ERROR: Data size mismatch!")
        return False
    
    # Extract RGB565 and Alpha data
    rgb565_data = data[:expected_rgb565_bytes]
    alpha_data = data[expected_rgb565_bytes:]
    
    print(f"  RGB565 data: {len(rgb565_data)} bytes")
    print(f"  Alpha data: {len(alpha_data)} bytes")
    
    # Verify RGB565 data (should be pairs of bytes)
    if len(rgb565_data) % 2 != 0:
        print(f"  ❌ ERROR: RGB565 data length is not even!")
        return False
    
    # Show some sample data
    print(f"  Sample RGB565 data (first 4 pixels):")
    for i in range(0, min(8, len(rgb565_data)), 2):
        rgb565 = (rgb565_data[i] << 8) | rgb565_data[i + 1]
        r = (rgb565 >> 11) & 0x1F
        g = (rgb565 >> 5) & 0x3F
        b = rgb565 & 0x1F
        print(f"    Pixel {i//2}: RGB565=0x{rgb565:04x} (R={r}, G={g}, B={b})")
    
    print(f"  Sample Alpha data (first 4 pixels):")
    for i in range(min(4, len(alpha_data))):
        print(f"    Pixel {i}: Alpha=0x{alpha_data[i]:02x} ({alpha_data[i]})")
    
    print(f"  ✅ Format verification passed!")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_format.py <c_file1> [c_file2] ...")
        return 1
    
    all_passed = True
    
    for filename in sys.argv[1:]:
        parsed_data = parse_c_file(filename)
        if not verify_format(parsed_data):
            all_passed = False
    
    if all_passed:
        print(f"\n✅ All files verified successfully!")
        return 0
    else:
        print(f"\n❌ Some files failed verification!")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 