import numpy as np
import struct
import os
import sys
from PIL import Image
import math
from sklearn.cluster import KMeans
import time
from multiprocessing import Pool, cpu_count
import heapq
from collections import defaultdict


def floyd_steinberg_dithering(img, bit_depth=4):
    """Apply Floyd-Steinberg dithering and quantize to specified bit depth."""
    pixels = np.array(img, dtype=np.int32)
    height, width = pixels.shape
    
    # Calculate quantization levels based on bit depth
    num_levels = 2 ** bit_depth
    step = 256 // (num_levels - 1)
    
    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = pixels[y, x]
            new_pixel = round(old_pixel / step) * step
            pixels[y, x] = new_pixel
            error = old_pixel - new_pixel
            pixels[y, x + 1] += error * 7 / 16
            pixels[y + 1, x - 1] += error * 3 / 16
            pixels[y + 1, x] += error * 5 / 16
            pixels[y + 1, x + 1] += error * 1 / 16

    np.clip(pixels, 0, 255, out=pixels)
    return pixels.astype(np.uint8)


def generate_palette(bit_depth=4):
    """Generate a grayscale palette based on bit depth."""
    num_colors = 2 ** bit_depth
    palette = []
    for i in range(num_colors):
        level = int(i * 255 / (num_colors - 1))
        palette.append((level, level, level, 0))  # (B, G, R, 0)
    return palette


def process_row(args):
    """Process a single row of pixels."""
    y, pixels, width, bit_depth, palette, row_padded = args
    row = []
    if bit_depth == 4:
        for x in range(0, width, 2):
            p1 = pixels[y, x] // 17  # 0-15
            if x + 1 < width:
                p2 = pixels[y, x + 1] // 17
            else:
                p2 = 0
            byte = (p1 << 4) | p2
            row.append(byte)
    else:  # 8-bit
        for x in range(width):
            color = pixels[y, x]
            index = find_closest_color(color, palette)
            row.append(index)
    
    while len(row) < row_padded:
        row.append(0)  # padding
    return row


def save_bmp(filename, pixels, bit_depth=4):
    """Save a numpy array as a BMP file with specified bit depth."""
    if bit_depth == 8:
        # For 8-bit color images, use RGB channels
        height, width, _ = pixels.shape
    else:
        # For 4-bit grayscale images
        height, width = pixels.shape

    bits_per_pixel = bit_depth
    bytes_per_pixel = bits_per_pixel // 8
    row_size = (width * bits_per_pixel + 7) // 8
    row_padded = (row_size + 3) & ~3  # 4-byte align each row

    # BMP Header (14 bytes)
    bfType = b'BM'
    bfSize = 14 + 40 + (2 ** bits_per_pixel) * 4 + row_padded * height
    bfReserved1 = 0
    bfReserved2 = 0
    bfOffBits = 14 + 40 + (2 ** bits_per_pixel) * 4

    bmp_header = struct.pack('<2sIHHI', bfType, bfSize, bfReserved1, bfReserved2, bfOffBits)

    # DIB Header (BITMAPINFOHEADER, 40 bytes)
    biSize = 40
    biWidth = width
    biHeight = height
    biPlanes = 1
    biBitCount = bits_per_pixel
    biCompression = 0
    biSizeImage = row_padded * height
    biXPelsPerMeter = 3780
    biYPelsPerMeter = 3780
    biClrUsed = 2 ** bits_per_pixel
    biClrImportant = 2 ** bits_per_pixel

    dib_header = struct.pack('<IIIHHIIIIII',
                             biSize, biWidth, biHeight, biPlanes, biBitCount,
                             biCompression, biSizeImage,
                             biXPelsPerMeter, biYPelsPerMeter,
                             biClrUsed, biClrImportant)

    # Generate appropriate palette based on bit depth
    if bit_depth == 8:
        # For 8-bit color images, generate a color palette
        palette = generate_color_palette(pixels)
    else:
        # For 4-bit grayscale images, use grayscale palette
        palette = generate_palette(bit_depth)

    palette_data = b''.join(struct.pack('<BBBB', *color) for color in palette)

    # Pixel Data
    # Start timing
    start_time = time.time()
    
    # Prepare parallel processing arguments
    process_args = [(y, pixels, width, bit_depth, palette, row_padded) 
                   for y in range(height - 1, -1, -1)]
    
    # Use process pool for parallel processing
    with Pool(processes=cpu_count()) as pool:
        rows = pool.map(process_row, process_args)
    
    # Merge processing results
    pixel_data = bytearray()
    for row in rows:
        pixel_data.extend(row)

    # End timing and print
    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"Processing completed: Image size {width}x{height}, Total time: {execution_time:.3f} seconds")

    with open(filename, 'wb') as f:
        f.write(bmp_header)
        f.write(dib_header)
        f.write(palette_data)
        f.write(pixel_data)
    print(f"{os.path.basename(filename)}")


def generate_color_palette(pixels):
    """Generate a 256-color palette from the image using median cut algorithm."""
    # Reshape pixels to 2D array of RGB values
    pixels_2d = pixels.reshape(-1, 3)
    
    # Count unique colors
    unique_colors = np.unique(pixels_2d, axis=0)
    num_colors = min(len(unique_colors), 256)
    
    if num_colors < 256:
        # If we have fewer than 256 unique colors, use them directly
        colors = unique_colors
    else:
        # Use k-means clustering to find representative colors
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10).fit(pixels_2d)
        colors = kmeans.cluster_centers_.astype(np.uint8)
    
    # Convert to palette format (B, G, R, A)
    palette = []
    for color in colors:
        palette.append((color[2], color[1], color[0], 255))  # BGR format
    
    # Pad palette to 256 colors if necessary
    while len(palette) < 256:
        palette.append((0, 0, 0, 255))
    
    return palette


def find_closest_color(color, palette):
    """Find the index of the closest color in the palette."""
    min_dist = float('inf')
    closest_index = 0
    
    for i, palette_color in enumerate(palette):
        # Calculate Euclidean distance in RGB space using uint8 arithmetic
        r_diff = int(color[0]) - int(palette_color[2])
        g_diff = int(color[1]) - int(palette_color[1])
        b_diff = int(color[2]) - int(palette_color[0])
        dist = r_diff * r_diff + g_diff * g_diff + b_diff * b_diff
        
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    
    return closest_index


def convert_gif_to_bmp(gif_path, output_dir, bit_depth=4):
    """Convert GIF frames to BMPs with specified bit depth."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(gif_path))[0]

    with Image.open(gif_path) as im:
        frame = 0
        try:
            while True:
                if bit_depth == 8:
                    # For 8-bit, keep original colors
                    frame_image = im.convert('RGB')
                    # Convert to numpy array while preserving colors
                    pixels = np.array(frame_image)
                    output_path = os.path.join(output_dir, f"{base_name}_{frame:04d}.bmp")
                    save_bmp(output_path, pixels, bit_depth)
                else:
                    # For 4-bit, convert to grayscale and dither
                    gray_frame = im.convert('L')
                    dithered_pixels = floyd_steinberg_dithering(gray_frame, bit_depth)
                    output_path = os.path.join(output_dir, f"{base_name}_{frame:04d}.bmp")
                    save_bmp(output_path, dithered_pixels, bit_depth)
                
                frame += 1
                im.seek(frame)
        except EOFError:
            pass


def create_header(width, height, splits, split_height, lenbuf, ext, bit_depth=4):
    """Creates the header for the output file based on the format.
    
    Args:
        width: Image width
        height: Image height
        splits: Number of splits
        split_height: Height of each split
        lenbuf: List of split lengths
        ext: File extension
        bit_depth: Bit depth (4 or 8)
    """
    header = bytearray()

    if ext.lower() == '.bmp':
        header += bytearray('_S'.encode('UTF-8'))

    # 6 BYTES VERSION
    header += bytearray(('\x00V1.00\x00').encode('UTF-8'))
    
    # 1 BYTE BIT DEPTH
    header += bytearray([bit_depth])

    # WIDTH 2 BYTES
    header += width.to_bytes(2, byteorder='little')

    # HEIGHT 2 BYTES
    header += height.to_bytes(2, byteorder='little')

    # NUMBER OF ITEMS 2 BYTES
    header += splits.to_bytes(2, byteorder='little')

    # SPLIT HEIGHT 2 BYTES
    header += split_height.to_bytes(2, byteorder='little')

    for item_len in lenbuf:
        # LENGTH 2 BYTES
        header += item_len.to_bytes(2, byteorder='little')

    return header


def rte_compress(data):
    """Simple RTE (Run-Time Encoding) compression: [count, value]"""
    if not data:
        return bytearray()

    compressed = bytearray()
    prev = data[0]
    count = 1

    for b in data[1:]:
        if b == prev and count < 255:
            count += 1
        else:
            compressed.extend([count, prev])
            prev = b
            count = 1

    # Don't forget the last run
    compressed.extend([count, prev])
    return compressed


def generate_palette_from_image(im, bit_depth=4):
    """Extracts or generates a palette based on bit depth.
    
    Args:
        im: PIL Image object
        bit_depth: Bit depth for the palette (4 or 8)
    
    Returns:
        tuple: (palette_bytes, palette_list)
            - palette_bytes: Byte array containing the palette
            - palette_list: List of RGB tuples
    """
    num_colors = 2 ** bit_depth
    palette_bytes = bytearray()
    
    if bit_depth == 8:
        # For 8-bit color images
        if im.mode == 'RGB':
            # Convert to numpy array for color analysis
            pixels = np.array(im)
            # Count unique colors
            unique_colors = np.unique(pixels.reshape(-1, 3), axis=0)
            num_unique = min(len(unique_colors), 256)
            
            if num_unique < 256:
                # If we have fewer than 256 unique colors, use them directly
                colors = unique_colors
            else:
                # Use k-means clustering to find representative colors
                kmeans = KMeans(n_clusters=num_unique, random_state=0, n_init=10).fit(pixels.reshape(-1, 3))
                colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # Convert colors to palette format (B, G, R, A)
            for color in colors:
                palette_bytes.extend([color[2], color[1], color[0], 255])  # BGR format
            
            # Pad palette to 256 colors if necessary
            while len(palette_bytes) < 256 * 4:
                palette_bytes.extend([0, 0, 0, 255])
        else:
            # If not RGB, convert to RGB first
            im = im.convert('RGB')
            return generate_palette_from_image(im, bit_depth)
    else:
        # For 4-bit grayscale images
        if im.mode == 'P':
            # If image already has a palette, use it
            palette = im.getpalette()
            if palette is not None:
                # Extract the first `num_colors` colors from the palette
                palette = palette[:num_colors * 3]  # Each color is represented by 3 bytes (R, G, B)
                for i in range(0, len(palette), 3):
                    r, g, b = palette[i:i + 3]
                    palette_bytes.extend([r, g, b, 255])  # Add an alpha channel value of 255
        else:
            # Generate a grayscale palette based on bit depth
            for i in range(num_colors):
                level = int(i * 255 / (num_colors - 1))
                palette_bytes.extend([level, level, level, 255])  # (B, G, R, A)
    
    # Create palette list for color matching
    palette_list = []
    for i in range(0, len(palette_bytes), 4):
        b, g, r, _ = palette_bytes[i:i + 4]
        palette_list.append((r, g, b))
    
    return palette_bytes, palette_list


def find_palette_index(pixel_value, palette):
    """Finds the closest palette index for a pixel value."""
    # Calculate the squared difference for each color channel (R, G, B)
    def color_distance_squared(c1, c2):
        return sum((c1[i] - c2[i]) ** 2 for i in range(3))

    # Find the index of the closest color in the palette
    closest_index = min(range(len(palette)), key=lambda i: color_distance_squared(
        (pixel_value, pixel_value, pixel_value), palette[i]))

    # Debugging: Print the distance for each palette index
    # for i in range(len(palette)):
    #     diff = color_distance_squared((pixel_value, pixel_value, pixel_value), palette[i])
    #     print(f"Palette index {i}, diff: {diff}")

    return closest_index


def build_huffman_tree(data):
    """Build Huffman tree from data."""
    # Count frequency of each byte
    freq = defaultdict(int)
    for byte in data:
        freq[byte] += 1
    
    # Create priority queue
    heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Create encoding dictionary
    huffman_dict = {byte: code for byte, code in heap[0][1:]}
    return huffman_dict


def huffman_compress(data):
    """Compress data using Huffman coding."""
    if not data:
        return bytearray(), 0
    
    # Build Huffman tree and get encoding dictionary
    huffman_dict = build_huffman_tree(data)
    
    # Encode data
    encoded = ''.join(huffman_dict[byte] for byte in data)
    
    # Pad encoded string to multiple of 8
    padding = (8 - len(encoded) % 8) % 8
    encoded += '0' * padding
    
    # Convert to bytes
    result = bytearray()
    for i in range(0, len(encoded), 8):
        byte = encoded[i:i+8]
        result.append(int(byte, 2))
    
    return result, len(huffman_dict)


def process_block(args):
    """Process a single block of pixels."""
    top, bottom, width, height, pixels, bit_depth = args
    block_height = bottom - top
    block_data = bytearray()
    
    for y in range(block_height):
        row_start = (top + y) * width
        if bit_depth == 4:
            # Pack two pixels into one byte
            for x in range(0, width, 2):
                p1 = pixels[row_start + x]  # Already 0-15 index value
                if x + 1 < width:
                    p2 = pixels[row_start + x + 1]
                else:
                    p2 = 0
                packed_byte = (p1 << 4) | p2
                block_data.append(packed_byte)
        else:  # 8-bit
            # Use pixel value directly as index
            for x in range(width):
                block_data.append(pixels[row_start + x])
    
    # RTE compress this block
    rte_compressed = rte_compress(block_data)
    
    # Huffman compress this block
    huffman_compressed, dict_size = huffman_compress(rte_compressed)
    
    # Print compression comparison
    rte_size = len(rte_compressed)
    huffman_size = len(huffman_compressed) + dict_size * 2  # Add dictionary size (2 bytes per entry)
    original_size = len(block_data)
    
    print(f"Block {top}-{bottom} | Original: {original_size}B | RTE: {rte_size}B | Huffman: {huffman_size}B | Dict: {dict_size} entries")
    
    return rte_compressed, huffman_compressed, dict_size  # Return both compression results


def split_bmp(im, block_size, input_dir=None, bit_depth=4):
    """Splits grayscale image into raw bitmap blocks with RTE compression.
    
    Args:
        im: PIL Image object (BMP file)
        block_size: Height of each block
        input_dir: Input directory (optional)
        bit_depth: Bit depth for the image (4 or 8)
    
    Returns:
        tuple: (width, height, splits, palette_bytes, split_data, lenbuf)
    """
    width, height = im.size
    splits = math.ceil(height / block_size) if block_size else 1

    # Read palette from file
    palette_size = 2 ** bit_depth * 4  # Each palette entry is 4 bytes (B,G,R,A)
    with open(im.filename, 'rb') as f:
        f.seek(54)  # Skip BMP header (14 + 40 bytes)
        palette_bytes = f.read(palette_size)

    # Read pixel data
    pixels = list(im.getdata())
    row_size = (width * bit_depth + 7) // 8
    row_padded = (row_size + 3) & ~3  # 4-byte align each row

    # Calculate original data size
    original_size = row_padded * height

    # Prepare parallel processing arguments
    process_args = []
    for i in range(splits):
        top = i * block_size
        bottom = min((i + 1) * block_size, height)
        process_args.append((top, bottom, width, height, pixels, bit_depth))

    # Use process pool for parallel processing
    split_data = bytearray()
    huffman_data = bytearray()
    total_dict_size = 0
    lenbuf = []
    with Pool(processes=cpu_count()) as pool:
        compressed_blocks = pool.map(process_block, process_args)
        
        # Collect processing results
        for rte_block, huffman_block, dict_size in compressed_blocks:
            split_data.extend(rte_block)
            huffman_data.extend(huffman_block)
            total_dict_size += dict_size
            lenbuf.append(len(rte_block))

    # Calculate compression ratios
    rte_size = len(split_data)
    huffman_size = len(huffman_data) + total_dict_size * 2  # Add total dictionary size
    compression_ratio_rte = (1 - rte_size / original_size) * 100
    compression_ratio_huffman = (1 - huffman_size / original_size) * 100

    # Print statistics in one line with colored ratios
    ratio_color_rte = '\033[31m' if compression_ratio_rte < 0 else '\033[32m'
    ratio_color_huffman = '\033[31m' if compression_ratio_huffman < 0 else '\033[32m'
    print(f"Frame {width}x{height} | Original: {original_size}B")
    print(f"RTE: {rte_size}B | Ratio: {ratio_color_rte}{compression_ratio_rte:+.2f}%\033[0m")
    print(f"Huffman: {huffman_size}B | Ratio: {ratio_color_huffman}{compression_ratio_huffman:+.2f}%\033[0m | Dict: {total_dict_size} entries")
    print(f"Splits: {splits}")

    return width, height, splits, palette_bytes, split_data, lenbuf


def save_image(output_file_path, header, split_data, palette_bytes):
    """Save the final packaged image with header, palette, and split data."""
    with open(output_file_path, 'wb') as f:
        f.write(header)
        # print("Header saved.")

        # Write the palette
        f.write(palette_bytes)
        # print("Palette saved.")

        # Write the split data
        f.write(split_data)
        # print("Split data saved.")


def process_bmp(input_file, output_file, split_height, bit_depth=4):
    """Main function to process the image and save it as the packaged file."""
    try:
        SPLIT_HEIGHT = int(split_height)
        if SPLIT_HEIGHT <= 0:
            raise ValueError('Height must be a positive integer')
    except ValueError as e:
        print('Error:', e)
        sys.exit(1)

    input_dir, input_filename = os.path.split(input_file)
    base_filename, ext = os.path.splitext(input_filename)
    OUTPUT_FILE_NAME = base_filename
    # print(f'Processing {input_filename}')
    # print(f'Input directory: {input_dir}')
    # print(f'Output file name: {OUTPUT_FILE_NAME}')
    # print(f'File extension: {ext}')

    try:
        im = Image.open(input_file)
    except Exception as e:
        print('Error:', e)
        sys.exit(0)

    # Split the image into blocks based on the specified split height
    width, height, splits, palette_bytes, split_data, lenbuf = split_bmp(im, SPLIT_HEIGHT, input_dir, bit_depth)

    # Create header based on image properties
    header = create_header(width, height, splits, SPLIT_HEIGHT, lenbuf, ext, bit_depth)

    # Save the final packaged file
    output_file_path = os.path.join(output_file, OUTPUT_FILE_NAME + '.sbmp')
    save_image(output_file_path, header, split_data, palette_bytes)

    print('Completed', input_filename, '->', os.path.basename(output_file_path))


def process_images_in_directory(input_dir, output_dir, split_height, bit_depth=4):
    """Process all BMP images in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to store processed image hashes and their corresponding output filenames
    processed_images = {}

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.bmp')):
            input_file = os.path.join(input_dir, filename)

            # Compute a hash of the input file to check for duplicates
            with open(input_file, 'rb') as f:
                file_hash = hash(f.read())

            # Check if the image has already been processed
            if file_hash in processed_images:
                # Modify the output filename based on the extension
                if filename.lower().endswith('.bmp'):
                    output_file_path = os.path.join(output_dir, filename[:-4] + '.sbmp')
                else:
                    output_file_path = os.path.join(output_dir, 's' + filename)

                # Write the already processed filename string to the current file
                with open(output_file_path, 'wb') as f:
                    converted_filename = os.path.splitext(processed_images[file_hash])[0] + '.sbmp'
                    f.write("_R".encode('UTF-8'))
                    filename_length = len(converted_filename)
                    f.write(bytearray([filename_length]))
                    f.write(converted_filename.encode('UTF-8'))
                # print(f"Duplicate file: {filename} matches {converted_filename}.")
                continue

            # Process the image
            process_bmp(input_file, output_dir, split_height, bit_depth)

            # Save the processed filename in the dictionary
            processed_images[file_hash] = filename


def main():
    if len(sys.argv) != 5:
        print("Usage: python gif_to_bmp_and_split.py input_folder output_folder split_height bit_depth")
        print("bit_depth: 4 for 4-bit grayscale, 8 for 8-bit grayscale")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    split_height = sys.argv[3]
    bit_depth = int(sys.argv[4])

    if bit_depth not in [4, 8]:
        print("Error: bit_depth must be either 4 or 8")
        sys.exit(1)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.gif'):
                gif_path = os.path.join(root, file)
                convert_gif_to_bmp(gif_path, output_dir, bit_depth)

    process_images_in_directory(output_dir, output_dir, split_height, bit_depth)


if __name__ == "__main__":
    main()