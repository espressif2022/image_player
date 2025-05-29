import os
import argparse
from dataclasses import dataclass
import logging
import re
from collections import defaultdict

@dataclass
class PackModelsConfig:
    target_path: str
    image_file: str
    assets_path: str

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('frame_merge.log'),
            logging.StreamHandler()
        ]
    )

def compute_checksum(data):
    """Compute a simple checksum of the data."""
    return sum(data) & 0xFFFFFFFF

def get_frame_info(filename):
    """Extract frame name and number from filename."""
    match = re.search(r'(.+)_(\d+)\.sbmp$', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, 0

def sort_key(filename):
    """Sort files by frame name and number."""
    name, number = get_frame_info(filename)
    return (name, number) if name else ('', 0)

def pack_assets(config: PackModelsConfig):
    """
    Pack models based on the provided configuration.
    """
    setup_logging()
    target_path = config.target_path
    out_file = config.image_file
    assets_path = config.assets_path

    merged_data = bytearray()
    frame_info_list = []  # List of (frame_number, offset, size, is_repeated, original_frame)
    frame_map = {}  # Store frame offsets and sizes by frame number

    # First pass: process all frames and collect information
    file_list = sorted(os.listdir(target_path), key=sort_key)
    for filename in file_list:
        if not filename.lower().endswith('.sbmp'):
            continue

        file_path = os.path.join(target_path, filename)
        try:
            file_size = os.path.getsize(file_path)
            frame_name, frame_number = get_frame_info(filename)
            if not frame_name:
                logging.warning(f"Invalid filename format: {filename}")
                continue
            
            # Read file content to check for _R prefix
            with open(file_path, 'rb') as bin_file:
                bin_data = bin_file.read()
                if not bin_data:
                    logging.warning(f"Empty file '{filename}'")
                    continue
                
                # Check if this is a repeated frame
                if bin_data.startswith(b'_R'):
                    # Extract the original frame name from content
                    try:
                        # Format: _R + filename_length(1 byte) + original_filename
                        filename_length = bin_data[2]  # Get filename length (1 byte)
                        original_frame = bin_data[3:3+filename_length].decode('utf-8')
                        original_frame_name, original_frame_num = get_frame_info(original_frame)
                        
                        logging.info(f"Repeated {frame_name}_{frame_number} referencing {original_frame_name}_{original_frame_num}")
                        frame_info_list.append((frame_number, 0, file_size, True, original_frame_num))
                    except (ValueError, IndexError) as e:
                        logging.error(f"Invalid repeated frame format in {filename}: {str(e)}")
                    continue

                # Process original frame
                logging.info(f"Original {frame_name}_{frame_number} with size {file_size} bytes")
                # Add 0x5A5A prefix to merged_data
                merged_data.extend(b'\x5A' * 2)
                merged_data.extend(bin_data)
                # Update frame info with correct offset and size (including prefix)
                #                   frame_number,     offset,                           size,          is_repeated, original_frame_num
                frame_info_list.append((frame_number, len(merged_data) - file_size - 2, file_size + 2, False, None))
                frame_map[frame_number] = (len(merged_data) - file_size - 2, file_size + 2)

        except IOError as e:
            logging.error(f"Could not read file '{filename}': {str(e)}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error processing file '{filename}': {str(e)}")
            continue

    # Second pass: update repeated frame offsets and recalculate
    file_info_list = []
    new_merged_data = bytearray()
    new_offset = 0

    # First add all original frames to new_merged_data
    for frame_number, offset, size, is_repeated, original_frame in frame_info_list:
        if not is_repeated:
            frame_data = merged_data[offset:offset+size]
            new_merged_data.extend(frame_data)
            # Align to 4 bytes
            # padding = (4 - (len(new_merged_data) % 4)) % 4
            # if padding > 0:
            #     new_merged_data.extend(b'\x00' * padding)
            # Update frame map with new offset
            frame_map[frame_number] = (new_offset, size)
            print(f" O [{frame_number}] frame_data: 0x{new_offset:08x} ({size})")
            file_info_list.append((new_offset, size))
            new_offset = len(new_merged_data)
        else:
            if original_frame in frame_map:
                orig_offset, orig_size = frame_map[original_frame]
                file_info_list.append((orig_offset, orig_size))
                print(f" R [{frame_number}] frame_data: 0x{orig_offset:08x} ({orig_size})")

    total_files = len(file_info_list)
    if total_files == 0:
        logging.error("No .sbmp files found to process")
        return

    mmap_table = bytearray()
    for i, (offset, file_size) in enumerate(file_info_list):
        mmap_table.extend(file_size.to_bytes(4, byteorder='little'))
        mmap_table.extend(offset.to_bytes(4, byteorder='little'))
        logging.info(f"[{i + 1}] frame_data: 0x{offset:08x} ({file_size})")

    # Align mmap_table to 4 bytes
    padding = (4 - (len(mmap_table) % 4)) % 4
    if padding > 0:
        mmap_table.extend(b'\x00' * padding)

    combined_data = mmap_table + new_merged_data
    combined_checksum = compute_checksum(combined_data)
    combined_data_length = len(combined_data).to_bytes(4, byteorder='little')
    header_data = total_files.to_bytes(4, byteorder='little') + combined_checksum.to_bytes(4, byteorder='little')
    final_data = header_data + combined_data_length + combined_data

    try:
        with open(out_file, 'wb') as output_bin:
            output_bin.write(final_data)
        logging.info(f"\nSuccessfully packed {total_files} .sbmp files into {out_file}")
        logging.info(f"Total size: {len(final_data)} bytes")
        logging.info(f"Header size: {len(header_data)} bytes")
        logging.info(f"Table size: {len(mmap_table)} bytes")
        logging.info(f"Data size: {len(new_merged_data)} bytes")
    except IOError as e:
        logging.error(f"Failed to write output file: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error writing output file: {str(e)}")

def process_directory(input_dir):
    """Process all .sbmp files in the directory and group them by name."""
    # Group files by their base name
    file_groups = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.sbmp'):
            name, _ = get_frame_info(filename)
            if name:
                file_groups[name].append(filename)
    
    # Process each group
    for name, files in file_groups.items():
        if not files:
            continue
            
        # Create output filename based on the group name
        output_file = os.path.join(input_dir, f"{name}.aaf")
        
        # Create a temporary directory for this group
        temp_dir = os.path.join(input_dir, f"temp_{name}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy files to temporary directory
        for file in files:
            src = os.path.join(input_dir, file)
            dst = os.path.join(temp_dir, file)
            os.link(src, dst)  # Use hard link to save space
        
        # Process the group
        config = PackModelsConfig(
            target_path=temp_dir,
            image_file=output_file,
            assets_path=temp_dir
        )
        
        pack_assets(config)
        
        # Clean up temporary directory
        for file in files:
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Pack .sbmp files into .aaf files')
    parser.add_argument('input_dir', help='Input directory containing .sbmp files')
    args = parser.parse_args()

    # Ensure input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return

    print("\nProcessing directory:", args.input_dir)
    print("-" * 50)
    
    process_directory(args.input_dir)
    
    print("\nProcessing completed!")
    print("-" * 50)

if __name__ == '__main__':
    main()