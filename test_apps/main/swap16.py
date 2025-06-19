import re
import sys
import os

def extract_c_array_and_struct(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 匹配 uint8_t 数组
    array_match = re.search(r'(const\s+)?uint8_t\s+(\w+)\s*\[\]\s*=\s*\{([^}]*)\};', content, re.DOTALL)
    if not array_match:
        raise ValueError("No valid uint8_t array found in input")

    header = content[:array_match.start()]
    var_name = array_match.group(2)
    data_str = array_match.group(3)

    # 匹配结构体
    struct_match = re.search(r'const\s+lv_image_dsc_t\s+(\w+)\s*=\s*\{([^}]*\})\s*;', content[array_match.end():], re.DOTALL)
    struct_var_name = struct_match.group(1).strip() if struct_match else None
    struct_body = struct_match.group(2).strip() if struct_match else None

    # 提取 uint8_t 数组的值
    data_str = re.sub(r'/\*.*?\*/', '', data_str)
    byte_values = [int(x.strip(), 16) for x in data_str.split(',') if x.strip()]
    if len(byte_values) % 2 != 0:
        print("Warning: Odd number of bytes, truncating last.")
        byte_values = byte_values[:-1]

    return header, var_name, byte_values, struct_var_name, struct_body

def swap_rgb565_bytes(byte_values):
    return [byte_values[i + 1] if i + 1 < len(byte_values) else 0 for i in range(0, len(byte_values), 2)] + \
           [byte_values[i] if i + 1 < len(byte_values) else 0 for i in range(0, len(byte_values), 2)]

def format_array(byte_values, indent=4, per_line=12):
    lines = []
    for i in range(0, len(byte_values), per_line):
        line = ', '.join(f'0x{b:02x}' for b in byte_values[i:i + per_line])
        lines.append(' ' * indent + line + ',')
    return '\n'.join(lines)

def write_swapped_file(output_path, header, var_name, swapped_bytes, struct_var_name, struct_body):
    swapped_var_name = var_name + '_swap'
    swapped_struct_name = struct_var_name + '_swap' if struct_var_name else None

    with open(output_path, 'w') as f:
        f.write(header)
        f.write(f'const uint8_t {swapped_var_name}[] = {{\n')
        f.write(format_array(swapped_bytes))
        f.write('\n};\n\n')

        if struct_body:
            # 替换结构体中变量名为新的数组名和结构体名
            struct_body = re.sub(r'\.data\s*=\s*\s*' + var_name, f'.data = {swapped_var_name}', struct_body)
            f.write(f'const lv_image_dsc_t {swapped_struct_name} = {{\n{struct_body}\n}};\n')

def main():
    if len(sys.argv) < 2:
        print("Usage: python swap16.py <input_file.c>")
        return

    input_file = sys.argv[1]
    if not input_file.endswith('.c'):
        print("Error: Input must be a .c file")
        return

    output_file = input_file.replace('.c', '_swap.c')

    header, var_name, byte_values, struct_var_name, struct_body = extract_c_array_and_struct(input_file)
    swapped_bytes = swap_rgb565_bytes(byte_values)
    write_swapped_file(output_file, header, var_name, swapped_bytes, struct_var_name, struct_body)

    print(f"Done: wrote swapped data to {output_file}")

if __name__ == '__main__':
    main()
