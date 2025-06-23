# PNG to RGB565A8 Converter

这个脚本将PNG图片转换为RGB565A8格式的C文件，适用于LVGL图形库。

## 功能特性

- 将PNG图片转换为RGB565A8格式
- 支持Alpha透明度通道
- 可选的字节交换（swap16）功能
- 自动生成符合LVGL格式的C文件
- 支持自定义变量名和输出文件名
- **RGB565和Alpha数据分开存储**：先存储所有RGB565数据，再存储所有Alpha数据

## 安装依赖

```bash
pip install Pillow
```

## 使用方法

### 基本用法

```bash
python png_to_rgb565a8.py input.png
```

这将生成 `input.c` 文件，变量名为 `input`。

### 高级用法

```bash
python png_to_rgb565a8.py input.png -o output.c -n my_image --swap16
```

参数说明：
- `input.png`: 输入的PNG文件
- `-o output.c`: 指定输出文件名
- `-n my_image`: 指定变量名
- `--swap16`: 启用字节交换

### 参数详解

- `input`: 输入的PNG文件路径（必需）
- `-o, --output`: 输出C文件路径（可选，默认为输入文件名.c）
- `-n, --name`: 变量名（可选，默认为输入文件名）
- `--swap16`: 启用RGB565字节交换（可选）

## 输出格式

生成的C文件包含：

1. **RGB565A8数据数组**: 先存储所有RGB565数据，再存储所有Alpha数据
2. **LVGL图像描述符**: 包含图像尺寸、格式等信息

### 数据格式

- **RGB565**: 16位颜色格式（5位红，6位绿，5位蓝）
- **Alpha**: 8位透明度（0=透明，255=不透明）
- **字节顺序**: RGB565可以是大端序或小端序（通过swap16控制）
- **存储顺序**: 
  - 前 `width * height * 2` 字节：所有像素的RGB565数据
  - 后 `width * height` 字节：所有像素的Alpha数据

## 示例

### 输入PNG文件
```
my_icon.png (32x32, RGBA)
```

### 生成的C文件
```c
const uint8_t my_icon_map[] = {
    // RGB565 data (32*32*2 = 2048 bytes)
    0xf8, 0x1f, 0xff, 0xf8, 0x1f, 0xff, ...
    
    // Alpha data (32*32 = 1024 bytes)
    0xff, 0xff, 0xff, 0xff, ...
};

const lv_image_dsc_t my_icon = {
    .header.cf = GFX_COLOR_FORMAT_RGB565A8,
    .header.magic = LV_IMAGE_HEADER_MAGIC,
    .header.w = 32,
    .header.h = 32,
    .data_size = 3072,  // 2048 + 1024
    .data = my_icon_map,
};
```

### 数据布局示例（4x4图像）
```
RGB565数据 (32字节):
像素(0,0) RGB565: [0x12, 0x34]
像素(0,1) RGB565: [0x56, 0x78]
像素(0,2) RGB565: [0x9a, 0xbc]
...
像素(3,3) RGB565: [0xfe, 0xdc]

Alpha数据 (16字节):
像素(0,0) Alpha: [0xff]
像素(0,1) Alpha: [0xee]
像素(0,2) Alpha: [0xdd]
...
像素(3,3) Alpha: [0x11]
```

## 测试

运行测试脚本：

```bash
python test_png_converter.py
```

这将创建一个测试图像并验证转换功能。

## 注意事项

1. 输入图片必须是PNG格式
2. 脚本会自动将图片转换为RGBA模式
3. RGB565格式会损失一些颜色精度
4. 生成的C文件可以直接在LVGL项目中使用
5. 字节交换功能适用于不同的硬件平台
6. **数据存储格式**：RGB565和Alpha数据是分开存储的，不是交错存储

## 错误处理

- 如果输入文件不存在，脚本会报错并退出
- 如果输入文件不是PNG格式，会显示警告但继续处理
- 如果输出文件无法写入，会显示错误信息

## 兼容性

- Python 3.6+
- Pillow 8.0+
- 支持所有PNG格式（包括RGBA、RGB、灰度等） 