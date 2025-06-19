# PNG to RGB565A8 转换工具总结

## 工具概述

这是一个完整的PNG图片转RGB565A8格式C文件的工具集，专门为LVGL图形库设计。

## 文件说明

### 核心文件
- **`png_to_rgb565a8.py`** - 主转换脚本
- **`README_png_converter.md`** - 详细使用说明
- **`SUMMARY.md`** - 本总结文档

### 测试和示例文件
- **`test_png_converter.py`** - 基本功能测试
- **`example_usage.py`** - 使用示例
- **`verify_format.py`** - 格式验证工具

## 核心功能

### 1. 数据格式
- **RGB565A8格式**: 每个像素3字节数据
- **存储顺序**: 先存储所有RGB565数据，再存储所有Alpha数据
- **字节交换**: 支持swap16选项来交换RGB565字节顺序

### 2. 转换流程
```
PNG图片 → RGBA像素数据 → RGB565转换 → 字节分离 → C文件生成
```

### 3. 输出格式
```c
const uint8_t image_map[] = {
    // RGB565数据 (width * height * 2 字节)
    0xf8, 0x1f, 0xff, 0xf8, ...
    
    // Alpha数据 (width * height 字节)
    0xff, 0xff, 0xff, ...
};

const lv_image_dsc_t image = {
    .header.cf = LV_COLOR_FORMAT_RGB565A8,
    .header.magic = LV_IMAGE_HEADER_MAGIC,
    .header.w = width,
    .header.h = height,
    .data_size = total_bytes,
    .data = image_map,
};
```

## 使用方法

### 基本用法
```bash
python png_to_rgb565a8.py input.png
```

### 高级用法
```bash
python png_to_rgb565a8.py input.png -o output.c -n my_image --swap16
```

### 参数说明
- `input.png`: 输入PNG文件
- `-o output.c`: 输出文件名
- `-n my_image`: 变量名
- `--swap16`: 启用字节交换

## 测试和验证

### 运行测试
```bash
python test_png_converter.py
```

### 运行示例
```bash
python example_usage.py
```

### 验证格式
```bash
python verify_format.py generated_file.c
```

## 数据布局示例

### 4x4图像的数据布局
```
总数据大小: 48字节 (4*4*3)

RGB565数据 (32字节):
像素(0,0): [0x12, 0x34]
像素(0,1): [0x56, 0x78]
像素(0,2): [0x9a, 0xbc]
...
像素(3,3): [0xfe, 0xdc]

Alpha数据 (16字节):
像素(0,0): [0xff]
像素(0,1): [0xee]
像素(0,2): [0xdd]
...
像素(3,3): [0x11]
```

## 技术细节

### RGB565转换
```python
def rgb888_to_rgb565(r, g, b):
    r = (r >> 3) & 0x1F  # 5位红
    g = (g >> 2) & 0x3F  # 6位绿
    b = (b >> 3) & 0x1F  # 5位蓝
    return (r << 11) | (g << 5) | b
```

### 字节交换
```python
def rgb565_to_bytes(rgb565, swap16=False):
    high_byte = (rgb565 >> 8) & 0xFF
    low_byte = rgb565 & 0xFF
    
    if swap16:
        return [low_byte, high_byte]  # 小端序
    else:
        return [high_byte, low_byte]  # 大端序
```

## 兼容性

- **Python**: 3.6+
- **依赖**: Pillow 8.0+
- **输入格式**: PNG (RGBA, RGB, 灰度等)
- **输出格式**: C文件，兼容LVGL

## 错误处理

- 输入文件验证
- 格式转换错误处理
- 输出文件写入错误处理
- 详细的错误信息输出

## 性能特点

- 内存高效的流式处理
- 支持大尺寸图像
- 快速转换速度
- 格式验证功能

## 使用场景

1. **嵌入式GUI开发**: LVGL项目中的图像资源
2. **游戏开发**: 需要透明度的精灵图像
3. **UI设计**: 图标和界面元素
4. **图像处理**: 批量转换PNG图像

## 扩展性

工具设计为模块化，可以轻松扩展：
- 支持其他颜色格式
- 添加图像预处理功能
- 支持批量处理
- 集成到构建系统

## 总结

这个工具集提供了完整的PNG到RGB565A8转换解决方案，具有以下优势：

✅ **格式正确**: 符合LVGL RGB565A8格式要求  
✅ **功能完整**: 支持透明度、字节交换等  
✅ **易于使用**: 简单的命令行界面  
✅ **验证工具**: 内置格式验证功能  
✅ **文档完善**: 详细的使用说明和示例  
✅ **可扩展**: 模块化设计便于扩展  

适用于需要将PNG图像转换为嵌入式图形库格式的开发场景。 