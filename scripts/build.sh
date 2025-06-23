#!/bin/bash

# 构建脚本
echo "构建 Image Player 组件..."

# 检查是否在 ESP-IDF 环境中
if [ -z "$IDF_PATH" ]; then
    echo "错误: 请先设置 ESP-IDF 环境"
    echo "请运行: . \$HOME/esp/esp-idf/export.sh"
    exit 1
fi

# 构建组件
idf.py build

echo "构建完成!"
