#!/bin/bash

# 测试脚本
echo "运行 Image Player 测试..."

# 运行单元测试
cd tests/unit
idf.py build
idf.py flash monitor

echo "测试完成!"
