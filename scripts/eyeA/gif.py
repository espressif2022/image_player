from PIL import Image
import glob
import os

# 匹配所有以 xx_ 开头、.jpg 结尾的文件
jpg_files = sorted(glob.glob("happy_*.jpg"))

# 检查是否有文件
if not jpg_files:
    print("未找到匹配的 JPG 文件")
    exit()

# 打开第一张图作为起始帧
first_frame = Image.open(jpg_files[0])

# 其余帧
frames = [Image.open(f) for f in jpg_files[1:]]

# 保存为 GIF
output_path = "output.gif"
first_frame.save(
    output_path,
    format="GIF",
    save_all=True,
    append_images=frames,
    duration=200,   # 每帧持续时间（毫秒）
    loop=0          # 无限循环
)

print(f"GIF 已保存为: {output_path}")
