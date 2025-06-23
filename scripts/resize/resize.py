from PIL import Image
import os
import glob

# 获取当前目录下所有的 GIF 文件
gif_files = glob.glob("*.gif")

# 新尺寸
new_width = 300
new_height = 300

# 处理每个 GIF 文件
for gif_file in gif_files:
    print(f"Processing {gif_file}...")
    
    # 打开 GIF
    im = Image.open(gif_file)
    
    # 输出的帧列表
    frames = []
    
    # 逐帧处理
    for frame in range(0, im.n_frames):
        im.seek(frame)
        frame_image = im.copy()
        
        # 确保图像有 alpha 通道
        if frame_image.mode != 'RGBA':
            frame_image = frame_image.convert('RGBA')
        
        # 创建黑色背景
        black_bg = Image.new('RGBA', frame_image.size, (0, 0, 0, 255))
        
        # 将原图合成到黑色背景上
        frame_image = Image.alpha_composite(black_bg, frame_image)
        
        # resize
        resized_frame = frame_image.resize((new_width, new_height), Image.LANCZOS)
        frames.append(resized_frame)
    
    # 生成输出文件名
    output_filename = f"resized_{gif_file}"
    
    # 保存为 GIF
    frames[0].save(output_filename, save_all=True, append_images=frames[1:], 
                  loop=0, duration=im.info["duration"], disposal=2)
    print(f"Saved as {output_filename}")

print("All GIF files have been processed!")
