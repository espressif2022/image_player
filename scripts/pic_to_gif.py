from PIL import Image
import os

# 设置输入目录、文件名模式和输出 GIF 文件名
input_dir = './frame_20'   # 你的帧文件所在目录
output_gif = 'output.gif'
frame_pattern = 'test_{:02d}.png'  # 文件名序号格式：frame_0001.png, frame_0002.png, ...

# 设置帧范围
start_frame = 1
end_frame = 20  # 假设总共有 100 帧

# 加载帧
frames = []
for i in range(start_frame, end_frame + 1):
    frame_path = os.path.join(input_dir, frame_pattern.format(i))
    if os.path.exists(frame_path):
        img = Image.open(frame_path).convert('RGBA')  # 可根据需要使用 'RGB'
        frames.append(img)
    else:
        print(f"Warning: {frame_path} not found.")

# 保存为 GIF
if frames:
    # duration: 每帧时间 (毫秒)；loop=0: 无限循环
    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"GIF saved as: {output_gif}")
else:
    print("No frames loaded!")
