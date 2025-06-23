# Image Player API 参考

## 概述
Image Player 是一个用于播放动画图像的组件，支持多种图像格式和渲染功能。

## 主要模块

### 动画播放器 (anim_player)
- `anim_player_init()` - 初始化播放器
- `anim_player_deinit()` - 反初始化播放器
- `anim_player_set_src_data()` - 设置源数据
- `anim_player_update()` - 更新播放状态

### 字体标签 (ft_label)
- `gfx_lable_new_font()` - 创建新字体
- `gfx_lable_set_text()` - 设置文本内容
- `gfx_lable_render_mask()` - 渲染文本遮罩

### 图像混合 (ft_blend)
- `blend_sw_img_draw()` - 软件图像绘制
- 支持多种混合模式

### 图形对象 (object)
- `gfx_label_create()` - 创建标签对象
- `gfx_image_create()` - 创建图像对象
- `gfx_obj_set_pos()` - 设置对象位置
