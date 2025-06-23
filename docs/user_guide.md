# Image Player 用户指南

## 快速开始

### 1. 初始化播放器
```c
anim_player_config_t config = ANIM_PLAYER_INIT_CONFIG();
config.flush_cb = your_flush_callback;
config.update_cb = your_update_callback;

anim_player_handle_t player = anim_player_init(&config);
```

### 2. 设置动画数据
```c
esp_err_t ret = anim_player_set_src_data(player, anim_data, data_len);
```

### 3. 开始播放
```c
anim_player_update(player, PLAYER_ACTION_START);
```

## 添加标签叠加
```c
gfx_lable_cfg_t label_cfg = {
    .name = "font1",
    .mem = font_data,
    .mem_size = font_size
};

gfx_obj_t *label = gfx_label_create(&label_cfg, player);
gfx_obj_set_pos(label, 100, 100);
```
