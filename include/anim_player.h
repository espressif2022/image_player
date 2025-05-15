#pragma once

#include <stdbool.h>
#include "esp_err.h"
#include "esp_mmap_assets.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *anim_player_handle_t;

typedef void (*flush_cb_t)(int x, int y, int width, int height, const void *data);

typedef enum {
    PLAYER_ACTION_STOP = 0,
    PLAYER_ACTION_START,
} player_event_t;

typedef struct {
    const char *partition_label;  ///< Partition label for assets
    int max_files;               ///< Maximum number of files
    uint32_t checksum;           ///< Checksum for verification
    flush_cb_t flush_cb;         ///< Callback function for flushing decoded data
} anim_player_config_t;

/**
 * @brief Initialize animation player
 *
 * @param config Player configuration
 * @return anim_player_handle_t Player handle, NULL on error
 */
anim_player_handle_t anim_player_init(const anim_player_config_t *config);

/**
 * @brief Deinitialize animation player
 *
 * @param handle Player handle
 */
void anim_player_deinit(anim_player_handle_t handle);

/**
 * @brief Update player settings
 *
 * @param handle Player handle
 * @param event New event
 * @param start_index New start index
 * @param end_index New end index
 * @param repeat New repeat setting
 * @param fps New frame rate in frames per second
 */
void anim_player_update(anim_player_handle_t handle, player_event_t event, int start_index, int end_index, bool repeat, int fps);

/**
 * @brief Check if flush is ready
 *
 * @param handle Player handle
 * @return bool True if the flush is ready, false otherwise
 */
bool anim_player_flush_ready(anim_player_handle_t handle);

#ifdef __cplusplus
}
#endif