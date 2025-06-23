# Get all source files recursively from src directory
file(GLOB_RECURSE SRC_FILES "src/*.c")

idf_component_register(
    SRCS ${SRC_FILES}
    INCLUDE_DIRS
        "include"
        "include_priv"
    REQUIRES esp_timer
)
