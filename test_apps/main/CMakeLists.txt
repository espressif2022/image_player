
idf_component_register(
    SRC_DIRS "."
    INCLUDE_DIRS "."
)

spiffs_create_partition_assets(
    assets_4bit
    "../test_4bit"
    FLASH_IN_PROJECT
    MMAP_FILE_SUPPORT_FORMAT ".aaf"
)

spiffs_create_partition_assets(
    assets_8bit
    "../test_8bit"
    FLASH_IN_PROJECT
    MMAP_FILE_SUPPORT_FORMAT ".aaf"
)

set(URL "https://github.com/espressif/esp-docs/raw/f036a337d8bee5d1a93b2264ecd29255baec4260/src/esp_docs/fonts/DejaVuSans.ttf")
set(FILE "DejaVuSans.ttf")
set(SPIFFS_DIR "${CMAKE_BINARY_DIR}/spiffs_assets")
file(MAKE_DIRECTORY ${SPIFFS_DIR})
file(DOWNLOAD ${URL} ${SPIFFS_DIR}/${FILE} SHOW_PROGRESS)

spiffs_create_partition_assets(assets ${SPIFFS_DIR} FLASH_IN_PROJECT MMAP_FILE_SUPPORT_FORMAT ".ttf")