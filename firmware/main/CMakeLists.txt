idf_component_register(SRCS "main.cpp" "async_file.cpp" "sd_card.cpp" "sd_speed.cpp" "usb_upload.cpp" "sd_bench.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES lfm M5Unified M5GFX fatfs sdmmc esp_driver_sdmmc esp_driver_sdspi esp_partition esp_timer esp_driver_usb_serial_jtag)
