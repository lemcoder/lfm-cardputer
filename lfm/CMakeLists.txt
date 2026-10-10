# Portable inference core. Built two ways:
#   - as an ESP-IDF component (firmware/ pulls it in via EXTRA_COMPONENT_DIRS)
#   - as a plain static library for the host reference build (host/)
file(GLOB LFM_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

if(ESP_PLATFORM)
    idf_component_register(SRCS ${LFM_SRCS} INCLUDE_DIRS "include")
    target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
else()
    add_library(lfm STATIC ${LFM_SRCS})
    target_include_directories(lfm PUBLIC include)
    target_compile_features(lfm PUBLIC cxx_std_20)
endif()
