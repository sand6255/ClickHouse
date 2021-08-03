option(ENABLE_LIBTORCH "Enable libtorch" ${ENABLE_LIBRARIES})

if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/pytorch/CMakeLists.txt")
    message(WARNING "submodule contrib/pytorch is missing. to fix try run: \n git submodule update --init --recursive")
endif()

if (ENABLE_LIBTORCH)
    set (USE_LIBTORCH 1)
endif()

message (STATUS "Using libtorch: ${USE_LIBTORCH}")
