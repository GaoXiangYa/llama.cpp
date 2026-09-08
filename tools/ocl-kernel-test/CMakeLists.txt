# ggml-ocl kernel test harness (DESIGN.md section 23)

find_package(OpenCL REQUIRED)

set(TARGET ocl-op-test)

add_executable(${TARGET}
    main.cpp
    cases/add.cpp
    cases/set_rows.cpp
    cases/get_rows.cpp
    cases/mul_mat.cpp
    cases/softmax.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-ocl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)
target_link_libraries(${TARGET} PRIVATE
    ggml-ocl
    ggml
    ${OpenCL_LIBRARIES}
)
target_compile_definitions(${TARGET} PRIVATE GGML_OCL_TARGET_VERSION=${GGML_OCL_TARGET_VERSION})

install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
