cmake_minimum_required(VERSION 3.10)
project(roadglyph_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── ONNX Runtime ───────────────────────────────────────────────────────────────
set(ORT_SRC "/media/nvidia/GLAD/ort_build/onnxruntime" CACHE PATH "ORT source tree")
set(ORT_BUILD "/media/nvidia/GLAD/ort_build/build/Release" CACHE PATH "ORT build dir")

set(ORT_INCLUDE ${ORT_SRC}/include/onnxruntime/core/session)
set(ORT_LIB_DIR ${ORT_BUILD})

if(NOT EXISTS ${ORT_INCLUDE}/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found: ${ORT_INCLUDE}")
endif()

find_library(ORT_LIB onnxruntime HINTS ${ORT_LIB_DIR} REQUIRED)

# ── CUDA (optional) ────────────────────────────────────────────────────────────
option(USE_CUDA "Enable CUDA support" ON)
if(USE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}  (${CUDA_TOOLKIT_ROOT_DIR})")
        enable_language(CUDA)
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "CUDA not found → CPU-only build")
        set(USE_CUDA OFF)
    endif()
endif()

# ── DriveWorks SDK ─────────────────────────────────────────────────────────────
set(DW_ROOT "/usr/local/driveworks-2.2/targets/aarch64-Linux" CACHE PATH "DriveWorks SDK root")
set(DW_INCLUDE ${DW_ROOT}/include)
set(DW_LIB_DIR ${DW_ROOT}/lib)

if(NOT EXISTS ${DW_INCLUDE}/dw/Driveworks.h)
    message(FATAL_ERROR "DriveWorks SDK not found: ${DW_INCLUDE}")
endif()

find_library(DW_LIB driveworks HINTS ${DW_LIB_DIR} REQUIRED)
find_library(DW_VIZ_LIB driveworks_visualization HINTS ${DW_LIB_DIR} REQUIRED)
find_library(GLFW_LIB glfw HINTS ${DW_LIB_DIR} REQUIRED)

# GLFW headers (bundled with DW SDK 3rdparty)
set(GLFW_INCLUDE "/usr/local/driveworks-2.2/samples/3rdparty/src/glfw/3.2-screen/include"
    CACHE PATH "GLFW include path")

# ── Helper macro: link ORT + CUDA ──────────────────────────────────────────────
macro(link_ort_cuda target_name)
    target_include_directories(${target_name} PRIVATE ${ORT_INCLUDE})
    target_link_libraries(${target_name} PRIVATE ${ORT_LIB})
    if(USE_CUDA AND CUDA_FOUND)
        target_link_libraries(${target_name} PRIVATE
            ${CUDA_LIBRARIES}
            ${CUDA_cudart_static_LIBRARY}
        )
        target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    endif()
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH   "${ORT_LIB_DIR};${DW_LIB_DIR}"
        INSTALL_RPATH "${ORT_LIB_DIR};${DW_LIB_DIR}"
    )
endmacro()

# ── Target: inference_v3 ───────────────────────────────────────────────────────
if(USE_CUDA AND CUDA_FOUND)
    add_executable(inference_v3 inference_v3.cpp preprocess_cuda.cu)
else()
    add_executable(inference_v3 inference_v3.cpp)
endif()
link_ort_cuda(inference_v3)
target_include_directories(inference_v3 PRIVATE
    ${DW_INCLUDE}
    ${GLFW_INCLUDE}
)
target_link_libraries(inference_v3 PRIVATE
    ${DW_LIB}
    ${DW_VIZ_LIB}
    ${GLFW_LIB}
    EGL
    GLESv2
    X11
    /usr/lib/aarch64-linux-gnu/libXcursor.so.1
    /usr/lib/aarch64-linux-gnu/libXrandr.so.2
    /usr/lib/aarch64-linux-gnu/libXinerama.so.1
)

message(STATUS "ORT_ROOT  : ${ORT_ROOT}")
message(STATUS "ORT_LIB   : ${ORT_LIB}")
message(STATUS "DW_ROOT   : ${DW_ROOT}")
message(STATUS "DW_LIB    : ${DW_LIB}")
message(STATUS "USE_CUDA  : ${USE_CUDA}")
