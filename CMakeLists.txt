# SPDX-FileCopyrightText: 2026 Kaito Udagawa <umireon@kaito.tokyo>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)

### Options
set(PLUGIN_DISPLAY_NAME "OBS Background Removal" CACHE STRING "Display name of plugin")
set(PLUGIN_AUTHOR "Roy Shilkrot" CACHE STRING "Author of plugin")
set(PLUGIN_EMAIL "roy.shil@gmail.com" CACHE STRING "Author email of plugin")
set(PLUGIN_BUILD_NUMBER "1" CACHE STRING "Build number in plugin metadata")
set(PLUGIN_BUNDLE_ID "com.royshilkrot.obs-backgroundremoval" CACHE STRING "Bundle ID of plugin installer")
set(PACKAGE_VERSION_SUFFIX "" CACHE STRING "Version suffix for plugin package")

### Project
block(PROPAGATE PLUGIN_VERSION)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_CONTENT)
  string(STRIP "${VERSION_CONTENT}" PLUGIN_VERSION)
endblock()

project(obs-backgroundremoval VERSION "${PLUGIN_VERSION}" LANGUAGES C CXX)
set(PLUGIN_NAME "${CMAKE_PROJECT_NAME}")

### Global configurations
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_C_EXTENSIONS FALSE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

if(NOT APPLE)
  message(FATAL_ERROR "This fork builds for macOS only.")
endif()

add_compile_options(-fno-omit-frame-pointer -fstack-protector-strong)

set(CMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO RelWithDebInfo Release None)

### Configure paths
if(DEFINED ENV{PLUGIN_BUILD_DIR})
  set(PLUGIN_BUILD_DIR "$ENV{PLUGIN_BUILD_DIR}")
else()
  set(PLUGIN_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

list(
  PREPEND
  CMAKE_PREFIX_PATH
  "${PLUGIN_BUILD_DIR}/.deps/obs-deps-prebuilt"
  "${PLUGIN_BUILD_DIR}/.deps/obs-deps-qt6"
  "${PLUGIN_BUILD_DIR}/obs_installed"
  "${PLUGIN_BUILD_DIR}/obs_installed/Frameworks"
  "${PLUGIN_BUILD_DIR}/vcpkg_ort_installed/universal-osx"
  "${PLUGIN_BUILD_DIR}/ort_installed"
  "${PLUGIN_BUILD_DIR}/vcpkg_installed/universal-osx-obs"
)

### Package dependencies
find_package(PkgConfig QUIET)

find_package(libobs REQUIRED)
find_package(obs-frontend-api REQUIRED)

block(PROPAGATE OpenCV_FOUND)
  find_package(OpenCV CONFIG QUIET)

  if(NOT OpenCV_FOUND AND PkgConfig_FOUND)
    pkg_check_modules(PC_OPENCV opencv4 IMPORTED_TARGET)
    if(PC_OPENCV_FOUND)
      add_library(OpenCV::opencv_core ALIAS PkgConfig::PC_OPENCV)
      add_library(OpenCV::opencv_imgproc ALIAS PkgConfig::PC_OPENCV)
      set(OpenCV_FOUND TRUE)
    endif()
  endif()

  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found via CMake or pkg-config.")
  endif()

  if(NOT TARGET OpenCV::opencv_core)
    add_library(OpenCV::opencv_core ALIAS opencv_core)
  endif()

  if(NOT TARGET OpenCV::opencv_imgproc)
    add_library(OpenCV::opencv_imgproc ALIAS opencv_imgproc)
  endif()

  if(TARGET opencv_geometry AND NOT TARGET OpenCV::opencv_geometry)
    add_library(OpenCV::opencv_geometry ALIAS opencv_geometry)
  endif()
endblock()

block(PROPAGATE onnxruntime_FOUND)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

  if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
  endif()

  find_package(onnxruntime CONFIG)

  if(NOT onnxruntime_FOUND AND PkgConfig_FOUND)
    pkg_check_modules(PC_onnxruntime onnxruntime IMPORTED_TARGET)
    if(PC_onnxruntime_FOUND)
      add_library(onnxruntime::onnxruntime ALIAS PkgConfig::PC_onnxruntime)
      set(onnxruntime_FOUND TRUE)
    endif()
  endif()

  if(NOT onnxruntime_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found via CMake or pkg-config.")
  endif()

  if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime ALIAS onnxruntime)
  endif()
endblock()

### Plugin main
add_library(obs-backgroundremoval_options INTERFACE)

target_compile_definitions(obs-backgroundremoval_options INTERFACE SIMDE_ENABLE_OPENMP)
target_compile_options(obs-backgroundremoval_options INTERFACE -fopenmp-simd -Wall -Wextra -Werror)

add_library(${CMAKE_PROJECT_NAME} MODULE)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    obs-backgroundremoval_options
    OBS::libobs
    OBS::obs-frontend-api
    onnxruntime::onnxruntime
    OpenCV::opencv_core
    OpenCV::opencv_imgproc
    $<$<TARGET_EXISTS:OpenCV::opencv_geometry>:OpenCV::opencv_geometry>
)

target_sources(
  ${CMAKE_PROJECT_NAME}
  PRIVATE
    src/plugin-main.c
    src/plugin-support.c
    src/ort-utils/ort-session-utils.cpp
    src/obs-utils/obs-utils.cpp
    src/background-filter-info.c
    src/background-filter.cpp
)

### Hide symbols
target_link_options(
  ${CMAKE_PROJECT_NAME}
  PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/exported_symbols_macos.txt"
)

### Install configurations
set(
  EFFECTS_FILES
  data/effects/kawase_blur.effect
  data/effects/mask_alpha_filter.effect
)

set(LOCALE_FILES data/locale/en-US.ini)

set(
  MODELS_FILES
  data/models/mediapipe.with_runtime_opt.ort
  data/models/mediapipe.with_runtime_opt.ort.license
  data/models/rvm_mobilenetv3_fp32.with_runtime_opt.ort
  data/models/rvm_mobilenetv3_fp32.with_runtime_opt.ort.license
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/packaging/pkg/Distribution.xml.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Distribution"
  @ONLY
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES BUNDLE TRUE BUNDLE_EXTENSION "plugin")
set_source_files_properties(${EFFECTS_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/effects")
set_source_files_properties(${LOCALE_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/locale")
set_source_files_properties(${MODELS_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/models")
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${EFFECTS_FILES} ${LOCALE_FILES} ${MODELS_FILES})

install(
  TARGETS "${CMAKE_PROJECT_NAME}"
  BUNDLE DESTINATION "Library/Application Support/obs-studio/plugins"
  LIBRARY DESTINATION "Library/Application Support/obs-studio/plugins"
)

### Export variables
block()
  set(SH_CONTENT "")

  foreach(VAR_NAME PLUGIN_BUNDLE_ID PLUGIN_DISPLAY_NAME PLUGIN_NAME PLUGIN_VERSION)
    set(VAR_VALUE "${${VAR_NAME}}")
    if(NOT VAR_VALUE MATCHES [[^[-A-Za-z0-9._ ]+$]])
      message(FATAL_ERROR "${VAR_NAME} contains unsupported characters: ${VAR_VALUE}")
    endif()
    string(APPEND SH_CONTENT "${VAR_NAME}=\'${VAR_VALUE}\'\n")
  endforeach()

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake_export.sh" "${SH_CONTENT}")
endblock()
