cmake_minimum_required(VERSION 3.28)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_VERBOSE_MAKEFILE ON)

include(cmake/prelude.cmake)

project(
    speech-vs-non-speech
    VERSION 0.1.0
    DESCRIPTION "Short description"
    HOMEPAGE_URL "https://github.com/mrdcvlsc/speech-vs-non-speech/"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)
include(FetchContent)

# libtorch

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message("Operating System Type: LINUX")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcpu.zip
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message("Operating System Type: APPLE")
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
      message(STATUS "x86_64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.5.0.zip
      )
  elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
      message(STATUS "arm64 architecture detected")
      FetchContent_Declare(Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.0.zip
      )
  else()
      message(FATAL_ERROR "Other architecture detected")
  endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message("Operating System Type: WINDOWS")
  FetchContent_Declare(Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.0%2Bcpu.zip
  )
else()
  message("Operating System Type: Unknown")
  message(FATAL_ERROR "Cannot be supported")
endif()
message("--------------------------")

# libtorch - make the libtorch library available to the project
FetchContent_MakeAvailable(Torch)

# libtorch - add the lib paths of the downloaded library from FetchContent
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")

# libtorch - find the libtorch lib and require it for the project
find_package(Torch REQUIRED)

# libtorch - link the libtorch libraries
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---- Declare library ----

FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.1
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

add_library(
    speech-vs-non-speech_lib OBJECT
    source/lib.cpp
)

target_include_directories(
    speech-vs-non-speech_lib ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(speech-vs-non-speech_lib PUBLIC cxx_std_17)

# ---- Declare executable ----

add_executable(speech-vs-non-speech_exe source/main.cpp)
add_executable(speech-vs-non-speech::exe ALIAS speech-vs-non-speech_exe)

set_property(TARGET speech-vs-non-speech_exe PROPERTY OUTPUT_NAME speech-vs-non-speech)

target_compile_features(speech-vs-non-speech_exe PRIVATE cxx_std_17)

target_link_libraries(speech-vs-non-speech_exe PRIVATE speech-vs-non-speech_lib SFML::Audio "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET speech-vs-non-speech_exe
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:speech-vs-non-speech_exe>)
endif (MSVC)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT speech-vs-non-speech_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of speech-vs-non-speech"
  )
endif()

include(cmake/dev-mode.cmake)
