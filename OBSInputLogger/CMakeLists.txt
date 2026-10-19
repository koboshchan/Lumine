cmake_minimum_required(VERSION 3.28)
project(lumine-input-logger VERSION 0.1.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "Lumine Input Logger currently supports Windows only")
endif()

# Same imported targets as obs-plugintemplate; no Qt dependency.
find_package(libobs 32 REQUIRED CONFIG)
find_package(obs-frontend-api REQUIRED CONFIG)
add_library(lumine-input-logger MODULE src/plugin-main.cpp)
target_compile_features(lumine-input-logger PRIVATE cxx_std_17)
target_compile_definitions(lumine-input-logger PRIVATE UNICODE _UNICODE _WIN32_WINNT=0x0A00)
target_link_libraries(lumine-input-logger PRIVATE OBS::libobs OBS::obs-frontend-api user32)
if(MSVC)
  target_compile_options(lumine-input-logger PRIVATE /W4 /permissive- /utf-8 /EHsc)
else()
  target_compile_options(lumine-input-logger PRIVATE -Wall -Wextra -Wpedantic)
endif()
set_target_properties(lumine-input-logger PROPERTIES PREFIX "" OUTPUT_NAME "lumine-input-logger")
install(TARGETS lumine-input-logger RUNTIME DESTINATION obs-plugins/64bit
        LIBRARY DESTINATION obs-plugins/64bit)
