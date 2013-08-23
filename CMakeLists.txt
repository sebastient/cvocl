cmake_minimum_required(VERSION 2.8)

project(cvocl C)

add_definitions(-std=c99 -Wall)

include(FindPkgConfig)

pkg_check_modules(GLIB glib-2.0)

find_path(CF4OCL_INCLUDE_DIRS clutils.h HINTS ${CMAKE_FIND_ROOT_PATH})
find_library(CF4OCL_LIBRARIES cf4ocl)

if (APPLE)
    find_path(CL_INCLUDE_DIRS OpenCL/cl.h HINTS ${CMAKE_FIND_ROOT_PATH})
    find_library(CL_LIBRARIES OpenCL)
else()
    find_path(CL_INCLUDE_DIRS CL/cl.h HINTS ${CMAKE_FIND_ROOT_PATH})
    find_library(CL_LIBRARY NAMES OpenCL libOpenCL)
    find_library(CLC_LIBRARY NAMES CLC libCLC)
    find_library(GAL_LIBRARY NAMES GAL libGAL)
    set(CL_LIBRARIES ${CL_LIBRARY} ${CLC_LIBRARY} ${GAL_LIBRARY})
endif()

message("Glib ${GLIB_INCLUDE_DIRS} ${GLIB_LIBRARIES}")
message("OpenCL ${CL_INCLUDE_DIRS} ${CL_LIBRARIES}")

link_directories(${GLIB_LIBRARY_DIRS})
include_directories(${CF4OCL_INCLUDE_DIRS} ${GLIB_INCLUDE_DIRS})

add_executable(cvocl cvocl.c)
target_link_libraries(cvocl ${GLIB_LIBRARIES} ${CF4OCL_LIBRARIES} ${CL_LIBRARIES})
install(TARGETS cvocl RUNTIME DESTINATION bin)

