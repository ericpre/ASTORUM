cmake_minimum_required(VERSION 3.10)
project(BlockWiseSmoothNMFlib VERSION 0.1.0 LANGUAGES CXX)

##### Set default behavior #####
set(DEFAULT_USE_OMP Yes)
set(DEFAULT_USE_TEST No)
set(DEFAULT_COVERAGE No)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

##### Check compiler Version #####
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 7.0.0)
		message(FATAL_ERROR "g++ >= 7.0.0 is required.")
	endif()

	if("${CMAKE_HOST_SYSTEM_PROCESSOR}" MATCHES "^x86.*")
		set(DEFAULT_OPT_FLAGS "-mtune=native -march=native -mfpmath=both")
	
    else()
		set(DEFAULT_OPT_FLAGS "-mtune=native -march=native")
    endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	message(STATUS "MSVC_VERSION = ${MSVC_VERSION}")

	if("${MSVC_VERSION}" VERSION_LESS 1900)
		message(FATAL_ERROR "MSVC >= 2015 is required.")
	endif()

elseif (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
	set(DEFAULT_USE_OMP No)

	if(${COVERAGE})
		message(FATAL_ERROR "Coverage is not supported for clang. Do not set COVERAGE=Yes on this environment.")
	endif()

else()
	message(FATAL_ERROR "Unsupported Compiler.")

endif()

##### Update with given flags #####
if(NOT DEFINED USE_OMP)
	set(USE_OMP ${DEFAULT_USE_OMP})
endif()

if(NOT DEFINED USE_TEST)
	set(USE_TEST ${DEFAULT_USE_TEST})
endif()

if(NOT DEFINED COVERAGE)
	set(COVERAGE ${DEFAULT_COVERAGE})
endif()

if(NOT DEFINED OPT_FLAGS)
	set(OPT_FLAGS ${DEFAULT_OPT_FLAGS})
endif()

message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR = ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "USE_OMP = ${USE_OMP}")
message(STATUS "USE_TEST = ${USE_TEST}")
message(STATUS "COVERAGE = ${COVERAGE}")
message(STATUS "OPT_FLAGS = ${OPT_FLAGS}")

#Include directory
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/external")

####### External libraries #######
find_package(pybind11 CONFIG REQUIRED)

##### Set warnings #####
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(WARNING_C "-Wall -Wextra -Werror=undef -Wlogical-op -Wmissing-include-dirs \
        -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
        -Werror=missing-prototypes -Werror=implicit-function-declaration -Werror=missing-declarations -Werror=return-type \
        -Werror=incompatible-pointer-types -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
        -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
        -Wnested-externs -fdiagnostics-color=auto")
    set(WARNING_CPP "-Wall -Wextra -Wlogical-op -Wmissing-include-dirs \
        -Wpointer-arith -Winit-self -Wfloat-equal -Wsuggest-attribute=noreturn \
        -Werror=missing-declarations -Werror=return-type \
        -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
        -Wimplicit-fallthrough=5 -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
        -fdiagnostics-color=auto")

elseif (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    set(WARNING_C "-Wall -Wextra -Werror=undef -Wmissing-include-dirs \
        -Wpointer-arith -Winit-self -Wfloat-equal \
        -Werror=missing-prototypes -Werror=implicit-function-declaration -Werror=missing-declarations -Werror=return-type \
        -Werror=incompatible-pointer-types -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
        -Wimplicit-fallthrough -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
        -Wnested-externs -fdiagnostics-color=auto")
    set(WARNING_CPP "-Wall -Wextra -Wmissing-include-dirs \
        -Wpointer-arith -Winit-self -Wfloat-equal \
        -Werror=missing-declarations -Werror=return-type \
        -Werror=format=2 -Wredundant-decls -Wmissing-noreturn \
        -Wimplicit-fallthrough -Wshadow -Wendif-labels -Wstrict-aliasing=2 -Wwrite-strings -Werror=overflow -Wdate-time \
        -fdiagnostics-color=auto")
endif()

##### Set flags #####
if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)

	# Set C++ standard
	set(CMAKE_CXX_STANDARD 17)
	set(PYBIND11_CPP_STANDARD -std=c++17)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    set(CMAKE_CXX_FLAGS "-O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

	# Enable pthread
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

	# Enable imaginary literals for C++
	if(NOT MINGW AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals")
	endif()

	# Enable OpenMP
	if(USE_OMP)
 		find_package(OpenMP QUIET REQUIRED)
   		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()

	# Add optimisation flags
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

	# Add coverage flags
	if(COVERAGE)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
	endif()

elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})

elseif(MSVC)
	#Compile csim with cpp compiler since VC does not completely support C99 complex values
	SET(CMAKE_C_COMPILER ${CMAKE_CXX_COMPILER})
	add_compile_options("/TP")
	add_compile_options("/std:c++17")
	add_compile_options("/permissive")

	# Ignore warning
	## ignore warning about template export
	add_compile_options(/wd4251)
	## ignore warning on non-Unicode files
	add_compile_options(/wd4819)

	# enable OpenMP
	if(USE_OMP)
		add_compile_options("/openmp")
	endif()

	# Static link to multithread runtimes
	set(variables
		CMAKE_CXX_FLAGS_DEBUG
		CMAKE_CXX_FLAGS_RELEASE
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		CMAKE_CXX_FLAGS_MINSIZEREL
	)

	foreach(variable ${variables})
		if(${variable} MATCHES "/MD")
			string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
		endif()
	endforeach()
	
else()
	message(FATAL "Unsupported environment")
endif()


##### show configurations #####
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG = ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}")

set(SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/BlockWiseSmoothNMF.cpp
    ${CMAKE_SOURCE_DIR}/src/dichotomy.cpp
    ${CMAKE_SOURCE_DIR}/src/EDXSDataset.cpp
    ${CMAKE_SOURCE_DIR}/src/EELSDataset.cpp
    ${CMAKE_SOURCE_DIR}/src/measures.cpp
    ${CMAKE_SOURCE_DIR}/src/modelutils.cpp
    ${CMAKE_SOURCE_DIR}/src/SmoothNMF.cpp
    ${CMAKE_SOURCE_DIR}/src/surrogates.cpp
    ${CMAKE_SOURCE_DIR}/src/updates.cpp
    )

set(INCLUDE_FILES 
    ${CMAKE_SOURCE_DIR}/include/BlockWiseSmoothNMF.h
    ${CMAKE_SOURCE_DIR}/include/constants.hpp
    ${CMAKE_SOURCE_DIR}/include/dichotomy.h
    ${CMAKE_SOURCE_DIR}/include/EDXSDataset.h
    ${CMAKE_SOURCE_DIR}/include/EELSDataset.h
    ${CMAKE_SOURCE_DIR}/include/measures.h
    ${CMAKE_SOURCE_DIR}/include/modelutils.h
    ${CMAKE_SOURCE_DIR}/include/SmoothNMF.h
    ${CMAKE_SOURCE_DIR}/include/surrogates.h
    ${CMAKE_SOURCE_DIR}/include/updates.h
    )

set(PY_BINDING_FILE
    ${CMAKE_SOURCE_DIR}/pybind/SmoothNMF_binding.cpp
)

pybind11_add_module(BlockWiseSmoothNMFlib
    ${SRC_FILES}
    ${INCLUDE_FILES}
    ${PY_BINDING_FILE}
)

target_link_libraries(BlockWiseSmoothNMFlib PUBLIC pybind11::embed)

install(TARGETS BlockWiseSmoothNMFlib
    COMPONENT python
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/out")