cmake_minimum_required (VERSION 3.0)

find_package(CUDA REQUIRED)

add_executable(PackHeaders internal/pack_headers.cpp)
add_custom_target(Run_PackHeaders
COMMAND PackHeaders
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running PackHeaders in ${CMAKE_CURRENT_SOURCE_DIR}"
SOURCES internal/pack_headers.cpp
)

set(LIB_SOURCES

internal/md5.c
internal/nvtrc_wrapper.cpp

TRTCContext.cpp
DVVector.cpp
fake_vectors/DVConstant.cpp
fake_vectors/DVCounter.cpp
fake_vectors/DVDiscard.cpp
fake_vectors/DVPermutation.cpp
fake_vectors/DVReverse.cpp
fake_vectors/DVTransform.cpp
functor.cpp
fill.cpp
replace.cpp
for_each.cpp
adjacent_difference.cpp
sequence.cpp
tabulate.cpp
transform.cpp
gather.cpp
scatter.cpp
)

set(LIB_HEADERS
TRTC_api.h
DeviceViewable.h
TRTCContext.h
DVVector.h
functor.h
fill.h
replace.h
for_each.h
adjacent_difference.h
sequence.h
tabulate.h
transform.h
gather.h
scatter.h
)

set(LIB_HEADERS_FAKEVECTORS
fake_vectors/DVConstant.h
fake_vectors/DVCounter.h
fake_vectors/DVDiscard.h
fake_vectors/DVPermutation.h
fake_vectors/DVReverse.h
fake_vectors/DVTransform.h
)

set(INTERNAL_HEADERS
internal/Timing.h
internal/md5.h
internal/nvtrc_wrapper.h
)

set (INCLUDE_DIR 
.
./internal
)


if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
-D"_CRT_SECURE_NO_WARNINGS"
)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

SET(CMAKE_INSTALL_RPATH ../lib)

cuda_add_library (ThrustRTC_static ${LIB_SOURCES} ${LIB_HEADERS} ${LIB_HEADERS_FAKEVECTORS} ${INTERNAL_HEADERS})
cuda_add_library (ThrustRTC SHARED ${LIB_SOURCES} ${LIB_HEADERS} ${LIB_HEADERS_FAKEVECTORS} ${INTERNAL_HEADERS})

add_dependencies(Run_PackHeaders PackHeaders)
add_dependencies(ThrustRTC_static Run_PackHeaders)
add_dependencies(ThrustRTC Run_PackHeaders)

if (WIN32) 
target_compile_definitions(ThrustRTC PRIVATE THRUST_RTC_DLL_EXPORT)
endif()

target_link_libraries(ThrustRTC cuda)

cuda_add_executable(test_trtc test/test_trtc.cpp)
cuda_add_executable(test_for test/test_for.cpp)
cuda_add_executable(test_fill test/test_fill.cpp)
cuda_add_executable(test_replace test/test_replace.cpp)
cuda_add_executable(test_for_each test/test_for_each.cpp)
cuda_add_executable(test_adjacent_difference test/test_adjacent_difference.cpp)
cuda_add_executable(test_sequence test/test_sequence.cpp)
cuda_add_executable(test_tabulate test/test_tabulate.cpp)
cuda_add_executable(test_transform test/test_transform.cpp)
cuda_add_executable(test_gather test/test_gather.cpp)
cuda_add_executable(test_scatter test/test_scatter.cpp)

cuda_add_executable(test_constant test/test_constant.cpp)
cuda_add_executable(test_counter test/test_counter.cpp)
cuda_add_executable(test_discard test/test_discard.cpp)
cuda_add_executable(test_permutation test/test_permutation.cpp)
cuda_add_executable(test_reverse test/test_reverse.cpp)
cuda_add_executable(test_transform_iter test/test_transform_iter.cpp)

if (WIN32) 
target_compile_definitions(test_trtc PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_for PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_fill PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_replace PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_for_each PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_adjacent_difference PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_sequence PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_tabulate PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_transform PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_gather PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_scatter PUBLIC THRUST_RTC_DLL_IMPORT)

target_compile_definitions(test_constant PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_counter PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_discard PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_permutation PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_reverse PUBLIC THRUST_RTC_DLL_IMPORT)
target_compile_definitions(test_transform_iter PUBLIC THRUST_RTC_DLL_IMPORT)
endif()

target_link_libraries(test_trtc ThrustRTC)
target_link_libraries(test_for ThrustRTC)
target_link_libraries(test_fill ThrustRTC)
target_link_libraries(test_replace ThrustRTC)
target_link_libraries(test_for_each ThrustRTC)
target_link_libraries(test_adjacent_difference ThrustRTC)
target_link_libraries(test_sequence ThrustRTC)
target_link_libraries(test_tabulate ThrustRTC)
target_link_libraries(test_transform ThrustRTC)
target_link_libraries(test_gather ThrustRTC)
target_link_libraries(test_scatter ThrustRTC)

target_link_libraries(test_constant ThrustRTC)
target_link_libraries(test_counter ThrustRTC)
target_link_libraries(test_discard ThrustRTC)
target_link_libraries(test_permutation ThrustRTC)
target_link_libraries(test_reverse ThrustRTC)
target_link_libraries(test_transform_iter ThrustRTC)

install(FILES ${LIB_HEADERS} DESTINATION include)
install(FILES ${LIB_HEADERS_FAKEVECTORS} DESTINATION include/fake_vectors)
install(TARGETS ThrustRTC_static DESTINATION lib)

if (WIN32) 
install(TARGETS ThrustRTC 
		RUNTIME DESTINATION test_cpp
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib 
		)
else()
install(TARGETS ThrustRTC DESTINATION lib)
endif()

install(TARGETS 
test_trtc 
test_for 
test_fill 
test_replace 
test_for_each 
test_adjacent_difference 
test_sequence
test_tabulate
test_transform
test_gather
test_scatter

test_constant
test_counter
test_discard
test_permutation
test_reverse
test_transform_iter

RUNTIME DESTINATION test_cpp)

install(DIRECTORY DESTINATION test_cpp/__ptx_cache__)

add_subdirectory(python)
