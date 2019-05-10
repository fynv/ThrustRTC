cmake_minimum_required (VERSION 3.0)

add_executable(PackHeaders internal/pack_headers.cpp)
add_custom_target(Run_PackHeaders
COMMAND PackHeaders
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running PackHeaders in ${CMAKE_CURRENT_SOURCE_DIR}"
SOURCES internal/pack_headers.cpp
)

set(LIB_SOURCES
thirdparty/crc64/crc64.cpp
internal/launch_calc.cpp
internal/cuda_wrapper.cpp
internal/nvtrc_wrapper.cpp
internal/general_reduce.cpp
internal/general_scan.cpp

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
copy.cpp
swap.cpp
count.cpp
reduce.cpp
equal.cpp
extrema.cpp
inner_product.cpp
transform_reduce.cpp
logical.cpp
scan.cpp
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
copy.h
swap.h
count.h
reduce.h
equal.h
extrema.h
inner_product.h
transform_reduce.h
logical.h
scan.h
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
thirdparty/crc64/crc64.h
internal/Timing.h
internal/launch_calc.h
internal/cuda_wrapper.h
internal/nvtrc_wrapper.h
internal/general_reduce.h
internal/general_scan.h
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
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

SET(CMAKE_INSTALL_RPATH ../lib)

add_library (ThrustRTC_static ${LIB_SOURCES} ${LIB_HEADERS} ${LIB_HEADERS_FAKEVECTORS} ${INTERNAL_HEADERS})
add_library (ThrustRTC SHARED ${LIB_SOURCES} ${LIB_HEADERS} ${LIB_HEADERS_FAKEVECTORS} ${INTERNAL_HEADERS})

if (WIN32) 
else()
target_link_libraries(ThrustRTC -ldl)
endif()


add_dependencies(Run_PackHeaders PackHeaders)
add_dependencies(ThrustRTC_static Run_PackHeaders)
add_dependencies(ThrustRTC Run_PackHeaders)

if (WIN32) 
target_compile_definitions(ThrustRTC PRIVATE THRUST_RTC_DLL_EXPORT)
endif()

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

add_subdirectory(test)
add_subdirectory(python)
