cmake_minimum_required (VERSION 3.0)

project(PyThrustRTC)

add_executable(PackHeaders ../internal/pack_headers.cpp)
add_custom_target(Run_PackHeaders
COMMAND PackHeaders
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
COMMENT "Running PackHeaders in ${CMAKE_CURRENT_SOURCE_DIR}/.."
SOURCES ../internal/pack_headers.cpp
)

add_custom_target(Run_CFFIBuild
COMMAND python ThrustRTC/cffi_build.py
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running cffi_build")

set(LIB_SOURCES
../thirdparty/crc64/crc64.cpp
../internal/launch_calc.cpp
../internal/cuda_wrapper.cpp
../internal/nvtrc_wrapper.cpp
../internal/general_reduce.cpp
../internal/general_scan.cpp
../internal/general_scan_by_key.cpp
../internal/general_copy_if.cpp
../internal/general_find.cpp
../internal/merge_sort.cpp
../internal/radix_sort.cpp

../TRTCContext.cpp
../DVVector.cpp
../DVTuple.cpp
../fake_vectors/DVRange.cpp
../fake_vectors/DVConstant.cpp
../fake_vectors/DVCounter.cpp
../fake_vectors/DVDiscard.cpp
../fake_vectors/DVPermutation.cpp
../fake_vectors/DVReverse.cpp
../fake_vectors/DVTransform.cpp
../fake_vectors/DVZipped.cpp
../fake_vectors/DVCustomVector.cpp
../functor.cpp
../fill.cpp
../replace.cpp
../for_each.cpp
../adjacent_difference.cpp
../sequence.cpp
../tabulate.cpp
../transform.cpp
../gather.cpp
../scatter.cpp
../copy.cpp
../swap.cpp
../count.cpp
../reduce.cpp
../equal.cpp
../extrema.cpp
../inner_product.cpp
../transform_reduce.cpp
../logical.cpp
../scan.cpp
../transform_scan.cpp
../scan_by_key.cpp
../remove.cpp
../unique.cpp
../partition.cpp
../find.cpp
../mismatch.cpp
../binary_search.cpp
../merge.cpp
../sort.cpp
)

set(LIB_HEADERS
../TRTC_api.h
../DeviceViewable.h
../built_in.h
../TRTCContext.h
../DVVector.h
../DVTuple.h
../functor.h
../fill.h
../replace.h
../for_each.h
../adjacent_difference.h
../sequence.h
../tabulate.h
../transform.h
../gather.h
../scatter.h
../copy.h
../swap.h
../count.h
../reduce.h
../equal.h
../extrema.h
../inner_product.h
../transform_reduce.h
../logical.h
../scan.h
../transform_scan.h
../remove.h
../unique.h
../partition.h
../find.h
../mismatch.h
../binary_search.h
../merge.h
../sort.h
)

set(LIB_HEADERS_FAKEVECTORS
../fake_vectors/DVRange.h
../fake_vectors/DVConstant.h
../fake_vectors/DVCounter.h
../fake_vectors/DVDiscard.h
../fake_vectors/DVPermutation.h
../fake_vectors/DVReverse.h
../fake_vectors/DVTransform.h
../fake_vectors/DVZipped.h
../fake_vectors/DVCustomVector.h
)

set(INTERNAL_HEADERS
../thirdparty/crc64/crc64.h
../internal/impl_context.inl
../internal/Timing.h
../internal/launch_calc.h
../internal/cuda_wrapper.h
../internal/nvtrc_wrapper.h
../internal/general_reduce.h
../internal/general_scan.h
../internal/general_scan_by_key.h
../internal/general_copy_if.h
../internal/general_find.h
../internal/merge_sort.h
../internal/radix_sort.h
)

set (SRC
api.h
api_utils.cpp
api_Context.cpp
api_DeviceViewable.cpp
api_DVVector.cpp
api_DVTuple.cpp
api_FakeVectors.cpp
api_Functor.cpp
api_Transformations.cpp
api_Copying.cpp
api_Reductions.cpp
api_PrefixSums.cpp
api_Reordering.cpp
api_Searching.cpp
api_Merging.cpp
api_Sorting.cpp
)


set (INCLUDE_DIR 
.
../thirdparty/crc64
../thirdparty/unqlite
..
../internal
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

add_library(unqlite STATIC ../thirdparty/unqlite/unqlite.h ../thirdparty/unqlite/unqlite.c)
add_library(PyThrustRTC SHARED ${LIB_SOURCES} ${LIB_HEADERS} ${LIB_HEADERS_FAKEVECTORS} ${INTERNAL_HEADERS} ${SRC})

target_link_libraries(PyThrustRTC unqlite)

add_dependencies(Run_PackHeaders PackHeaders)
add_dependencies(PyThrustRTC Run_PackHeaders)

if (WIN32) 
target_compile_definitions(PyThrustRTC PRIVATE THRUST_RTC_DLL_EXPORT)
endif()

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX  ../install CACHE PATH "Install path" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX  ../install CACHE PATH "Install path" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

install(FILES ${LIB_HEADERS} DESTINATION include)
install(FILES ${LIB_HEADERS_FAKEVECTORS} DESTINATION include/fake_vectors)


if (WIN32) 
install(TARGETS PyThrustRTC 
		RUNTIME DESTINATION test_python
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib 
		)
else()
install(TARGETS PyThrustRTC DESTINATION test_python)
endif()

set(PYTHON
	ThrustRTC/__init__.py
	ThrustRTC/cffi.py
	ThrustRTC/Native.py
	ThrustRTC/utils.py
	ThrustRTC/Context.py
	ThrustRTC/DeviceViewable.py
	ThrustRTC/DVVector.py
	ThrustRTC/DVTuple.py
	ThrustRTC/FakeVectors.py
	ThrustRTC/Functor.py
	ThrustRTC/Transformations.py
	ThrustRTC/Copying.py
	ThrustRTC/Reductions.py
	ThrustRTC/PrefixSums.py
	ThrustRTC/Reordering.py
	ThrustRTC/Searching.py
	ThrustRTC/Merging.py
	ThrustRTC/Sorting.py
)

install(FILES ${PYTHON} DESTINATION test_python/ThrustRTC)
install(FILES setup.py README.md DESTINATION test_python)

set(INCLUDE_TESTS false CACHE BOOL "Include tests")

if (INCLUDE_TESTS)
add_subdirectory(test)
endif()
