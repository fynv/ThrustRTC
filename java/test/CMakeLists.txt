
find_package(Java REQUIRED)
include(UseJava)

set(JAVA_TESTS
test_trtc.java
test_for.java
test_fill.java
test_replace.java
test_for_each.java
test_adjacent_difference.java
test_sequence.java
test_tabulate.java
test_transform.java
test_gather.java
test_scatter.java
test_swap.java
test_count.java
test_reduce.java
test_equal.java
test_extrema.java
test_inner_product.java
test_transform_reduce.java
test_logical.java
test_scan.java
test_scan_by_key.java
test_transform_scan.java
test_copy.java
test_remove.java
test_unique.java
test_partition.java
test_find.java
test_mismatch.java
test_binary_search.java
test_merge.java
test_sort.java
)

add_jar(JThrustRTCTests SOURCES ${JAVA_TESTS} INCLUDE_JARS JThrustRTC)
install_jar(JThrustRTCTests test_java)

install(FILES run_tests.bat run_tests.sh DESTINATION test_java )