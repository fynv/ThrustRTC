set(PYTHON_TEST
	test_trtc.py
	test_for.py
	test_fill.py
	test_replace.py
	test_for_each.py
	test_adjacent_difference.py
	test_sequence.py
	test_tabulate.py
	test_transform.py
	test_gather.py
	test_scatter.py
	test_copy.py
	test_swap.py
	test_count.py
	test_reduce.py
	test_equal.py
	test_extrema.py
	test_inner_product.py
	test_transform_reduce.py
	test_logical.py
	test_scan.py
	test_transform_scan.py
	test_scan_by_key.py
	test_remove.py
	test_unique.py
	test_partition.py
	test_find.py
	test_mismatch.py
	test_binary_search.py
	test_merge.py	
	test_sort.py
	test_numba.py

	test_constant.py
	test_counter.py
	test_discard.py
	test_permutation.py
	test_reverse.py
	test_transform_iter.py
	test_zipped.py
	test_custom_vector.py

	../demo/histogram.py
	../demo/k-means.py
)

install(FILES ${PYTHON_TEST} DESTINATION test_python )