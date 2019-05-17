add_subdirectory(PyThrustRTC)

set(PYTHON
	ThrustRTC/__init__.py
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
)

install(FILES ${PYTHON} DESTINATION test_python/ThrustRTC)

set(PYTHON_TEST
	test/test_trtc.py
	test/test_for.py
	test/test_fill.py
	test/test_replace.py
	test/test_for_each.py
	test/test_adjacent_difference.py
	test/test_sequence.py
	test/test_tabulate.py
	test/test_transform.py
	test/test_gather.py
	test/test_scatter.py
	test/test_copy.py
	test/test_swap.py
	test/test_count.py
	test/test_reduce.py
	test/test_equal.py
	test/test_extrema.py
	test/test_inner_product.py
	test/test_transform_reduce.py
	test/test_logical.py
	test/test_scan.py
	test/test_transform_scan.py

	test/test_constant.py
	test/test_counter.py
	test/test_discard.py
	test/test_permutation.py
	test/test_reverse.py
	test/test_transform_iter.py
	test/test_zipped.py
)

install(FILES ${PYTHON_TEST} DESTINATION test_python )


