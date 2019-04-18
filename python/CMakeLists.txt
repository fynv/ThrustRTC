add_subdirectory(PyThrustRTC)

set(PYTHON
	ThrustRTC/__init__.py
	ThrustRTC/Context.py
	ThrustRTC/DeviceViewable.py
	ThrustRTC/DVVector.py
	ThrustRTC/Fill.py
	ThrustRTC/Functor.py
	ThrustRTC/Replace.py
	ThrustRTC/ForEach.py
	ThrustRTC/AdjacentDifference.py
	ThrustRTC/Sequence.py
	ThrustRTC/Tabulate.py
	ThrustRTC/Transform.py
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
)

install(FILES ${PYTHON_TEST} DESTINATION test_python )


