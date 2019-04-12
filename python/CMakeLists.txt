set(PYTHON
	__init__.py
	Context.py
	DeviceViewable.py
	DVVector.py
	ForLoop.py
	Fill.py
	Functor.py
	Replace.py
)

install(FILES ${PYTHON} DESTINATION test/ThrustRTC)

set(PYTHON_TEST
	test/test_trtc.py
	test/test_for.py
	test/test_fill.py
	test/test_replace.py
)

install(FILES ${PYTHON_TEST} DESTINATION test )


