md build_python
cd build_python
cmake ../python -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
nmake install
cd ../install/test_python
"%PYTHON%" setup.py install
