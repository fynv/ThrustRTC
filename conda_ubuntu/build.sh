mkdir build_python
cd build_python
cmake ../python -DCMAKE_BUILD_TYPE=Release
make
make install
cd ../install/test_python
$PYTHON setup.py install
