md build
cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install
nmake
nmake install
cd ../python
"%PYTHON%" setup.py install
