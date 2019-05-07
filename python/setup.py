#!/usr/bin/python3

from setuptools import setup, Extension
from codecs import open
import os
import platform

lib_dirs = ['../install/lib']
libs = ['ThrustRTC_static']

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extra_compile_args=[]
if os.name != 'nt':
	extra_compile_args = ['-std=c++11']

module_PyThrustRTC = Extension(
	'ThrustRTC.PyThrustRTC',
	sources = ['PyThrustRTC/PyThrustRTC.cpp'],
	include_dirs = ['PyThrustRTC', '../'],
	libraries = libs,
    library_dirs = lib_dirs,
	extra_compile_args=extra_compile_args)

setup(
	name = 'ThrustRTC',
	version = '0.0.3',
	description = 'Thrust for Python based on NVRTC',
	long_description=long_description,
	long_description_content_type='text/markdown',  
	url='https://github.com/fynv/ThrustRTC',
	license='Anti 996',
	author='Fei Yang',
	author_email='hyangfeih@gmail.com',
	keywords='GPU CUDA Thrust',
	packages=['ThrustRTC'],
	ext_modules=[module_PyThrustRTC],
)

