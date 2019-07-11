#!/usr/bin/python3

from setuptools import setup, Extension
from codecs import open
import os
import platform

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extra_compile_args=[]
define_macros = []

if os.name == 'nt':
	define_macros += [
		('THRUST_RTC_DLL_EXPORT', None),
		('_CRT_SECURE_NO_DEPRECATE', None),
		('_SCL_SECURE_NO_DEPRECATE', None),
		('_CRT_SECURE_NO_WARNINGS', None)
	]
else:
	extra_compile_args = ['-std=c++11']

sources = [ 
'PyThrustRTC.cpp',
'../thirdparty/unqlite/unqlite.c',
'../thirdparty/crc64/crc64.cpp',
'../internal/launch_calc.cpp',
'../internal/cuda_wrapper.cpp',
'../internal/nvtrc_wrapper.cpp',
'../internal/general_reduce.cpp',
'../internal/general_scan.cpp',
'../internal/general_scan_by_key.cpp',
'../internal/general_copy_if.cpp',
'../internal/general_find.cpp',
'../internal/merge_sort.cpp',
'../internal/radix_sort.cpp',
'../TRTCContext.cpp',
'../DVVector.cpp',
'../DVTuple.cpp',
'../fake_vectors/DVRange.cpp',
'../fake_vectors/DVConstant.cpp',
'../fake_vectors/DVCounter.cpp',
'../fake_vectors/DVDiscard.cpp',
'../fake_vectors/DVPermutation.cpp',
'../fake_vectors/DVReverse.cpp',
'../fake_vectors/DVTransform.cpp',
'../fake_vectors/DVZipped.cpp',
'../fake_vectors/DVCustomVector.cpp',
'../functor.cpp',
'../fill.cpp',
'../replace.cpp',
'../for_each.cpp',
'../adjacent_difference.cpp',
'../sequence.cpp',
'../tabulate.cpp',
'../transform.cpp',
'../gather.cpp',
'../scatter.cpp',
'../copy.cpp',
'../swap.cpp',
'../count.cpp',
'../reduce.cpp',
'../equal.cpp',
'../extrema.cpp',
'../inner_product.cpp',
'../transform_reduce.cpp',
'../logical.cpp',
'../scan.cpp',
'../transform_scan.cpp',
'../scan_by_key.cpp',
'../remove.cpp',
'../unique.cpp',
'../partition.cpp',
'../find.cpp',
'../mismatch.cpp',
'../binary_search.cpp',
'../merge.cpp',
'../sort.cpp'
]


module_PyThrustRTC = Extension(
	'PyThrustRTC',
	sources = sources,
	include_dirs = ['.', '../thirdparty/crc64', '../thirdparty/unqlite', '..', '../internal'],
	define_macros = define_macros,
	extra_compile_args=extra_compile_args)

setup(
	name = 'ThrustRTC',
	version = '0.2.0',
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
	install_requires = ['numpy']
)

