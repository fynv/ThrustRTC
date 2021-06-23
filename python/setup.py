from setuptools import setup
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
	name = 'ThrustRTC',
	version = '0.3.12',
	description = 'Thrust for Python based on NVRTC',
	long_description=long_description,
	long_description_content_type='text/markdown',  
	url='https://github.com/fynv/ThrustRTC',
	license='Anti 996',
	author='Fei Yang',
	author_email='hyangfeih@gmail.com',
	keywords='GPU CUDA Thrust',
	packages=['ThrustRTC'],
	package_data = { 'ThrustRTC': ['*.dll', '*.so']},
	install_requires = ['cffi','numpy'],	
)

