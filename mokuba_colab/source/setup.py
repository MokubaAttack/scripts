from setuptools import setup, find_packages

setup(
	name='mokucola',
	version='2.10.128',
	packages=find_packages(),
	include_package_data=True,
	author='mokuba_attack',
	description='This is a script that I use when I create images by diffusers.',
	url='https://github.com/MokubaAttack/scripts',
	license='BSD-3-Clause',
	classifiers=[
		'License :: OSI Approved :: BSD License',
		'Programming Language :: Python :: 3.12',
	],
	install_requires=[
		"pyexiv2",
		"compel",
		"torch @ https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
		"torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
		"diffusers==0.37.0",
		"realesrgan",
		"accelerate",
		"PEFT",
		"torchao>=0.16.0",
	],
)
