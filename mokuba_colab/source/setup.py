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
        "torchsde",
        "xformers @ https://download.pytorch.org/whl/cu128/xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl",
        "torch @ https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=628e89bd5110ced7debee2a57c69959725b7fbc64eab81a39dd70e46c7e28ba5",
        "torchvision @ https://download-r2.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=1255a0ca2bf987acf9f103b96c5c4cfe3415fc4a1eef17fa08af527a04a4f573",
        "diffusers==0.37.0",
        "realesrgan",
        "tomesd",
        "torchao==0.16.0",
    ],
)