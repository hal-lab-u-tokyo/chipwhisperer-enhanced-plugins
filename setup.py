from setuptools import setup, find_packages

setup(
    name='sca_tools',
    version='1.0.0',
    license='MIT',
    description='python tools for power analysis-based side-channel attack',

    author='Takuya Kojima',
    author_email='tkojima@hal.ipc.i.u-tokyo.ac.jp',
    url='https://www.tkojima.me',

    install_requires=[
        "PyUSB>=1.2.1",
        "pyvisa>=1.13.0",
        "ftd2xx>=1.3.3",
        "pycryptodome>=3.19.0",
        "matplotlib>=3.8.0",
        "numpy>=1.25.0",
    ],

    packages=find_packages(where="lib"),
    package_dir={'sca_tools': 'lib/sca_tools'},

    scripts=[]

)

