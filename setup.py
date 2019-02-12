from setuptools import setup, Extension

setup(
    name='pyvrft',
    version='0.1',
    description='Virtual Reference Feedback Tuning',
    packages=['vrft'],
    install_requires=['numpy','scipy'],
    author='Diego Eckhard',
    author_email='diego@eckhard.com.br',
    url='http://github.com/datadrivencontrol/pyvrft',
    license='MIT',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
