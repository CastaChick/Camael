from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Camael',
    packages=['camael'],

    version='1.0.0',

    license='MIT',

    install_requires=['numpy', 'cvxopt'],

    author='Casta46',
    author_email='casta46chick@gmail.com',

    url='https://github.com/CastaChick/Camael',

    description='Machine learing library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='python MachineLearning DeepLearning',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
