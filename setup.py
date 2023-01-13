from setuptools import find_packages, setup
setup(
    name='defog',
    packages=find_packages(),
    version='0.1.0',
    description='Defog is a Python library that helps you generate SQL queries from natural language questions.',
    author='Full Stack Data Pte. Ltd.',
    license='MIT',
    install_requires=[
        'requests',
    ],
)