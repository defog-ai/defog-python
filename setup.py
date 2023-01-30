from setuptools import find_packages, setup
setup(
    name='defog',
    packages=find_packages(),
    version='0.9.0',
    description='Defog is a Python library that helps you generate data queries from natural language questions.',
    author='Full Stack Data Pte. Ltd.',
    license='MIT',
    install_requires=[
        'requests',
        'psycopg2',
        'mysql-connector-python',
        'pymongo',
        'google-cloud-bigquery'
    ],
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description='Defog is a Python library that helps you generate data queries from natural language questions.',
    long_description_content_type='text/markdown'
)