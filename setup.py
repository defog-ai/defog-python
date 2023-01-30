from setuptools import find_packages, setup
setup(
    name='defog',
    packages=find_packages(),
    version='0.9.2',
    description='Defog is a Python library that helps you generate data queries from natural language questions.',
    author='Full Stack Data Pte. Ltd.',
    license='MIT',
    install_requires=[
        'requests>=2.28.2',
        'psycopg2-binary>=2.9.5',
        'mysql-connector-python>=8.0.32',
        'pymongo>=4.3.3',
        'google-cloud-bigquery>=3.4.2'
    ],
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description='Defog is a Python library that helps you generate data queries from natural language questions.',
    long_description_content_type='text/markdown'
)