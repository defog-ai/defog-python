from setuptools import find_packages, setup
setup(
    name='defog',
    packages=find_packages(),
    version='0.4.0',
    description='Defog is a Python library that helps you generate SQL queries from natural language questions.',
    author='Full Stack Data Pte. Ltd.',
    license='MIT',
    install_requires=[
        'requests',
    ],
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description='Defog is a Python library that helps you generate SQL queries from natural language questions. To use it, please sign up for an account at https://defog.ai. Then, you can use the API key to generate queries. You can view the source code for this library [https://github.com/defog-ai/defog-python](on GitHub).',
    long_description_content_type='text/markdown'
)