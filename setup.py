from setuptools import find_packages, setup

# with open("./requirements.txt", "r") as f:
#     requirements = f.read().splitlines()

setup(
    name="defog",
    packages=find_packages(),
    version="0.37.0",
    description="Defog is a Python library that helps you generate data queries from natural language questions.",
    author="Full Stack Data Pte. Ltd.",
    license="MIT",
    # install_requires=requirements,
    install_requires=[
        "requests>=2.28.2",
        "psycopg2-binary>=2.9.5",
    ],
    entry_points={
        "console_scripts": [
            "defog=defog.cli:main",
        ],
    },
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description="Defog is a Python library that helps you generate data queries from natural language questions.",
    long_description_content_type="text/markdown",
)
