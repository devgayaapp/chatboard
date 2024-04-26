from setuptools import setup, find_namespace_packages

setup(
    name='chatboard-common',
    version='1.0',
    author='Pavel Schudel',
    author_email='pavel1860@gmail.com',
    description='Common utilities for my_package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pavel1860/chatboard',
    packages=find_namespace_packages(include=['chatboard.*']),
    install_requires=[
        "langchain==0.1.9",
        "pydantic==1.10.4",
        "tiktoken==0.5.2",
        "pinecone-client==3.0.1",
        "pinecone-text==0.7.0",
        "scipy==1.11.4",
        "boto3==1.24.47",
        "openai==1.23.2",
        "GitPython==3.1.31",
        # Dependencies listed here, e.g., 'requests >= 2.19.1'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum version of Python required
)
