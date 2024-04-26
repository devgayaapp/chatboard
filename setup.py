from setuptools import find_namespace_packages, find_packages, setup

setup(
    name='chatboard',
    version='1.0',
    description='Installs all components of chatboard',
    author='Pavel Schudel',
    author_email='pavel1860@gmail.com',
    url='https://github.com/pavel1860/chatboard',
    packages=[
        "chatboard.chatboard_text", 
        # "chatboard.chatboard_scrape", 
        # "chatboard.chatboard_media"
    ]
    # packages=find_namespace_packages(include=['chatboard.*']),
    # packages=find_packages(), 
    # install_requires=[
    #     'chatboard-common==0.1',
    #     'chatboard-text==0.1',
    #     'chatboard-media==0.1',
    #     # Make sure versions are managed to prevent conflicts
    # ],
    # extras_require={
    #     'full': ['some-optional-dependency']
    # }
)