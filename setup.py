from setuptools import setup, find_packages

setup(
    name='lineoptim',
    version='0.1.0',
    description='A line optimization module',
    author='David Senoner',
    author_email='david.senoner@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)