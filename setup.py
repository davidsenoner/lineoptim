from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='lineoptim',
    version='1.2.1',
    description='A comprehensive package for electric power line simulation and optimization.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='David Senoner',
    author_email='david.senoner@gmail.com',
    url='https://github.com/davidsenoner/lineoptim',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "matplotlib>=3.0.0",
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "torch>=1.7.0",
        "networkx>=2.5"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.8',
            'black>=20.8b1',
            'isort>=5.0'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
    ],
    keywords='electrical engineering, power systems, optimization, simulation, pytorch',
    project_urls={
        'Bug Reports': 'https://github.com/davidsenoner/lineoptim/issues',
        'Source': 'https://github.com/davidsenoner/lineoptim',
        'Documentation': 'https://github.com/davidsenoner/lineoptim/blob/main/README.md',
    },
    include_package_data=True,
    zip_safe=False,
)
