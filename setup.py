from setuptools import setup, find_packages

def fetch_readme() -> str:
  """
  This function reads the README.md file in the current directory.

  Returns:
      The lines in the README file.
  """
  with open("README.md", encoding="utf-8") as f:
    return f.read()

setup(
  name='lineoptim',
  version='1.0.0',
  description='A line optimization package for electrical conductors.',
  author='David Senoner',
  author_email='david.senoner@gmail.com',
  long_description=fetch_readme(),
  long_description_content_type="text/markdown",
  packages=find_packages(),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
  install_requires=[
    'setuptools',
    'wheel',
    'numpy',
    'pandas',
    'torch',
    'matplotlib'
  ],
)
