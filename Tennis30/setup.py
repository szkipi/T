"""Tennis30 Setup"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='tennis30',
    version='0.1.0',
    description='Multi-pass precision tennis tracking for 3D reconstruction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Tennis30 Project',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'tennis30=Tennis30.cli:main',
        ],
    },
)
