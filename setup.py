"""
AquaSOT: End-to-End RGB+Sonar Single-Object Tracking

Setup configuration for pip installation.
"""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from file, skipping comments and pip options."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line and not line.startswith(('#', '-'))]


# Read long description from README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='aquasot',
    version='1.0.1',
    description='End-to-End RGB+Sonar Single-Object Tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luhang Hong',
    url='https://github.com/rohanhong/AquaSOT',
    packages=find_packages(include=['lib', 'lib.*']),
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.1.0',
            'mypy>=1.5.0',
            'isort>=5.12.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='underwater tracking sonar rgb multi-modal object-tracking deep-learning',
    project_urls={
        'Documentation': 'https://github.com/rohanhong/AquaSOT/tree/main/docs',
        'Source': 'https://github.com/rohanhong/AquaSOT',
        'Tracker': 'https://github.com/rohanhong/AquaSOT/issues',
    },
)
