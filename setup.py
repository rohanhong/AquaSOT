"""
AquaSOT-3D: End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation

Setup configuration for pip installation.
"""

from setuptools import setup, find_packages
import os


def read_requirements(filename):
    """Read requirements from file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


# Read long description from README
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ''
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='aquasot3d',
    version='0.1.0',
    description='End-to-End RGB+Sonar Single-Object Tracking with Weak 3D Estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AquaSOT-3D Team',
    author_email='aquasot3d@example.com',
    url='https://github.com/yourusername/AquaSOT-3D',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
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
    entry_points={
        'console_scripts': [
            'aquasot3d-train=scripts.train:main',
            'aquasot3d-eval=scripts.eval:main',
            'aquasot3d-demo=scripts.demo:main',
        ],
    },
    keywords='underwater tracking sonar rgb multi-modal object-tracking deep-learning',
    project_urls={
        'Documentation': 'https://github.com/yourusername/AquaSOT-3D/docs',
        'Source': 'https://github.com/yourusername/AquaSOT-3D',
        'Tracker': 'https://github.com/yourusername/AquaSOT-3D/issues',
    },
)
