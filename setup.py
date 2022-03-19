#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
import pathlib
import schedgym  # noqa
try:
    from Cython.Build import cythonize
except (NameError, ModuleNotFoundError):
    def cythonize(*args, **kwargs):
        pass

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.rst').read_text(encoding='utf-8')

extras = {
    'render': [
        'matplotlib',
        'pyglet',
    ],
    'test': [
        'pytest',
        'coverage',
    ],
    'docs': [
        'Sphinx',
        'docutils',
        'nbsphinx',
    ]
}

extras['all'] = [item for group in extras.values() for item in group]

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='sched-rl-gym',
    description='OpenAI Gym environment for HPC job scheduling',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/renatolfc/sched-rl-gym',
    author='Renato L. de F. Cunha',
    author_email='renatocunha@acm.org',

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='gym, reinforcement learning, artificial intelligence',

    package_dir={'schedgym': 'schedgym'},
    packages=find_packages(),
    python_requires='>=3.6, <4',

    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'gym',
        'numpy',
        'cython',
        'intervaltree>=3.0',
        'parallelworkloads',
    ],

    extras_require=extras,

    ext_modules=cythonize([
        Extension('schedgym.job', ['schedgym/job.py']),
        Extension('schedgym.pool', ['schedgym/pool.py']),
        Extension('schedgym.simulator', ['schedgym/simulator.py']),
        Extension('schedgym.resource', ['schedgym/resource.py']),
        Extension('schedgym.cluster', ['schedgym/cluster.py']),
        Extension('schedgym.envs.workload', ['schedgym/envs/workload.py']),
        Extension('schedgym.envs.simulator', ['schedgym/envs/simulator.py']),
        Extension('schedgym.envs.compact_env', ['schedgym/envs/compact_env.py']),
        Extension('schedgym.envs.base', ['schedgym/envs/base.py']),
        Extension('schedgym.envs.deeprm_env', ['schedgym/envs/deeprm_env.py']),
        Extension('schedgym.scheduler.backfilling_scheduler', ['schedgym/scheduler/backfilling_scheduler.py']),
        Extension('schedgym.scheduler.null_scheduler', ['schedgym/scheduler/null_scheduler.py']),
        Extension('schedgym.scheduler.easy_scheduler', ['schedgym/scheduler/easy_scheduler.py']),
        Extension('schedgym.scheduler.fifo_scheduler', ['schedgym/scheduler/fifo_scheduler.py']),
        Extension('schedgym.scheduler.packer_scheduler', ['schedgym/scheduler/packer_scheduler.py']),
        Extension('schedgym.scheduler.random_scheduler', ['schedgym/scheduler/random_scheduler.py']),
        Extension('schedgym.scheduler.sjf_scheduler', ['schedgym/scheduler/sjf_scheduler.py']),
        Extension('schedgym.scheduler.tetris_scheduler', ['schedgym/scheduler/tetris_scheduler.py']),
        Extension('schedgym.workload.base', ['schedgym/workload/base.py']),
        Extension('schedgym.workload.trace', ['schedgym/workload/trace.py']),
        Extension('schedgym.workload.distribution', ['schedgym/workload/distribution.py']),
        Extension('schedgym.workload.swf_parser', ['schedgym/workload/swf_parser.py']),
    ], language_level=3),

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/renatolfc/sched-rl-gym/issues',
        'Say Thanks!': 'https://saythanks.io/to/renatolfc',
        'Source': 'https://github.com/renatolfc/sched-rl-gym',
    },
)
