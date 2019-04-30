# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# import glob
# for directory in glob.glob('./**/', recursive=True):
#     print(directory)

setup(
    name='pyneuroner',

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.7',

    description='NeuroNER',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/Franck-Dernoncourt/NeuroNER',

    # Author details
    # author='',
    # author_email='',

    # Choose your license
    # license='MIT',

    # What does your project relate to?
    keywords='Named-entity recognition using neural networks',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    # packages=find_packages(exclude=['contrib', 'docs', 'tests','env',
    #     'output']),
    packages=['neuroner'],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    # py_modules=[''],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'matplotlib>=3.0.2',
        'networkx>=2.2',
        'pycorenlp>=0.3.0',
        'scikit-learn>=0.20.2',
        'scipy>=1.2.0',
        'spacy>=2.0.18',
        ],

    # allow user to select flavour of TensorFlow 
    # https://github.com/tensorflow/tensorflow/issues/7166
    extras_require={
        "cpu": ["tensorflow>=1.12.0"],
        "gpu": ["tensorflow-gpu>=1.0.0"],
    },

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    zip_safe=False,
    # package_dir={'neuroner': 'neuroner'},
    include_package_data = True,
    package_data={'data': ['data/**'], 
        'trained_models': ['trained_models/**']
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # data_files=[('neuroner', ['conlleval'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'neuroner = neuroner.__main__:main',
        ],
    },

)