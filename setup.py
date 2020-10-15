
import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Abdul Rehman",
    author_email="alabdulrehman@hotmail.fr",
    name='formantfeatures',
    license="MIT",
    description='Extract formant characteristics from speech wav files.',
    version='v1.0.3',
    long_description='Please go to: https://github.com/tabahi/formantfeatures',
    url='https://github.com/tabahi/formantfeatures',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'scipy', 'h5py', 'numba', 'wavio'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
