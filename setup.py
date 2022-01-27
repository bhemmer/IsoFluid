DESCRIPTION = "Isotopic analysis of fluid inclusions"
LONG_DESCRIPTION = """\

"""

DISTNAME = 'IsoFluid'
MAINTAINER = 'Benedikt Hemmer'
MAINTAINER_EMAIL = 'benedikt.hemmer@iup.uni-heidelberg.de'
URL = 'https://github.com/bhemmer/IsoFluid'
LICENSE = 'GNU GPLv3'
DOWNLOAD_URL = 'https://github.com/bhemmer/IsoFluid'
VERSION = '1.0'

INSTALL_REQUIRES = [
    'numpy>=1.16.5',
    'datetime>=3.7',
    'scipy>=1.3',
]

PACKAGES = [
    'IsoFluid',
]

CLASSIFIERS = [
    'Programming Language :: Python :: 3.7',
    'Development Status :: 4 - Beta',
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
