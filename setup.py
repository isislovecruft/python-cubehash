#!/usr/bin/env python

"""This file is part of python-cubehash, a CubeHash reference implementation
in pure Python.

:authors: Isis <isis@torproject.org> 0xA3ADB67A2CDB8B35
:copyright: (c) 2014, Isis Agora Lovecruft
:license: see LICENSE for licensing information
"""

from __future__ import print_function

import setuptools
import os

try:
    # setup automatic versioning (see top-level versioneer.py file):
    import versioneer
except (ImportError, NameError):
    print("Could not initiate automatic versioning tool, versioneer.")
else:
    versioneer.VCS = 'git'
    versioneer.versionfile_source = 'cubehash/_version.py'
    versioneer.versionfile_build = 'cubehash/_version.py'
    # source tarballs should unpack to a directory like 'cubehash-6.6.6'
    versioneer.parentdir_prefix = 'cubehash-'
    # when creating a release, tags should be prefixed with 'cubehash-', like so:
    #
    #     git checkout -b release-6.6.6 develop
    #     [do some stuff, merge whatever, test things]
    #     git tag -S cubehash-6.6.6
    #     git push origin --tags
    #     git checkout master
    #     git merge -S --no-ff release-6.6.6
    #     git checkout develop
    #     git merge -S --no-ff master
    #     git branch -d release-6.6.6
    #
    versioneer.tag_prefix = 'cubehash-'

try:
    version = versioneer.get_version()
except:
    version = 'unknown'

# Use the ReStructured Text from the README file for PyPI:
with open(os.path.join(os.getcwd(), 'README')) as readme:
    long_description = readme.read()


def get_cmdclass():
    """Get our cmdclass dictionary for use in setuptool.setup().

    This must be done outside the call to setuptools.setup() because we need
    to add our own classes to the cmdclass dictionary, and then update that
    dictionary with the one returned from versioneer.get_cmdclass().
    """
    cmdclass = {'test': runTests}
    try:
        cmdclass.update(versioneer.get_cmdclass())
    except NameError:
        pass
    return cmdclass

def get_requirements():
    """Extract the list of requirements from our requirements.txt.

    :rtype: 2-tuple
    :returns: Two lists, the first is a list of requirements in the form of
        pkgname==version. The second is a list of URIs or VCS checkout strings
        which specify the dependency links for obtaining a copy of the
        requirement.
    """
    import os

    requirements_file = os.path.join(os.getcwd(), 'requirements.txt')
    requirements = []
    links=[]
    try:
        with open(requirements_file) as reqfile:
            for line in reqfile.readlines():
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line.startswith(
                        ('https://', 'git://', 'hg://', 'svn://')):
                    links.append(line)
                else:
                    requirements.append(line)

    except (IOError, OSError) as error:
        print(error)

    return requirements, links

class runTests(setuptools.Command):
    """Run unittests.

    Based on setup.py from mixminion, which is based on setup.py from Zooko's
    pyutil package, which is in turn based on:
    http://mail.python.org/pipermail/distutils-sig/2002-January/002714.html
    """
    description = "Run unittests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        build = self.get_finalized_command('build')
        self.build_purelib = build.build_purelib
        self.build_platlib = build.build_platlib

    def run(self):
        import sys
        self.run_command('build')
        old_path = sys.path[:]
        sys.path[0:0] = [self.build_purelib, self.build_platlib]
        try:
            testmod = __import__("cubehash.test", globals(), "", [])
            testmod.Tests.main()
        finally:
            sys.path = old_path

#requires, deplinks = get_requirements()

setuptools.setup(
    name='cubehash',
    version=version,
    description="A reference implementation of DJB's CubeHash in pure Python",
    long_description=long_description,
    license="Public Domain",
    author='isis',
    author_email='isis@torproject.org',
    maintainer='isis',
    maintainer_email='isis@torproject.org 0xA3ADB67A2CDB8B35',
    url='http://cubehash.cr.yp.to',
    download_url='https://github.com/isislovecruft/python-cubehash.git',
    packages=['cubehash'],
    cmdclass=get_cmdclass(),
    scripts=['versioneer.py'],
    #include_package_data=True,
    #install_requires=requires,
    #dependency_links=deplinks,
    platforms="Linux",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Public Domain",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
    ]
)
