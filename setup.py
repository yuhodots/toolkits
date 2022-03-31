import os
from setuptools import setup, find_packages


def configuration(parent_package="", top_path=None):
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage("toolkits")
    return config


def setup_package():
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

    metadata = dict(
        name='dl-toolkits',
        version='1.1.1',
        long_description=long_description,
        long_description_content_type='text/markdown',
        description='Deep Learning Analysis Toolkits',
        author='Yuho Jeong',
        url='https://github.com/yuhodots/toolkits',
        install_requires=[],
        packages=find_packages(exclude=[]),
        keywords=['dl', 'toolkits'],
        python_requires='>=3',
    )
    metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
