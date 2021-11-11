import os
from setuptools import setup


def configuration(parent_package="", top_path=None):
    if os.path.exists("MANIFEST"):
        os.remove("MANIFEST")

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config.add_subpackage("toolkits")
    config.add_subpackage("toolkits/cluster")
    config.add_subpackage("toolkits/pprint")
    config.add_subpackage("toolkits/utils")
    config.add_subpackage("toolkits/viz")
    return config


def setup_package():
    metadata = dict(
        name='dl-toolkits',
        version='1.0.0.7',
        description='Deep Learning Analysis Toolkits',
        author=['Yuho Jeong', 'Solang Kim'],
        url='https://github.com/yuhodots/toolkits',
        install_requires=[],
        packages=['toolkits'],
        keywords=['dl', 'toolkits'],
        python_requires='>=3',
    )
    metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
