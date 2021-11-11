import os
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("toolkits", parent_package, top_path)
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_subpackage("cluster")
    config.add_subpackage("pprint")
    config.add_subpackage("utils")
    config.add_subpackage("viz")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())
