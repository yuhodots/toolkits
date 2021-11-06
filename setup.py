from setuptools import setup, find_packages

setup(
    name='dl-toolkits',
    version='1.0',
    description='dl-toolkits',
    author='Yuho Jeong',
    url='https://github.com/yuhodots/toolkits',
    install_requires=[],
    packages=find_packages(exclude=['docs', 'tests*']),
    keywords=['dl', 'toolkits'],
    python_requires='>=3',
)