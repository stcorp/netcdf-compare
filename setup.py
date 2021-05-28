from setuptools import setup, find_packages

setup(
    name='netcdf-compare',
    version='1.0',
    author='S[&]T',
    author_email="info@stcorp.nl",
    url="http://stcorp.nl/",
    description='NetCDF comparison tool',
    license="BSD",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'netcdf-compare = netcdf_compare:main',
        ],
    },
    install_requires=[
        "numpy",
        "netCDF4",
    ],
)
