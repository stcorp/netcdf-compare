# netcdf-compare

Netcdf-compare is a netCDF comparison tool. It comes with a variety of command-line options to determine what exactly should be compared and how. For example, it is possible to compare two specific variables, using specific absolute and/or relative tolerance limits.

Netcdf-compare is implemented in Python, using the numpy and netCDF4 libraries.

# usage

    netcdf-compare [-h] [-a PATH] [-A PATH] [-v PATH] [-V PATH] [-g PATH]
                          [-G PATH] [-n] [-s] [-X] [-Y] [-W] [-q] [--atol FLOAT]
                          [--rtol FLOAT] [--combined_tolerance]
                          [--max-values NUMBER] [--verbose]
                          file1 file2

    positional arguments:
      file1                 NetCDF file to compare
      file2                 NetCDF file to compare

    optional arguments:
      -h, --help            show this help message and exit
      -a PATH, --attribute PATH
                            Attribute(s) to include
      -A PATH, --exclude-attribute PATH
                            Attribute(s) to exclude
      -v PATH, --variable PATH
                            Variable(s) to include
      -V PATH, --exclude-variable PATH
                            Variable(s) to exclude
      -g PATH, --group PATH
                            Group(s) to include
      -G PATH, --exclude-group PATH
                            Group(s) to exclude
      -n, --non-recursive   Non-recursively compare group(s)
      -s, --structure       Only compare structure (not content)
      -X, --skip-variables  Exclude all variables
      -Y, --skip-attributes
                            Exclude all attributes
      -W, --no-warnings     Hide warnings
      -q, --quiet           Do not show detailed differences.
      --atol FLOAT          Absolute tolerance (default 1e-6)
      --rtol FLOAT          Relative tolerance (default 1e-3)
      --combined_tolerance  Combine absolute/relative tolerance checking
      --max-values NUMBER   Number of example differences (default 1)
      --verbose             Verbose mode

    Variables, groups and attributes can be referenced as follows:

    -by name
    -by absolute path, for example "/some/group" or "/group/group2/varname"
    -by relative path, for example "group" or "group2/varname"

    When not referenced by absolute path, there may be multiple matches.
