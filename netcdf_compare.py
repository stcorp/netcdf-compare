#!/usr/bin/env python3
from __future__ import print_function

import argparse
import collections
import itertools
import os
import warnings
import sys

import numpy as np
import netCDF4


DESC = 'A netCDF comparison tool'

EPILOG = """

Variables, groups and attributes can be referenced as follows:

-by name
-by absolute path, for example "/some/group" or "/group/group2/varname"
-by relative path, for example "group" or "group2/varname"

When not referenced by absolute path, there may be multiple matches.

"""


def parse_args():
    parser = argparse.ArgumentParser(
                 description=DESC,
                 epilog=EPILOG,
                 formatter_class=argparse.RawDescriptionHelpFormatter
             )

    parser.add_argument('file1',
                        help='NetCDF file to compare')

    parser.add_argument('file2',
                        help='NetCDF file to compare')

    parser.add_argument('-a', '--attribute',
                        dest='attributes', action='append', default=[],
                        metavar='PATH',
                        help='Attribute(s) to include')

    parser.add_argument('-A', '--exclude-attribute',
                        dest='exclude_attributes', action='append', default=[],
                        metavar='PATH',
                        help='Attribute(s) to exclude')

    parser.add_argument('-v', '--variable',
                        dest='variables', action='append', default=[],
                        metavar='PATH',
                        help='Variable(s) to include')

    parser.add_argument('-V', '--exclude-variable',
                        dest='exclude_variables', action='append', default=[],
                        metavar='PATH',
                        help='Variable(s) to exclude')

    parser.add_argument('-g', '--group',
                        dest='groups', action='append', default=[],
                        metavar='PATH',
                        help='Group(s) to include')

    parser.add_argument('-G', '--exclude-group',
                        dest='exclude_groups', action='append', default=[],
                        metavar='PATH',
                        help='Group(s) to exclude')

    parser.add_argument('-n', '--non-recursive',
                        action='store_true',
                        help='Non-recursively compare group(s)')

    parser.add_argument('-s', '--structure',
                        action='store_true',
                        help='Only compare structure (not content)')

    parser.add_argument('-X', '--skip-variables',
                        action='store_true',
                        help='Exclude all variables')

    parser.add_argument('-Y', '--skip-attributes',
                        action='store_true',
                        help='Exclude all attributes')

    parser.add_argument('-W', '--no-warnings',
                        action='store_true',
                        help='Hide warnings')

    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Do not show detailed differences.')

    parser.add_argument('--atol',
                        type=float, default=1e-6,
                        metavar = 'FLOAT',
                        help='Absolute tolerance (default 1e-6)')

    parser.add_argument('--rtol',
                        type=float, default=1e-3,
                        metavar = 'FLOAT',
                        help='Relative tolerance (default 1e-3)')

    parser.add_argument('--combined_tolerance', action='store_true',
                        help='Combine absolute/relative tolerance checking')

    parser.add_argument('--max-values',
                        type=int, default=1,
                        metavar = 'NUMBER',
                        help='Number of example differences (default 1)')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose mode')

    return parser.parse_args()


def path_match(args, option, path, matches):
    # name, absolute or relative path matching
    exprs = getattr(args, option)

    split_path = path.split('/')
    for expr in exprs:
        split_expr = expr.split('/')
        if split_path[-len(split_expr):] == split_expr:
            matches[option].add(expr)
            return True
    return False


def check_missing(args, path, names1, names2, objtype, differences, indent, matches):
    # check if groups/vars/attrs are missing on one side
    left = names2 - names1
    right = names1 - names2

    for file_nr, names in ((1, left), (2, right)):
        for name in names:
            objpath = os.path.join(path, name)  # TODO windows
            exclude_option = 'exclude_%ss' % objtype.lower()
            if not path_match(args, exclude_option, objpath, matches):
                difference = '  FILE %d MISSES %s: %s' % (file_nr, objtype, name)
                differences.append(indent + difference)


def compare_attribute(obj1, obj2, path, attr_name, args, indent, matches):
    # compare a single attribute
    differences = []

    attr_path = os.path.join(path, attr_name)  # TODO windows

    if args.attributes and not path_match(args, 'attributes', attr_path,
                                          matches):
        return differences

    if path_match(args, 'exclude_attributes', attr_path, matches):
        return differences

    try:
        a = getattr(obj1, attr_name)
        b = getattr(obj2, attr_name)
    except:
        if not args.no_warnings:
            warnings.warn('cannot retrieve content for attribute %s' % \
                          attr_path)
        return differences

    # TODO strings: show from position with difference (and cut off)

    try:
        different = bool(a != b)
    except:
        if not args.no_warnings:
            warnings.warn('cannot compare content for attribute %s' % \
                          attr_path)
        return differences

    if different:
        difference = '    FILE 1: %s' % a
        differences.append(indent + difference)
        difference = '    FILE 2: %s' % b
        differences.append(indent + difference)

    return differences


def compare_variable(v1, v2, args, indent, matches):
    # compare a single variable
    differences = []

    var_path = os.path.join(v1.group().path, v1.name)  # TODO windows

    if args.variables and not path_match(args, 'variables', var_path, matches):
        return differences

    if path_match(args, 'exclude_variables', var_path, matches):
        return differences

    # compare attributes
    attr_differences = compare_attributes(v1, v2, var_path, args,
                                          indent+'  ', matches)

    if args.attributes:
        differences.extend(attr_differences)
        return differences

    # content

    a = v1[:]
    b = v2[:]

    max_values = args.max_values

    # handle scalar string variables differently
    #   these are not read as Numpy data types but as Python str objects
    if isinstance(a, str):
        if a != b:
            difference = '    DIFFERENT SCALAR STRING CONTENT (FILE 1: %s, '\
                'FILE 2: %s)' % (a, b)
            differences.append(indent + difference)
        return differences

    if a.shape != b.shape:
        difference = '    DIFFERENT SHAPE (FILE 1: %s, FILE 2: %s)' % \
                         (a.shape, b.shape)
        differences.append(indent + difference)

    if a.dtype != b.dtype:
        difference = '    DIFFERENT TYPE (FILE 1: %s, FILE 2: %s)' % \
                         (a.dtype, b.dtype)
        differences.append(indent + difference)

    if differences:
        return differences

    if not np.issubdtype(a.dtype, np.number):  # TODO other types
        if not args.no_warnings:
            warnings.warn('unsupported data type for variable %s, skipping' % \
                          var_path)
        return differences

    # make scalars 1d, so we can use indexing below (e.g. aa[both_nan] = 1)
    if len(a.shape) == 0:
        if a is np.ma.masked:
            a = np.ma.MaskedArray(np.nan)
        if b is np.ma.masked:
            b = np.ma.MaskedArray(np.nan)

        a = np.atleast_1d(a)
        b = np.atleast_1d(b)

    # compare nan/inf/-inf
    aa = a.copy()
    bb = b.copy()

    # don't compare both masked
    # TODO would be better to use numpy indexing for this and the following instead
    if isinstance(a, np.ma.MaskedArray) and isinstance(b, np.ma.MaskedArray):
        both_masked = (a.mask & b.mask)
        aa[both_masked] = 1
        bb[both_masked] = 1

    # (nan is not equal to nan)
    both_nan = (np.isnan(aa) & np.isnan(bb)).nonzero()
    aa[both_nan] = 1
    bb[both_nan] = 1

    # (only compare non-finite)
    both_finite = (np.isfinite(aa) & np.isfinite(bb)).nonzero()
    aa[both_finite] = 1
    bb[both_finite] = 1

    violations_array = np.array(~np.equal(aa, bb))  # this converts the array to a non-masked array
    violations = violations_array.nonzero()

    if len(violations[0]):
        difference = '    %d NON-FINITE DIFFERENCE(S)' % \
                         len(violations[0])
        differences.append(indent + difference)
        difference = '      FIRST %s OCCURRENCE(S):' % \
                            min(max_values, len(violations[0]))
        differences.append(indent + difference)
        for t in itertools.islice(sorted(zip(*violations)), 0, max_values):
            difference = '      %s: %s, %s' % (t, a[t], b[t])
            differences.append(indent + difference)

    # compare using absolute tolerance
    aa = a.copy()
    bb = b.copy()

    # (avoid divide-by-zero later)
    both_zero = (np.equal(aa, 0) & np.equal(bb, 0)).nonzero()
    aa[both_zero] = 1
    bb[both_zero] = 1

    # (ignore nan/inf here)
    notfinite = (~np.isfinite(aa) | ~np.isfinite(bb)).nonzero()
    aa[notfinite] = 1
    bb[notfinite] = 1

    absaminb = abs(aa-bb)
    aviolations = (absaminb > args.atol).nonzero()

    if len(aviolations[0]):
        amax = np.amax(absaminb)
        difference = '    MAXIMUM ABSOLUTE VIOLATION: %s' % amax
        differences.append(indent + difference)

        maxidcs = (absaminb == amax).nonzero()
        for t in itertools.islice(sorted(zip(*maxidcs)), 0, max_values):
            if len(aa.shape) == 0:
                difference = '      %s: %s, %s' % (t, a, b)
            else:
                difference = '      %s: %s, %s' % (t, a[t], b[t])
            differences.append(indent + difference)

    # compare using relative tolerance
    reldiff = absaminb / np.minimum(np.abs(aa), np.abs(bb))
    rviolations = (reldiff > args.rtol).nonzero()

    if len(rviolations[0]):
        rmax = np.amax(reldiff)
        difference = '    MAXIMUM RELATIVE VIOLATION: %s' % rmax
        differences.append(indent + difference)

        maxidcs = (reldiff == rmax).nonzero()
        for t in itertools.islice(sorted(zip(*maxidcs)), 0, max_values):
            if len(aa.shape) == 0:
                difference = '      %s: %s, %s' % (t, a, b)
            else:
                difference = '      %s: %s, %s' % (t, a[t], b[t])
            differences.append(indent + difference)

    # logically combine absolute/relative checks

    if args.combined_tolerance:
        violations = set(zip(*aviolations)) & set(zip(*rviolations))
    else:
        violations = set(zip(*aviolations)) | set(zip(*rviolations))
    violations = sorted(violations)

    if violations:
        difference = '    TOTAL NUMBER OF VIOLATIONS: %s' % \
                         len(violations)
        differences.append(indent + difference)
        difference = '      FIRST %s OCCURRENCE(S):' % \
                            min(max_values, len(violations))
        differences.append(indent + difference)
        for t in itertools.islice(violations, 0, max_values):
            if len(aa.shape) == 0:
                difference = '      %s: %s, %s' % (t, a, b)
            else:
                difference = '      %s: %s, %s' % (t, a[t], b[t])
            differences.append(indent + difference)

    differences.extend(attr_differences)
    return differences


def compare_attributes(obj1, obj2, path, args, indent, matches):
    # compare group/variable attributes
    differences = []

    attrs1 = set(obj1.ncattrs())
    attrs2 = set(obj2.ncattrs())

    if not args.attributes:
        check_missing(args, path, attrs1, attrs2, 'ATTRIBUTE', differences, indent, matches)

    if not args.structure and not args.skip_attributes:
        for attr in attrs1 & attrs2:
            attr_differences = compare_attribute(obj1, obj2, path, attr,
                                                 args, indent, matches)
            if attr_differences or args.verbose:
                attr_differences.insert(0, indent + '  ATTRIBUTE %s' % attr)
                differences.extend(attr_differences)

    return differences


def compare_variables(group1, group2, args, indent, matches):
    # compare variables, including attributes
    differences = []

    vars1 = set(group1.variables)
    vars2 = set(group2.variables)

    if not args.variables:  # TODO if variables specified, check missing
        check_missing(args, group1.path, vars1, vars2, 'VARIABLE', differences, indent, matches)

    if not args.structure and not args.skip_variables:
        for var in vars1 & vars2:
            v1, v2 = group1[var], group2[var]

            var_differences = compare_variable(v1, v2, args, indent, matches)

            if var_differences or args.verbose:
                var_differences.insert(0, indent + '  VAR %s' % var)
                differences.extend(var_differences)

    return differences


def compare_group(group1, group2, args, check_group, matches, indent=''):
    # compare group (recursively)
    differences = []

    group_path = group1.path
    group_name = os.path.basename(group_path)

    if path_match(args, 'groups', group_path, matches):
        check_group = True

    if path_match(args, 'exclude_groups', group_path, matches):
        return differences

    if check_group:
        var_differences = compare_variables(group1, group2, args, indent,
                                            matches)
        differences.extend(var_differences)

        if not args.variables:
            attr_differences = compare_attributes(group1, group2, group1.path,
                                              args, indent, matches)
            differences.extend(attr_differences)

    groups1 = set(group1.groups)
    groups2 = set(group2.groups)

    if not args.groups:
        check_missing(args, group_path, groups1, groups2, 'GROUP', differences, indent, matches)

    for group in groups1 & groups2:
        new_indent = indent + '  '
        if args.non_recursive:
            check_group = False
        g1, g2 = group1[group], group2[group]
        group_differences = compare_group(g1, g2, args, check_group,
                                          matches, new_indent)
        differences.extend(group_differences)

    if differences or args.verbose:
        difference = 'GROUP %s' % (group1.path.split('/')[-1] or '/')
        differences.insert(0, indent + difference)

    return differences


def main():
    args = parse_args()

    ds1 = netCDF4.Dataset(args.file1)  # TODO suppress warnings if requested
    ds2 = netCDF4.Dataset(args.file2)

    matches = collections.defaultdict(set)
    differences = compare_group(ds1, ds2, args, not args.groups, matches)
    if not args.quiet:
        for difference in differences:
            print(difference)
    elif differences:
        print('Files {} and {} differ'.format(args.file1, args.file2))

    # check that filter options were used
    option_unused = False
    for option in (
        'variables',
        'groups',
        'attributes',
    ):
        unused = set(getattr(args, option)) - matches[option]
        for expr in unused:
            print('NO MATCH FOR FILTER: --%s=%s' % (option[:-1], expr), file=sys.stderr)
            option_unused = True

    ds1.close()
    ds2.close()

    if differences or option_unused:
        sys.exit(1)


if __name__ == '__main__':
    main()
