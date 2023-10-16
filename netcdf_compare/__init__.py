#!/usr/bin/env python3
from __future__ import print_function

import argparse
import collections
import itertools
import os
import warnings
import sys

import numpy as np
import numpy.ma as ma
import netCDF4


DESC = 'A netCDF comparison tool'

EPILOG = """

Variables, groups and attributes can be referenced as follows:

-by name
-by absolute path, for example "/some/group" or "/group/group2/varname"
-by relative path, for example "group" or "group2/varname"

When not referenced by absolute path, there may be multiple matches.

Compound types and vlen/ragged arrays are not supported at the moment,
and will cause various warnings.


"""

# TODO test coverage


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
        different = not np.array_equal(a, b)
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


def show_violations(v1, a, b, idcs, indent, differences):
    for t in idcs:
        if not isinstance(a, dict) and len(a.shape) == 0:
            difference = '      %s: %s, %s' % (t, a, b)
        else:
            if v1.dtype is str:
                difference = '      %s: "%s", "%s"' % (t, a[t], b[t])
            else:
                difference = '      %s: %s, %s' % (t, a[t], b[t])
        differences.append(indent + difference)


def compare_variable(v1, v2, args, indent, matches):
    # compare a single variable
    differences = []

    # filter for cmd-line args
    var_path = os.path.join(v1.group().path, v1.name)  # TODO windows

    if args.variables and not path_match(args, 'variables', var_path, matches):
        return differences

    if path_match(args, 'exclude_variables', var_path, matches):
        return differences

    # compare attributes
    attr_differences = compare_attributes(v1, v2, var_path, args,
                                          indent+'  ', matches)
    differences.extend(attr_differences)
    if args.attributes:
        return differences

    # compare structure
    if v1.shape != v2.shape:
        difference = '    DIFFERENT SHAPE (FILE 1: %s, FILE 2: %s)' % \
                         (v1.shape, v2.shape)
        differences.append(indent + difference)

    if v1.dtype != v2.dtype:  # TODO compare .datatype?
        difference = '    DIFFERENT TYPE (FILE 1: %s, FILE 2: %s)' % \
                         (v1.dtype, v2.dtype)
        differences.append(indent + difference)

    if differences:
        return differences

    if isinstance(v1.datatype, netCDF4._netCDF4.CompoundType):
        for field in v1.datatype.dtype.names:
            field_diffs = []
            compare_array(v1, v2, args, field_diffs, indent, field=field)
            if field_diffs:
                differences.append(indent + '    FIELD ' + field)
            differences.extend(['  '+d for d in field_diffs])
    else:
        compare_array(v1, v2, args, differences, indent)

    return differences


def compare_array(v1, v2, args, differences, indent, field=None):
    vlen_violations = None

    # compare scalar data
    if len(v1.shape) == 0:
        a = v1[:]
        b = v2[:]

        # handle scalar string variables differently
        #   these are not read as Numpy data types but as Python str objects
        if isinstance(a, str):
            if a != b:
                difference = '    DIFFERENT SCALAR STRING CONTENT (FILE 1: %s, '\
                    'FILE 2: %s)' % (a, b)
                differences.append(indent + difference)
            return differences

        # make scalars 1d, so we can use indexing below (e.g. aa[both_nan] = 1)
        if a is ma.masked:
            a = ma.MaskedArray(np.nan)
        if b is ma.masked:
            b = ma.MaskedArray(np.nan)
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)

        (
            nonfin_violations, nonfin_idcs,
            abs_max_violation, abs_max_idcs,
            rel_max_violation, rel_max_idcs,
            combined_violations, combined_idcs,

        ) = compare_chunk(a, b, args)

    # compare array data
    else:
        # determine chunk size
        chunk = v1.chunking()

        if chunk == 'contiguous':
            if len(v1.shape) == 1:
                chunk = (1000000,)
            else:
                # TODO fine-tune for arbitrary shapes
                chunk = tuple([1] * (len(v1.shape)-2) + [1000, 1000])  # TODO add test

        # compare netcdf chunks (hyperslabs) individually
        # (avoiding insane memory usage for large arrays)
        nonfin_violations = abs_max_violation = rel_max_violation = combined_violations = None
        all_abs_max_idcs = []
        all_rel_max_idcs = []
        all_combined_idcs = []
        all_nonfin_idcs = []
        vlen_idcs = []

        a = {}
        b = {}

        dimpos = [range(0, dim, chunkdim) for dim, chunkdim in zip(v1.shape, chunk)]
        for pos in itertools.product(*dimpos):
            hyperslice = [slice(i,i+j) for i, j in zip(pos, chunk)]
            chunka = v1[hyperslice]
            chunkb = v2[hyperslice]

            # compound array: get specified field
            if field is not None:  # TODO don't load all fields
                chunka = chunka[field]
                chunkb = chunkb[field]

            # vlen (str/object) array
            if chunka.dtype == object:
                idcs = []
                for t in zip(*[idcs.flat for idcs in np.indices(chunka.shape)]):  # TODO slow for now: per-vlen-object
                    full_pos = tuple(pos[i]+t[i] for i in range(len(pos)))

                    if v1.dtype is str:
                        if chunka[t] != chunkb[t]:
                            vlen_violations = (vlen_violations or 0) + 1
                            a[full_pos] = chunka[t]
                            b[full_pos] = chunkb[t]
                            vlen_idcs.append(full_pos)
                    else:
                        chunka_arr = np.asarray(chunka[t])
                        chunkb_arr = np.asarray(chunkb[t])
                        inequal_idcs = (~np.isclose(chunka_arr, chunkb_arr, args.rtol, args.atol)).nonzero()
                        if len(inequal_idcs[0]) > 0:
                            for u in sorted(zip(*inequal_idcs)):
                                full_pos2 = full_pos + u
                                a[full_pos2] = chunka_arr[u]
                                b[full_pos2] = chunkb_arr[u]
                                vlen_violations = (vlen_violations or 0) + 1
                                vlen_idcs.append(full_pos2)

                continue

            # regular scalar array
            (
                nonfin_violations_, nonfin_idcs_,
                abs_max_violation_, abs_max_idcs_,
                rel_max_violation_, rel_max_idcs_,
                combined_violations_, combined_idcs_,
            ) = compare_chunk(chunka, chunkb, args)

            # collect results
            if nonfin_violations_ is not None:
                nonfin_violations = (nonfin_violations or 0) + nonfin_violations_

                for t in nonfin_idcs_:
                    full_pos = tuple(pos[i]+t[i] for i in range(len(pos)))
                    a[full_pos] = chunka[t]
                    b[full_pos] = chunkb[t]
                    all_nonfin_idcs.append(full_pos)

            if abs_max_violation_ is not None:
                if abs_max_violation is None or abs_max_violation_ > abs_max_violation:
                    abs_max_violation = abs_max_violation_

                for t in abs_max_idcs_:
                    full_pos = tuple(pos[i]+t[i] for i in range(len(pos)))
                    a[full_pos] = chunka[t]
                    b[full_pos] = chunkb[t]
                    all_abs_max_idcs.append((abs_max_violation_, full_pos))

            if rel_max_violation_ is not None:
                if rel_max_violation is None or rel_max_violation_ > rel_max_violation:
                    rel_max_violation = rel_max_violation_

                for t in rel_max_idcs_:
                    full_pos = tuple(pos[i]+t[i] for i in range(len(pos)))
                    a[full_pos] = chunka[t]
                    b[full_pos] = chunkb[t]
                    all_rel_max_idcs.append((rel_max_violation_, full_pos))

            if combined_violations_ is not None:
                combined_violations = (combined_violations or 0) + combined_violations_

                for t in combined_idcs_:
                    full_pos = tuple(pos[i]+t[i] for i in range(len(pos)))
                    a[full_pos] = chunka[t]
                    b[full_pos] = chunkb[t]
                    all_combined_idcs.append(full_pos)

        # merge results
        abs_max_idcs = [idx for max_, idx in all_abs_max_idcs if max_ == abs_max_violation]
        abs_max_idcs = sorted(abs_max_idcs)[:args.max_values]

        rel_max_idcs = [idx for max_, idx in all_rel_max_idcs if max_ == rel_max_violation]
        rel_max_idcs = sorted(rel_max_idcs)[:args.max_values]

        combined_idcs = sorted(all_combined_idcs)[:args.max_values]

        nonfin_idcs = sorted(all_nonfin_idcs)[:args.max_values]

    # summarize differences
    if vlen_violations is not None:
        difference = '    %d OBJECT DIFFERENCE(S):' % vlen_violations
        differences.append(indent + difference)
        show_violations(v1, a, b, vlen_idcs, indent, differences)

    if nonfin_violations is not None:
        difference = '    %d NON-FINITE DIFFERENCE(S)' % nonfin_violations
        differences.append(indent + difference)
        difference = '      FIRST %s OCCURRENCE(S):' % len(nonfin_idcs)
        differences.append(indent + difference)
        show_violations(v1, a, b, nonfin_idcs, indent, differences)

    if abs_max_violation is not None:
        difference = '    MAXIMUM ABSOLUTE VIOLATION: %s' % abs_max_violation
        differences.append(indent + difference)
        show_violations(v1, a, b, abs_max_idcs, indent, differences)

    if rel_max_violation is not None:
        difference = '    MAXIMUM RELATIVE VIOLATION: %s' % rel_max_violation
        differences.append(indent + difference)
        show_violations(v1, a, b, rel_max_idcs, indent, differences)

    if combined_violations is not None:
        difference = '    TOTAL NUMBER OF VIOLATIONS: %s' % combined_violations
        differences.append(indent + difference)
        difference = '      FIRST %s OCCURRENCE(S):' % len(combined_idcs)
        differences.append(indent + difference)
        show_violations(v1, a, b, combined_idcs, indent, differences)

    return differences


def compare_chunk(a, b, args):
    max_values = args.max_values

    # compare nan/inf/-inf
    aa = a.copy()
    bb = b.copy()

    # don't compare both masked
    # TODO would be better to use numpy indexing for this and the following instead
    if isinstance(a, ma.MaskedArray) and isinstance(b, ma.MaskedArray):
        both_masked = (a.mask & b.mask)
        aa[both_masked] = 1
        bb[both_masked] = 1
        both_masked = None

    # (nan is not equal to nan)
    both_nan = ma.filled(np.isnan(aa) & np.isnan(bb), False)
    aa[both_nan] = 1
    bb[both_nan] = 1
    both_nan = None

    # (only compare non-finite)
    both_finite = ma.filled(np.isfinite(aa) & np.isfinite(bb), False)
    aa[both_finite] = 1
    bb[both_finite] = 1
    both_finite = None

    violations_array = np.array(~np.equal(aa, bb))  # this converts the array to a non-masked array
    violations = violations_array.nonzero()
    violations_array = None

    if len(violations[0]):
        nonfin_violations = len(violations[0])
        nonfin_idcs = sorted(zip(*violations))[:max_values]
    else:
        nonfin_violations = None
        nonfin_idcs = None

    # compare absolute/relative
    aa = a.copy()
    bb = b.copy()

    # (avoid divide-by-zero later)
    both_zero = ma.filled(np.equal(aa, 0) & np.equal(bb, 0), False)
    aa[both_zero] = 1
    bb[both_zero] = 1
    both_zero = None

    # (ignore nan/inf here)
    notfinite = ma.filled(~np.isfinite(aa) | ~np.isfinite(bb), False)
    aa[notfinite] = 1
    bb[notfinite] = 1
    notfinite = None

    if np.issubdtype(a.dtype, np.integer):
        absaminb = abs(aa.astype(np.int64) - bb)  # avoid overflowing type
    else:
        absaminb = abs(aa - bb)

    # compare using absolute tolerance
    aviolations = (absaminb > args.atol).nonzero()

    if len(aviolations[0]):
        abs_max_violation = np.amax(absaminb)
        abs_max_idcs = (absaminb == abs_max_violation).nonzero()
        abs_max_idcs = sorted(zip(*abs_max_idcs))[:max_values]
    else:
        abs_max_violation = None
        abs_max_idcs = None

    # compare using relative tolerance
    reldiff = absaminb / np.minimum(np.abs(aa), np.abs(bb))
    absaminb = None
    rviolations = (reldiff > args.rtol).nonzero()

    if len(rviolations[0]):
        rel_max_violation = np.amax(reldiff)
        rel_max_idcs = (reldiff == rel_max_violation).nonzero()
        rel_max_idcs = sorted(zip(*rel_max_idcs))[:max_values]
    else:
        rel_max_violation = None
        rel_max_idcs = None

    reldiff = None

    # logically combine absolute/relative checks
    if args.combined_tolerance:
        violations = set(zip(*aviolations)) & set(zip(*rviolations))
    else:
        violations = set(zip(*aviolations)) | set(zip(*rviolations))
    violations = sorted(violations)

    if violations:
        combined_violations = len(violations)
        combined_idcs = violations[:max_values]

    else:
        combined_violations = None
        combined_idcs = None

    return (
        nonfin_violations, nonfin_idcs,
        abs_max_violation, abs_max_idcs,
        rel_max_violation, rel_max_idcs,
        combined_violations, combined_idcs,
    )


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
