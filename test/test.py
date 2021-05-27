#!/usr/bin/env python
import subprocess
import sys

import numpy as np

import netCDF4

PYTHON = 'python3' if sys.version_info[0] == 3 else 'python2'


class TestNetCDFCompare:
    def setup(self):
        self.left = netCDF4.Dataset('left.nc', 'w')
        self.left.createDimension('t', 3)
        self.left.createDimension('t4', 4)

        self.right = netCDF4.Dataset('right.nc', 'w')
        self.right.createDimension('t', 3)
        self.right.createDimension('t4', 4)

    def compare(self, options=''):
        self.left.close()
        self.right.close()

        process = subprocess.Popen('%s ../netcdf_compare.py left.nc right.nc %s' % (PYTHON, options),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, errs = process.communicate()
        for line in output.splitlines():
            print(line)
        for line in errs.splitlines():
            print(line)
        assert not errs
        return output

    def check_output(self, output, msg):
        assert msg in output

    def test_scalar_equal(self):
        lvar = self.left.createVariable('x', float)
        lvar[:] = [1.1]

        rvar = self.right.createVariable('x', float)
        rvar[:] = [1.1]

        output = self.compare()
        assert not output

    def test_scalar_different(self):
        lvar = self.left.createVariable('x', float)
        lvar[:] = [1.1]

        rvar = self.right.createVariable('x', float)
        rvar[:] = [4.4]

        output = self.compare()
        self.check_output(output,
            b'    MAXIMUM ABSOLUTE VIOLATION: 3.3000000000000003\n'
            b'      (0,): 1.1, 4.4\n'
            b'    MAXIMUM RELATIVE VIOLATION: 3.0\n'
            b'      (0,): 1.1, 4.4\n'
            b'    TOTAL NUMBER OF VIOLATIONS: 1\n'
            b'      FIRST 1 OCCURRENCE(S):\n'
            b'      (0,): 1.1, 4.4\n')

    def test_scalar_one_empty(self):
        lvar = self.left.createVariable('x', float)
        lvar[:] = [1.1]

        rvar = self.right.createVariable('x', float)

        output = self.compare()
        self.check_output(output,
            b'    1 NON-FINITE DIFFERENCE(S)\n'
            b'      FIRST 1 OCCURRENCE(S):\n'
            b'      (0,): 1.1, nan\n')

    def test_scalar_both_empty(self):
        lvar = self.left.createVariable('x', float)
        rvar = self.right.createVariable('x', float)

        output = self.compare()
        assert not output


    def test_float_equal(self):
        lvar = self.left.createVariable('x', float, ('t',))
        lvar[:] = [1.1, 2.2, 3.3]

        rvar = self.right.createVariable('x', float, ('t',))
        rvar[:] = [1.1, 2.2, 3.3]

        output = self.compare()
        assert not output

    def test_float_different(self):
        lvar = self.left.createVariable('x', float, ('t',))
        lvar[:] = [-1, 2.2, 10]

        rvar = self.right.createVariable('x', float, ('t',))
        rvar[:] = [4, 2.3, 16]

        output = self.compare()

        self.check_output(output,
            b'    MAXIMUM ABSOLUTE VIOLATION: 6.0\n'
            b'      (2,): 10.0, 16.0\n'
            b'    MAXIMUM RELATIVE VIOLATION: 5.0\n'
            b'      (0,): -1.0, 4.0\n'
            b'    TOTAL NUMBER OF VIOLATIONS: 3\n'
            b'      FIRST 1 OCCURRENCE(S):\n'
            b'      (0,): -1.0, 4.0\n')

    def test_float_different_fillvalue(self):
        lvar = self.left.createVariable('x', float, ('t',), fill_value=1e30)
        lvar[:] = [1.1, 2.2, 1e30]

        rvar = self.right.createVariable('x', float, ('t',), fill_value=1e31)
        rvar[:] = [1.1, 2.2, 1e31]

        output = self.compare('--skip-attributes')
        assert not output

    def test_float_fillvalue_masked(self):
        lvar = self.left.createVariable('x', float, ('t',))
        lvar[:] = [1.1, 2.2, 3]

        rvar = self.right.createVariable('x', float, ('t',))
        rvar[:] = [1.1, 2.2, 3]
        rvar[1:] = np.ma.masked

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  VAR x\n'
            b'    2 NON-FINITE DIFFERENCE(S)\n'
            b'      FIRST 1 OCCURRENCE(S):\n'
            b'      (1,): 2.2, --\n')

    def test_float_specials_equal(self):
        lvar = self.left.createVariable('x', float, ('t4',))
        lvar[:] = [0, np.nan, np.inf, -np.inf]

        rvar = self.right.createVariable('x', float, ('t4',))
        rvar[:] = [0, np.nan, np.inf, -np.inf]

        output = self.compare()
        assert not output

    def test_float_specials_different(self):
        lvar = self.left.createVariable('x', float, ('t4',))
        lvar[:] = [np.nan, np.inf, -np.inf, 0]

        rvar = self.right.createVariable('x', float, ('t4',))
        rvar[:] = [0, np.nan, np.inf, -np.inf]

        output = self.compare('--max-values=10')
        self.check_output(output,
            b'    4 NON-FINITE DIFFERENCE(S)\n'
            b'      FIRST 4 OCCURRENCE(S):\n'
            b'      (0,): nan, 0.0\n'
            b'      (1,): inf, nan\n'
            b'      (2,): -inf, inf\n'
            b'      (3,): 0.0, -inf\n')

    def test_group_extra(self):
        self.left.createGroup('extraleft')
        self.right.createGroup('extraright')

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  FILE 1 MISSES GROUP: extraright\n'
            b'  FILE 2 MISSES GROUP: extraleft\n')

    def test_variable_extra(self):
        lvar = self.left.createVariable('x', float, ('t4',))
        rvar = self.right.createVariable('y', float, ('t4',))

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  FILE 1 MISSES VARIABLE: y\n'
            b'  FILE 2 MISSES VARIABLE: x\n')

    def test_attribute_extra(self):
        self.left.some_attr = 'something something'
        self.right.some_attr2 = 'something something'

        lvar = self.left.createVariable('x', float, ('t4',))
        lvar.some_attr = 'bla'
        rvar = self.right.createVariable('x', float, ('t4',))
        rvar.some_attr2 = 'bla'

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  VAR x\n'
            b'    FILE 1 MISSES ATTRIBUTE: some_attr2\n'
            b'    FILE 2 MISSES ATTRIBUTE: some_attr\n'
            b'  FILE 1 MISSES ATTRIBUTE: some_attr2\n'
            b'  FILE 2 MISSES ATTRIBUTE: some_attr\n')

    def test_attribute_different(self):
        self.left.some_attr = 'something something'
        self.right.some_attr = 'something something more'

        lvar = self.left.createVariable('x', float, ('t4',))
        lvar.some_attr = 'bla'
        rvar = self.right.createVariable('x', float, ('t4',))
        rvar.some_attr = 'bla bla'

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  VAR x\n'
            b'    ATTRIBUTE some_attr\n'
            b'      FILE 1: bla\n'
            b'      FILE 2: bla bla\n'
            b'  ATTRIBUTE some_attr\n'
            b'    FILE 1: something something\n'
            b'    FILE 2: something something more\n')

    def test_type_different(self):
        lvar = self.left.createVariable('x', float, ('t',))
        rvar = self.right.createVariable('x', int, ('t',))

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  VAR x\n'
            b'    DIFFERENT TYPE (FILE 1: float64, FILE 2: int64)\n')

    def test_shape_different(self):
        lvar = self.left.createVariable('x', int, ('t',))
        rvar = self.right.createVariable('x', int, ('t4',))

        output = self.compare()
        self.check_output(output,
            b'GROUP /\n'
            b'  VAR x\n'
            b'    DIFFERENT SHAPE (FILE 1: (3,), FILE 2: (4,))\n')

if __name__ == '__main__':
    print("""This file is not meant to run directly, please use
    
    python -m nose test.py
    """)
    sys.exit(1)