"""
Compute the general form the covariance matrix using symbolic expressions?
"""

# study example from https://stackoverflow.com/questions/55307308/convert-a-function-defined-in-numpy-to-sympy

import ast, inspect
import numpy as np
import sympy as sp
from types import FunctionType


translate = {'array': 'Array'}

class np_to_sp(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id=='np':
            node = ast.copy_location(ast.Name(id='sp', ctx=node.ctx), node)
        return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.value.id=='sp' and node.attr in translate:
            fields = {k: getattr(node, k) for k in node._fields}
            fields['attr'] = translate[node.attr]
            node = ast.copy_location(ast.Attribute(**fields), node)
        return node



def f(a):
    return np.array([np.sin(a), np.cos(a)])

def f2(a):
    return np.array([1, np.sin((f2bis(a)))])

def f2bis(a):
    return a**2

def f3(a):
    return f(a) + f2(a)


def g(a):
    return a*np.eye(3)

def g2(a):
    # NOT WORKING!!!! Turn the whole code in sympy...
    return np.linalg.inv(g(a))


# F_TO_SYM = f3
F_TO_SYM = g2

for fn in F_TO_SYM.__code__.co_names:
    print(fn)
    fo = globals()[fn]
    if not isinstance(fo, FunctionType):
        continue
    z = ast.parse(inspect.getsource(fo))
    np_to_sp().visit(z)
    exec(compile(z, '', 'exec'))

x = sp.Symbol('x')
print(F_TO_SYM(x))