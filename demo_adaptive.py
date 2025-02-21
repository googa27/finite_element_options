import skfem as fem
import skfem.helpers as fh
import skfem.visuals as femv
import numpy as np

import src.plots as splt

from skfem.visuals.matplotlib import plot as femplot

### INITIAL DEFINITIONS ###

e = fem.ElementTriP1()
x_max = 1
y_max = 1


def f(x, y):
    return np.exp((x - 0.3)**2 + (y - 0.2)**2)

### FORMS ###


@fem.LinearForm
def b_lin(v, w):
    x, y = w.x
    return f(x, y)*v


@fem.BilinearForm
def a_bil(u, v, _):
    return fh.dot(u.grad, v.grad) + 0.1*u*v


def solve_system(m):
    Vh = fem.Basis(m, e)
    print(f'Number of elements: {Vh.N}')
    A = a_bil.assemble(Vh)
    b = b_lin.assemble(Vh)
    A, b = fem.enforce(A, b,
                       D=Vh.get_dofs(['y_min', 'y_max']))
    u = fem.solve(A, b)
    return u


def _eval_estimator(m, u):
    fbasis = [fem.InteriorFacetBasis(m, e, side=i) for i in [0, 1]]
    w = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}

    @fem.Functional
    def edge_jump(w):
        h = w.h
        n = w.n
        dw1 = fh.grad(w['u1'])
        dw2 = fh.grad(w['u2'])
        return h * ((dw1[0] - dw2[0]) * n[0] +
                    (dw1[1] - dw2[1]) * n[1]) ** 2

    eta_E = edge_jump.elemental(fbasis[0], **w)

    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(.5 * tmp[m.t2f], axis=0)

    return eta_E


def mesh_init(x_max, y_max, n=3):
    m = (fem.MeshTri()
         .init_tensor(x=np.linspace(0, x_max, n),
                      y=np.linspace(0, y_max, n)
                      )
         .with_boundaries({'x_min': lambda x: x[0] == 0,
                           'x_max': lambda x: x[0] == x_max,
                           'y_min': lambda x: x[1] == 0,
                           'y_max': lambda x: x[1] == y_max
                           }
                          )
         )

    return m


def adapt_mesh(m, u):
    m = m.refined(fem.adaptive_theta(_eval_estimator(m, u))).smoothed()
    m = m.with_boundaries({'x_min': lambda x: x[0] == 0,
                           'x_max': lambda x: x[0] == x_max,
                           'y_min': lambda x: x[1] == 0,
                           'y_max': lambda x: x[1] == y_max
                           }
                          )
    return m


### MAIN LOOP ###
m = mesh_init(x_max, y_max, 3)
u = solve_system(m)

for i in range(5):
    m = adapt_mesh(m, u)
    u = solve_system(m)

Vh = fem.Basis(m, e)
splt.plot_2d(Vh, u, title='Solution')
