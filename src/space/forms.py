import skfem as fem
import skfem.helpers as fhl


class Forms:

    def __init__(self, prm):
        self.is_call = prm.is_call
        self.bsopt = prm.bsopt
        self.dynh = prm.dh

    @staticmethod
    def id_bil():
        @fem.BilinearForm
        def Id_bil(u, v, _):
            return u*v
        return Id_bil

    def l_bil(self):
        @fem.BilinearForm
        def L_bil(u, v, w):
            x, y = w.x
            A = self.dynh.A(x, y)
            dA = self.dynh.dA(x, y)
            b = self.dynh.b(x, y)
            mu = [b_i - dA_i/2
                  for b_i, dA_i in zip(b, dA)]
            return (-(1/2)*fhl.dot(fhl.grad(v), fhl.mul(A, fhl.grad(u)))
                    + v*fhl.dot(mu, fhl.grad(u))
                    - self.dynh.r*v*u)
        return L_bil

    def b_lin(self):
        @fem.LinearForm
        def b_lin(v, w):
            x, y = w.x
            th = w.th

            v_avg = self.dynh.mean_variance(th, y)
            A = self.dynh.A(x, y)
            d_sig_d_v = 1/(2*v_avg**0.5)

            if self.is_call:
                du = [self.bsopt.call_delta(th, x, v_avg),
                      self.bsopt.vega(th, x, v_avg)*d_sig_d_v]
            else:
                du = [self.bsopt.put_delta(th, x, v_avg),
                      self.bsopt.vega(th, x, v_avg)*d_sig_d_v]

            return (1/2)*v*fhl.dot(w.n, fhl.mul(A, du))
        return b_lin
