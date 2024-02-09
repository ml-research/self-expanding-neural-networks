from typing import Any, Callable, Sequence, Optional
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax
from jax import numpy as jnp
from jax import vmap, jit
from jax.tree_util import tree_map, tree_reduce
import flax



def sqtree(tree):
    return tree_map(lambda v: v**2, tree)

def sqlen(tree):
    sq = tree_map(lambda v: jnp.sum(v**2), tree)
    return tree_reduce(lambda a, b: a + b, sq)

def zeros_like_tree(tree):
    return tree_map(lambda arr: jnp.zeros_like(arr), tree)

def calc_update(tau, old, new):
    return (new-old)/tau

def tree_update(old, direction, scale=1.):
    return tree_map(lambda a, b: a + b*scale, old, direction)

def tree_scale(scale, tree):
    return tree_map(lambda arr: scale*arr, tree)

def state_exp_update(key, tau, state, new):
    old = state[key]
    update = tree_map(partial(calc_update, tau), old, new)
    # umag = sqlen(update)
    newstate = state.copy({key: tree_update(old, update)})
    return newstate

def tree_axis(tree, axis):
    return tree_map(lambda: axis, tree)

@dataclass
class EMA:
    key: str
    tau: Optional[float]
    initial: float = 0.
    sqtau: Optional[float] = None
    t: int = 0

    @property
    def sqkey(self):
        return f'_{self.key}_sq'

    @property
    def tkey(self):
        return f'_{self.key}_t'

    def init(self, state):
        if 'ema' not in state:
            state = state.copy({'ema': {}})
        return state.copy({'ema': state['ema'].copy({
            self.key: self.initial,
            self.sqkey: sqtree(self.initial),
            self.tkey: self.t
        })})

    def debias(self, tree, tau, t):
        if tau is None:
            return tree
        correction = 1/(1 - ((tau-1)/tau)**t)
        return tree_map(lambda arr: arr*correction, tree)

    def get_t(self, state):
        return state['ema'][self.tkey]

    def increment_t(self, state):
        return state.copy({'ema': state['ema'].copy({
            self.tkey: self.get_t(state) + 1
        })})

    def mean(self, state):
        return self.debias(state['ema'][self.key], self.tau, self.get_t(state))

    def mean_sq(self, state):
        sqtau = self.tau if self.sqtau is None else self.sqtau
        return self.debias(state['ema'][self.sqkey], sqtau, self.get_t(state))

    def sqmag(self, state):
        return sqlen(self.mean(state))

    def variance(self, state):
        return tree_map(lambda a, b: a - b, state['ema'][self.sqkey], sqtree(self.mean(state)))

    def scalarvar(self, state):
        return sqlen(self.variance(state))

    def update(self, state, value, batch_axis=None):
        sqval = sqtree(value)
        if batch_axis is not None:
            batch_mean = lambda v: jnp.mean(v, axis=batch_axis)
            value = tree_map(batch_mean, value)
            sqval = tree_map(batch_mean, sqval)
        if self.tau is None:
            state = state.copy({'ema': state['ema'].copy({
                self.key: value,
                self.sqkey: sqval
            })})
        else:
            sqtau = self.tau if self.sqtau is None else self.sqtau
            state = state.copy({'ema': state_exp_update(self.key, self.tau, state['ema'], value)})
            state = state.copy({'ema': state_exp_update(self.sqkey, sqtau, state['ema'], sqval)})
        state = self.increment_t(state)
        return state



class Excalibur:

    def __init__(self, cfg, state):
        self.tau = cfg['tau'].get(None)
        self.LR = cfg['lr'].get(0.1)

        self.jl = EMA('jl', self.tau, state['params'])
        # state = self.jl.init(state)
        self.fp = EMA('fp', self.tau, state['params'])
        # state = self.fp.init(state)

        self.curv = EMA('curv', self.tau, 0.)

        # JP, FL, JFP
        self.lprods = EMA('lprods', self.tau, jnp.zeros((3,)))
        self.prods = EMA('prods', self.tau, jnp.zeros((3, 3)))

        self.lsqmag = EMA('lsqmag', self.tau, 1.)

    def init(self, state):
        for avg in [self.jl, self.fp, self.curv, self.lprods, self.prods]:
            state = avg.init(state)
        return state

    @staticmethod
    def alpha(func, state):
        tan = lambda params: jax.jvp(func, (params,), (state['p'],))
        (Y, JP), (_, JP2) = jax.jvp(tan, (state['params'],), (state['p'],))
        return Y, JP, JP2
        # for ema, val in zip([self.Y, self.JP, self.JP2], [Y, JP, JP2]):
            # state = ema.update(state, val)

    @staticmethod
    def beta(lossfn, state, Y, JP):
        (loss, L), (L_JP, L2) = jax.jvp(jax.value_and_grad(lossfn), (Y,), (JP,))
        return loss, L, L_JP, L2

    @staticmethod
    def gamma(func, state, cotangents):
        _, JT = jax.vjp(func, state['params'])
        # print('1.3.1')
        return list(map(JT, cotangents))

    def precondition(self, state, fp):
        # disable due to poor performance
        return fp
        loss_sqmag = self.lsqmag.mean(state)
        efdiag = self.jl.mean_sq(state)
        # efdiag = tree_map(lambda e, f: e.at[f == 0.].set(1.), efdiag, fp)
        return tree_map(lambda v, efd: v / (efd+1e-40) * loss_sqmag, fp, efdiag)

    def observe(self, model, lossfn, restrict_grad, xs, labels, state):
        # print('1')
        func = lambda x, params: model.apply(state.copy({'params': params}), x)
        # print('1.1')
        Y, JP, JP2 = vmap(lambda x: self.alpha(partial(func, x), state))(xs)
        # print('1.2')
        loss, L, L_JP, L2 = vmap(lambda l, Y_, JP_: self.beta(partial(lossfn, l), state, Y_, JP_))(labels, Y, JP)
        # print('1.3')
        (fp,), (jl,) = vmap(lambda x, JP_, L_: self.gamma(partial(func, x), state, [JP_, L_]))(xs, JP, L)
        rg = lambda tan: restrict_grad({'params': tan})['params']
        fp = rg(fp)
        jl = rg(jl)
        # print('1.4')

        state = self.lsqmag.update(state, vmap(lambda l: jnp.sum(l**2))(L), batch_axis=0)
        # print('2')
        # fp = vmap(partial(self.precondition, state))(fp)
        state = self.fp.update(state, fp, batch_axis=0)
        state = self.jl.update(state, jl, batch_axis=0)
        fp, jl = rg(self.fp.mean(state)), rg(self.jl.mean(state))
        fp = self.precondition(state, fp)
        # print('3')

        push_tan = lambda x, t: jax.jvp(partial(func, x), (state['params'],), (t,))[1]
        JFP = vmap(lambda x: push_tan(x, fp))(xs)
        FL = vmap(lambda x: push_tan(x, jl))(xs)
        # print('4')

        y_axis_count = len(Y.shape) - 1
        y_axes = list(-(i + 1) for i in range(y_axis_count))
        vectors = jnp.stack([JP, FL, JFP], axis=1)
        lprodu = jnp.sum(L[:,None,...]*vectors, axis=y_axes)
        produ = jnp.sum(vectors[:,:,None,...] * vectors[:,None,:,...], axis=y_axes)
        # print('5')
        # lprodu = jnp.array([dot(L, b) for b in vectors]).moveaxis(-1, 0)
        # produ = jnp.array([[dot(a, b) for b in vectors] for a in vectors]).moveaxis(-1, 0)

        state = self.lprods.update(state, lprodu, batch_axis=0)
        state = self.prods.update(state, produ, batch_axis=0)
        # state = self.jl.update(state, jl, batch_axis=None)
        # state = self.fp.update(state, fp, batch_axis=None)
        # print('6')
        def dot(a, b):
            return vmap(lambda a, b: jnp.sum(a*b))(a, b)
        curv = dot(JP2, L) + dot(JP, L2)
        state = self.curv.update(state, curv, batch_axis=0)
        # print('7')
        return state

    def update_params(self, state):
        recip_tau = 1. if self.tau is None else 1/self.tau
        curv = jnp.abs(self.curv.mean(state)) + 1e-3
        #curvvar = jnp.abs(self.curv.variance(state))
        scale = -self.lprods.mean(state)[0] / curv
        scale = scale * recip_tau * self.LR
        return state.copy({'params': tree_update(state['params'], state['p'], scale)})

    def update_p(self, state):
        # raise NotImplementedError
        mat = self.prods.mean(state)
        vec = self.lprods.mean(state)
        coeffs = jnp.linalg.pinv(mat)@vec
        recip_tau = 1. if self.tau is None else 1/self.tau
        jl_scale = coeffs[1] * recip_tau
        fp_scale = coeffs[2] * recip_tau

        old = state['p']
        new = tree_map(lambda v: v*coeffs[0], old)
        new = tree_update(new, self.jl.mean(state), jl_scale)
        new = tree_update(new, self.precondition(state, self.fp.mean(state)), fp_scale)

        new = tree_map(lambda v: v/jnp.sqrt(sqlen(new)), new)

        return state.copy({'p': new})

    def step(self, func, lossfn, state):
        raise NotImplementedError
        state = self.observe(func, lossfn, state)
        state = self.update_params(state)
        state = self.update_p(state)
        return state



class SimpleGradient:

    def __init__(self, cfg, params, name='grad'):
        self.tau = cfg['tau'].get(None)
        self.sqtau = cfg['sqtau'].get(None)
        self.epsilon = 1e-8

        self.grad = EMA(name, self.tau, zeros_like_tree(params), self.sqtau)

    def init(self, state):
        return self.grad.init(state)

    def observe(self, func, restrict_grad, xs, labels, params, state):
        maybe_restrict_grad = lambda x: restrict_grad(x) if restrict_grad is not None else x
        grad_for_x = lambda x, l: maybe_restrict_grad(jax.grad(func)(params, x, l))
        # grad_for_x = lambda x, l: restrict_grad(jax.grad(func)(params, x, l))
        grads = jax.vmap(grad_for_x)(xs, labels)
        return self.grad.update(state, grads, batch_axis=0)

    def read(self, state):
        return self.grad.mean(state)

    def adam(self, state):
        grad = self.grad.mean(state)
        sqgrad = self.grad.mean_sq(state)
        return tree_map(
            lambda a, b: a/(jnp.sqrt(b)+self.epsilon),
            grad,
            sqgrad
            )



class KrylovInverter:

    def __init__(self, cfg, params):
        self.tau = cfg['tau'].get(None)
        self.order = cfg['order'].get()

        initial = tree_map(lambda arr: jnp.zeros((self.order,) + arr.shape), params)

        self.krylov_powers = EMA('krylov_powers', self.tau, initial)

        initial_interleaved = jnp.zeros((2*self.order,))
        self.interleaved = EMA('interleaved', self.tau, initial_interleaved)

    def init(self, state):
        state = self.krylov_powers.init(state)
        state = self.interleaved.init(state)
        return state

    def observe(self, operator, restrict_grad, xs, tangent, state):
        maybe_restrict_grad = lambda p: restrict_grad(p) if restrict_grad is not None else p
        def calc_powers(x):
            func = lambda p, _: (maybe_restrict_grad(operator(p, x)),)*2
            # func = lambda p, _: (restrict_grad(operator(p, x)),)*2
            init = tangent
            dummies = jnp.zeros((self.order,))
            return jax.lax.scan(func, init, dummies)[1]
        krylov_powers = jax.vmap(calc_powers)(xs)
        state = self.krylov_powers.update(state, krylov_powers, batch_axis=0)

        krylov_powers = tree_map(lambda v: jnp.mean(v, axis=0), krylov_powers)
        dot = partial(tree_map, lambda a, b: jnp.sum(a*b))
        shift = lambda kry, tan: jnp.roll(kry, 1, axis=0).at[0].set(tan)
        shifted_krylov = tree_map(shift, krylov_powers, tangent)

        evens = tree_reduce(jax.lax.add, jax.vmap(dot)(shifted_krylov, shifted_krylov))
        odds = tree_reduce(jax.lax.add, jax.vmap(dot)(shifted_krylov, krylov_powers))
        interleaved = jnp.zeros((2*self.order,)).at[0::2].set(evens)\
                                                .at[1::2].set(odds)
        state = self.interleaved.update(state, interleaved)
        return state

    def inv_pow(self, tangent, exponent, state):
        assert exponent == 1
        dot = partial(tree_map, lambda a, b: jnp.sum(a*b))
        shift = lambda kry, tan: jnp.roll(kry, 1, axis=0).at[0].set(tan)
        krylov_powers = self.krylov_powers.mean(state)
        shifted_krylov = tree_map(shift, krylov_powers, tangent)
#
        # evens = tree_reduce(jax.lax.add, jax.vmap(dot)(shifted_krylov, shifted_krylov))
        # odds = tree_reduce(jax.lax.add, jax.vmap(dot)(shifted_krylov, krylov_powers))
        # interleaved = jnp.zeros((2*self.order,)).at[0::2].set(evens)\
                                                # .at[1::2].set(odds)
        interleaved = self.interleaved.mean(state)
        # print(tangent)
        # print(krylov_powers)
        # exit()
        tanprods = jax.vmap(partial(dot, tangent))(shifted_krylov)
        prods_init = jnp.zeros((self.order,))
        tanprods = tree_reduce(lambda a, b: a + b, tanprods, prods_init)
        print(f'tanprods: {tanprods}')
        # tanprods = interleaved[:self.order]
        # print(f'tanprods: {tanprods}')
        # crossprods = jax.vmap(lambda vec: jax.vmap(partial(dot, vec))(krylov_powers))(krylov_powers)
        # crossprods = jax.vmap(lambda vec: jax.vmap(partial(dot, vec))(krylov_powers))(shifted_krylov)
        # cross_init = jnp.zeros((self.order,)*2)
        # crossprods = tree_reduce(lambda a, b: a + b, crossprods, cross_init)
        # THIS IS NOT ALWAYS SYMMETRIC DUE TO NUMERICAL ERROR SO SYMMETRISE
        indices = jnp.arange(self.order)[:,None] + (jnp.arange(self.order)+1)[None,:]
        # print(indices)
        # exit()
        crossprods = interleaved[indices]
        power_penalty = 1.1
        penalties = jnp.power(power_penalty, jnp.arange(self.order))
        dindices = jnp.diag_indices(self.order)
        new_diag = crossprods[dindices] * penalties
        crossprods = crossprods.at[dindices].set(new_diag)
        print(f'crossprods: {crossprods}')
        # crossprods = 0.5*(crossprods+jnp.transpose(crossprods))
        # tancrossprod = dot(tangent, tangent)
        # tancrossprod = tree_reduce(lambda a, b: a + b, tancrossprod)
        # print(f'tancrossprod: {tancrossprod}')
        print(crossprods.dtype)
        w, v = jnp.linalg.eigh(crossprods)
        print(f'eig: {w}')
        winv = jax.nn.relu(jnp.reciprocal(w))
        coeffs = jnp.sum(tanprods[None,:]@v * winv * v, axis=-1)
        print(f'coeffs: {coeffs}')
        # print(f'pinvprods: {jnp.linalg.pinv(crossprods)}')
        # coeffs = jnp.linalg.pinv(crossprods)@tanprods
        # coeffs = jnp.linalg.solve(crossprods, tanprods)
        # print(f'coeffs: {coeffs}')
        print(f'theory out dot: {jnp.sum(coeffs*tanprods)}')
        # print(f'inverse out dot: {jnp.sum(tanprods*(crossprods@tanprods))}')

        out = jax.vmap(lambda a, b: tree_map(partial(jnp.multiply, a), b))(coeffs, shifted_krylov)
        out = tree_map(lambda arr: jnp.sum(arr, axis=0), out)
        print(f'actual out dot: {tree_reduce(jax.lax.add, dot(tangent, out))}')
        # tangent_coeff = coeffs[0]
        # initial_out = tree_map(lambda tan: tangent_coeff*tan, tangent)
        # if self.order == 1:
            # out = initial_out
        # else:
            # krylov_coeffs = jnp.roll(coeffs, -1).at[-1].set(0.)
            # print(f'theory in dot: {jnp.sum(krylov_coeffs*tanprods) + tangent_coeff*tancrossprod}')
            # # out = tree_map(lambda tan, kry: tangent_coeff*tan + jnp.average(kry,
                                                                # # axis=0,
                                                                # # weights=krylov_coeffs)*jnp.sum(krylov_coeffs),
                       # # tangent,
                       # # krylov_powers)
            # rest = jax.vmap(lambda a, b: tree_map(partial(jnp.multiply, a), b))\
                # (krylov_coeffs, krylov_powers)
            # out = tree_map(lambda init, rest: init + jnp.sum(rest, axis=0), initial_out, rest)
            # # out = jax.lax.associative_scan(func, initial_out, (krylov_coeffs, krylov_powers))[1]
        return out



class CGInverter:

    def __init__(self, cfg, params):
        # self.tau = None
        self.tau = cfg['soln_tau'].get(None)
        self.sqtau = cfg['soln_sqtau'].get(None)
        self.epsilon = 1e-8
        self.order = cfg['order'].get()
        self.method = cfg['method'].get('cg')
        assert self.method in ['cg', 'gmres'], f"requested method: {self.method} not recognised"
        initial = tree_map(jnp.zeros_like, params)
        self.soln = EMA('soln', self.tau, initial, self.sqtau)

    def init(self, state):
        state = self.soln.init(state)
        return state

    def observe(self, operator, restrict_grad, xs, tangent, state):
        maybe_rg = lambda p: restrict_grad(p) if restrict_grad is not None else p
        F_x = lambda p, x: maybe_rg(operator(maybe_rg(p), x))
        F = lambda p: tree_map(lambda arr: arr.mean(axis=0), vmap(partial(F_x, p))(xs))
        if self.method == 'cg':
            soln, _ = jax.scipy.sparse.linalg.cg(F, tangent, maxiter=self.order)
        elif self.method == 'gmres':
            soln, _ = jax.scipy.sparse.linalg.gmres(F, tangent, maxiter=self.order, restart=self.order)
        state = self.soln.update(state, soln)
        return state

    def read(self, state):
        return self.soln.mean(state)

    def adam(self, state):
        grad = self.soln.mean(state)
        sqgrad = self.soln.mean_sq(state)
        return tree_map(
            lambda a, b: a/(jnp.sqrt(b)+self.epsilon),
            grad,
            sqgrad
            )



class FisherNorm:

    def __init__(self, cfg, params):
        self.tau = None
        self.Fsqmag = EMA('Fsqmag', self.tau)

    def init(self, state):
        state = self.Fsqmag.init(state)
        return state

    def observe(self, func, xs, tangent, params, state):
        fwd_jac = lambda tan, x: jax.jvp(lambda p: func(p, x), (params,), (tan,))[1]
        sqmag_x = lambda x: sqlen(fwd_jac(tangent, x))
        sqmag = vmap(sqmag_x)(xs).mean(axis=0)
        state = self.Fsqmag.update(state, sqmag)
        return state

    def read(self, state):
        return self.Fsqmag.mean(state)

    def apply(self, tangent, state):
        mag = jnp.sqrt(self.read(state))
        return tree_map(lambda arr: arr/mag, tangent)



class KrylovNG:

    def __init__(self, cfg, params):
        self.SG = SimpleGradient(cfg, params)
        self.KI = KrylovInverter(cfg, params)
        self.key = 'nat_grad'
        self.adam = cfg['use_adam'].get(False)

    def init(self, state=flax.core.frozen_dict.freeze({})):
        state = self.SG.init(state)
        state = self.KI.init(state)
        return state

    def observe(self, func, lossfn, restrict_grad, xs, ls, params, state):
        tree_double = lambda tree: tree_map(lambda arr: arr.astype('float64'), tree)
        params=tree_double(params)
        # print(func(params, xs[0]))
        # print(lossfn(ls[0], func(params, xs[0])))
        # exit()
        state = self.SG.observe(lambda params, x, l: lossfn(l, func(params, x)),
                                restrict_grad,
                                xs,
                                ls,
                                params,
                                state)
        fwd_jac = lambda tan, x: jax.jvp(lambda p: func(p, x), (params,), (tan,))[1]
        bkd_jac = lambda tan, x: jax.vjp(lambda p: func(p, x), params,)[1](tan)[0]
        # sqjacob = lambda tan, x: bkd_jac(fwd_jac(tan, x), x)
        ID_MUL = 0e1
        sqjacob = lambda tan, x: tree_map(lambda a ,b: ID_MUL*a+b, tan, bkd_jac(fwd_jac(tan, x), x))
        if self.adam:
            grad = tree_double(self.SG.adam(state))
        else:
            grad = tree_double(self.SG.read(state))
        # print(params)
        # print(grad)
        # tan_ = fwd_jac(grad, xs[0])
        # # print(tan_)
        # # JT = jax.vjp(lambda p: func(p, xs[0]), params)[1]
        # # print(JT)
        # # tan_2 = JT(tan_)
        # tan_2 = bkd_jac(tan_, xs[0])
        # print(tan_2)
        # exit()
        state = self.KI.observe(sqjacob, restrict_grad, xs, grad, state)
        nat_grad = self.KI.inv_pow(grad, 1, state)
        return state.copy({self.key: nat_grad})

    def read(self, state):
        return state[self.key]



class CGNG:

    def __init__(self, cfg, params):
        self.SG = SimpleGradient(cfg, params)
        self.CGinv = CGInverter(cfg, params)
        self.tikhonov = cfg['tikhonov'].get(1e0)
        self.adam = cfg['soln_adam'].get(False)
        self.norm = FisherNorm(cfg, params)
        self.gnorm = FisherNorm(cfg, params)
        self.param_Fnorm = FisherNorm(cfg, params)
        self.l2 = cfg['l2_regularization'].get(0e0)

    def init(self, state=flax.core.frozen_dict.freeze({})):
        state = self.SG.init(state)
        state = self.CGinv.init(state)
        state = self.norm.init(state)
        state = self.gnorm.init(state)
        state = self.param_Fnorm.init(state)
        return state

    def observe(self, func, lossfn, restrict_grad, xs, ls, params, state):
        tree_double = lambda tree: tree_map(lambda arr: arr.astype('float64'), tree)
        params=tree_double(params)
        state = self.SG.observe(lambda params, x, l: lossfn(l, func(params, x)),
                                restrict_grad,
                                xs,
                                ls,
                                params,
                                state)
        fwd_jac = lambda tan, x: jax.jvp(lambda p: func(p, x), (params,), (tan,))[1]
        bkd_jac = lambda tan, x: jax.vjp(lambda p: func(p, x), params,)[1](tan)[0]
        sqjacob = lambda tan, x: bkd_jac(fwd_jac(tan, x), x)

        rparams = restrict_grad(params)
        grad = self.SG.read(state)
        grad = tree_map(lambda g, p: g + self.l2*p, grad, rparams)
        grad = tree_double(grad)

        # gradlen = jnp.sqrt(sqlen(grad))
        state = self.gnorm.observe(func, xs, grad, params, state)
        eps = self.tikhonov
        # id_coeff = eps * self.gnorm.read(state)/sqlen(grad)
        id_coeff = eps
        tikhonov = lambda tan, x: tree_map(jax.lax.add, sqjacob(tan, x), tree_scale(id_coeff, tan))

        state = self.CGinv.observe(tikhonov, restrict_grad, xs, grad, state)

        state = self.norm.observe(func, xs, self.read(state), params, state)
        state = self.param_Fnorm.observe(func, xs, rparams, rparams, state)
        return state

    def raw_grad(self, state):
        if self.adam:
            return self.SG.adam(state)
        else:
            return self.SG.read(state)

    def read(self, state):
        if self.adam:
            return self.CGinv.adam(state)
        else:
            return self.CGinv.read(state)

    def read_normed(self, state):
        return self.norm.apply(self.read(state), state)



class KrylovNM:

    def __init__(self, cfg, params):
        self.SG = SimpleGradient(cfg, params)
        self.KI = KrylovInverter(cfg, params)

    def init(self, state):
        state = self.SG.init(state)
        state = self.KI.init(state)
        return state

    def observe(self, func, lossfn, params, xs, state):
        assert False
        #need to change sqjacob to hessian, then implement powers > 1
        state = self.SG.observe(state, lambda params, x: lossfn(func(params, x)))
        fwd_jac = lambda tan, x: jax.jvp(lambda p: func(p, x), (params,), (tan,))[1][0]
        bkd_jac = lambda tan, x: jax.vjp(lambda p: func(p, x), (params,))[1]((tan,))[0]
        sqjacob = lambda tan, x: bkd_jac(fwd_jac(tan, x), x)
        grad = self.SG.read(state)
        state = self.KI.observe(sqjacob, xs, grad, state)
        h2inv_grad = self.KI.inv_pow(grad, 2, state)
        return state.copy({self.key: nat_grad})

    def read(self, state):
        return state[self.key]
