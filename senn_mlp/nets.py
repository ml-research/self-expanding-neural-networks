from typing import Any, Callable, Sequence, Optional
from abc import ABC, abstractmethod
from functools import partial
from itertools import count
import numpy as np

import jax
from jax import lax, random, numpy as jnp
from jax.tree_util import tree_map as jtm, tree_reduce as jtr
import flax
from flax import linen as nn
from flax.linen import initializers
from flax.traverse_util import t_identity as trav_identity

def_prec = 'highest'



def pad_axis(arr, count, axis=0):
    if count == 0:
        return arr
    assert count > 0
    pad_shape = arr.shape[:axis] + (count,) + arr.shape[axis:][1:]
    return jnp.concatenate([arr, jnp.zeros(pad_shape)], axis=axis)

def pad_target(current):
    return int(2**jnp.ceil(jnp.log2(current+1)))

def update_dict(target, args, func, is_leaf):
    for key, value in args.items():
        if is_leaf(value):
            target[key] = func(target[key], value)
        else:
            target[key] = update_dict(target[key], value, func, is_leaf)
    return target



class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x



class Passive(nn.Module):

    def insert_feature(self, feature, index, origin=0):
        func = lambda v, f: v.at[..., index].set(f[..., origin])
        return self.variables.copy({
            'params': jtm(func, self.variables['params'].unfreeze(), feature['params'])
        })

    def pad_features(self, count):
        trav = trav_identity['params'].tree()
        func = lambda p: pad_axis(p, count, axis=-1)
        return trav.update(func, self.variables.unfreeze())

    def null(self):
        return jnp.bool_(True)

    def restrict_params(self, in_dim, out_dim):
        params = self.variables["params"]
        restrict = lambda arr: arr.at[..., out_dim:].set(0.)
        return jtm(restrict, params)



class DConv(Passive):
    kernel_size: int = 3
    init_identity: bool = False
    #N.B. kernel param has shape [3,3,1,F]

    def identity_kernel(self, F):
        K = jnp.zeros((self.kernel_size, self.kernel_size, 1, F))
        cidx = self.kernel_size//2
        return K.at[cidx, cidx,...].set(1.)

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        kernel = (self.kernel_size,)*2
        if self.init_identity:
            init = lambda rng, shape, dtype: self.identity_kernel(x.shape[-1])
            x = nn.Conv(dim, kernel, padding='SAME', feature_group_count=dim, kernel_init=init, precision=def_prec)(x)
        else:
            lec_n = initializers.lecun_uniform()
            unif = initializers.uniform()
            def init(rng, shape, dtype):
                default = lec_n(rng, shape, dtype)
                return default / jnp.sqrt((default**2).sum(axis=-2, keepdims=True))
            x = nn.Conv(dim, kernel, padding='SAME', feature_group_count=dim, kernel_init=init, bias_init=unif, precision=def_prec)(x)
        return x

    def identity_params(self, F):
        params = self.variables["params"]
        return {'Conv_0': {
            'bias': jnp.zeros(F),
            'kernel': self.identity_kernel(F)
        }}



class Rational1D(Passive):
    residual: bool = True
    init_identity: bool = False
    epsilon: float = 1e-8

    @nn.compact
    def __call__(self, x):
        init = initializers.zeros if self.init_identity else random.normal
        v = self.param('w_vec', init, x.shape[-1:])
        c = self.param('w_const', init, x.shape[-1:])
        sumsq = v**2 + c**2 + self.epsilon
        if self.residual:
            init = initializers.ones if self.init_identity else random.normal
            l = self.param('w_lin', init, x.shape[-1:])
            sumsq = sumsq + l**2
        norm = jnp.sqrt(sumsq)
        #DISABLE NORMING:
        # norm = jnp.ones_like(norm)
        den = 1/(1 + x**2)
        self.sow("intermediates", "lin", x)
        self.sow("intermediates", "odd", 2.*x*den)
        self.sow("intermediates", "even", den)
        num = 2.*v*x + c
        out = num*den
        if self.residual:
            out = out + l*x
        out = out / norm
        self.sow("intermediates", "activations", out)
        return out

    def null(self):
        null = lambda name: self.variables["params"][name] == 0.
        return null('w_vec') & null('w_const') & null('w_lin')

    def identity_params(self, F, rank=None):
        params = {
            'w_vec': jnp.zeros(F),
            'w_const': jnp.zeros(F),
            'w_lin': jnp.ones(F)
        }
        if rank is not None:
            params['w_lin'] = params['w_lin'].at[rank:].set(0.)
        return params




class Active(ABC, Passive):
    @abstractmethod
    def pad_inputs(self, count):
        pass

    @abstractmethod
    def input_size(self):
        pass

    @abstractmethod
    def output_size(self):
        pass

    def pad_pow2(self, override=False):
        null = self.null() & ~jnp.bool_(override)
        target = pad_target((~null).sum())
        current = len(null)
        deficit = target - current
        if deficit > 0:
            params = self.pad_features(deficit)
            return params, target
        else:
            return self.variables, current



class Splittable(Active):
    @abstractmethod
    def split_basis_size(self):
        pass

    @abstractmethod
    def split_params(self, basis, shift):
        pass

    # @abstractmethod
    # def split_module(self):
        # pass



class DDense(nn.Dense, Splittable):
    init_zero: bool = False
    def __init__(self, features, init_zero = False):
        if init_zero:
            super().__init__(features, kernel_init=initializers.zeros, precision=def_prec)
        else:
            super().__init__(features, bias_init=initializers.normal(1e0), precision=def_prec)
            # init = initializers.lecun_uniform()
            # unif = initializers.uniform()
            # super().__init__(features, bias_init=unif, kernel_init=init, precision=def_prec)

    def __call__(self, x):
        self.sow("intermediates", "preactivations", x)
        x = nn.Dense.__call__(self, x)
        return x
    
    def pad_inputs(self, count):
        return self.variables.copy({
            "params": self.variables["params"].copy({
                "kernel": pad_axis(self.variables["params"]["kernel"], count, axis=-2)
            })
        })

    def input_size(self):
        return self.variables["params"]["kernel"].shape[-2]

    def output_size(self):
        return self.variables["params"]["kernel"].shape[-1]

    def null(self):
        null = (self.variables["params"]["kernel"] == 0.).all(axis=-2)
        return null & (self.variables["params"]["bias"] == 0.)

    def split_basis_size(self):
        return min(self.input_size(), self.output_size())

    # def split_params(self, basis, shift, kill_in, kill_out):
        # old = self.variables["params"]["kernel"]
        # in_dim, out_dim = old.shape
        # # basis_inv = jnp.linalg.pinv(basis)
        # # if in_dim-kill_in < out_dim-kill_out:
        # if in_dim < out_dim:
            # if kill_in > 0:
                # basis = basis.at[-kill_in:,:].set(0.)
                # basis = basis.at[:,-kill_in:].set(0.)
                # shift = shift.at[-kill_in:].set(0.)
            # basis_inv = jnp.linalg.pinv(basis)
            # first = basis
            # second = basis_inv@old
            # intermediate = in_dim
        # else:
            # if kill_out > 0:
                # basis = basis.at[-kill_out:,:].set(0.)
                # basis = basis.at[:,-kill_out:].set(0.)
                # shift = shift.at[-kill_out:].set(0.)
            # basis_inv = jnp.linalg.pinv(basis)
            # first = old@basis
            # second = basis_inv
            # intermediate = out_dim
        # first_params = {
            # "bias": shift,
            # "kernel": first
        # }
        # second_params = {
            # "bias": self.variables["params"]["bias"] - shift@second,
            # "kernel": second
        # }
        # return first_params, second_params

    def split_params(self, basis, inv_basis, shift):
        old = self.variables["params"]["kernel"]
        first = basis
        # print(old.shape)
        # print(inv_basis.shape)
        second = inv_basis@old[:inv_basis.shape[1],:]
        first_params = {
            "bias": shift,
            "kernel": first,
        }
        second_params = {
            "bias": self.variables["params"]["bias"] - shift@second,
            "kernel": second
        }
        return first_params, second_params

    def split_module(self):
        intermediate = min(self.input_size(), self.output_size())
        return DDense(intermediate), DDense(self.output_size())

    def identity_params(self):
        params = self.variables["params"]
        old_kernel = params["kernel"]
        diag = jnp.diag_indices(min(old_kernel.shape))
        return params.copy({
            "bias": jnp.zeros_like(params["bias"]),
            "kernel": jnp.zeros_like(old_kernel).at[diag].set(1.)
        })

    def restrict_params(self, in_dim, out_dim):
        params = super().restrict_params(in_dim, out_dim)
        # print(params)
        return params.copy({
            "kernel": params["kernel"].at[in_dim:, :].set(0.)
        })

    def extract_activations(self):
        return self.variables["intermediates"]["preactivations"]

    def extract_out_grads(self):
        return self.variables["params"]["bias"]
        



class Layer(Splittable):
    features: Optional[int]
    make_invariant: Sequence[nn.Module] = ()
    make_linear: Callable[[int], Active] = DDense
    make_equivariant: Sequence[Callable[[], Passive]] = (partial(Rational1D, True),)

    def setup(self):
        F = self.features
        self.linear = None if F is None else self.make_linear(F)
        self.equivariant = None if F is None else [func() for func in self.make_equivariant]
        self.invariant = [func() for func in self.make_invariant]

    nn.nowrap
    def dormant(self):
        return self.features is None

    def __call__(self, x):
        if self.features is not None:
            for module in self.invariant:
                x = module(x)
            x = self.linear(x)
            for module in self.equivariant:
                x = module(x)
        return x

    def pad_inputs(self, count):
        return self.variables.copy({
            "params": self.variables["params"].copy({
                "linear": self.linear.pad_inputs(count)["params"]
                })
            })

    def null(self):
        if self.dormant():
            return None
        else:
            out = self.linear.null()
            for i, module in enumerate(self.equivariant):
                if f'equivariant_{i}' in self.variables['params']:
                    out = out & module.null()
            return out

    def input_size(self):
        return self.linear.input_size()

    def output_size(self):
        return self.linear.output_size()

    def split_basis_size(self):
        return self.linear.split_basis_size()

    # def split_params(self, basis, shift, kill_in, kill_out):
        # first, second = self.linear.split_params(basis, shift, kill_in, kill_out)
        # # F = first['kernel'].shape[-1]
        # # equivariant_params = [
            # # eq.identity_params(F) for eq in self.equivariant
        # # ]
        # # first = self.variables["params"].copy({
            # # "linear": first
        # # })
        # # for i, p in enumerate(equivariant_params):
            # # first = first.copy({
                # # f"equivariant_{i}": p
            # # })
        # second = self.variables["params"].copy({
                # "linear": second
            # })
        # # assert "invariant_0" not in self.variables["params"]
        # return first, second

    def split_params(self, basis, inv_basis, shift):
        first, second = self.linear.split_params(basis, inv_basis, shift)
        second = self.variables["params"].copy({
            "linear": second
        })
        return first, second

    def split_identity(self, linear_params, rank=None):
        params = self.variables['params']
        params = params.copy({
            'linear': linear_params,
        })
        for i, eq in enumerate(self.equivariant):
            params = params.copy({
                f'equivariant_{i}': eq.identity_params(self.features, rank=rank)
            })
        assert 'invariant_0' not in params
        return params

    def restrict_params(self, in_dim, out_dim):
        params = self.variables["params"]
        params = params.copy({
            'linear': self.linear.restrict_params(in_dim, out_dim)
        })
        params = params.copy({
            f'equivariant_{i}': eq.restrict_params(in_dim, out_dim) \
            for i, eq in enumerate(self.equivariant) \
            if f'equivariant_{i}' in params
        })
        assert 'invariant_0' not in params
        return params

    def identity_params(self):
        return self.split_identity(self.linear.identity_params())



class Layers(nn.Module):
    layers: Sequence[Layer]

    def features(self):
        return [layer.features for layer in self.layers if layer.features]

    def nulls(self):
        return [layer.null() for layer in self.layers if layer.features]

    def lift(self, index, func, *args):
        return func(self.layers[index], *args)

    def __call__(self, x):
        for L in self.layers:
            x = L(x)
        return x

    def pad_features(self, index, count):
        variables = self.variables
        def layer_update(i, new_params):
            nonlocal variables
            variables = variables.copy({
                'params': variables['params'].copy({
                    f'layers_{i}': new_params['params']
                })
            })
        layer_update(index, self.layers[index].pad_features(count))
        subsequent = index + 1
        if subsequent < len(self.layers):
            layer_update(subsequent, self.layers[subsequent].pad_inputs(count))
        return variables

    def pad_pow2(self, index, override=False):
        features = self.features()
        increase = None
        old = features[index]
        updated, features[index] = self.layers[index].pad_pow2(override)
        variables = self.variables
        def layer_update(i, new_params):
            nonlocal variables
            variables = variables.copy({
                "params": variables["params"].copy({
                    f"layers_{i}": new_params["params"]
                })
            })
        layer_update(index, updated)
        layer_update(index+1, self.layers[index+1].pad_inputs(features[index] - old))
        return variables, features

    def split_layer(self, index, basis, shift, inserter):
        kill_out = sum(self.layers[index].null())
        kill_in = sum(self.layers[index-1].null()) if index > 0 else 0
        first_linear, second_layer = self.layers[index].split_params(basis, shift, kill_in, kill_out)
        first_layer = inserter(first_linear)
        params = self.variables['params']
        for i in reversed(range(index+1, len(self.layers))):
            params = params.copy({
                f'layers_{i+1}': params[f'layers_{i}']
            })
        params = params.copy({
            f'layers_{index}': first_layer,
            f'layers_{index+1}': second_layer
        })
        return params

    @nn.nowrap
    @staticmethod
    def get_nearest_active(dims, index):
        assert dims[index] is None
        for i, dim in enumerate(dims[index:]):
            if dim is not None:
                succeeding = index + i
                break
        preceding = None
        for i, dim in enumerate(reversed(dims[:index+1])):
            if dim is not None:
                preceding = index - i
                break
        return preceding, succeeding

    def activate_layer(self, dims, index, basis, shift):
        preceding, succeeding = self.get_nearest_active(dims, index)
        input_dim = self.layers[0].linear.input_size()
        # assert input_dim == 1
        new_rank = sum(~self.layers[preceding].null()) if preceding is not None else input_dim
        new_input_size = self.layers[index-1].linear.output_size() if index != 0 else input_dim
        basis = basis.at[new_rank:,:].set(0.)
        basis = basis.at[:,new_rank:].set(0.)
        shift = shift.at[new_rank:].set(0.)
        inv_basis = jnp.linalg.pinv(basis)
        basis = basis[:new_input_size,:]
        # kill_in = sum(self.layers[preceding].null()) if preceding is not None else 0
        # kill_out = sum(self.layers[succeeding].null())
        # print(inv_basis.shape)
        new_linear, new_succeeding = self.layers[succeeding] \
                                         .split_params(basis,
                                                       inv_basis,
                                                       shift)
        new_layer = self.layers[index].split_identity(new_linear, rank=new_rank)

        return self.variables.copy({
            'params': self.variables['params'].copy({
                f'layers_{index}': new_layer,
                f'layers_{succeeding}': new_succeeding
            })
        })

    def restrict_params(self, dims):
        state = self.variables
        params = state["params"]
        
        def modify(layer, in_dim, out_dim):
            if out_dim is None:
                return layer.identity_params()
            else:
                return layer.restrict_params(in_dim, out_dim)

        def get_key(i):
            return f"layers_{i}"

        input_size = self.layers[0].input_size()
        in_dims = [input_size] + dims[:-1]
        out_dims = dims
                
        return state.copy({
            "params": params.copy({
                get_key(i): modify(layer, in_dim, out_dim) \
                    for i, (layer, in_dim, out_dim) in enumerate(zip(self.layers, in_dims, out_dims))
            })
        })

    def restrict_grad(self, dims):
        params = self.variables["params"]
        # print(params)

        input_size = self.layers[0].input_size()
        # in_dims =  [input_size] + dims[:-1]
        in_dims = []
        carry = input_size
        for out_dim in dims:
            if out_dim is None:
                in_dims.append(None)
            else:
                in_dims.append(carry)
                carry = out_dim

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, dims)):
            key = f"layers_{i}"
            if out_dim is None:
                params = params.copy({
                    key: jtm(lambda arr: arr*0., params[key])
                })
            else:
                params = params.copy({
                    key: self.layers[i].restrict_params(in_dim, out_dim)
                })
        # print(params)
        # print(dims)
        # print('Done!')
        grad = self.variables.copy({
            "params": params
        })
        return grad

    def extract_activations(self):
        return [layer.linear.extract_activations() for layer in self.layers]

    def extract_out_grads(self):
            return [layer.linear.extract_out_grads() for layer in self.layers]




class Seq(nn.Module):
    modules: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x):
        for m in modules:
            x = m(x)
        return x



def push_tangent(func, t, x):
    return jax.jvp(func, [x], [t])

def push_curvature(func, t, x):
    return jax.jvp(partial(push_tangent, func, t), [x], [t])

def reject_from(a, b):
    # return the component of a orthogonal to b
    return a - b * jnp.sum(a*b) / jnp.sum(b*b)

def combine(a, b, c):
    # return coefficients for 'a' and 'b' to produce least squares solution for c
    # only accurate up to a scale factor
    alpha = (b*b).sum() * (a*c).sum() - (a*b).sum() * (b*c).sum()
    beta = (a*a).sum() * (b*c).sum() - (a*b).sum() * (a*c).sum()
    return alpha, beta

def tree_length(v):
    tsq = jtm(lambda v: jnp.sum(v**2), v)
    ssq = jtr(lambda a, b: a + b, tsq)
    return jnp.sqrt(ssq)

def pass_one(func, lossfn, params, p, x):
    func_x = lambda params: func(params, x)
    (y, Jp),(_, J2p) = push_curvature(func_x, p, params)
    # y, JT = jax.vjp(func_x, params)
    (loss, dL), (Jp_prod_dL, dLdJp) = jax.jvp(jax.value_and_grad(lossfn), [y], [Jp])

    dL_p = Jp_prod_dL
    d2L_p = jnp.sum(J2p*dL) + jnp.sum(Jp*dLdJp)

    return y, loss, dL_p, d2L_p, dL, Jp, Jp_prod_dL#, JT

def pass_two(func, params, err, x):
    func_x = lambda params: func(params, x)
    _, JT = jax.vjp(func_x, params)
    return JT(err)

def pass_three(func, params, p, x):
    func_x = lambda params: func(params, x)
    return push_tangent(func_x, p[0], params)[1]

def batch_ng(func, lossfn, metric, params, x, labels, p, axis=0):
    y, loss, dL_p, d2L_p, dL, Jp, corr = jax.vmap(lambda x, l: pass_one(func, partial(lossfn, l), params, p, x), 0)(x, labels)

    orth_grad = reject_from(dL, Jp)
    im_reject = jax.vmap(partial(pass_two, func, params), 0)(orth_grad, x)
    delta_p = jtm(lambda v: jnp.sum(v, axis=0), im_reject)
    delta_Jp = jax.vmap(partial(pass_three, func, params, delta_p), 0)(x)

    alpha, beta = combine(Jp, delta_Jp, dL)
    pu_rescale = beta/alpha * tree_length(p)
    p_update = jtm(lambda v: v * pu_rescale, delta_p)

    #curv = len(d2L_p)*jnp.sqrt(jnp.mean(d2L_p**2))
    curv = jnp.abs(jnp.sum(d2L_p))
    #curv = jnp.sum(jnp.abs(d2L_p))
    p_rescale = -jnp.sum(dL_p) / curv
    param_update = jtm(lambda v: v * p_rescale, p)

    speed = jnp.sqrt((Jp**2).sum())
    utility = corr.sum() / jnp.sqrt((dL**2).sum()) * speed
    distance = speed / p_rescale

    avg_loss = jnp.mean(loss)
    if metric is None:
        return avg_loss, param_update, p_update, utility
    else:
        return avg_loss, param_update, p_update, utility, metric(y, labels), curv, distance

    #goodness = dL . Jp / (|dL| |Jp|)
