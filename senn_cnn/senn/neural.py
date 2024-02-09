"""NN Modules and associated second order and expansion methods"""
from functools import partial, wraps
from compose import compose
from collections.abc import Callable
from typing import Tuple, Any, Optional
from math import prod

import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.tree_util import tree_map, Partial
from tensorflow_probability.substrates import jax as tfp
import flax
from flax import linen as nn

from senn import linalg


def with_dummy_cotan(mdl):
    def f(x, cotan):
        return x

    def fwd(x, cotan):
        y, vjp = nn.vjp(mdl.apply, mdl, x)
        params_t = vjp(cotan)
        return x, params_t

    def bwd(params_t, x_t):
        return params_t, x_t


def homogenize_last_dim(x):
    """inserts a constant input of 1.0 at the first channel index"""
    ones = jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)
    return jnp.concatenate((ones, x), axis=-1)


class Sherman(nn.Module):
    """Tracks second order statistics for a linear transform.

    Assumes that the last dimension of the kernel is output, and the rest are input.
    Uses exponential averaging, with weight for new observation given by "decay".
    """

    decay: Any = None

    @nn.compact
    def update_and_track(self, x, fake_bias):
        # vec = x.reshape((self.in_size(),))
        vec = jnp.ravel(x)
        init_fn = linalg.SecondMoment.init_identity
        self.variable("kron", "in", init_fn, vec.size)
        self.variable("kron", "out", init_fn, fake_bias.size)
        kron_in = self.get_variable("kron", "in")
        kron_in = kron_in.rank_one_update(vec, decay=self.decay)
        self.put_variable("kron", "in", kron_in)

        # assert fake_bias.shape == (self.features,)
        return self.perturb("out", fake_bias)

    def update_kron_out(self):
        x = self.get_variable("perturbations", "out")
        # vec = x.reshape((self.features,))
        vec = jnp.ravel(x)
        kron_out = self.get_variable("kron", "out")
        kron_out = kron_out.rank_one_update(vec, decay=self.decay)
        self.put_variable("kron", "out", kron_out)

    def get_kron(self):
        kron_in = self.get_variable("kron", "in")
        kron_out = self.get_variable("kron", "out")
        return kron_in, kron_out

    @nn.nowrap
    def kron_mul(self, Kin, Kout, kernel):
        in_size = Kin.shape[-1]
        out_size = Kout.shape[-1]
        reshaped = kernel.reshape((in_size, out_size))
        reshaped = Kin @ reshaped @ (jnp.conj(Kout).T)
        return reshaped.reshape(kernel.shape)

    def ichol_mul(self, kernel):
        Kin, Kout = self.get_kron()
        kernel = self.kron_mul(Kin.ichol, Kout.ichol, kernel)
        return kernel

    def inv_mul(self, kernel):
        Kin, Kout = self.get_kron()
        kernel = self.kron_mul(Kin.inv, Kout.inv, kernel)
        return kernel


class Kronify(nn.Module):
    """Wraps a linear module, for which it tracks second order statistics.

    Optionally, appends a nonlinearity to the module and/or adds a homogeneous coordinate to
    the input. This coordinate simulates the use of a bias while introducing no new logic.
    """

    linear: nn.Module
    sherman: Sherman = Sherman()
    homogenize: bool = False
    nonlin: Optional[Callable] = None

    def reduced_linear_variables(self):
        params = self.linear.variables["params"]
        assert "bias" not in params, "bias not supported, use homogenize instead"
        params = tree_map(lambda arr: arr[..., [0]], params)
        return {"params": params}

    def update_kron_out(self):
        self.sherman.update_kron_out()

    def __call__(self, x):
        if self.homogenize:
            x = homogenize_last_dim(x)
        self.sow("intermediates", "kernel_in", x)

        if self.has_rng("noisy_params"):
            assert not self.is_initializing(), "do not use noise when initializing"

            lin, linvars = self.linear.unbind()
            kernel = self.linear.get_variable("params", "kernel")
            key = self.make_rng("noisy_params")
            white_noise = jax.random.normal(key, kernel.shape, kernel.dtype)
            unwhite_noise = self.sherman.ichol_mul(white_noise)
            noised_kernel = jax.lax.stop_gradient(unwhite_noise) + kernel
            y = lin.apply(linvars.copy({"params": {"kernel": noised_kernel}}), x)

        else:
            y = self.linear(x)

        if self.has_rng("hutchinson"):
            dummy = self.linear.clone(features=1).bind(self.reduced_linear_variables())
            dummy_y, vjp = nn.vjp(lambda mdl: mdl(x), dummy)
            hutch_key = self.make_rng("hutchinson")
            hutch_cotan = jax.random.rademacher(hutch_key, dummy_y.shape, dummy_y.dtype)
            (params_t,) = vjp(hutch_cotan)
            kron_in_sample = params_t["params"]["kernel"]

            fake_bias = jnp.zeros(y.shape[-1:], dtype=y.dtype)
            assert hutch_cotan.shape[-1] == 1
            tracked_bias = self.sherman.update_and_track(kron_in_sample, fake_bias)
            y = y + hutch_cotan * tracked_bias
        if self.nonlin is not None:
            y = self.nonlin(y)
        return y


class ScannedKronify(Kronify):
    initial_count: int = 1

    @nn.compact
    def __call__(self, x):
        assert (
            x.shape[-1] == self.linear.features
        ), "we require feature size to remain unchanged"
        length = self.initial_count if self.is_initializing else None

        def body_fun(module, carry, _):
            return super(type(self), module).__call__(carry), None

        scan = nn.scan(
            body_fun,
            variable_axes={True: 0},
            variable_broadcast=False,
            split_rngs={True: True},
            length=length,
        )
        x, _ = scan(self, x, None)
        return x


def reduced_variables(mdl):
    variables = mdl.variables
    params = variables["params"]
    new_params = tree_map(lambda arr: arr[..., [0]], params)
    return variables.copy(dict(params=new_params))


def record_input_sensitivity(mdl, x, hutch):
    dummy = mdl.clone(features=1).bind(reduced_variables(mdl))
    dummy_y, vjp = nn.vjp(lambda dum: super(type(dum), dum).__call__(x), dummy)
    (vars_cotan,) = vjp(hutch)
    for key, value in vars_cotan["params"].items():
        mdl.put_variable("hutch_in", key, value)


def record_output_sensitivity(mdl, y, hutch):
    assert hutch.shape[:-1] == y.shape[:-1]
    fake_bias = jnp.zeros(shape=y.shape[-1:], dtype=y.dtype)
    for name in mdl.variables["params"].keys():
        fake_bias = mdl.perturb(name, fake_bias, collection="hutch_out")
    return y + fake_bias * hutch


def make_hutch_for(mdl, x):
    key = mdl.make_rng("hutchinson")
    hutch_shape = x.shape[:-1] + (1,)
    return jax.random.rademacher(key, shape=hutch_shape, dtype=x.dtype)


class KDense(nn.Dense):
    homogenize: bool = False
    nonlin: Optional[Callable] = None
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        if self.homogenize:
            x = homogenize_last_dim(x)

        y = super().__call__(x)

        hutch = make_hutch_for(self, y)
        record_input_sensitivity(self, x, hutch)
        y = record_output_sensitivity(self, y, hutch)

        y = y if self.nonlin is None else self.nonlin(y)
        return y


class KKronify(nn.Module):
    """Wraps a linear module, for which it tracks second order statistics.

    Optionally, appends a nonlinearity to the module and/or adds a homogeneous coordinate to
    the input. This coordinate simulates the use of a bias while introducing no new logic.
    """

    linear: nn.Module
    homogenize: bool = False
    nonlin: Optional[Callable] = None

    def reduced_linear_variables(self):
        params = self.linear.variables["params"]
        assert "bias" not in params, "bias not supported, use homogenize instead"
        params = tree_map(lambda arr: arr[..., [0]], params)
        return {"params": params}

    def update_kron_out(self):
        self.sherman.update_kron_out()

    def __call__(self, x):
        if self.homogenize:
            x = homogenize_last_dim(x)
        self.sow("intermediates", "kernel_in", x)

        y = self.linear(x)

        if self.has_rng("hutchinson"):
            dummy = self.linear.clone(features=1).bind(self.reduced_linear_variables())
            dummy_y, vjp = nn.vjp(lambda mdl: mdl(x), dummy)
            hutch_key = self.make_rng("hutchinson")
            hutch_cotan = jax.random.rademacher(hutch_key, dummy_y.shape, dummy_y.dtype)
            (params_t,) = vjp(hutch_cotan)
            kron_in_sample = params_t["params"]["kernel"]

            fake_bias = jnp.zeros(y.shape[-1:], dtype=y.dtype)
            assert hutch_cotan.shape[-1] == 1
            tracked_bias = self.sherman.update_and_track(kron_in_sample, fake_bias)
            y = y + hutch_cotan * tracked_bias
        if self.nonlin is not None:
            y = self.nonlin(y)
        return y


def _homogenized(cls, argnums=(0,)):
    old_call = cls.__call__

    @wraps(old_call)
    def new_call(self, *args, **kwargs):
        maybe_homog = (
            lambda tup: homogenize_last_dim(tup[1]) if tup[0] in argnums else tup[1]
        )
        return old_call(self, *map(maybe_homog, enumerate(args)), **kwargs)

    cls.__call__ = new_call
    return cls


def _instrumented(cls):
    old_call = cls.__call__

    def new_call(self, *args):
        pass


def value_grad_curv(fn, x):
    ones_jvp = lambda x: jax.jvp(fn, (x,), (jnp.ones_like(x),))
    (y, dy), (_, ddy) = jax.jvp(ones_jvp, (x,), (jnp.ones_like(x),))
    return y, dy, ddy


def general_hperturb(fn, transform=jnp.sign):
    @jax.custom_vjp
    def inner(x, *args, perturb):
        return fn(x, *args)

    def inner_fwd(x, *args, perturb):
        y, vjp = jax.vjp(fn, x, *args)
        ddy = jax.grad(jax.grad(fn))(x, *args)
        return vjp, ddy, perturb

    def inner_bwd(res, g):
        vjp, ddy, perturb = res
        stop_g = vjp(jax.lax.stop_gradient(g))
        stop_vjp = jax.lax.stop_gradient(vjp)(g)
        scale = transform(ddy * g)

        def combine(a, b):
            return a + scale * b - scale * jax.lax.stop_gradient(b)

        x_grad, *args_grad = map(combine, stop_vjp, stop_g)
        return x_grad + jnp.sqrt(scale * jnp.abs(g * ddy)) * perturb, *args_grad, None

    inner.defvjp(inner_fwd, inner_bwd)
    return inner


def hperturb(fn, elementwise=True, chol_rank=None):
    @jax.custom_vjp
    def inner(key, mag, x):
        return fn(x)

    def inner_fwd(key, mag, x):
        noise = jax.random.rademacher(key, x.shape, x.dtype)
        if elementwise:
            y, dy, ddy = value_grad_curv(fn, x)
            assert y.shape == x.shape
            assert dy.shape == x.shape
            assert ddy.shape == x.shape
            perturbation = noise * mag

            @Partial
            def vjp(g):
                return (dy * g,)

        else:
            EPS = 1e-4
            assert len(x.shape) == 1
            (full_rank,) = x.shape
            y, vjp = jax.vjp(fn, x)
            hess = jax.hessian(fn)(x)
            hess = hess + EPS * jnp.identity(full_rank)
            max_rank = full_rank if chol_rank is None else chol_rank
            max_rank = full_rank
            low_rank, _, _ = tfp.math.low_rank_cholesky(hess, max_rank)
            perturbation = low_rank @ (noise[..., :max_rank] * mag)
            ddy = 1.0
        res = vjp, ddy, perturbation
        return y, res

    def inner_bwd(res, g):
        vjp, ddy, perturbation = res
        perturbation = jnp.sqrt(jnp.abs(ddy * g) + 1e-12) * perturbation
        # (grad,) = vjp(g)
        (grad_stopped_g,) = vjp(jax.lax.stop_gradient(g))
        (grad_stopped_vjp,) = jax.lax.stop_gradient(vjp)(g)
        zero_unstopped_vjp = grad_stopped_g - jax.lax.stop_gradient(grad_stopped_g)
        grad_out = grad_stopped_vjp + jnp.sign(ddy * g) * zero_unstopped_vjp
        return (
            None,
            None,
            grad_out + perturbation,
        )

    inner.defvjp(inner_fwd, inner_bwd)
    return inner


class HPerturb(nn.Module):
    fn: Callable
    rng_name: str = "nonlin"

    @nn.compact
    def __call__(self, x):
        probe = self.perturb("nonlin", jnp.zeros_like(x), collection="probes")
        if self.has_variable("probes", "nonlin"):
            key = PRNGKey(0) if self.is_initializing() else self.make_rng("nonlin")
            y = hperturb(self.fn)(key, probe, x)
        else:
            y = self.fn(x)
        return y
