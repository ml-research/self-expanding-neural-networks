import wandb
import math
from functools import partial, wraps
from compose import compose
from typing import Any
import jax
from jax import numpy as jnp
from flax import struct
from jax.experimental.jet import jet
from jax.tree_util import tree_map


def jet_taylorify(fn, basepoint, order=1):
    def inner(x, *args):
        def curried(x):
            return fn(x, *args)

        tangent = tree_map(lambda b, x: x - b, basepoint, x)
        zero = tree_map(jnp.zeros_like, tangent)
        tangents = list(tangent if i == 0 else zero for i in range(order))
        y, derivs = jet(curried, (basepoint,), (tangents,))
        coeff = 1.0
        for i, deriv in zip(range(order), derivs):
            coeff = coeff / (i + 1)
            y = tree_map(lambda y, d: y + coeff * d, y, deriv)
        return y

    return inner


def taylorify(fn, basepoint, order=1):
    """expand fn about its first argument at basepoint"""

    @wraps(fn)
    def inner(x, *args, **kwargs):
        tangent = tree_map(lambda x, b: x - b, x, basepoint)

        def curried(x):
            return fn(x, *args, **kwargs), ()

        def expand(fn):
            def inner(basepoint):
                primout, tanout, tup = jax.jvp(
                    fn, (basepoint,), (tangent,), has_aux=True
                )
                tanout = tree_map(lambda t: t / (len(tup) + 1), tanout)
                return tanout, tup + (primout,)

            return inner

        expansion = curried
        for i in range(order):
            expansion = expand(expansion)
        a, b = expansion(basepoint)
        terms = b + (a,)
        return sum(terms)

    return inner


def taylor_grad(train_state, tasks, weights, key):
    @jax.grad
    def direct_grad(params):
        preds = train_state.apply_fn(dict(params=params), tasks.x)
        losses = jax.vmap(tasks.lossfn)(tasks.label, preds)
        return jnp.sum(losses * weights)

    grads = direct_grad(train_state.params)
    return grads


def taylor_step(train_state, *args):
    grads = taylor_grad(train_state, *args)
    train_state = train_state.apply_gradients(grads=grads, hgrads=grads)
