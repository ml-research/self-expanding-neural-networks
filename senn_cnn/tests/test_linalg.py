import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from functools import partial
from compose import compose

import pytest

from senn import linalg as sennlinalg


def weighted_samples(key, dim=8, count=128):
    init = jnp.eye(dim)
    vec_key, weight_key = jax.random.split(key)
    vecs = jax.vmap(
        partial(
            jax.random.normal,
            shape=(8,),
        )
    )(jax.random.split(vec_key, count))
    weights = jax.vmap(jax.random.uniform)(jax.random.split(weight_key, count))
    return init, vecs, weights


def scan_updates(fn, init, vecs):
    """fn: M, v --> newM, init: M, vecs: [v]"""
    as_pair = lambda a: (a, a)
    _, intermediates = jax.lax.scan(compose(as_pair, fn), init, vecs)
    return intermediates


def check_rank_one_update_fn(updatefn, check_vs_direct):
    key = jax.random.PRNGKey(seed=0)
    init, vecs, weights = weighted_samples(key)

    directs = scan_updates(sennlinalg.direct_update, init, vecs)
    incrementals = scan_updates(updatefn, init, vecs)

    for i in range(len(weights)):
        check_vs_direct(incrementals[i], directs[i])


def approx_identity(direct):
    return pytest.approx(jnp.eye(direct.shape[-1]), abs=1e-5)


def check_chol(chol, direct):
    assert tfp.math.hpsd_quadratic_form_solve(direct, chol) == approx_identity(direct)


def test_chol_update():
    check_rank_one_update_fn(sennlinalg.chol_update, check_chol)


def check_inv(inv, direct):
    assert inv @ direct == approx_identity(direct)


def test_inv_update():
    check_rank_one_update_fn(sennlinalg.inv_update, check_inv)


def check_ichol(ichol, direct):
    assert ichol.T @ direct @ ichol == approx_identity(direct)


def test_ichol_update():
    check_rank_one_update_fn(sennlinalg.ichol_update, check_ichol)


def test_cholupdate():
    dim = 8
    count = 128
    I = jnp.eye(8)
    key = jax.random.PRNGKey(seed=0)
    samples = jax.vmap(partial(jax.random.normal, shape=(8,)))(
        jax.random.split(key, count)
    )
    C = I + jnp.sum(jax.vmap(lambda v: jnp.outer(v, v))(samples), axis=0)

    actual_chol = jnp.linalg.cholesky(C)

    def f(carry, x):
        return tfp.math.cholesky_update(carry, x), None

    update_chol, _ = jax.lax.scan(f, I, samples)

    assert update_chol == pytest.approx(actual_chol, rel=1e-4)

    actual_ichol = jnp.linalg.cholesky(jnp.linalg.inv(C))

    def g(carry, x):
        y = carry.T @ x
        mult = -1 / (1 + jnp.inner(y, y))
        return tfp.math.cholesky_update(carry, carry @ y, multiplier=mult), None

    update_ichol, _ = jax.lax.scan(g, I, samples)

    assert update_ichol == pytest.approx(actual_ichol, rel=1e-4)
