from functools import partial
from compose import compose

from flax import linen as nn
from senn import neural
import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.tree_util import tree_map
import numpy as np
import pytest
from tensorflow_probability.substrates import jax as tfp


def make_hdense(num_features=8):
    key = jax.random.PRNGKey(seed=0)
    in_shape = (num_features + 3,)
    out_shape = (num_features,)
    # model = neural.HDense(num_features)
    model = neural.Kronify(nn.Dense(num_features, use_bias=False))
    state = model.init({"params": key, "hutchinson": key}, jnp.zeros(in_shape))
    return model, state, in_shape, out_shape


def test_hdense_init(num_features=8):
    model, state, in_shape, out_shape = make_hdense()
    assert "params" in state
    assert "linear" in state["params"]
    assert "kernel" in state["params"]["linear"]
    assert state["params"]["linear"]["kernel"].shape == in_shape + out_shape


def test_hdense_sow():
    model, state, in_shape, out_shape = make_hdense()
    x = np.random.normal(size=in_shape)
    y1 = model.apply(state, x)
    y2, aux = model.apply(state, x, mutable="intermediates")

    assert y1 == pytest.approx(y2)
    assert aux["intermediates"]["kernel_in"][0] == pytest.approx(x)


def test_hdense_stats():
    model, state, in_shape, out_shape = make_hdense()
    hutch_key = PRNGKey(0)
    x = jnp.array(np.random.normal(size=in_shape))
    yct = np.random.normal(size=out_shape)
    fwd = partial(model.apply, mutable="kron", rngs={"hutchinson": hutch_key})
    y, aux = fwd(state, x)
    M = jnp.identity(in_shape[0]) + jnp.outer(x, x)
    approx = partial(pytest.approx, rel=1e-4)
    assert "sherman" in aux["kron"]
    kron_in = aux["kron"]["sherman"]["in"]
    assert kron_in.direct == approx(M)
    assert kron_in.inv == approx(jnp.linalg.inv(M))
    assert kron_in.chol == approx(jnp.linalg.cholesky(M))
    calc_ichol = compose(jnp.linalg.cholesky, jnp.linalg.inv)
    assert kron_in.ichol == approx(calc_ichol(M))


def test_hdense_grad_perturb():
    model, state, in_shape, out_shape = make_hdense()
    assert state["perturbations"]["sherman"]["out"] == pytest.approx(
        jnp.zeros(out_shape)
    )
    x = np.random.normal(size=in_shape)
    yct = np.random.normal(size=out_shape)
    fwd = partial(model.apply, mutable="kron", rngs={"hutchinson": PRNGKey(0)})
    y, VJP, aux = jax.vjp(fwd, state, x, has_aux=True)
    state_grad, x_grad = VJP(yct)
    assert state_grad["params"]["linear"]["kernel"] == pytest.approx(jnp.outer(x, yct))
    assert state_grad["perturbations"]["sherman"]["out"] == pytest.approx(yct)


def test_hdense_perturb_rank_one_update():
    model, state, in_shape, out_shape = make_hdense()
    assert state["perturbations"]["sherman"]["out"] == pytest.approx(
        jnp.zeros(out_shape)
    )
    x = np.random.normal(size=in_shape)
    yct = np.random.normal(size=out_shape)
    fwd = partial(model.apply, mutable="kron", rngs={"hutchinson": PRNGKey(0)})
    y, VJP, aux = jax.vjp(fwd, state, x, has_aux=True)
    state = state.copy(aux)
    state_grad, x_grad = VJP(yct)

    state_and_pgrad = state.copy({"perturbations": state_grad["perturbations"]})
    _, aux2 = model.apply(state_and_pgrad, method=model.update_kron_out, mutable="kron")

    M = jnp.identity(out_shape[0]) + jnp.outer(yct, yct)
    approx = partial(pytest.approx, rel=1e-4)
    kron_out = aux2["kron"]["sherman"]["out"]
    assert kron_out.direct == approx(M)
    assert kron_out.inv == approx(jnp.linalg.inv(M))
    assert kron_out.chol == approx(jnp.linalg.cholesky(M))
    calc_ichol = compose(jnp.linalg.cholesky, jnp.linalg.inv)
    assert kron_out.ichol == approx(calc_ichol(M))


def test_hperturb_transform(fn=jnp.tanh):
    approx = partial(pytest.approx, rel=1e-6, abs=1e-6)
    hfn = neural.hperturb(fn)
    key = jax.random.PRNGKey(seed=0)
    x = np.random.normal(size=(3,))
    true_y = fn(x)
    yct = np.random.normal(size=x.shape)
    true_y, true_vjp = jax.vjp(fn, x)
    (true_grad,) = true_vjp(yct)
    mags = np.random.normal(size=x.shape)

    def calc(mag):
        my_y, my_vjp = jax.vjp(partial(hfn, key, mag), x)
        return my_y, my_vjp(yct)[0]

    my_y, my_grad = calc(jnp.zeros_like(mags))
    assert my_y == approx(true_y)
    assert my_grad == approx(true_grad)

    _, (zero, my_vec) = jax.jvp(calc, (jnp.zeros_like(mags),), (mags,))
    assert zero == approx(jnp.zeros_like(zero))
    hess_diag = jax.vmap(jax.hessian(fn))(x)
    abs_hess_diag = jnp.abs(hess_diag * yct)
    assert jnp.square(my_vec) == approx(jnp.square(mags) * abs_hess_diag)


def test_hperturb_cholesky(fn=jax.scipy.special.logsumexp):
    dim = 8
    approx = partial(pytest.approx, rel=1e-1, abs=1e-2)
    hfn = neural.hperturb(fn, elementwise=False)
    keys = jax.random.split(PRNGKey(0), 1024)
    x = np.random.normal(size=dim)

    def gen_outer(key):
        def calc(mag):
            return jax.grad(hfn, argnums=2)(key, mag, x)

        _, vec = jax.jvp(calc, (0.0,), (1.0,))
        return jnp.outer(vec, vec)

    outers = jax.vmap(gen_outer)(keys)
    estimated_hess = jnp.mean(outers, axis=0)

    direct_hess = jax.hessian(fn)(x)
    assert estimated_hess == approx(direct_hess)


def make_scanned_kronify(initial_count=1, hidden_dim=3):
    linear = nn.Dense(hidden_dim, use_bias=False)
    kronify = neural.Kronify(linear, nonlin=None)
    scanned = neural.ScannedKronify(linear, nonlin=None, initial_count=initial_count)
    x = np.random.normal(size=(hidden_dim,))
    rngs = {"params": PRNGKey(0), "hutchinson": PRNGKey(0)}
    kvars = kronify.init(rngs, x)
    svars = scanned.init(rngs, x)
    return (kronify, kvars), (scanned, svars)


@pytest.mark.parametrize("initial_count", [7, 0, 1])
def test_scanned_init(initial_count):
    (kronify, kvars), (scanned, svars) = make_scanned_kronify(initial_count)

    def check(k, s):
        assert (initial_count,) + k.shape == s.shape

    tree_map(check, kvars, svars)


@pytest.mark.parametrize("initial_count", [7, 0, 1])
def test_scanned_apply(initial_count, hidden_dim=3):
    _, (scanned, svars) = make_scanned_kronify(initial_count, hidden_dim)
    x = np.random.normal(size=(hidden_dim,))
    y = scanned.apply(svars, x)
    approx = partial(pytest.approx, rel=1e-4)
    if initial_count == 0:
        assert y == approx(x)


def test_noisy_params(hidden_dim=3):
    (kronify, kvars), _ = make_scanned_kronify(1, hidden_dim)
    key1, key2 = jax.random.split(PRNGKey(0), 2)
    x = np.random.normal(size=(hidden_dim,))
    y1 = kronify.apply(kvars, x, rngs={"noisy_params": key1})
    y2 = kronify.apply(kvars, x, rngs={"noisy_params": key2})
    y1_1 = kronify.apply(kvars, x, rngs={"noisy_params": key1})
    approx = partial(pytest.approx, rel=1e-4)
    assert y1 == approx(y1_1)
    assert y1 != approx(y2)


def make_kdense(*args, x, **kwargs):
    key = PRNGKey(0)
    model = neural.KDense(*args, **kwargs)
    variables = model.init(rngs={"params": key, "hutchinson": key}, x=x)
    return model, variables


def make_vec(dim):
    return np.random.normal(size=(dim,))


def test_kdense_init():
    x = make_vec(3)
    mdl, state = make_kdense(features=3, x=x)
    assert state["params"].keys() == {"kernel"}
    assert state["hutch_out"].keys() == {"kernel"}
    y, aux = mdl.apply(state, x, mutable="hutch_in", rngs={"hutchinson": PRNGKey(0)})
    assert aux["hutch_in"].keys() == {"kernel"}
