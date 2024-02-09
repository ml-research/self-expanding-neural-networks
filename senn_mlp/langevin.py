import numpy as np
import jax
from jax import flatten_util
import jax.numpy as jnp
from jax.tree_util import tree_map as jtm, tree_reduce as jtr
from compose import compose
from functools import partial


def tree_normal_like(tree, key):
    treevec, unflatten = jax.flatten_util.ravel_pytree(tree)
    noise = jax.random.normal(key, treevec.shape)
    return unflatten(noise)


def tree_inner(tree1, tree2):
    prods = jtm(jnp.multiply, tree1, tree2)
    sumprods = jtm(jnp.sum, prods)
    return jtr(jnp.add, sumprods)


def mala_step(lossgradfn, priorvar, old_state, key, lr, temp=1e0, legacy=False):
    def reglossgrad(state):
        loss, grad = lossgradfn(state)
        prior_loss = jtr(jnp.add, jtm(lambda s, p: jnp.sum(s**2 / p), state, priorvar))
        return prior_loss + loss, grad

    noise_key, accept_key = jax.random.split(key)
    old_loss, old_grad = reglossgrad(old_state)
    noise = tree_normal_like(old_state, noise_key)

    def _delta(prior, state, grad, noise):
        return -lr * (grad / temp + state / prior) + jnp.sqrt(2. * lr) * noise * jnp.sqrt(prior)

    delta = lambda s, g, n: jtm(_delta, priorvar, s, g, n)

    half_delta = jtm(partial(jnp.multiply, 0.5), delta(old_state, old_grad, noise))
    half_state = jtm(jnp.add, old_state, half_delta)
    half_loss, half_grad = reglossgrad(half_state)

    prop_state = jtm(jnp.add, old_state, delta(half_state, half_grad, noise))
    prop_loss, prop_grad = reglossgrad(prop_state)

    mean_grad = jtm(compose(partial(jnp.multiply, 0.5), jnp.add), old_grad, prop_grad)

    actual_improvement = old_loss - prop_loss
    expected_improvement = -tree_inner(mean_grad, delta(old_state, mean_grad, noise))
    if not legacy:
        improvement_gap = (actual_improvement - expected_improvement)/temp
    else:
        improvement_gap = actual_improvement - expected_improvement

    accept_prob = jnp.minimum(1., jnp.exp(improvement_gap))
    accept_decision = jax.random.bernoulli(accept_key, p=accept_prob, shape=())
    next_state = jtm(partial(jax.lax.select, accept_decision), prop_state, old_state)
    return next_state, accept_prob


def mala_steps(lossgradfn, priorvar, state, key, lr, steps, temp=1e0, legacy=False):
    def f(state, key):
        return mala_step(lossgradfn, priorvar, state, key, lr, temp=temp, legacy=legacy)

    keys = jax.random.split(key, steps)
    final_state, probs = jax.lax.scan(f, init=state, xs=keys)
    accept_rate = jnp.mean(probs)
    return final_state, accept_rate


def vgd_step(lossgradfn, old_state, lr):
    loss, grad = lossgradfn(old_state)
    delta = jtm(partial(jnp.multiply, -lr), grad)
    new_state = jtm(jnp.add, old_state, delta)
    return new_state
