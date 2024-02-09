from functools import partial
import functools
from compose import compose
from math import prod
from typing import Any, Optional, Callable

import jax
from jax.random import PRNGKey
from jax import numpy as jnp
import numpy as np
from jax.tree_util import (
    tree_map,
    tree_structure,
    tree_transpose,
    tree_flatten,
    tree_unflatten,
    tree_leaves,
    tree_all,
    tree_reduce,
)
from jax.flatten_util import ravel_pytree
import flax
from flax import linen as nn
from flax import struct
from flax import traverse_util
from flax.traverse_util import flatten_dict, unflatten_dict, path_aware_map
from flax.core import frozen_dict

from tensorflow_probability.substrates import jax as tfp

import optax

from senn import neural, linalg
from time import time
import wandb


def random_split_like_tree(rng_key, target=None, treedef=None, is_leaf=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target, is_leaf=is_leaf)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def nested_dict_inject(new, old):
    assert set(new) <= set(
        old
    ), f"some keys of new were not in old: {set(new) - set(old)}"
    unflat = unflatten_dict({**flatten_dict(old), **flatten_dict(new)})
    return frozen_dict.freeze(unflat)


def nested_dict_subset(pred, tree):
    flat = flatten_dict(tree)
    unflat = unflatten_dict({k: v for k, v in flat.items() if pred(k)})
    return frozen_dict.freeze(unflat)


class TrainState(struct.PyTreeNode):
    tx: optax.GradientTransformationExtraArgs = struct.field(pytree_node=False)
    step: int
    params: Any
    probes: Any
    opt_state: Any
    apply_fn: Callable = struct.field(pytree_node=False)
    batch_stats: Any
    # add_width_fn: Callable = struct.field(pytree_node=False)
    model: nn.Module = struct.field(pytree_node=False)
    # traversal: Any = struct.field(pytree_node=False)
    path_pred: Callable = struct.field(pytree_node=False)
    dummy_input: Any = None

    def subset(self, nested):
        return nested_dict_subset(self.path_pred, nested)

    @property
    def subparams(self):
        return self.subset(self.params)

    @classmethod
    def create(
        cls,
        tx,
        params,
        probes,
        apply_fn,
        path_pred=lambda p: True,
        batch_stats={},
        **kwargs,
    ):
        opt_state = tx.init(nested_dict_subset(path_pred, params))
        return cls(
            tx,
            0,
            params,
            probes,
            opt_state,
            apply_fn,
            batch_stats,
            path_pred=path_pred,
            **kwargs,
        )

    @jax.jit
    def apply_gradients(self, *, grads, hgrads, **kwargs):
        subgrads, subhgrads = map(self.subset, (grads, hgrads))
        updates, new_opt_state = self.tx.update(
            subgrads, self.opt_state, params=self.subparams, hgrads=subhgrads
        )
        new_params = optax.apply_updates(self.subparams, updates)
        new_params = nested_dict_inject(new_params, self.params)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def sample_posterior(self, key, params=None, scale_tangent=None):
        if wandb.config.fast_turbo:
            return self.params
        params = self.params if params is None else params
        tangent = self.sample_tangent(key)
        if scale_tangent is not None:
            tangent = tree_map(lambda arr: arr * scale_tangent, tangent)
        sample = tree_map(jnp.add, params, tangent)
        return sample
        # return self.tx.sample_posterior(params, self.opt_state, key)

    def sample_tangent(self, key):
        tan = self.tx.sample_tangent(self.subparams, self.opt_state, key=key)
        return nested_dict_inject(tan, tree_map(jnp.zeros_like, self.params))

    @jax.jit
    def eval(self, task, key=None):
        if key is not None:
            key, dropout_key = jax.random.split(key)
        params = self.params if key is None else self.sample_posterior(key)
        # variables = {"params": params, "probes": self.probes}
        variables = {
            "params": params,
            "batch_stats": self.batch_stats,
        }  # this should disable HPerturb modules
        rngs = {}
        if key is not None and wandb.config.use_dropout:
            rngs = dict(**rngs, dropout=dropout_key)
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P, Mesh
        @partial(
            shard_map,
            mesh=Mesh(jax.devices(), axis_names=("gpus",)),
            in_specs=P("gpus", None, None, None),
            out_specs=P("gpus"), check_rep=False)
        def fwd(x):
            return self.apply_fn(variables, x, mutable=False, rngs=rngs)
        
        #y = self.apply_fn(variables, task.x, mutable=False, rngs=rngs)
        y = fwd(task.x)
        return y

    @partial(jax.jit, static_argnums=3)
    def eval_marginalized(self, task, key, num_samples=8):
        ys = jax.vmap(partial(self.eval, task))(jax.random.split(key, num_samples))
        ys = jax.nn.log_softmax(ys)
        y = tfp.math.reduce_logmeanexp(ys, axis=0)
        return y

    def get_metrics(self):
        return self.tx.get_metrics(self.subparams, self.opt_state)

    def pin_prior(self):
        new_opt_state = self.tx.pin_prior(self.subparams, self.opt_state)
        return self.replace(opt_state=new_opt_state)

    def maybe_reinit(self, key):
        can_key, apply_key = jax.random.split(key)
        can_reinit = self.tx.reinit_allowed(self.subparams, self.opt_state, key=can_key)
        _, can_reinit = self.model.apply(
            dict(params=self.params, allowed=can_reinit),
            method=self.model.bud_reinit_allowed,
            mutable="allowed",
        )
        can_reinit = can_reinit.get("allowed")
        variables = self.model.init(apply_key, self.dummy_input)
        # _, variables = self.apply_fn(
        #    dict(), self.dummy_input, rngs=dict(params=apply_key), mutable="params"
        # )
        fresh_params = variables["params"]
        new_params = tree_map(
            lambda new, old, can: jnp.where(can, new, old),
            fresh_params,
            self.params,
            can_reinit,
        )
        new_opt_state = self.tx.process_reinit(
            self.subset(new_params), self.opt_state, self.subset(can_reinit)
        )
        return self.replace(params=new_params, opt_state=new_opt_state)

    def maybe_expand_width(self, key, builder, add_width=None):
        add_width = (
            self.tx.should_add_width(self.subparams, self.opt_state, key=key)
            if add_width is None
            else add_width
        )
        # no_width_added = tree_all(tree_map(lambda arr: arr == 0, add_width))
        # if no_width_added:
        #    return self
        was_padded = tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.bool_), self.params)
        variables = dict(
            params=self.params,
            probes=self.probes,
            add_width=add_width,
            was_padded=was_padded,
        )
        # widthss, variables = self.add_width_fn(variables, rngs=dict(params=key))
        widthss, variables = self.model.apply(
            variables,
            method=self.model.maybe_add_width,
            mutable=True,
            rngs=dict(params=key),
        )
        if widthss is None:  # widthss is None means no expansion occurred
            # return immediately to avoid overwriting functions and causing recompilation
            return self
        new_opt_state = self.tx.process_add_width(
            self.subparams, self.opt_state, self.subset(variables["was_padded"])
        )
        new_model = builder(widthss)
        print(new_model.tabulate(PRNGKey(0), self.dummy_input))
        new_apply_fn = new_model.apply
        new_add_width_fn = partial(
            new_model.apply, method=new_model.maybe_add_width, mutable=True
        )
        return self.replace(
            params=variables["params"],
            probes=variables["probes"],
            opt_state=new_opt_state,
            apply_fn=new_apply_fn,
            # add_width_fn = new_add_width_fn,
            model=new_model,
        )

    def init_prune(self):
        subparams = self.tx.init_prune_params(self.subparams)
        params = nested_dict_inject(subparams, self.params)
        opt_state = self.tx.init_prune_opt_state(subparams, self.opt_state)
        return self.replace(params=params, opt_state=opt_state)

    def insert_layer(self, builder, bidx, lidx, key=PRNGKey(0)):
        # choose bidx and lidx for where to insert
        was_padded = tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.bool_), self.params)
        variables = dict(params=self.params, probes=self.probes, was_padded=was_padded)
        (new_widthss, new_layer), variables = self.model.apply(
            variables,
            bidx,
            lidx,
            method=self.model.activate_bud,
            rngs=dict(params=key),
            mutable=True,
        )
        # adjust opt_state for new padding
        new_opt_state = self.tx.process_add_width(
            self.subparams, self.opt_state, self.subset(variables["was_padded"])
        )
        # insert new_layer variables at appropriate location
        params, probes = variables["params"], variables["probes"]
        new_layer_params, new_layer_probes = new_layer["params"], new_layer["probes"]
        new_params = self.model.insert_into_tree(params, bidx, lidx, new_layer_params)
        new_probes = self.model.insert_into_tree(probes, bidx, lidx, new_layer_probes)
        # create new opt_state for layer
        new_layer_opt_state = self.tx.init(self.subset(new_layer_params))
        # insert new opt_state at appropriate location
        new_opt_state = self.model.insert_into_tree(
            new_opt_state, bidx, lidx, new_layer_opt_state
        )
        # construct new model for widthss
        new_model = builder(new_widthss)
        # write new train_state with updated contents
        return self.replace(
            params=new_params,
            probes=new_probes,
            opt_state=new_opt_state,
            apply_fn=new_model.apply,
            # add_width_fn = partial(new_model.apply, method=new_model.maybe_add_width, mutable=True),
            model=new_model,
        )

    def maybe_insert_layer(self, builder, key=PRNGKey(0)):
        depth_scores = self.tx.get_depth_score(self.subparams, self.opt_state)
        score, (bidx, lidx) = self.model.apply(
            dict(score=depth_scores), method=self.model.argmax_score
        )
        current_score = self.tx.global_score(self.subparams, self.opt_state)
        if (score > wandb.config.depth_score_abs_thresh) and \
            (score > current_score*wandb.config.depth_score_rel_thresh):
            print(f"Inserting layer at block {bidx}, layer {lidx}...")
            new_self = self.insert_layer(builder, bidx, lidx, key)
            print("Completed layer insertion.")
            return new_self
        else:
            print(f"Declined best layer insertion score at ({bidx}, {lidx}) of {score} (rel {score / current_score})")
            return self

    def tx_reinit_changed_shapes(self):
        opt_state = self.tx.reinit_changed_shapes(self.subparams, self.opt_state)
        return self.replace(opt_state=opt_state)


class HessTracker(struct.PyTreeNode):
    def init(self, params):
        raise NotImplementedError

    def solve(self, grad, key):
        raise NotImplementedError

    def rescale(self, factor, soln=None):
        raise NotImplementedError

    def rank_one_update(self, hgrad, key, soln=None):
        raise NotImplementedError

    def iroot_mul(self, tangent, key):
        raise NotImplementedError


class DiagTracker(HessTracker):
    hdiag: Any = None
    eps: float = 1e-12
    initial_precision: float = 1.0

    def init(self, params):
        hdiag = jnp.full_like(params, fill_value=self.initial_precision)
        return self.replace(hdiag=hdiag)

    def _inv_diag(self):
        return jnp.reciprocal(self.hdiag + self.eps)

    def solve(self, grad, key):
        return self._inv_diag() * grad

    def rescale(self, factor, soln=None):
        assert soln is None
        delta_logdet = jnp.sum(jnp.log(factor) * jnp.ones_like(self.hdiag))
        new_hdiag = self.hdiag * factor
        return self.replace(hdiag=new_hdiag), delta_logdet

    def rank_one_update(self, hgrad, key, soln=None):
        assert soln is None
        delta_logdet = jnp.log1p(hgrad * self._inv_diag() * hgrad)
        new_hdiag = self.hdiag + jnp.square(hgrad)
        return self.replace(hdiag=new_hdiag), delta_logdet

    def iroot_mul(self, tangent, key):
        return tangent * jnp.sqrt(self._inv_diag())


class ICholTracker(HessTracker):
    ichol: Any = None
    eps: float = 1e-12

    def init(self, params, initial_precision=1.0):
        ichol = jnp.reciprocal(initial_precision) * jnp.identity(params.size)
        return self.replace(ichol=ichol)

    def solve(self, grad):
        return self.ichol @ (self.ichol.T @ grad)

    def rescale(self, factor, soln=None):
        assert soln is None
        delta_logdet = jnp.sum(jnp.log(factor) * jnp.ones_like(jnp.diag(self.ichol)))
        new_ichol = self.ichol * jnp.reciprocal(jnp.sqrt(factor))
        return self.replace(ichol=new_ichol)

    def rank_one_update(self, hgrad, soln=None):
        assert soln is None
        whitened = self.ichol.T @ hgrad
        white_mag = jnp.inner(whitened, whitened.conj())
        delta_logdet = jnp.log1p(white_mag)
        multiplier = -jnp.exp(-delta_logdet)
        inv_update_vec = self.ichol @ whitened
        new_ichol = tfp.math.cholesky_update(
            self.ichol, inv_update_vec, multiplier=multiplier
        )
        return self.replace(ichol=new_ichol)

    def iroot_mul(self, tangent):
        return self.ichol @ tangent

    def whiten(self, tangent):
        return self.ichol.T @ tangent

    def ldet(self):
        return -jnp.sum(jnp.log(jnp.diag(self.ichol)))

    def trace_inv(self):
        return jnp.sum(jnp.square(jnp.abs(self.ichol)))


class ICholDummy(ICholTracker):
    """This is actually a kronecker factored diagonal approximation.
    It is useful because it does not use a cholesky update and so can
    be used to test performance impact of tfp.math.cholesky_update"""

    def rank_one_update(self, hgrad, soln=None):
        whitened = self.ichol.T @ hgrad
        white_mag = jnp.square(whitened)
        delta_logdet = jnp.sum(jnp.log1p(white_mag))
        new_ichol = self.ichol * jnp.reciprocal(1.0 + white_mag)
        return self.replace(ichol=new_ichol)

    def rank_n_update(self, hgrads, soln=None):
        whitened = hgrads @ self.ichol
        white_mag = jnp.sum(jnp.square(jnp.abs(whitened)), axis=0)
        delta_logdet = jnp.sum(jnp.log1p(white_mag))
        new_ichol = self.ichol * jnp.sqrt(jnp.reciprocal(1.0 + white_mag))
        return self.replace(ichol=new_ichol)


class IRootTracker(ICholTracker):
    def rank_one_update(self, hgrad, soln=None):
        assert soln is None
        whitened = self.ichol.T @ hgrad
        white_mag = jnp.inner(whitened, whitened.conj())
        delta_logdet = jnp.log1p(white_mag)
        multiplier = -jnp.exp(-delta_logdet)
        inv_update_vec = self.ichol @ whitened
        asymm = jnp.outer(inv_update_vec, whitened)
        new_ichol = self.ichol + 0.25 * multiplier * (asymm + asymm.T)
        return self.replace(ichol=new_ichol)

    def rank_n_update(self, hgrads, soln=None):
        assert soln is None
        whitened = hgrads @ self.ichol
        white_mag = jnp.sum(jnp.square(jnp.abs(whitened)))
        delta_logdet = jnp.log1p(white_mag)

        # multiplier = -0.5*jnp.exp(-delta_logdet)
        def multiplier_old(mag):
            return -0.5 * jnp.exp(-jnp.log1p(mag))

        def multiplier(mag):
            x = -mag * jnp.reciprocal(1.0 + mag)
            x = jnp.expm1(0.5 * jnp.log1p(x))
            return x * jnp.reciprocal(self.eps + mag)

        asymm = self.ichol @ whitened.T @ whitened
        # new_ichol = self.ichol + 0.5*multiplier*(asymm+asymm.T)
        new_ichol = self.ichol + multiplier(white_mag) * asymm
        return self.replace(ichol=new_ichol)


class KronTracker(HessTracker):
    h_in: HessTracker = IRootTracker()
    h_out: HessTracker = IRootTracker()
    shape: Any = None
    initial_precision: float = 1.0
    prune_on_init: float = 0.0

    def renorm(self):
        raise NotImplementedError

    def in_map(self, fn):
        return jax.vmap(fn, in_axes=-1, out_axes=-1)

    def out_map(self, fn):
        return jax.vmap(fn, in_axes=-2, out_axes=-2)

    def init(self, params):
        in_sample = params[:, 0]
        out_sample = params[0, :]
        h_in = self.h_in.init(
            in_sample, initial_precision=jnp.sqrt(self.initial_precision)
        )
        # init with pruned weights
        to_keep = int(len(in_sample) * (1.0 - self.prune_on_init))
        h_in = h_in.replace(
            iroot=h_in.iroot.at[to_keep:, :].set(0.0).at[:, to_keep:].set(0.0)
        )
        h_in = h_in.replace(mask=jnp.arange(len(in_sample)) < to_keep)
        # h_out = self.h_out.replace(initial_precision=len(in_sample)).init(out_sample)
        h_out = self.h_out.init(
            out_sample,
            initial_precision=len(in_sample) * jnp.sqrt(self.initial_precision),
        )
        return self.replace(h_in=h_in, h_out=h_out, shape=params.shape)

    def solve(self, grad, key):
        soln = self.in_map(self.h_in.solve)(grad)
        soln = self.out_map(self.h_out.solve)(soln)
        return soln

    def rescale(self, factor, soln=None):
        assert soln is None
        in_dim, out_dim = self.shape
        # extra factor to halt determinant drift due to redundancy in parameterization
        # extra = jnp.exp(0.5 * (self.h_in.ldet() / in_dim - self.h_out.ldet() / out_dim))
        extra = jnp.sqrt((self.h_out.trace_inv()) / (self.h_in.trace_inv()))
        # factor = jnp.sqrt(factor)
        h_in = self.h_in.rescale(factor / extra)
        h_out = self.h_out.rescale(factor * extra)
        # delta_logdet = delta_in * out_dim + delta_out * in_dim
        return self.replace(h_in=h_in, h_out=h_out), 0.0

    def rank_one_update(self, hgrad, key, soln=None):
        NUM_SAMPLES = None
        RANK_N_UPDATE = True
        in_key, out_key = jax.random.split(key)
        assert soln is None

        def scan_update(hess, vec):
            hess, delta_logdet = hess.rank_one_update(vec)
            return hess, delta_logdet

        def update(hess, vecs, key):
            if NUM_SAMPLES is not None:
                samples = jax.random.normal(
                    key, (NUM_SAMPLES, vecs.shape[0]), vecs.dtype
                )
                vecs = samples @ vecs / jnp.sqrt(NUM_SAMPLES)
            if RANK_N_UPDATE:
                return hess.rank_n_update(vecs)
            else:
                return jax.lax.scan(scan_update, hess, vecs)

        in_dim, out_dim = self.shape
        in_dim = jnp.sum(self.h_in.mask)
        white_out = self.out_map(self.h_out.whiten)(hgrad)
        h_in = update(self.h_in, white_out.T / jnp.sqrt(out_dim), in_key)
        white_in = self.in_map(self.h_in.whiten)(hgrad)
        h_out = update(self.h_out, white_in / jnp.sqrt(in_dim), out_key)
        # delta_logdet = jnp.sum(deltas_in * out_dim) + jnp.sum(deltas_out * in_dim)
        return self.replace(h_in=h_in, h_out=h_out), 0.0

    def whiten_out(self, hgrad):
        white_out = self.out_map(self.h_out.whiten)(hgrad)
        return white_out

    def whiten_in(self, hgrad):
        return self.in_map(self.h_in.whiten)(hgrad)

    def whiten(self, hgrad):
        return self.whiten_out(self.whiten_in(hgrad))

    def iroot_mul(self, tangent, key):
        tangent = self.in_map(partial(self.h_in.iroot_mul))(tangent)
        tangent = self.out_map(partial(self.h_out.iroot_mul))(tangent)
        return tangent

    def gmres_solve(self, tangent):
        tangent = self.h_out.gmres_solve(tangent)
        tangent = self.h_in.gmres_solve(tangent.T).T
        return tangent


class GaussianPrior(struct.PyTreeNode):
    precision: float = 1.0

    def __call__(self, params, key):
        noise = jax.random.normal(key, params.shape, params.dtype)
        hgrad = jnp.sqrt(self.precision) * noise
        grad = self.precision * params
        return grad, hgrad


class FanInPrior(GaussianPrior):
    precision: float = 1.0

    def __call__(self, params, key):
        fan_in = params.shape[0]
        noise = jax.random.normal(key, params.shape, params.dtype)
        hgrad = jnp.sqrt(self.precision * fan_in) * noise
        grad = self.precision * fan_in * params
        return grad, hgrad


class KronPrior(struct.PyTreeNode):
    root_in: Any
    root_out: Any
    center: Any

    @classmethod
    def init(cls, params, precision=1.0):
        scale = jnp.sqrt(jnp.sqrt(precision))
        size_in, size_out = params.shape
        root_in = jnp.identity(size_in) * scale
        root_out = jnp.identity(size_out) * scale
        center = jnp.zeros_like(params)
        return cls(root_in=root_in, root_out=root_out, center=center)

    def __call__(self, params, key):
        noise = jax.random.normal(key, params.shape, params.dtype)
        hgrad = self.root_in @ noise @ self.root_out.T
        grad = (
            self.root_in
            @ self.root_in.T
            @ (params - self.center)
            @ self.root_out
            @ self.root_out.T
        )
        return grad, hgrad

    def distance(self, params):
        transformed = self.root_in.T @ (params - self.center) @ self.root_out
        return 0.5 * jnp.mean(jnp.square(jnp.abs(transformed)))


class EMA(struct.PyTreeNode):
    mu: Any

    @classmethod
    def init_zero(cls, shape=()):
        return cls(mu=jnp.zeros(shape=shape))

    def update(self, obs, rate):
        new_mu = self.mu + rate * obs - rate * self.mu
        return self.replace(mu=new_mu)


class EMVar(EMA):
    sq: EMA

    @classmethod
    def init_zero(cls, shape=()):
        mu = jnp.zeros(shape=shape)
        sq = jnp.ones(shape=shape)
        return cls(mu=mu, sq=EMA(sq))

    def update(self, obs, rate):
        out = super().update(obs, rate)
        sqobs = jnp.square(obs)
        out = out.replace(sq=out.sq.update(sqobs, rate))
        return out


class EMT(struct.PyTreeNode):
    mu: Any
    mass: Any

    @classmethod
    def init_with_obs(cls, obs):
        return cls(mu=obs, mass=1.0)

    def update(self, obs, rate):
        new_mu = self.mu + obs - rate * self.mu
        new_mass = self.mass + 1.0 - rate * self.mass
        return self.replace(mu=new_mu, mass=new_mass)

    def mean(self):
        return self.mu / self.mass

    def set_zero(self, where):
        new_mu = jnp.where(where, self.mu, jnp.zeros_like(self.mu))
        return self.replace(mu=new_mu)

    def process_add_width(self, was_padded, fill=0.0):
        fresh = jnp.full(was_padded.shape, fill_value=fill, dtype=self.mu.dtype)
        new_mu = recover_from_padding(was_padded, self.mu, fresh)
        return self.replace(mu=new_mu)


class EMT_N(struct.PyTreeNode):
    mus: Any
    mass: Any

    @classmethod
    def init_with_obs(cls, obs, N):
        mus = jnp.array((obs,) * (N + 1))
        return cls(mus=mus, mass=1.0)

    def update(self, obs, rate):
        N = len(self.mus)
        leq = jnp.arange(N)[:, None] <= jnp.arange(N)
        L = jnp.where(leq, -rate, 0.0)
        delta_mus = (self.mus.T @ L.T).T + obs
        new_mus = self.mus + delta_mus
        new_mass = self.mass + 1.0 - rate * self.mass
        return self.replace(mus=new_mus, mass=new_mass)

    def mean(self):
        return jnp.sum(self.mus, axis=0) / self.mass

    def process_add_width(self, was_padded, fill=0.0):
        was_padded = was_padded[None, ...]
        new_shape = was_padded.shape
        fresh = jnp.full(new_shape, fill_value=fill, dtype=self.mus.dtype)
        new_mus = recover_from_padding(was_padded, self.mus, fresh)
        return self.replace(mus=new_mus)


def tree_method(fn, is_leaf=None, outputs=1):
    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        example_tree = args[0] if len(args) > 0 else kwargs[next(kwargs.keys())]
        treedef = tree_structure(example_tree, is_leaf=is_leaf)
        if kwargs:
            if "key" in kwargs:
                kwargs = {
                    **kwargs,
                    "key": random_split_like_tree(kwargs["key"], treedef=treedef),
                }
            outerdef = tree_structure(
                kwargs, is_leaf=lambda t: tree_structure(t, is_leaf=is_leaf) == treedef
            )
            kwargs = tree_transpose(outerdef, treedef, kwargs)

            def call(dummy, kwargs, *args):
                return fn(self, *args, **kwargs)

            out = tree_map(call, example_tree, kwargs, *args)
        else:

            def call(dummy, *args):
                return fn(self, *args)

            out = tree_map(call, example_tree, *args)
        if outputs > 1:
            out = tuple(
                tree_map(lambda _, tup: tup[i], example_tree, out)
                for i in range(outputs)
            )
        return out

    return inner


def ravelled_method(fn, keep_axis=-2, unravel_out=False, ignore_args=()):
    if isinstance(ignore_args, int):
        ignore_args = (ignore_args,)

    def _ravel(arr):
        return jax.vmap(jnp.ravel, in_axes=keep_axis, out_axes=keep_axis)(arr)

    def _unravel(arr, shape):
        dummy = jnp.zeros(shape=shape)
        return jax.vmap(
            lambda a, d: a.reshape(d.shape), in_axes=keep_axis, out_axes=keep_axis
        )(arr, dummy)

    @functools.wraps(fn)
    def inner(self, *args, **kwargs):
        shapes = tuple(arg.shape for i, arg in enumerate(args) if i not in ignore_args)
        args = tuple(
            _ravel(arg) if i not in ignore_args else arg for i, arg in enumerate(args)
        )
        # kwshapes = {key: value.shape for key, value in kwargs.items()}
        kwargs = {key: _ravel(value) for key, value in kwargs.items()}
        out = fn(self, *args, **kwargs)
        if unravel_out:
            shape = shapes[0]
            out = _unravel(out, shape)
        return out

    return inner


ravelled_method_unravel = partial(ravelled_method, unravel_out=True)


class SimpleOpt(struct.PyTreeNode):
    rate: float = 1e-3

    class State(struct.PyTreeNode):
        grad: EMT
        hess: EMT

    @tree_method
    def init(self, params):
        grad = EMT.init_with_obs(jnp.zeros_like(params))
        hess = EMT.init_with_obs(jnp.ones_like(params))
        return self.State(grad=grad, hess=hess)

    @partial(tree_method, outputs=2)
    def update(self, grads, state, params=None, *, hgrads, **kwargs):
        del kwargs
        state = state.replace(grad=state.grad.update(grads, 1e-1))
        hess_obs = jnp.square(jnp.abs(1e-1 * grads)) + jnp.square(jnp.abs(hgrads))
        state = state.replace(hess=state.hess.update(hess_obs, 1e-2))
        updates = -self.rate * state.grad.mean() / jnp.sqrt(state.hess.mean() + 1e-12)
        return updates, state
        curv_scale = jnp.maximum(
            1e0, jnp.mean(jnp.square(jnp.abs(hgrads)) + jnp.square(jnp.abs(grads)))
        )
        return -1e-3 * grads / curv_scale, state

    def _sample_tangent(self, params, state, *, key):
        noise = jax.random.normal(key=key, shape=params.shape, dtype=params.dtype)
        hess = state.hess.mean()
        return noise / jnp.sqrt(hess + 1e-12)

    @tree_method
    def sample_posterior(self, params, state, *, key):
        return params + self._sample_tangent(params, state, key=key)

    @tree_method
    def sample_tangent(self, params, state, *, key):
        return self._sample_tangent(params, state, key=key)

    @tree_method
    def get_metrics(self, params, state):
        return dict()

    @tree_method
    def pin_prior(self, params, state):
        return state


class ScalarCurv(struct.PyTreeNode):
    rate: float = 1e-3
    eps: float = 1e-6

    def init(self, params):
        return EMT.init_with_obs(1.0)

    def update(self, grads, state, params, *, hgrads):
        return state.update(jnp.mean(jnp.square(jnp.abs(hgrads))), self.rate)

    def update_n(self, grads, state, params, *, hgrads):
        return self.update(grads, state, params, hgrads=hgrads)

    def iroot_mul(self, state, x):
        return x / jnp.sqrt(state.mean() + self.eps)

    def root_mul(self, state, x):
        return x * jnp.sqrt(state.mean() + self.eps)

    def whiten(self, state, x):
        return x / jnp.sqrt(state.mean() + self.eps)

    def solve(self, state, x):
        return x / (state.mean() + self.eps)


class SimpleCurv(struct.PyTreeNode):
    rate: float = 1e-3
    eps: float = 1e0

    def init(self, params):
        return EMT.init_with_obs(jnp.ones_like(params))

    def update(self, grads, state, params, *, hgrads):
        return state.update(jnp.square(jnp.abs(hgrads)), self.rate)

    def update_n(self, grads, state, params, *, hgrads):
        return state.update(jnp.sum(jnp.square(jnp.abs(hgrads)), axis=0), self.rate)

    def iroot_mul(self, state, x):
        return x / jnp.sqrt(state.mean() + self.eps)

    def i4root_mul(self, state, x):
        return x / jnp.sqrt(jnp.sqrt(state.mean() + self.eps))

    def whiten(self, state, x):
        return x / jnp.sqrt(state.mean() + self.eps)

    def solve(self, state, x):
        return x / (state.mean() + self.eps)

    def root_mul(self, state, x):
        return x * jnp.sqrt(state.mean() + self.eps)

    def process_add_width(self, was_padded, state):
        return state.process_add_width(was_padded, fill=1.0)


class IRootCurv(struct.PyTreeNode):
    rate: float = 1e-3

    def init(self, params):
        return linalg.MaskedWhitener.init_identity(params.shape[-1], diag_fraction=1e-1)

    def update(self, grads, state, params, *, hgrads):
        return self.update_n(None, state, None, hgrads=hgrads[None, :])

    def update_n(self, grads, state, params, *, hgrads):
        state = state.rescale(1.0 - self.rate)
        vecs = jnp.sqrt(self.rate) * hgrads
        state = state.rank_n_update(vecs)
        return state

    def iroot_mul(self, state, x):
        return state.iroot_mul(x)

    def whiten(self, state, x):
        return state.whiten(x)

    def solve(self, state, x):
        return state.solve(x)

    def root_mul(self, state, x):
        x = state.direct_mul(x)
        return self.whiten(state, x)

    def process_add_width(self, was_padded, state):
        (dim,) = was_padded.shape
        fresh = jnp.identity(dim)
        fresh_mask = jnp.zeros_like(was_padded, dtype=jnp.bool_)
        square_padded = was_padded[:, None] | was_padded[None, :]
        iroot = recover_from_padding(square_padded, state.iroot, fresh)
        direct = recover_from_padding(square_padded, state.direct, fresh)
        mask = recover_from_padding(was_padded, state.mask, fresh_mask)
        return state.replace(iroot=iroot, direct=direct, mask=mask)


class FactoredCurv(struct.PyTreeNode):
    out_curv: SimpleCurv = SimpleCurv()
    in_curv: SimpleCurv = SimpleCurv()
    diag_curv: SimpleCurv = SimpleCurv()
    scalar_curv: ScalarCurv = ScalarCurv()

    @ravelled_method
    def init(self, params):
        return (
            self.in_curv.init(params[:, 0]),
            self.out_curv.init(params[0, :]),
            self.diag_curv.init(params),
            self.scalar_curv.init(params),
        )

    @partial(ravelled_method, ignore_args=(1, 3))
    def update(self, grads, state, params, key, *, hgrads):
        def norm(arr):
            return jnp.sqrt(jnp.sum(jnp.square(jnp.abs(arr))))

        in_key, out_key = jax.random.split(key)
        in_dim, out_dim = grads.shape
        in_probe = jax.random.normal(
            in_key,
            (
                1,
                out_dim,
            ),
        ) / jnp.sqrt(out_dim)
        in_probe = in_probe / norm(in_probe)
        out_probe = jax.random.normal(
            out_key,
            (
                1,
                in_dim,
            ),
        ) / jnp.sqrt(in_dim)
        out_probe = out_probe / norm(out_probe)
        in_curv, out_curv, diag_curv, scalar_curv = state

        # new_diag = self.diag_curv.update(None, diag_curv, None, hgrads=hgrads)
        def maybe_print(name, value):
            should_print = ~jnp.isfinite(value).all()
            jax.lax.cond(
                should_print,
                lambda v: jax.debug.print(name + ": {}", value),
                lambda v: None,
                value,
            )

        def suppress_nans(arr):
            return jnp.where(jnp.isfinite(arr), arr, 0.0)

        # hgrads = suppress_nans(hgrads)
        maybe_print("hgrads0", hgrads)
        new_scalar = self.scalar_curv.update(None, scalar_curv, None, hgrads=hgrads)
        # hgrads = self.diag_curv.whiten(new_diag, hgrads)
        hgrads = self.scalar_curv.whiten(new_scalar, hgrads)
        # hgrads = suppress_nans(hgrads)
        # jax.debug.print("hgrads1: {}", hgrads)
        maybe_print("hgrads1", hgrads)
        # new_out = self.out_curv.update_n(None, out_curv, None, hgrads=hgrads)
        new_out = self.out_curv.update_n(
            None, out_curv, None, hgrads=out_probe @ hgrads
        )
        hgrads = self.out_curv.whiten(new_out, hgrads)
        # hgrads = suppress_nans(hgrads)
        # jax.debug.print("hgrads2: {}", hgrads)
        maybe_print("hgrads2", hgrads)
        new_in = self.in_curv.update_n(
            grads.T, in_curv, params.T, hgrads=in_probe @ hgrads.T
        )
        # new_in = self.in_curv.update_n(grads.T, in_curv, params.T, hgrads=hgrads.T)
        hgrads = self.in_curv.whiten(new_in, hgrads.T).T
        # hgrads = suppress_nans(hgrads)
        # jax.debug.print("hgrads3: {}", hgrads)
        maybe_print("hgrads3", hgrads)
        new_diag = self.diag_curv.update(None, diag_curv, None, hgrads=hgrads)
        return new_in, new_out, new_diag, new_scalar

    @partial(ravelled_method_unravel, ignore_args=0)
    def iroot_mul(self, state, x):
        in_curv, out_curv, diag_curv, scalar_curv = state
        x = self.diag_curv.iroot_mul(diag_curv, x)
        x = self.in_curv.iroot_mul(in_curv, x.T).T
        x = self.out_curv.iroot_mul(out_curv, x)
        x = self.scalar_curv.iroot_mul(scalar_curv, x)
        return x

    @partial(ravelled_method_unravel, ignore_args=0)
    def whiten(self, state, x):
        in_curv, out_curv, diag_curv, scalar_curv = state
        x = self.scalar_curv.whiten(scalar_curv, x)
        x = self.out_curv.whiten(out_curv, x)
        x = self.in_curv.whiten(in_curv, x.T).T
        x = self.diag_curv.whiten(diag_curv, x)
        return x

    @partial(ravelled_method_unravel, ignore_args=0)
    def solve(self, state, x):
        x = self.whiten(state, x)
        x = self.iroot_mul(state, x)
        return x

    @partial(ravelled_method, ignore_args=0)
    def current_score(self, state, grads):
        in_curv, out_curv, diag_curv, scalar_curv = state
        grads = self.scalar_curv.whiten(scalar_curv, grads)
        grads = self.out_curv.whiten(out_curv, grads)
        ngrad = in_curv.cg_solve(grads.T)
        mag = jnp.abs(jnp.sum(grads.T * ngrad))
        return mag

    @partial(ravelled_method, ignore_args=0)
    def freeze_prune_thaw_scores(self, state, grads, params):
        in_curv, out_curv, diag_curv, scalar_curv = state
        grads = self.scalar_curv.whiten(scalar_curv, grads)
        grads = self.out_curv.whiten(out_curv, grads)
        ngrad = in_curv.cg_solve(grads.T)
        params = self.scalar_curv.root_mul(scalar_curv, params)
        params = self.out_curv.root_mul(out_curv, params)
        # if wandb.config.freeze_is_prune:
        #    params = self.scalar_curv.root_mul(scalar_curv, params)
        #    ngrad = ngrad + params.T
        #    grads = grads + in_curv.direct @ params
        mag = jnp.abs(jnp.sum(grads.T * ngrad))
        return mag, in_curv.freeze_prune_thaw_scores(grads.T, params.T, ngrad=ngrad)

    @partial(ravelled_method_unravel, ignore_args=(0, 3))
    def prune(self, state, grads, params, where):
        in_curv, out_curv, diag_curv, scalar_curv = state
        param_diff = jnp.where(where[:, None], -params, jnp.zeros_like(params))
        extra_diff = in_curv.cg_project(-param_diff.T).T
        params = jnp.where(where[:, None], jnp.zeros_like(params), params)
        return params + extra_diff

    @partial(ravelled_method, ignore_args=1)
    def process_add_width(self, was_padded, state):
        in_curv, out_curv, diag_curv, scalar_curv = state
        in_padded = jax.vmap(jnp.all, in_axes=-2)(was_padded)
        out_padded = jax.vmap(jnp.all, in_axes=-1)(was_padded)
        in_curv = self.in_curv.process_add_width(in_padded, in_curv)
        out_curv = self.out_curv.process_add_width(out_padded, out_curv)
        diag_curv = self.diag_curv.process_add_width(was_padded, diag_curv)
        # print(f"scalar_curv state: {scalar_curv}")
        scalar_curv = scalar_curv
        return in_curv, out_curv, diag_curv, scalar_curv


class SlowGrad(struct.PyTreeNode):
    grad_rate: float = 1e-3

    @ravelled_method
    def init(self, params):
        return EMT.init_with_obs(jnp.zeros_like(params))

    @partial(ravelled_method, ignore_args=1)
    def update(self, grads, state, params):
        return state.update(grads, self.grad_rate)

    @partial(ravelled_method_unravel, ignore_args=1)
    def mean(self, _, state):
        return state.mean()

    @partial(ravelled_method_unravel, ignore_args=1)
    def implicit_var(self, grads, state):
        return jnp.sqrt(self.grad_rate) * (grads - state.mean())

    @partial(ravelled_method, ignore_args=0)
    def set_zero(self, state, where):
        return state.set_zero(where)

    @partial(ravelled_method, ignore_args=1)
    def process_add_width(self, was_padded, state):
        return state.process_add_width(was_padded)


class MyAdam(struct.PyTreeNode):
    lr: Any
    mom1: float = 1e-1
    mom2: float = 1e-2
    weight_decay: float = 0.0
    noise_std: float = 0.0
    order: int = 0
    eps: float = 1e-6

    def init(self, params):
        mean_grad = jnp.zeros_like(params)
        grad = EMT_N.init_with_obs(mean_grad, self.order)
        # grad = EMT.init_with_obs(mean_grad)
        grad2 = EMT.init_with_obs(jnp.ones_like(params))
        count = 0
        return grad, grad2, count

    def update(self, grads, state, params, **kwargs):
        grad, grad2, count = state
        grad = grad.update(grads, self.mom1)
        grad2 = grad2.update(grads**2, self.mom2)
        updates = grad.mean() / jnp.sqrt(grad2.mean() + self.eps)
        updates = updates + self.weight_decay * params
        noise = jax.random.normal(PRNGKey(count), updates.shape)
        updates = updates + noise * self.noise_std
        updates = -self.lr(count) * updates
        return updates, (grad, grad2, count + 1)

    def process_add_width(self, was_padded, state):
        grad, grad2, count = state
        grad = grad.process_add_width(was_padded, fill=0.0)
        grad2 = grad2.process_add_width(was_padded, fill=1.0)
        return grad, grad2, count


def finite_warn(arr, msg):
    jax.lax.cond(
        jnp.isfinite(arr).all(), lambda x: None, lambda x: jax.debug.print(msg), 0
    )


def recover_from_padding(was_padded, original, fresh):
    indices = (~was_padded).nonzero()
    combined = fresh.at[indices].set(original.reshape(fresh.at[indices].get().shape))
    # print(f"{was_padded.shape} - {original.shape} - {fresh.shape} - {combined.shape}")
    return combined


class WrappedFirstOrder(struct.PyTreeNode):
    tx: optax.GradientTransformation = optax.adam(learning_rate=1e-3, b2=0.99)
    curv: SimpleCurv = FactoredCurv(IRootCurv(), IRootCurv())
    slow_grad: SlowGrad = SlowGrad()
    diag_curv: SimpleCurv = SimpleCurv()

    class State(struct.PyTreeNode):
        curv: EMT
        opt_state: Any
        slow_grad: EMT
        diag_curv: EMT
        time_untouched: Any
        count: int = 0

    def init(self, params):
        if wandb.config.fast_turbo:
            return self.tx.init(params)
        else:
            return self._init(params)

    @tree_method
    def _init(self, params):
        return self._raw_init(params)

    def _raw_init(self, params):
        opt_state = self.tx.init(params)
        curv = self.curv.init(params)
        slow_grad = self.slow_grad.init(params)
        diag_curv = self.diag_curv.init(params)
        time_untouched = jnp.zeros_like(params, dtype=jnp.int32)
        return self.State(
            curv=curv,
            opt_state=opt_state,
            slow_grad=slow_grad,
            diag_curv=diag_curv,
            time_untouched=time_untouched,
        )

    def fast_update(self, grads, state, params, **kwargs):
        updates, opt_state = self.tx.update(grads, state.opt_state, params, **kwargs)
        state = state.replace(opt_state=opt_state)
        return updates, state

    def update(self, grads, state, params=None, **kwargs):
        if wandb.config.fast_turbo:
            return self.tx.update(grads, state, params, **kwargs)
        else:
            return self._update(grads, state, params, **kwargs)

    def _update(self, grads, state, params=None, *, hgrads, **kwargs):
        updates, state = self._main_update(grads, state, params, hgrads=hgrads, **kwargs)
        pre_prune_state = state
        if not wandb.config.freeze_thaw_disable:
            new_params = tree_map(jnp.add, params, updates)
            global_score = self.global_score(new_params, state)
            global_score_tree = tree_map(lambda p: global_score, new_params)
            new_params, state = self._maybe_freeze_thaw_prune(new_params, state, global_score=global_score_tree)
            updates = tree_map(jnp.subtract, new_params, params)
        updates = self._cull_updates(updates, pre_prune_state)
        return updates, state


    @partial(tree_method, outputs=2)
    def _main_update(self, grads, state, params=None, *, hgrads, **kwargs):
        if wandb.config.fast:
            return self.fast_update(grads, state, params, **kwargs)
        finite_warn(grads, "non-finite grads")
        finite_warn(hgrads, "non-finite hgrads")
        mask = state.curv[0].mask
        key = PRNGKey(state.count)
        key, noise_key = jax.random.split(key)
        slow_grad = self.slow_grad.update(grads, state.slow_grad, params)
        slow_grads_mean = self.slow_grad.mean(grads, state.slow_grad)
        diag_curv = self.diag_curv.update(grads, state.diag_curv, params, hgrads=grads)
        curv_grad = hgrads
        noise = jax.random.normal(noise_key, params.shape)
        if wandb.config.add_unit_normal_curvature:
            curv_grad = curv_grad + noise
        if wandb.config.grad_update_as_curvature:
            grad_diff = self.slow_grad.implicit_var(grads, state.slow_grad)
            # grad_diff = grads - self.slow_grad.mean(grads, state.slow_grad)
            grad_diff_curv = jnp.sqrt(wandb.config.grad_curvature_mul) * grad_diff
            curv_grad = curv_grad + grad_diff_curv
        elif wandb.config.grad_as_curvature:
            extra = grads * jnp.sqrt(wandb.config.grad_curvature_mul)
            if wandb.config.root_of_grad_for_curvature:
                extra = self.diag_curv.i4root_mul(state.diag_curv, extra)
            curv_grad = curv_grad + extra
        curv = self.curv.update(grads, state.curv, params, key, hgrads=curv_grad)
        ngrads = self.curv.solve(curv, 0.0 * hgrads + grads + params)
        updates, opt_state = self.tx.update(grads, state.opt_state, params, **kwargs)
        # noise_scale = 1e-2/jnp.sqrt(prod(updates.shape) / updates.shape[-2])
        # updates = updates + jax.random.normal(noise_key, updates.shape)*noise_scale

        eta = self.tx.lr(state.count)
        scale = state.count / 78100
        # scale = jnp.exp(-7*(78100 - state.count)/78100)
        scale = 0.0
        #updates = -eta * ngrads #+ scale * (
        #    jnp.sqrt(2 * eta) - eta
        # ) * self.curv.iroot_mul(curv, noise)

        updates = jnp.where(mask[:, None], updates, jnp.zeros_like(updates))
        if wandb.config.pruned_lr_rescale:
            rescale = len(mask) / (1 + jnp.sum(mask))
            updates = updates * rescale

        time_untouched = jnp.where(grads == 0.0, state.time_untouched + 1, 0)
        state = state.replace(
            curv=curv,
            opt_state=opt_state,
            slow_grad=slow_grad,
            diag_curv=diag_curv,
            time_untouched=time_untouched,
        )
        state = state.replace(count=state.count + 1)
        if not (True or wandb.config.freeze_thaw_disable):
            new_params = params + updates
            expansion_allowed = (state.count < wandb.config.expansion_max_step) & (state.count >= wandb.config.expansion_min_step)
            new_params, state = jax.lax.cond(
                expansion_allowed,
                lambda tup: self._freeze_thaw_prune(*tup),
                lambda tup: tup,
                (new_params, state),
            )
            # new_params, state = self._freeze_thaw_prune(new_params, state)
            updates = new_params - params
        #updates = jnp.where(mask[:, None], updates, jnp.zeros_like(updates))
        return updates, state

    @partial(tree_method, outputs=2)
    def _maybe_freeze_thaw_prune(self, new_params, state, global_score):
        not_too_late = state.count < wandb.config.expansion_max_step
        not_too_early = state.count >= wandb.config.expansion_min_step
        new_params, state = jax.lax.cond(
                not_too_late & not_too_early,
                lambda tup: self._freeze_thaw_prune(tup[0], tup[1], global_score),
                lambda tup: tup,
                (new_params, state),
            )
        return new_params, state

    @tree_method
    def _cull_updates(self, updates, state):
        """zero out updates of frozen weights"""
        mask = state.curv[0].mask
        updates = jnp.where(mask[:, None], updates, jnp.zeros_like(updates))
        return updates

    def _sample_tangent(self, params, state, *, key):
        noise = jax.random.normal(key=key, shape=params.shape, dtype=params.dtype)
        return self.curv.iroot_mul(state.curv, noise)

    @tree_method
    def sample_posterior(self, params, state, *, key):
        return params + self._sample_tangent(params, state, key=key)

    @tree_method
    def sample_tangent(self, params, state, *, key):
        return self._sample_tangent(params, state, key=key)

    def _grad_mag(self, vec_shape, state):
        grad = self.slow_grad.mean(vec_shape, state.slow_grad)
        wgrad = self.curv.iroot_mul(state.curv, grad)
        mag = jnp.sum(jnp.square(jnp.abs(wgrad)))
        return mag

    @tree_method
    def get_depth_score(self, params, state):
        c, n, b, s = self._get_depth_score(params, state)
        return b

    def _get_depth_score(self, params, state):
        if wandb.config.bud_width is None:
            return 1., 1e-3, 1e-3, 1e-3
        grads = state.slow_grad.mean()
        current_score, (frz, pru, tha) = self.curv.freeze_prune_thaw_scores(
            state.curv, grads, params
        )
        iw = wandb.config.ignore_width
        assert iw > 0
        k = wandb.config.depth_score_max_k
        normal = tha[:-iw]
        if len(normal) > k:
            normal, _ = jax.lax.approx_max_k(normal, k)
        bud = tha[-iw:]
        if len(bud) > k:
            bud, _ = jax.lax.approx_max_k(bud, k)
        nscore = jnp.sum(normal)
        bscore = jnp.sum(bud)
        final_score = bscore / (nscore + wandb.config.depth_score_add_to_current_score)
        return current_score, nscore, bscore, final_score

    def get_metrics(self, params, state):
        if wandb.config.fast_turbo:
            return frozen_dict.freeze(dict())
        else:
            return self._get_metrics(params, state)
    @tree_method
    def _get_metrics(self, params, state):
        metrics = {"mag": self._grad_mag(params, state)}
        if not wandb.config.fast and isinstance(self.curv, FactoredCurv) and isinstance(
            state.curv[0], linalg.MaskedWhitener
        ):
            # grads = state.opt_state[0].mu
            grads = state.opt_state[0].mean()
            grads = state.slow_grad.mean()
            mask = state.curv[0].mask
            mag, frz_pru_tha = self.curv.freeze_prune_thaw_scores(
                state.curv, grads, params
            )
            grads_abs = jax.vmap(jnp.sum, in_axes=-1, out_axes=-1)(jnp.abs(grads) ** 2)
            metrics = {**metrics, "mag": mag, "grads": jnp.log1p(grads_abs)}
            frz, pru, tha = map(lambda arr: jnp.median(arr), frz_pru_tha)
            unfrozen = jnp.sum(state.curv[0].mask)
            metrics = dict(**metrics, frz=frz, pru=pru, tha=tha, unfrozen=unfrozen)
            frzm, prum, _ = map(
                lambda arr: jnp.min(arr, where=mask, initial=1e6), frz_pru_tha
            )
            tha_mask = ~mask
            if wandb.config.ignore_width > 0 and self.has_bud(params):
                tha_mask = tha_mask.at[-wandb.config.ignore_width :].set(False)
            tham = jnp.max(frz_pru_tha[-1], where=tha_mask, initial=1e-6)
            metrics = dict(**metrics, frzm=frzm, prum=prum, tham=tham)
            w = state.curv[0]
            health = jnp.max(jnp.diag(w.iroot.T @ w.direct @ w.iroot))
            metrics = dict(**metrics, health=health)

            in_mags = jax.vmap(
                compose(jnp.ravel, jnp.linalg.norm), in_axes=-2, out_axes=-2
            )(params)
            unpruned = jnp.sum(in_mags > 1e-6)
            metrics = dict(**metrics, unpruned=unpruned)
            can_reinit = jnp.sum(self.can_reinit(state))
            metrics = dict(**metrics, can_reinit=can_reinit)
            metrics = dict(**metrics, active_neurons=jnp.sum(~self.can_reinit(state)))

            cscore, nscore, bscore, fscore = self._get_depth_score(params, state)
            metrics = dict(
                **metrics, depth_score=fscore, main_score=nscore, bud_score=bscore
            )

            if (iw := wandb.config.ignore_width) > 0 and self.has_bud(params):
                btha = frz_pru_tha[-1][-iw:]
                budmax = jnp.max(btha)
                budmean = jnp.mean(btha)
                budmed = jnp.median(btha)
                metrics = dict(**metrics, budmax=budmax, budmean=budmean, budmed=budmed)
        return metrics

    def pin_prior(self, params, state):
        if wandb.config.fast_turbo:
            print("WARNING: pin_prior not implemented.")
            return state
        else:
            return self._pin_prior(params, state)

    @tree_method
    def _pin_prior(self, params, state):
        print("WARNING: pin_prior not implemented.")
        return state

    @partial(jax.jit, static_argnums=0)
    @partial(tree_method, outputs=2)
    def freeze_thaw_prune(self, params, state):
        raise NotImplementedError("need to provide global_score somehow")
        return self._freeze_thaw_prune(params, state, global_score)

    def has_bud(self, params):
        # assume that params with shape of final layer has no bud
        return params.shape[-1] != wandb.config.num_classes

    def global_score(self, params, state):
        current_scores = self._leaf_score(params, state)
        if wandb.config.global_score_is_max_not_sum:
            global_score = tree_reduce(jnp.maximum, current_scores, 0.)
        else:
            global_score = tree_reduce(jnp.add, current_scores, 0.)
        return global_score

    @tree_method
    def _leaf_score(self, params, state):
        grads = self.slow_grad.mean(params, state.slow_grad)
        current_score = self.curv.current_score(state.curv, grads)
        return current_score

    def _freeze_thaw_prune(self, params, state, global_score):
        # THRESH=1e1
        # current_score = self._grad_mag(params, state)
        frz_key, tha_key = jax.random.split(PRNGKey(state.count))
        old_params, old_state = params, state
        grads = self.slow_grad.mean(params, state.slow_grad)
        current_score, (frz, pru, tha) = self.curv.freeze_prune_thaw_scores(
            state.curv, grads, params
        )
        if wandb.config.use_global_expansion_score:
            current_score = global_score
        # TESTING
        # if wandb.config["freeze_is_prune"]:
        #    frz = pru

        curv = state.curv
        whitener = curv[0]
        fthresh = jnp.maximum(
            wandb.config.freeze_thresh, wandb.config.freeze_thresh_rel * current_score
        )
        to_freeze = whitener.mask & (frz < fthresh)
        if wandb.config["freeze_is_prune"]:
            to_freeze = to_freeze & (pru < 1e-0 * fthresh)
        # jax.debug.print("{}",to_freeze)
        # to_freeze = jnp.argmax(to_freeze) == jnp.arange(len(to_freeze))
        frz_p = 1e-3
        frz_samples = jax.random.bernoulli(frz_key, p=1e-3, shape=to_freeze.shape)
        to_freeze = to_freeze & frz_samples
        # jax.debug.print("{}",to_freeze)
        to_freeze = to_freeze & (jnp.sum(whitener.mask) > wandb.config.minimum_width)
        # jax.debug.print("{}",to_freeze)
        health = jnp.max(jnp.diag(whitener.iroot.T @ whitener.direct @ whitener.iroot))
        tthresh = jnp.maximum(
            wandb.config.thaw_thresh, wandb.config.thaw_thresh_rel * current_score
        )
        to_thaw = ~whitener.mask & (tha > tthresh) & (health < 1e1)
        tha_p = 1e-3
        if wandb.config.thaw_prob_size_compensate:
            tha_p = (
                tha_p
                * len(whitener.mask)
                / jnp.maximum(1.0, 1.0 * jnp.sum(~whitener.mask))
            )
        tha_samples = jax.random.bernoulli(tha_key, p=tha_p, shape=to_thaw.shape)
        to_thaw = to_thaw & tha_samples
        if wandb.config.ignore_width > 0 and self.has_bud(params):
            to_thaw = to_thaw.at[-wandb.config.ignore_width :].set(False)
        #jax.debug.print("thawed {} from sample {}", jnp.sum(to_thaw), jnp.sum(tha_samples))
        whitener = whitener.freeze_many(to_freeze)
        # jax.debug.print("{}", whitener)
        whitener = whitener.thaw_many(to_thaw)
        # jax.debug.print("{}", whitener)
        curv = (whitener,) + curv[1:]
        state = state.replace(curv=curv)
        if wandb.config["freeze_is_prune"]:
            params = self.curv.prune(state.curv, grads, params, to_freeze)
        return params, state

    @tree_method
    def init_prune_params(self, params):
        in_dim = params.shape[-2]
        keep_number = wandb.config.minimum_width
        if self.has_bud(params):
            assert (
                in_dim > wandb.config.ignore_width
            ), "ignore_width larger than bud size?"
            keep_number = min(keep_number, in_dim - wandb.config.ignore_width)
        keep = jnp.arange(in_dim) < keep_number
        params = jnp.where(keep[:, None], params, jnp.zeros_like(params))
        return params

    @tree_method
    def init_prune_opt_state(self, params, state):
        in_dim = params.shape[-2]
        keep_number = wandb.config.minimum_width
        if self.has_bud(params):
            keep_number = min(keep_number, in_dim - wandb.config.ignore_width)
        keep = jnp.arange(in_dim) < keep_number
        in_curv_whitener = state.curv[0]
        in_curv_whitener = in_curv_whitener.replace(mask=keep).reset_iroot()
        curv_state = (in_curv_whitener,) + state.curv[1:]
        state = state.replace(curv=curv_state)
        return state

    def can_reinit(self, state):
        def ok(arr):
            return (arr >= wandb.config.untouched_thresh).all()

        return jax.vmap(ok, in_axes=-1, out_axes=-1)(state.time_untouched)

    @tree_method
    def reinit_allowed(self, params, state, *, key):
        allowed = self.can_reinit(state)
        # key = PRNGKey(state.count)
        bernoulli = jax.random.bernoulli(
            key, p=wandb.config.reinit_prob, shape=allowed.shape
        )
        allowed = allowed & bernoulli
        jax.debug.print("authorized {} reinitializations", jnp.sum(allowed))
        allowed = allowed & state.curv[0].mask[:, None]
        true = jnp.ones_like(params, dtype=jnp.bool_)
        false = jnp.zeros_like(params, dtype=jnp.bool_)
        return jnp.where(allowed, true, false)

    @tree_method
    def process_reinit(self, params, state, was_reinitialized):
        new_slow_grad = self.slow_grad.set_zero(state.slow_grad, was_reinitialized)
        return state.replace(slow_grad=new_slow_grad)

    @tree_method
    def should_add_width(self, params, state, *, key):
        if params.shape == (1, 1):
            return 0
        allowed = self.can_reinit(state)
        thresh = wandb.config.add_width_thresh
        factor = wandb.config.add_width_factor
        current = params.shape[-1]
        assert allowed.shape == params.shape[-1:]
        under_thresh = jnp.sum(allowed) / current < thresh
        new_size = int(current * factor)
        width_cap = wandb.config.maximum_width
        new_size = int(min(new_size, width_cap - current))
        if under_thresh and new_size > 0:
            jax.debug.print("authorized width expansion of size {}", new_size)
            return new_size
        else:
            return 0

    @tree_method
    def process_add_width(self, params, state, was_padded):
        new_time_untouched = recover_from_padding(
            was_padded,
            state.time_untouched,
            jnp.zeros_like(was_padded, dtype=jnp.int32),
        )
        new_curv = self.curv.process_add_width(was_padded, state.curv)
        new_opt_state = self.tx.process_add_width(was_padded, state.opt_state)
        new_slow_grad = self.slow_grad.process_add_width(was_padded, state.slow_grad)
        new_diag_curv = self.diag_curv.process_add_width(was_padded, state.diag_curv)
        return state.replace(
            curv=new_curv,
            opt_state=new_opt_state,
            slow_grad=new_slow_grad,
            diag_curv=new_diag_curv,
            time_untouched=new_time_untouched,
        )

    def reinit_changed_shapes(self, params, state):
        if wandb.config.fast_turbo:
            return self.tx.init(params)
        else:
            return self._reinit_changed_shapes(params, state)
    @tree_method
    def _reinit_changed_shapes(self, params, state):
        curv = state.diag_curv.mean()
        if curv.shape == params.shape: return state
        else:
            old_mask = state.curv[0].mask
            state = self._raw_init(params)
            in_curv_whitener = state.curv[0]
            in_curv_whitener = in_curv_whitener.replace(mask=old_mask)
            curv_state = (in_curv_whitener,) + state.curv[1:]
            state = state.replace(curv=curv_state)
            return state



class Pruner(struct.PyTreeNode):
    freeze: EMA
    prune: EMA
    thaw: EMA
    freeze_bound: EMA
    grad_mag: float = 1e0
    eps: float = 1e-3

    @classmethod
    def init(cls, grads):
        # shape=grads.shape[-1]
        shape = grads.shape
        return cls(
            freeze=EMVar.init_zero(shape),
            prune=EMVar.init_zero(shape),
            thaw=EMVar.init_zero(shape),
            freeze_bound=EMVar.init_zero(shape[-1:]),
        )

    def get_scaling(self, whitener):
        iroot = whitener.iroot
        diag = whitener.diag_inv() + self.eps
        scaling = jnp.reciprocal(jnp.sqrt(diag))
        # line below is approximation erring on the side of freezing too much
        # dscaling = jnp.sqrt(jnp.diag(whitener.direct))
        # scaling = jnp.minimum(scaling, dscaling)
        scaling = jnp.where(whitener.mask, scaling, jnp.zeros_like(scaling))
        return scaling
        scaled_iroot = scaling[:, None] * iroot
        return scaled_iroot

    def update_freeze_prune(self, whitener, grads, params, rate, ngrad):
        iroot = whitener.iroot
        scaling = self.get_scaling(whitener)
        nat_grads = grads @ iroot @ iroot.T
        nat_grads = ngrad
        freeze_scores = nat_grads * scaling
        freeze_bound = jnp.maximum(jnp.sum(nat_grads * grads, axis=0), 0.0)
        new_freeze_bound = self.freeze_bound.update(freeze_bound, rate)
        new_freeze = self.freeze.update(freeze_scores, rate)
        prune_scores = (nat_grads - params) * scaling
        new_prune = self.prune.update(prune_scores, rate)
        return self.replace(
            freeze=new_freeze, prune=new_prune, freeze_bound=new_freeze_bound
        )

    def ghetto_project(self, direct, mask, vecs):
        normalizer = jnp.sqrt(jnp.diag(direct))
        normed = (direct / normalizer) / normalizer[:, None]
        fraction = 1.0 - jnp.max(normed, axis=0, where=mask, initial=0.0)
        return vecs * fraction

    def gmres_project(self, direct, mask, vecs, iroot):
        subvecs = jnp.where(mask, vecs, 0.0)
        subdirect = jnp.where(mask[:, None] & mask, direct, 0.0)
        precon = iroot @ iroot.T
        solns = jax.scipy.sparse.linalg.gmres(
            subdirect,
            subvecs.T,
            M=precon,
            maxiter=1,
            restart=20,
            solve_method="batched",
        )[0]
        return vecs - solns.T @ direct

    def update_thaw(self, whitener, grads, rate, ngrad):
        iroot = whitener.iroot
        direct = whitener.direct
        # vecs = grads - grads @ iroot @ iroot.T @ direct
        # vecs = self.ghetto_project(direct, whitener.mask, grads)
        # vecs = self.gmres_project(direct, whitener.mask, grads, iroot)
        vecs = grads - ngrad @ direct
        # center = direct - direct @ iroot @ iroot.T @ direct
        # center = jnp.diag(center)
        adj_center = jnp.diag(direct) - jnp.diag(
            direct
        ) * whitener.diag_inv() * jnp.diag(direct)
        # BELOW IS AN APPROXIMATION - IF PROBLEMS USE LINE ABOVE
        center = jnp.diag(direct)
        # center = jnp.maximum(1e-3*center, adj_center)
        scaling = jnp.reciprocal(jnp.sqrt(jnp.maximum(self.eps, jnp.abs(center))))
        scaling = jnp.where(whitener.mask, jnp.zeros_like(scaling), scaling)
        thaw_scores = vecs * scaling
        # FOR SOME REASON WE GET NON-FINITE, NON-NAN scores here
        # Maybe fixed by changing minimum to maximum on line using self.eps?
        thaw_scores = jnp.where(
            jnp.isfinite(thaw_scores), thaw_scores, jnp.zeros_like(thaw_scores)
        )
        new_thaw = self.thaw.update(thaw_scores, rate)
        return self.replace(thaw=new_thaw)

    def update(self, whitener, grads, params, rate):
        """vecs is whitened in other kron factors"""
        ngrad = whitener.gmres_solve(grads)
        grad_mag = jnp.maximum(0.0, jnp.sum(ngrad * grads))
        out = self.update_freeze_prune(whitener, grads, params, rate, ngrad=ngrad)
        out = out.update_thaw(whitener, grads, rate, ngrad=ngrad)
        out = out.replace(grad_mag=grad_mag)
        return out

    def freeze_which(self, active, freeze_scores, key):
        scores = jnp.where(active, freeze_scores, jnp.inf)
        to_freeze = jnp.argmin(scores) == jnp.arange(len(active))
        thresh = wandb.config.get("freeze_thresh", 1e0)
        rel_thresh = wandb.config.get("freeze_thresh_rel", 0.0) * self.grad_mag
        thresh = jnp.maximum(thresh, rel_thresh)
        # to_freeze = jnp.argmax(scores < thresh) == jnp.arange(len(active))
        not_too_small = jnp.sum(active) > wandb.config.get("minimum_width", 8)
        to_freeze = to_freeze & (jnp.min(scores) < thresh) & not_too_small
        # stochastic implementation:
        stoch = jax.random.bernoulli(key, p=1e-2, shape=active.shape)
        stoch = (scores < thresh) & stoch & not_too_small
        return stoch
        return to_freeze

    def thaw_which(self, active, thaw_scores, key):
        scores = jnp.where(active, 0.0, thaw_scores)
        to_thaw = jnp.argmax(scores) == jnp.arange(len(active))
        thresh = wandb.config.get("thaw_thresh", 1e1)
        rel_thresh = wandb.config.get("thaw_thresh_rel", 0.0) * self.grad_mag
        thresh = jnp.maximum(thresh, rel_thresh)
        stoch = jax.random.bernoulli(key, p=1e-2, shape=active.shape)
        stoch = (scores > thresh) & stoch
        return stoch
        to_thaw = (scores > thresh) & to_thaw
        return to_thaw

    def freeze_thaw(self, whitener, key):
        if wandb.config.freeze_thaw_disable:
            return whitener
        frz_key, tha_key = jax.random.split(key)
        active = whitener.mask
        freeze_scores = jnp.sum(jnp.square(jnp.abs(self.freeze.mu)), axis=0)
        # freeze_scores = self.freeze_bound.mu
        to_freeze = self.freeze_which(active, freeze_scores, key=frz_key)
        whitener = whitener.freeze_many(to_freeze)
        # whitener = jax.lax.cond(
        #    jnp.any(to_freeze),
        #    lambda b: whitener.freeze(jnp.argmax(b)),
        #    lambda b: whitener,
        #    to_freeze)
        thaw_scores = jnp.sum(jnp.square(jnp.abs(self.thaw.mu)), axis=0)
        to_thaw = self.thaw_which(active, thaw_scores, key=tha_key)
        whitener = whitener.thaw_many(to_thaw)
        # whitener = jax.lax.cond(
        #    jnp.any(to_thaw),
        #    lambda b: whitener.thaw(jnp.argmax(b)),
        #    lambda b: whitener,
        #    to_thaw)
        # jax.debug.print("freeze: {}", freeze_scores)
        jax.debug.print("froze {}, thawed {}", jnp.sum(to_freeze), jnp.sum(to_thaw))
        return whitener


class InnerState(struct.PyTreeNode):
    hess: HessTracker
    grad: Any = None
    nat_grad: Any = None
    key: PRNGKey = PRNGKey(0)
    # prior: GaussianPrior = GaussianPrior()
    prior: KronPrior = KronPrior(None, None, None)
    delta_logdet: float = 0.0
    excess: float = 1.0
    grad_excess: float = 1.0
    grad_mag: float = 1.0
    base_rate: float = 1e-2
    param_mul: float = 3e-1
    grad_mul: float = 1e-0
    speedup_cap: float = 1e1
    grad_var: Any = None
    adam_base_rate: float = 1e-2
    adam_power: float = 0.5
    freeze_score: Any = None
    prune_score: Any = None
    pruner: Pruner = Pruner(None, None, None, None)
    count: int = 1
    wgrad_track: EMT = None
    wgrad_sq_track: EMT = None


class InnerConfig(struct.PyTreeNode):
    decay_grad_by_lr: bool = True
    grad_update_as_curvature: bool = True
    grad_as_curvature: bool = False
    grad_curvature_mul: float = 1.0
    incremental_solve: bool = False
    use_white_adam: bool = False
    use_wgrad_snr_lr: bool = False
    eps: float = 1e-12
    init_prior_precision: float = 1.0
    nat_grad_fraction: float = 1.0


class InnerOpt(struct.PyTreeNode):
    """second order optimizer - only sees (in_dim, out_dim) dense arrays"""

    init_state: InnerState
    conf: InnerConfig = InnerConfig()
    init_key: PRNGKey = PRNGKey(0)

    def init(self, params):
        # init_grad = jnp.zeros_like(params)
        init_grad = jax.random.normal(self.init_key, params.shape, params.dtype)
        init_grad = init_grad / self.conf.init_prior_precision
        init_nat_grad = jnp.zeros_like(params)
        hess = self.init_state.hess.init(params)
        init_grad_var = jnp.ones_like(params)
        init_freeze_score = jnp.ones_like(params[:, 0])
        init_prune_score = jnp.ones_like(params[:, 0])
        init_pruner = self.init_state.pruner.init(params.T)
        init_prior = self.init_state.prior.init(
            params, precision=self.conf.init_prior_precision
        )
        init_wgrad_track = EMT.init_with_obs(init_grad)
        init_wgrad_sq_track = EMT.init_with_obs(jnp.square(jnp.abs(init_grad)))
        return self.init_state.replace(
            hess=hess,
            grad=init_grad,
            nat_grad=init_nat_grad,
            key=self.init_key,
            grad_var=init_grad_var,
            freeze_score=init_freeze_score,
            prune_score=init_prune_score,
            pruner=init_pruner,
            prior=init_prior,
            wgrad_track=init_wgrad_track,
            wgrad_sq_track=init_wgrad_sq_track,
        )

    @staticmethod
    def update_hess(hgrads, hess_state, hess_rate, key):
        hess_state, delta_logdet_total = hess_state.rescale(1.0 - hess_rate)
        for hgrad in hgrads:
            obs = jnp.sqrt(hess_rate) * hgrad
            hess_state, delta_logdet = hess_state.rank_one_update(obs, key=key)
            delta_logdet_total = delta_logdet_total + delta_logdet
        return hess_state, delta_logdet_total

    @staticmethod
    def get_rate_hess(state, conf):
        excess = state.excess * wandb.config.hess_excess_scale
        denom = 1.0 + state.speedup_cap * jnp.maximum(0.0, excess)
        numer = state.speedup_cap * state.base_rate
        # if conf.use_white_adam:
        #    adam = jnp.sqrt(state.grad_excess / (1e-12 + state.grad_mag))
        #    adam = jnp.maximum(adam, 1.)
        #    adam = jnp.minimum(adam, 1000.)
        #    numer = numer / adam
        return numer / denom

    @staticmethod
    def get_rate_grad(state, conf):
        inv_excess = jnp.reciprocal(state.grad_excess)
        return state.adam_base_rate * inv_excess

    @classmethod
    def get_rate(cls, state, conf):
        rate = cls.get_rate_hess(state, conf)
        if conf.use_white_adam:
            grate = cls.get_rate_grad(state, conf)
            grate = jnp.maximum(grate, 1.0 / state.count)
            rate = jnp.minimum(rate, grate)
            rate = jnp.maximum(rate, wandb.config.min_adam_mult)
        if conf.use_wgrad_snr_lr:
            rate = jnp.minimum(
                rate, cls.wgrad_rate(state, conf) * wandb.config.wgrad_snr_base_lr
            )
        return rate

    @staticmethod
    def update_excess(state, hgrads, rate):
        expected = rate * state.excess

        def for_one(hgrad):
            whitened = state.hess.whiten(hgrad)
            return jnp.mean(jnp.log1p(rate * jnp.square(whitened)))

        obs = sum(map(for_one, hgrads))
        new_excess = state.excess + obs - expected
        return state.replace(excess=new_excess)

    @staticmethod
    def white_grad_mag(state, grad, rate):
        EPS = 1e-12
        white = state.hess.whiten(grad)
        sqmag = jnp.sum(jnp.square(jnp.abs(white)))
        return jnp.log1p(rate * sqmag) / (rate + EPS)

    @staticmethod
    def update_grad_var(state, werr, rate):
        obs = jnp.square(jnp.abs(werr))
        new_grad_var = state.grad_var + rate * obs - rate * state.grad_var
        return state.replace(grad_var=new_grad_var)

    @classmethod
    def update_grad_excess(cls, state, grad, rate):
        err = grad - state.grad
        werr = state.hess.whiten(err)
        wgrad = state.hess.whiten(state.grad)
        state = cls.update_grad_var(state, werr, rate)
        wgrad_elem_mags = jnp.square(jnp.abs(wgrad))
        floored_elem_mags = wgrad_elem_mags  # + wandb.config.grad_mag_floor/grad.size
        obs = jnp.sum(floored_elem_mags * jnp.reciprocal(state.grad_var))
        obs = obs * (grad.size) ** (-state.adam_power)
        obs = jnp.reciprocal(jnp.sqrt(obs))
        # obs = cls.white_grad_mag(state, err, rate)
        # grad_mag = cls.white_grad_mag(state, state.grad, rate)
        grad_mag = jnp.sum(wgrad_elem_mags)
        new_grad_excess = state.grad_excess + obs - rate * state.grad_excess
        # new_grad_mag = state.grad_mag + 1.*grad_mag - 1.*state.grad_mag
        new_grad_mag = grad_mag
        return state.replace(grad_excess=new_grad_excess, grad_mag=new_grad_mag)

    @classmethod
    def wgrad_rate(cls, state, conf):
        wgrad_mean = state.wgrad_track.mean()
        wgrad_mean_sq = state.wgrad_sq_track.mean()
        wgrad_sq_mean = jnp.square(jnp.abs(wgrad_mean))
        # wgrad_var = jnp.maximum(conf.eps, wgrad_mean_sq - wgrad_sq_mean)
        dim = jnp.size(wgrad_mean)
        wgrad_mean_sq = jnp.maximum(conf.eps, wgrad_mean_sq)
        sqrate = jnp.sum(wgrad_sq_mean * jnp.reciprocal(wgrad_mean_sq)) / jnp.sqrt(dim)
        sqrate = jnp.maximum(sqrate, conf.eps)

        rate = jnp.sqrt(sqrate)
        return rate

    def update_wgrad(self, state, wgrad, rate):
        state = state.replace(wgrad_track=state.wgrad_track.update(wgrad, rate))
        state = state.replace(
            wgrad_sq_track=state.wgrad_sq_track.update(jnp.square(jnp.abs(wgrad)), rate)
        )
        return state

    def update_freeze_score(self, state, rate):
        diag = state.hess.h_in.diag_inv()
        diag = jnp.reciprocal(jnp.sqrt(diag + self.conf.eps))
        nat_grad = state.nat_grad
        score = nat_grad * diag[:, None]
        total = jnp.sum(jnp.square(score), axis=-1)
        new_freeze_score = state.freeze_score + rate * total - rate * state.freeze_score
        state = state.replace(freeze_score=new_freeze_score)
        return state

    def update_prune_score(self, state, rate, params):
        diag = state.hess.h_in.diag_inv()
        diag = jnp.reciprocal(jnp.sqrt(diag + self.conf.eps))
        nat_grad = state.nat_grad - params
        score = nat_grad * diag[:, None]
        total = jnp.sum(jnp.square(score), axis=-1)
        new_prune_score = state.prune_score + rate * total - rate * state.prune_score
        state = state.replace(prune_score=new_prune_score)
        return state

    def update_pruner(self, state, rate, params):
        grads = state.hess.whiten_out(state.grad).T
        params = state.hess.whiten_out(params).T
        pruner_rate = 1.0
        state = state.replace(
            pruner=state.pruner.update(state.hess.h_in, grads, params, pruner_rate)
        )
        return state

    def freeze_thaw(self, state, params, key):
        new_h_in = state.pruner.freeze_thaw(state.hess.h_in, key)
        return state.replace(hess=state.hess.replace(h_in=new_h_in))

    def update(self, grads, state, params, hgrads):
        new_key, prior_key, update_key, solve_key = jax.random.split(state.key, 4)
        state = state.replace(key=new_key)

        rate = self.get_rate(state, self.conf)

        prior_grad, prior_hgrad = state.prior(params, key=prior_key)
        # update grad
        grad_update = state.grad_mul * rate * (grads + prior_grad - state.grad)
        state = state.replace(grad=state.grad + grad_update)
        state = self.update_grad_excess(state, grads + prior_grad, rate)

        # update hessian
        root_gcurv_mul = jnp.sqrt(self.conf.grad_curvature_mul)
        all_hgrads = (hgrads, prior_hgrad)
        if self.conf.grad_update_as_curvature:
            all_hgrads = all_hgrads + (grad_update * root_gcurv_mul / jnp.sqrt(rate),)
        if self.conf.grad_as_curvature:
            curv_grad = grads + prior_grad - state.grad
            curv_grad = root_gcurv_mul * curv_grad
            all_hgrads = all_hgrads + (curv_grad,)
        if self.conf.grad_update_as_curvature:
            exgradmul = wandb.config.grad_curvature_excess_mul
            ex_hgrads = all_hgrads[:-1] + (all_hgrads[-1] * jnp.sqrt(exgradmul),)
        else:
            ex_hgrads = all_hgrads
        state = self.update_excess(state, ex_hgrads, rate)
        hess_state, delta_logdet = self.update_hess(
            all_hgrads, state.hess, rate, key=update_key
        )
        state = state.replace(hess=hess_state)
        delta_logdet_update = rate * (delta_logdet - state.delta_logdet)
        state = state.replace(delta_logdet=state.delta_logdet + delta_logdet_update)

        # update natural gradient
        if self.conf.incremental_solve:
            # this is too noisy - should rescale deterministically, and update stochastically
            # this also needs to account for decay_grad_by_lr
            nat_grad_update = state.hess.solve(grad_update, key=solve_key)
            state = state.replace(nat_grad=state.nat_grad + nat_grad_update)
            raise NotImplementedError
        else:
            new_nat_grad = state.hess.solve(state.grad, solve_key)
            state = state.replace(nat_grad=new_nat_grad)
        # state = self.update_freeze_score(state, rate)
        # state = self.update_prune_score(state, rate, params)
        state = self.update_pruner(state, rate, params)
        state = self.freeze_thaw(state, params, key=update_key)

        wgrad = state.hess.whiten(state.grad)
        state = self.update_wgrad(state, wgrad, rate)

        # update params
        lr = state.param_mul * rate
        ngfrac = self.conf.nat_grad_fraction
        params_update = -lr * (ngfrac * state.nat_grad + (1.0 - ngfrac) * state.grad)
        if wandb.config.sgd_override:
            params_update = state.param_mul * state.grad
        if self.conf.decay_grad_by_lr:
            # this accounts for expected change in gradient due to the known hessian
            state = state.replace(grad=state.grad * (1.0 - lr))
        state = state.replace(count=state.count + 1)
        return params_update, state

    def sample_tangent(self, params, state, key):
        noise_key, iroot_mul_key = jax.random.split(key, 2)
        noise = jax.random.normal(noise_key, params.shape, params.dtype)
        tangent = state.hess.iroot_mul(noise, iroot_mul_key)
        return tangent

    def sample_posterior(self, params, state, key):
        return params + self.sample_tangent(params, state, key)

    def get_metrics(self, params, state):
        hrate = self.get_rate_hess(state, self.conf)
        grate = self.get_rate_grad(state, self.conf)

        # freeze_score = state.freeze_score
        def score(emvar):
            return jnp.square(jnp.abs(emvar.mu))
            raw = emvar.mu**4 / (1e-6 + emvar.sq.mu)
            raw = jnp.where(jnp.isfinite(raw), raw, jnp.zeros_like(raw))
            return raw

        freeze_score = jnp.sum(score(state.pruner.freeze), axis=0)
        ln1pfrz = np.log1p(freeze_score)
        # jax.debug.print("ln1pfrz: {}, {}", type(ln1pfrz), ln1pfrz)
        ln1pfrz = np.where(np.isfinite(ln1pfrz), ln1pfrz, np.zeros_like(ln1pfrz))
        # ln1ppru = np.log1p(state.prune_score)
        ln1ppru = np.log1p(jnp.sum(score(state.pruner.prune), axis=0))
        ln1ptha = np.log1p(jnp.sum(score(state.pruner.thaw), axis=0))
        ln1ptha = np.where(np.isfinite(ln1ptha), ln1ptha, np.zeros_like(ln1ptha))
        prior_distance = state.prior.distance(params)

        wgrad_mag = jnp.sum(jnp.square(jnp.abs(state.wgrad_track.mean())))
        grad_l2 = jnp.sum(jnp.square(jnp.abs(state.grad)))
        wgrad_snr_rate = self.wgrad_rate(state, self.conf)
        wgrad_snr = jnp.square(jnp.abs(state.wgrad_track.mean())) * jnp.reciprocal(
            state.wgrad_sq_track.mean()
        )
        wgrad_snr_mag = jnp.sum(
            wgrad_snr * jnp.square(jnp.abs(state.wgrad_track.mean()))
        )
        wgrad_snr = np.log(jnp.maximum(0.0, wgrad_snr))
        active = np.sum(state.hess.h_in.mask)
        inactive = np.sum(~state.hess.h_in.mask)

        gmres_ngrad = state.hess.gmres_solve(state.grad)
        gmres_grad_mag = jnp.sum(state.grad * gmres_ngrad)
        return dict(
            excess=state.excess,
            rate=hrate,
            grad_excess=state.grad_excess,
            grad_mag=state.grad_mag,
            rate_grad=grate,
            freeze=np.array(freeze_score),
            ln1pfrz=ln1pfrz,
            ln1ppru=ln1ppru,
            ln1ptha=ln1ptha,
            prior_distance=prior_distance,
            wgrad_mag=wgrad_mag,
            wgrad_snr_rate=wgrad_snr_rate,
            grad_l2=grad_l2,
            wgrad_snr_mag=wgrad_snr_mag,
            active=active,
            inactive=inactive,
            gmres_grad_mag=gmres_grad_mag,
        )

    def pin_prior(self, params, state):
        grad, nat_grad, hess = state.grad, state.nat_grad, state.hess
        root_in = hess.h_in.direct @ hess.h_in.iroot
        root_out = hess.h_out.direct @ hess.h_out.iroot
        new_prior = state.prior.replace(
            root_in=root_in, root_out=root_out, center=params
        )
        new_grad = jnp.zeros_like(grad)
        new_nat_grad = jnp.zeros_like(nat_grad)
        return state.replace(grad=new_grad, nat_grad=new_nat_grad, prior=new_prior)


class Flattener(struct.PyTreeNode):
    shapes: Any = struct.field(pytree_node=False)
    in_sizes: Any = struct.field(pytree_node=False)

    @staticmethod
    def _ravel(arr):
        return jax.vmap(jnp.ravel, in_axes=-2, out_axes=-2)(arr)

    @staticmethod
    def _unravel(arr, shape):
        dummy = jnp.zeros(shape=shape)
        return jax.vmap(lambda a, d: a.reshape(d.shape), in_axes=-2, out_axes=-2)(
            arr, dummy
        )

    @classmethod
    def create(cls, pytree):
        shapes = tree_map(jnp.shape, pytree)
        ravelled = tree_map(cls._ravel, pytree)
        in_sizes = tree_map(lambda arr: arr.shape[0], ravelled)
        concatenated = jnp.concatenate(tree_leaves(ravelled), axis=-2)
        return cls(shapes, in_sizes), concatenated

    def flatten(self, pytree):
        _, concatenated = self.create(pytree)
        return concatenated

    def unflatten(self, arr):
        in_sizes_list, tree_def = tree_flatten(self.in_sizes)
        in_sizes_array = np.array(in_sizes_list)
        split_indices = np.cumsum(in_sizes_array)[1:]
        ravelled = tree_unflatten(tree_def, jnp.split(arr, split_indices))
        # original = tree_map(jnp.reshape, ravelled, self.shapes)
        original = tree_map(self._unravel, ravelled, self.shapes)
        return original


class FlattenOpt(struct.PyTreeNode):
    inner: InnerOpt
    init_key: PRNGKey = PRNGKey(0)

    class FlattenState(struct.PyTreeNode):
        flattener: Flattener
        inner_state: InnerState

    def init(self, params):
        flattener, flat_params = Flattener.create(params)
        inner_state = self.inner.replace(init_key=self.init_key).init(flat_params)
        return self.FlattenState(flattener, inner_state)

    def update(self, grads, state, params, hgrads):
        grads, params, hgrads = map(state.flattener.flatten, (grads, params, hgrads))
        updates, inner_state = self.inner.update(
            grads, state.inner_state, params, hgrads
        )
        updates = state.flattener.unflatten(updates)
        return updates, state.replace(inner_state=inner_state)

    def sample_tangent(self, params, state, key):
        params = state.flattener.flatten(params)
        tangent = self.inner.sample_tangent(params, state.inner_state, key)
        return state.flattener.unflatten(tangent)

    def sample_posterior(self, params, state, key):
        params = state.flattener.flatten(params)
        sample = self.inner.sample_posterior(params, state.inner_state, key)
        return state.flattener.unflatten(sample)

    def get_metrics(self, params, state):
        params = state.flattener.flatten(params)
        return self.inner.get_metrics(params, state.inner_state)

    def pin_prior(self, params, state):
        flat_params = state.flattener.flatten(params)
        new_inner_state = self.inner.pin_prior(flat_params, state.inner_state)
        return state.replace(inner_state=new_inner_state)


class DiagOpt(struct.PyTreeNode):
    lr: float = 1e-2
    grad_decay: float = 0.90
    hess_decay: float = 0.90
    grad: Any = None
    hess: Any = None
    init_key: Any = PRNGKey(0)
    surprise: float = 3.0

    def flatten(self, arr):
        return jax.vmap(jnp.ravel, in_axes=-1, out_axes=-1)(arr)

    def init(self, params):
        flat = self.flatten(params)
        return self.replace(grad=jnp.zeros_like(flat), hess=jnp.ones_like(flat) * 1.0)

    def update(self, grads, state, params, hgrads):
        PRIOR_PRECISION = 1.0
        expected_grad = (1.0 - state.lr) * state.grad
        grad_rate = 1.0 - state.grad_decay
        grad_update = grad_rate * (
            self.flatten(grads) + self.flatten(params) * PRIOR_PRECISION - expected_grad
        )
        new_grad = expected_grad + grad_update
        hess_rate = 1.0 - state.hess_decay
        hess_obs = (
            self.flatten(jnp.square(hgrads)) + jnp.square(grad_update) + PRIOR_PRECISION
        )
        hess_update = hess_rate * (hess_obs - state.hess)
        new_hess = state.hess + hess_update

        EPS = 1e-12
        # lr = jnp.where(state.surprise < 2., state.lr, 0.)
        lr = state.lr
        params_update = -lr * new_grad * jnp.reciprocal(new_hess + EPS)
        if wandb.config.sgd_override:
            params_update = -lr * new_grad

        surprise = (
            jnp.mean(jnp.log1p(hess_rate * hess_obs / (state.hess + EPS))) / hess_rate
        )
        # new_hess_tau = 1/hess_rate + state.surprise - 1.
        # new_hess_rate = 1/new_hess_tau
        # new_hess_decay = 1. - new_hess_rate
        # new_hess_decay = state.hess_decay + hess_rate*(surprise*state.hess_decay - state.hess_decay)
        new_surprise = state.surprise + 0.1 * (surprise - state.surprise)

        return params_update.reshape(params.shape), state.replace(
            grad=new_grad, hess=new_hess, surprise=new_surprise
        )

    def sample_tangent(self, params, state, key):
        noise = self.flatten(jax.random.normal(key, shape=params.shape))
        colored = noise * jnp.reciprocal(jnp.sqrt(state.hess + 1e-12))
        return colored.reshape(params.shape)

    def sample_posterior(self, params, state, key):
        return params + self.sample_tangent(params, state, key)

    def get_metrics(self, params, state):
        grad_l2 = jnp.sum(jnp.square(jnp.abs(state.grad)))
        ngrad = state.grad * jnp.reciprocal(jnp.sqrt(state.hess + 1e-12))
        wgrad_mag = jnp.sum(jnp.square(jnp.abs(ngrad)))
        hess_mean_diag = jnp.mean(state.hess)
        return dict(
            grad_l2=grad_l2,
            wgrad_mag=wgrad_mag,
            surprise=state.surprise,
            hess_mean_diag=hess_mean_diag,
        )


class LeafState(struct.PyTreeNode):
    kron_in: Any
    kron_out: Any
    grad: Any
    nat_grad: Any
    key: Optional[PRNGKey] = None

    def update_kron_in(self, vec, **kwargs):
        kron_in, nat_grad = self.kron_in.rank_one_update(
            vec, soln=self.nat_grad, **kwargs
        )
        return self.replace(kron_in=kron_in, nat_grad=nat_grad)

    def update_kron_out(self, vec, **kwargs):
        kron_out, nat_grad_T = self.kron_out.rank_one_update(
            vec, soln=self.nat_grad.T, **kwargs
        )
        return self.replace(kron_out=kron_out, nat_grad=nat_grad_T.T)

    def update_grad_full(self, grad, multiplier=1.0, decay=None):
        old_w, new_w = (
            (1.0, multiplier) if decay is None else (decay, multiplier * (1.0 - decay))
        )
        new_grad = self.grad * old_w + grad * new_w
        Li, Lo = self.kron_in.ichol, self.kron_out.ichol
        new_nat_grad = Li @ Li.T @ new_grad @ Lo @ Lo.T
        return self.replace(grad=new_grad, nat_grad=new_nat_grad)

    def renorm(self):
        trin, trout = jnp.trace(self.kron_in.direct), jnp.trace(self.kron_out.direct)
        EPS = 1e-12
        ratio = (trin + EPS) / (trout + EPS)
        rescale = jnp.sqrt(ratio) + EPS
        new_in = self.kron_in.scale_by(1 / rescale)
        new_out = self.kron_out.scale_by(rescale)
        return self.replace(kron_in=new_in, kron_out=new_out)


class LeafOpt(struct.PyTreeNode):
    lr: float = 1e-2
    decay: float = 0.99
    init_key: PRNGKey = PRNGKey(0)

    def flatten(self, pytree):
        return jax.vmap(jnp.ravel, in_axes=-1, out_axes=-1)(pytree)

    def init(self, params):
        flat = self.flatten(params)
        return LeafState(
            linalg.SecondMoment.init_identity(flat.shape[0]),
            linalg.SecondMoment.init_identity(flat.shape[1]),
            jnp.zeros_like(flat),
            jnp.zeros_like(flat),
            key=self.init_key,
        )

    def update(self, grads, state, params, hgrads):
        out_key, in_key, prior_key, new_key = jax.random.split(state.key, 4)
        state = state.replace(key=new_key)

        state = state.replace(grad=state.grad * (1.0 - self.lr))

        hgrads = self.flatten(hgrads)
        PRIOR_PRECISION = 1.0
        prior_curv = (
            jax.random.normal(prior_key, hgrads.shape, hgrads.dtype) * PRIOR_PRECISION
        )
        hgrads = hgrads + jnp.sqrt(1.0 - self.decay) * prior_curv
        grads = grads + (1.0 - self.decay) * params * PRIOR_PRECISION
        in_size, out_size = hgrads.shape

        out_vec = jax.random.normal(out_key, (out_size,), hgrads.dtype)
        mul = 1 / len(out_vec)
        probe = state.kron_out.ichol @ out_vec
        state = state.update_kron_in(hgrads @ probe, multiplier=mul, decay=self.decay)

        in_vec = jax.random.normal(in_key, (in_size,), hgrads.dtype)
        mul = 1 / len(in_vec)
        probe = state.kron_in.ichol @ in_vec
        state = state.update_kron_out(probe @ hgrads, multiplier=mul, decay=self.decay)

        state = state.update_grad_full(
            self.flatten(grads), multiplier=1.0, decay=self.decay
        )
        state = state.renorm()
        # jax.debug.print("inner: {x}", x=jnp.sum(state.grad*state.nat_grad))
        # return -self.lr * state.grad.reshape(grads.shape), state

        return -self.lr * state.nat_grad.reshape(grads.shape), state

    def sample_tangent(self, params, state, key):
        white_noise = jax.random.normal(key, self.flatten(params).shape, params.dtype)
        colored_noise = state.kron_in.ichol @ white_noise @ state.kron_out.ichol.T
        return colored_noise.reshape(params.shape)

    def sample_posterior(self, params, state, key):
        return params + self.sample_tangent(params, state, key)


class TreeOpt(struct.PyTreeNode):
    leaf_opt: LeafOpt = LeafOpt()
    is_leaf: Optional[Callable[[Any], bool]] = None

    def init(self, params):
        key_tree = random_split_like_tree(
            self.leaf_opt.init_key, params, is_leaf=self.is_leaf
        )

        def init_leaf(param, key):
            return self.leaf_opt.replace(init_key=key).init(param)

        return tree_map(init_leaf, params, key_tree, is_leaf=self.is_leaf)

    def update(self, grads, state, params=None, *, hgrads, **kwargs):
        del kwargs
        updates_state = tree_map(
            self.leaf_opt.update, grads, state, params, hgrads, is_leaf=self.is_leaf
        )
        updates = tree_map(
            lambda grad, tup: tup[0], grads, updates_state, is_leaf=self.is_leaf
        )
        state = tree_map(
            lambda grad, tup: tup[1], grads, updates_state, is_leaf=self.is_leaf
        )
        return updates, state

    def sample_posterior(self, params, state, key):
        keys = random_split_like_tree(key, params, is_leaf=self.is_leaf)
        return tree_map(
            self.leaf_opt.sample_posterior, params, state, keys, is_leaf=self.is_leaf
        )

    def sample_tangent(self, params, state, key):
        keys = random_split_like_tree(key, params, is_leaf=self.is_leaf)
        return tree_map(
            self.leaf_opt.sample_tangent, params, state, keys, is_leaf=self.is_leaf
        )

    def get_metrics(self, params, state):
        return tree_map(self.leaf_opt.get_metrics, params, state, is_leaf=self.is_leaf)

    def pin_prior(self, params, state):
        return tree_map(self.leaf_opt.pin_prior, params, state, is_leaf=self.is_leaf)


class Task(struct.PyTreeNode):
    """lossfn: (label, y) -> scalar"""

    x: Any
    label: Any
    lossfn: Callable = struct.field(pytree_node=False)

    def loss(self, y):
        return self.lossfn(self.label, y)


def value_grad_hvp(fn, primals, tangents, **kwargs):
    """Returns *(value, grad, hvp). kwargs are forwarded to value_and_grad.
    e.g. with has_aux=True, output is *((value, aux), grad, hvp)"""
    value_and_grad = jax.value_and_grad(fn, **kwargs)

    def grad_and_value(*primals):
        value, grad = jax.value_and_grad(fn, **kwargs)(*primals)
        return grad, value

    grad, hvp, value = jax.jvp(grad_and_value, primals, tangents, has_aux=True)
    return value, grad, hvp


def universal_grad_hgrad(y, task, key, w=1.0):
    @jax.grad
    def pgrad(y, probe):
        return w * neural.hperturb(task.loss, elementwise=False)(key, probe, y)

    curried = partial(pgrad, y)
    grad, hgrad = jax.jvp(curried, primals=(0.0,), tangents=(1.0,))
    return grad, hgrad


def softmax_grad_hgrad(y, task, key, w=1.0):
    grad = w * jax.grad(task.loss)(y)
    hess_diag_bound = w * jax.nn.softmax(y)
    noise = jax.random.normal(key, y.shape, y.dtype)
    hgrad = noise * jnp.sqrt(hess_diag_bound)
    return grad, hgrad


class Stepper(struct.PyTreeNode):
    """grad_hgrad: y, task, key -> y_grad, y_hgrad_sample"""

    varinf: bool = True
    paired_varinf: bool = True
    hess_only_varinf: bool = False
    per_example_varinf_grad: bool = False
    per_example_varinf: bool = False
    varinf_sample_scale: float = 1.0
    grad_hgrad: Callable[[Any, Task, PRNGKey], Any] = universal_grad_hgrad

    @partial(jax.jit, static_argnums=0, static_argnames="wandb_config")
    def params_grad_hgrad(
        self, train_state, tasks, weights, key, scale=1.0, *, wandb_config=True
    ):
        curvature = wandb.config.curvature if wandb_config else "UBAH"
        use_dropout = wandb.config.use_dropout if wandb_config else False
        if curvature == "EMP_FISH":

            @jax.grad
            def sharded_grad(params):
                import jax.experimental.shard_map
                from jax.experimental.shard_map import shard_map
                from jax.sharding import PartitionSpec as P, Mesh
                @partial(
                    jax.experimental.shard_map.shard_map,
                    mesh=Mesh(jax.devices(), axis_names=("gpus",)),
                    in_specs=(P("gpus", None, None, None), P("gpus"), P("gpus")),
                    out_specs=P(None), check_rep=False)
                def total_loss(xs, labels, weights):
                    ys = train_state.apply_fn(dict(params=params), xs)
                    losses = jax.vmap(tasks.lossfn)(labels, ys)
                    loss = jnp.sum(losses * weights)
                    loss = jax.lax.psum(loss, 'gpus')
                    return loss

                return total_loss(tasks.x, tasks.label, weights)

            @jax.grad
            def direct_grad(params):
                losses = jax.vmap(tasks.lossfn)(
                    tasks.label, train_state.apply_fn(dict(params=params), tasks.x)
                )
                return jnp.sum(losses * weights)

            #grads = direct_grad(train_state.params)
            grads = sharded_grad(train_state.params)
            return grads, grads, {}
        params_key, nonlin_key, loss_key, dropout_key = jax.random.split(key, 4)

        # @partial(jax.experimental.maps.xmap,
        #    in_axes=({0: 'batch'}, {0: 'batch'}, {}),
        #    out_axes={}
        #    )
        @partial(jax.vmap, in_axes=(0, 0, None), out_axes=None, axis_name="batch")
        def probed_grad(tasks, weights, probe):
            def pred(params):
                probes = tree_map(
                    partial(jnp.full_like, fill_value=probe * wandb.config.ubah_mul),
                    train_state.probes,
                )
                variables = {"params": params, "batch_stats": train_state.batch_stats}
                if curvature == "UBAH":
                    variables = {**variables, "probes": probes}
                rngs = dict(nonlin=nonlin_key)
                if use_dropout:
                    local_dropout_key = jax.random.fold_in(
                        dropout_key, jax.lax.axis_index("batch")
                    )
                    rngs = dict(**rngs, dropout=local_dropout_key)
                y, aux = train_state.apply_fn(
                    variables, tasks.x, mutable="batch_stats", rngs=rngs
                )
                return y, aux

            if self.per_example_varinf:
                local_params_key = jax.random.fold_in(
                    params_key, jax.lax.axis_index("batch")
                )
            else:
                local_params_key = params_key
            params = (
                train_state.sample_posterior(local_params_key, scale_tangent=scale)
                if self.varinf
                else train_state.params
            )

            y, vjpfun, aux = jax.vjp(pred, params, has_aux=True, reduce_axes=("batch",))
            batch_size = y.shape[0]
            # loss_keys = jax.random.split(loss_key, batch_size)
            # y_grad, y_hgrad = jax.vmap(self.grad_hgrad)(y, tasks, loss_keys, w=weights)
            local_loss_key = jax.random.fold_in(loss_key, jax.lax.axis_index("batch"))
            y_grad, y_hgrad = self.grad_hgrad(y, tasks, local_loss_key, w=weights)
            (grad,) = vjpfun(jnp.array(y_grad + probe * y_hgrad, dtype=y.dtype))
            grad = tree_map(lambda arr: jax.lax.psum(arr, "batch"), grad)
            # return grad, {}
            return grad, aux

        curried_probed_grad = partial(probed_grad, tasks, weights)
        # grads, aux = curried_probed_grad(0.0)
        # return grads, grads, aux

        grads, hgrads, aux = jax.jvp(
            curried_probed_grad, primals=(0.0,), tangents=(1.0,), has_aux=True
        )
        return grads, hgrads, aux

    @staticmethod
    @partial(
        jax.experimental.maps.xmap,
        in_axes=({}, {0: "batch"}, {0: "batch"}, {}, {}),
        out_axes={},
    )
    def per_example_grad(train_state, tasks, weights, key, scale):
        key = jax.random.fold_in(key, jax.lax.axis_index("batch"))
        dropout_key, params_key = jax.random.split(key)

        def calc_loss(params):
            variables = {"params": params, "batch_stats": train_state.batch_stats}
            rngs = {}
            if wandb.config.use_dropout:
                rngs = dict(**rngs, dropout=dropout_key)
            y = train_state.apply_fn(variables, tasks.x, mutable=False, rngs=rngs)
            loss = tasks.loss(y) * weights
            return jax.lax.psum(loss, "batch")

        params = train_state.sample_posterior(params_key, scale_tangent=scale)
        grad = jax.grad(calc_loss, reduce_axes=("batch",))(params)
        return grad

    @partial(jax.jit, static_argnums=0, static_argnames="wandb_config")
    def __call__(self, train_state, *args, wandb_config=True):
        start = time()
        scale = self.varinf_sample_scale
        grads, hgrads, aux = self.params_grad_hgrad(
            train_state, *args, scale=scale, wandb_config=wandb_config
        )
        if self.varinf and self.paired_varinf:
            grads2, hgrads2, _ = self.params_grad_hgrad(
                train_state, *args, scale=-scale, wandb_config=wandb_config
            )
            grads = tree_map(lambda a, b: (a + b) / 2, grads, grads2)
            hgrads = tree_map(lambda a, b: (a + b) / 2, hgrads, hgrads2)
        if self.varinf and self.hess_only_varinf:
            grads3, hgrads3, _ = self.params_grad_hgrad(
                train_state, *args, scale=0.0, wandb_config=wandb_config
            )
            grads = grads3
        if self.varinf and self.per_example_varinf_grad:
            grads = self.per_example_grad(
                train_state, *args, scale, wandb_config=wandb_config
            )
            if self.paired_varinf:
                grads2 = self.per_example_grad(
                    train_state, *args, -scale, wandb_config=wandb_config
                )
                grads = tree_map(lambda a, b: (a + b) / 2, grads, grads2)
        grads = jax.block_until_ready(grads)
        end1 = time()
        train_state = train_state.apply_gradients(grads=grads, hgrads=hgrads, **aux)
        train_state = jax.block_until_ready(train_state)
        end2 = time()
        print(f"grad_hgrad: {end1 - start:.3f}s apply_gradients: {end2 - end1:.3f}s")
        return train_state


def step(train_state, task, key, batch=False, varinf=True):
    @partial(jax.grad, has_aux=True)
    def probed_grad(params, probe):
        params_key, nonlin_key, loss_key = jax.random.split(key, 3)
        model_probes = tree_map(
            partial(jnp.full_like, fill_value=probe), train_state.probes
        )
        params = (
            train_state.sample_posterior(params_key, params=params)
            if varinf
            else params
        )
        variables = {
            "params": params,
            "probes": model_probes,
        }
        y, aux = train_state.apply_fn(
            variables, task.x, mutable="probes", rngs={"nonlin": nonlin_key}
        )
        if batch:
            batch_size = y.shape[0]
            keys = jax.random.split(loss_key, batch_size)
            probes = jnp.ones((batch_size,), dtype=probe.dtype) * probe
            vec_loss = jax.vmap(
                lambda t, k, p, y: neural.hperturb(t.loss, elementwise=False)(k, p, y)
            )
            loss = jnp.sum(vec_loss(task, keys, probes, y), axis=0)
        else:
            loss = neural.hperturb(task.loss, elementwise=False)(loss_key, probe, y)
        return loss, aux

    curried = partial(probed_grad, train_state.params)

    (grads, hgrads, aux) = jax.jvp(
        curried, primals=(0.0,), tangents=(1.0,), has_aux=True
    )

    train_state = train_state.apply_gradients(grads=grads, hgrads=hgrads, **aux)
    return train_state


def classify_step(train_state, task, key, varinf=True):
    def probed_grad(params, probe):
        params_key, nonlin_key, loss_key = jax.random.split(key, 3)
        model_probes = tree_map(
            partial(jnp.full_like, fill_value=probe), train_state.probes
        )

        def pred(params):
            if varinf:
                params = train_state.sample_posterior(params_key, params=params)
            variables = {
                "params": params,
                "probes": model_probes,
            }
            return train_state.apply_fn(
                variables, task.x, mutable=False, rngs={"nonlin": nonlin_key}
            )

        y, vjpfun = jax.vjp(pred, params)
        ygrad = jax.grad(compose(jnp.sum, task.loss))(y)
        noise_scale = jnp.sqrt(jax.nn.softmax(y))
        yhgrad = (
            jax.random.normal(loss_key, shape=ygrad.shape, dtype=ygrad.dtype)
            * noise_scale
        )
        (grad,) = vjpfun(ygrad + probe * yhgrad)
        return grad

    curried = partial(probed_grad, train_state.params)

    grads, hgrads = jax.jvp(curried, primals=(0.0,), tangents=(1.0,))
    train_state = train_state.apply_gradients(grads=grads, hgrads=hgrads)
    return train_state


def hvp_step(train_state, task, key, batch_axis=None):
    def calc_loss(params):
        y = train_state.apply_fn({"params": params}, task.x)
        loss = task.loss(y)
        if batch_axis is not None:
            loss = jax.lax.psum(loss, batch_axis)
        return loss

    if batch_axis is not None:
        key = jax.random.fold_in(key, jax.lax.axis_index(batch_axis))
    posterior_key, tangent_key = jax.random.split(key)
    loss, grads, hgrads = value_grad_hvp(
        calc_loss,
        (train_state.sample_posterior(posterior_key),),
        (train_state.sample_tangent(tangent_key),),
    )
    train_state = train_state.apply_gradients(grads=grads, hgrads=hgrads)
    return train_state
