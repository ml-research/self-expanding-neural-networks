from collections.abc import Callable
from typing import Any, Optional, Sequence
from functools import partial
from compose import compose
from math import prod

import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.tree_util import (
    tree_map,
    tree_leaves,
    tree_flatten,
    tree_unflatten,
    tree_reduce,
)
from flax import struct, linen as nn
from flax.core import frozen_dict
from senn.opt import (
    Task,
    TreeOpt,
    TrainState,
    step as opt_step,
    Stepper,
    softmax_grad_hgrad,
    universal_grad_hgrad,
    DiagOpt,
)
import senn.opt
from senn.neural import HPerturb, hperturb

import tensorflow_datasets as tfds

from time import time
from tqdm import tqdm, trange

import wandb


class FakeBatchNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * self.param("gamma", nn.initializers.ones, (1, 1), jnp.float32)


def pad_vars(
    module,
    index,
    length,
    axis=-1,
    from_back=False,
    collection="params",
    init=nn.initializers.zeros,
    filt=lambda arr: True,
):
    def pad(arr, init=init):
        if not filt(arr):
            return arr
        if not (-len(arr.shape) <= axis < len(arr.shape)):
            return arr
        pad_shape = tuple(jnp.array(arr.shape).at[axis].set(length))
        key = module.make_rng("params")
        padding = init(key, pad_shape, arr.dtype)
        split_idx = arr.shape[axis] - index if from_back else index
        prefix, suffix = jnp.split(arr, [split_idx], axis=axis)
        new_arr = jnp.concatenate([prefix, padding, suffix], axis=axis)
        return new_arr

    for key, value in module.variables[collection].items():
        results = tree_map(pad, value)
        module.put_variable(collection, key, tree_map(pad, value))
        if collection == "params" and module.is_mutable_collection("was_padded"):
            # def fake_pad(arr):
            #    arr = jnp.zeros_like(arr, dtype=jnp.bool_)
            #    return pad(arr, init=nn.initializers.ones)

            if module.has_variable("was_padded", key):
                old = module.get_variable("was_padded", key)
            else:
                old = tree_map(lambda a: jnp.zeros_like(a, dtype=jnp.bool_), value)
            was_padded = tree_map(partial(pad, init=nn.initializers.ones), old)

            # was_padded = tree_map(fake_pad, value)
            # if module.has_variable("was_padded", key):
            #    old = module.get_variable("was_padded", key)
            #    was_padded = tree_map(jnp.logical_or, was_padded, old)
            module.put_variable("was_padded", key, was_padded)
    return None


def pad_vars_back(*args, **kwargs):
    return pad_vars(*args, **kwargs, from_back=True)


def pad_dense_inputs_back(mdl, idx, length):
    pad_vars_back(mdl, idx, length, collection="params", axis=-2)
    return None


class DenseLayer(nn.Module):
    linear: nn.Module
    nonlin: Optional[nn.Module]
    norm: Optional[nn.Module]
    in_paddable_collections: Sequence[str] = ("params",)
    out_paddable_collections: Sequence[str] = ("params", "probes")

    @nn.compact
    def __call__(self, fmaps):
        cat = jnp.concatenate(fmaps, axis=-1)
        y = self.linear(cat)
        y = y if self.norm is None else self.norm(y)
        y = y if self.nonlin is None else self.nonlin(y)
        return y


class XLayer(DenseLayer):
    def pad_back_inputs(self, idx, length):
        filt = lambda arr: len(arr.shape) > 1 and arr.shape[-2] > 1
        for col in self.variables.keys():
            if col in self.in_paddable_collections:
                pad_vars_back(self, idx, length, collection=col, filt=filt, axis=-2)
        return None

    def pad_back_outputs(self, idx, length):
        filt = lambda arr: len(arr.shape) > 0 and arr.shape[-1] > 1
        for col in self.variables.keys():
            if col in self.out_paddable_collections:
                pad_vars_back(self, idx, length, collection=col, filt=filt, axis=-1)
        return None

    def out_dim(self):
        def dim(arr):
            return 0 if len(arr.shape) < 1 else arr.shape[-1]

        dims = tree_map(dim, self.linear.variables["params"])
        return tree_reduce(jnp.maximum, dims, 0)

    def zero_params(self):
        for name, tree in self.variables["params"].items():
            new_tree = tree_map(lambda arr: jnp.zeros_like(arr), tree)
            self.put_variable("params", name, new_tree)


class Buddable(nn.Module):
    main: XLayer
    bud: Optional[XLayer]

    def __call__(self, fmaps):
        if self.bud is not None:
            fmaps = fmaps + (self.bud(fmaps),)
        return self.main(fmaps)

    def out_dim(self):
        return self.main.out_dim()

    def pad_back_inputs(self, idx, length):
        if self.bud is not None:
            self.bud.pad_back_inputs(idx, length)
            idx = idx + self.bud.out_dim()
        self.main.pad_back_inputs(idx, length)
        return None

    def pad_back_outputs(self, idx, length):
        self.main.pad_back_outputs(idx, length)

    def bud_reinit_allowed(self, col="allowed"):
        if self.bud is None:
            return None
        for key, param in self.main.variables["params"].items():
            bud_out = self.bud.out_dim()

            def crop(arr):
                if len(arr.shape) < 1 or arr.shape == (1, 1):
                    return arr
                bo_true = jnp.ones(shape=(bud_out,), dtype=jnp.bool_)
                in_ok = jnp.any(arr[..., :-bud_out, :], axis=-1, keepdims=True)
                return in_ok & bo_true

            allowed = self.main.variables[col].get(key)
            self.bud.put_variable(col, key, tree_map(crop, allowed))
        return None

    def new_layer_vars(self):
        mdl, bvars = self.bud.unbind()
        bud_length = self.bud.out_dim()
        _, new_bud = mdl.apply(bvars, method=mdl.zero_params, mutable=True)
        _, new_main = mdl.apply(
            bvars,
            idx=0,
            length=bud_length,
            method=mdl.pad_back_inputs,
            mutable=True,
            rngs=dict(params=self.make_rng("params")),
        )
        merged = frozen_dict.freeze(
            {
                col: {"bud": new_bud[col], "main": new_main[col]}
                for col in self.variables.keys()
            }
        )
        return merged

    def zero_bud(self):
        self.bud.zero_params()


def expandable_conv(features, **kwargs):
    return nn.Conv(
        features=features,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        use_bias=False,
        **kwargs,
    )


def expandable_pool(x):
    return nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")


class Block(nn.Module):
    layers: Sequence[Buddable]

    @nn.compact
    def __call__(self, x):
        fmaps = (x,)
        for layer in self.layers[:-1]:
            fmaps = fmaps + (layer(fmaps),)
        for layer in self.layers[-1:]:
            x = layer(fmaps)
        return x


class XBlock(Block):
    def pad_intermediate(self, idx, length):
        assert idx + 1 < len(self.layers)
        self.layers[idx].pad_back_outputs(0, length)
        index = 0
        for layer in self.layers[idx + 1 :]:
            layer.pad_back_inputs(index, length)
            index += layer.out_dim()
        return None

    def pad_final(self, length):
        self.layers[-1].pad_back_outputs(0, length)
        return None

    def pad_inputs(self, length):
        index = 0
        for layer in self.layers:
            layer.pad_back_inputs(index, length)
            index += layer.out_dim()
        return None

    def shift_old_to_new(self, idx):
        def maybe_write(source, dest, name, new_name=None):
            new_name = name if new_name is None else new_name
            if source.has_variable("old", name):
                v = source.get_variable("old", name)
                dest.put_variable("new", new_name, v)

        for i, (layer, next_layer) in enumerate(zip(layers, layers[1:])):
            if i < idx:
                maybe_write(layer, layer, "main")
                maybe_write(layer, layer, "bud")
            if i == idx:
                maybe_write(layer, layer, "bud", new_name="main")
                maybe_write(layer, next_layer, "main")
            if i > idx:
                maybe_write(layer, next_layer, "main")
                maybe_write(layer, next_layer, "bud")
        return None

    def activate_bud(self, idx):
        # collections: was_padded, params, probes
        new_layer = self.layers[idx].new_layer_vars()
        length = self.layers[idx].bud.out_dim()
        index = 0
        for layer in self.layers[idx:]:
            layer.pad_back_inputs(index, length)
            index += layer.out_dim()
        self.layers[idx].zero_bud()
        widths = list(layer.out_dim() for layer in self.layers)
        new_widths = widths[:idx] + [length] + widths[idx:]
        return new_widths, new_layer

    def insert_layer_vars(self, idx, new_layer):
        name = lambda i: f"layers_{i}"
        for i, layer in enumerate(self.layers[idx:]):
            for col in layer.variables.keys():
                self.put_variable(col, name(i + 1), self.get_variable(col, name(i)))
        for col, variable in new_layer.variables.items():
            self.put_variable(col, name(idx), variable)
        return None


def width_to_add(module):
    if "add_width" in module.variables:
        return tree_reduce(jnp.maximum, module.variables["add_width"], 0)
    else:
        return 0


class ExpandableDense(nn.Module):
    blocks: Sequence[XBlock]
    final: nn.Module

    @nn.nowrap
    @classmethod
    def build(cls, *, out, nonlin, widthss, maybe_bud_width=None):
        def layer(w):
            # norm = FakeBatchnorm
            # norm = IdentityModule
            return XLayer(linear=expandable_conv(w), nonlin=HPerturb(nonlin), norm=None)

        def bud():
            return None if maybe_bud_width is None else layer(maybe_bud_width)

        def buddable(w):
            return Buddable(main=layer(w), bud=bud())

        def block(widths):
            return XBlock(list(map(buddable, widths)))

        blocks = tuple(map(block, widthss))
        final = nn.Dense(out, use_bias=False)
        return cls(blocks, final)

    @nn.compact
    def __call__(self, x):
        pool = expandable_pool
        for block in self.blocks:
            x = block(x)
            x = pool(x)
        x = jnp.mean(x, axis=(-2, -3))
        x = self.final(x)
        return x

    def layer_widths(self):
        return tuple(
            tuple(layer.out_dim() for layer in block.layers) for block in self.blocks
        )

    def maybe_add_width(self):
        old_widths = self.layer_widths()
        for i, block in enumerate(self.blocks):
            for j, layer in enumerate(block.layers):
                to_add = width_to_add(layer.main.linear)
                if to_add <= 0:
                    continue
                jax.debug.print("adding {} neurons to block {}, layer {}", to_add, i, j)
                if j + 1 < len(block.layers):
                    self.blocks[i].pad_intermediate(j, to_add)
                elif j + 1 == len(block.layers) and i + 1 < len(self.blocks):
                    self.blocks[i].pad_final(to_add)
                    self.blocks[i + 1].pad_inputs(to_add)
                elif j + 1 == len(block.layers) and i + 1 == len(self.blocks):
                    self.blocks[i].pad_final(to_add)
                    pad_dense_inputs_back(self.final, 0, to_add)
                    # do not expand final layer
                    continue
        new_widths = self.layer_widths()
        if new_widths == old_widths:
            return None
        else:
            return new_widths
        # for col, value in self.variables.items():
        #    if self.is_mutable_collection(col):
        #        for key, variable in value.items():
        #            self.put_variable(col, key, variable)
        # if (to_add := width_to_add(self.dense)) > 0:
        #    pad_dense_inputs_back(self.dense, 0, to_add)
        #    self.blocks[-1].pad_final(to_add)

    def bud_reinit_allowed(self):
        for block in self.blocks:
            for layer in block.layers:
                layer.bud_reinit_allowed()

    def activate_bud(self, block_idx, layer_idx):
        # collections: params, probes, was_padded
        old_widths = self.layer_widths()
        widths, new_layer = self.blocks[block_idx].activate_bud(layer_idx)
        out_w = old_widths[:block_idx] + (widths,) + old_widths[block_idx + 1 :]
        return out_w, new_layer

    @nn.nowrap
    def insert_into_tree(self, tree, bidx, lidx, item):
        bname = lambda i: f"blocks_{i}"
        lname = lambda i: f"layers_{i}"
        new_block = {lname(lidx): item}
        old_block = tree[bname(bidx)]
        N = len(old_block.keys())
        for i in range(0, lidx):
            new_block[lname(i)] = old_block[lname(i)]
            # new_block = cursor(new_block)[lname(i)].set(tree[lname(i)])
        for i in range(lidx, N):
            new_block[lname(i + 1)] = old_block[lname(i)]
            # new_block = cursor(new_block)[lname(i+1)].set(tree[lname(i)])
        # return cursor(tree)[bname(bidx)].set(new_block)
        new_block = frozen_dict.freeze(new_block)
        return tree.copy({bname(bidx): new_block})

    def argmax_score(self, ignore_first=True):
        best = -jnp.inf
        coords = -1, -1
        for bidx, block in enumerate(self.blocks):
            if len(block.layers) >= wandb.config.block_size_hard_cap:
                continue
            for lidx, layer in enumerate(block.layers):
                if ignore_first and (bidx, lidx) == (0, 0):
                    continue
                score = layer.main.linear.get_variable("score", "kernel", -jnp.inf)
                if score > best:
                    best = score
                    coords = bidx, lidx
        return best, coords
