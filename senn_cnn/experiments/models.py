from collections.abc import Callable
from typing import Any, Optional, Sequence
from functools import partial
from compose import compose
from math import prod

import jax
from jax import numpy as jnp
import numpy as np
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


class FakeBatchNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * self.param("gamma", nn.initializers.ones, (1, 1), jnp.float32)


class SmallConvNet(nn.Module):
    out: int
    hidden: int = 32
    depth: int = 3
    dense_size: int = 128
    multiplicity: int = 1
    nonlin: Callable = jnp.tanh
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        pool = partial(nn.max_pool, window_shape=(2, 2), strides=(2, 2))

        def pix_flat(x):
            x = x.reshape(x.shape[:-3] + (prod(x.shape[-3:]),))
            return x

        def pix_mean(x):
            return jnp.mean(jnp.mean(x, axis=-2), axis=-2)

        for i in range(self.depth):
            for j in range(self.multiplicity):
                kernel_size = (7,) * 2 if i == 0 and j == 0 else (3,) * 2
                x = nn.Conv(self.hidden, kernel_size, use_bias=self.use_bias)(x)
                x = HPerturb(self.nonlin)(x)
            x = pool(x)
        x = pix_flat(x)
        x = nn.Dense(self.dense_size, use_bias=self.use_bias)(x)
        x = HPerturb(self.nonlin)(x)
        x = nn.Dense(self.out, use_bias=self.use_bias)(x)
        return x


class Perceptron(nn.Module):
    out: int
    hidden: int = 32
    depth: int = 2
    nonlin: Callable = jnp.tanh
    flatten_last_n: int = 1
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        assert self.flatten_last_n > 0
        x = x.reshape(
            x.shape[: -self.flatten_last_n] + (prod(x.shape[-self.flatten_last_n :]),)
        )
        for i in range(self.depth):
            x = nn.Dense(self.hidden, use_bias=self.use_bias)(x)
            x = HPerturb(self.nonlin)(x)
            # x = self.nonlin(x)
        x = nn.Dense(self.out, use_bias=self.use_bias)(x)
        return x


class AllCnnA(nn.Module):
    out: int
    nonlin: Callable = jax.nn.swish
    use_bias: bool = False
    early_channels: int = 96
    late_channels: int = 192

    @nn.compact
    def __call__(self, x):
        pool = lambda x: nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        conv = partial(nn.Conv, strides=(1, 1), padding="SAME", use_bias=self.use_bias)
        x = nn.Dropout(rate=0.2, deterministic=not self.has_rng("dropout"))(x)
        for channels in (self.early_channels, self.late_channels):
            x = conv(channels, (5, 5))(x)
            x = HPerturb(self.nonlin)(x)
            x = pool(x)
            x = nn.Dropout(rate=0.5, deterministic=not self.has_rng("dropout"))(x)
        x = conv(self.late_channels, (3, 3))(x)
        x = HPerturb(self.nonlin)(x)
        x = conv(self.late_channels, (1, 1))(x)
        x = HPerturb(self.nonlin)(x)
        x = conv(self.out, (1, 1))(x)
        x = HPerturb(self.nonlin)(x)
        x = jnp.mean(x, axis=(-2, -3))
        return x


class AllCnnC(nn.Module):
    out: int
    nonlin: Callable = jax.nn.swish
    use_bias: bool = False
    early_channels: int = 96
    late_channels: int = 192
    fake_batch_norm: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        del train
        pool = lambda x: nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        conv = partial(nn.Conv, strides=(1, 1), padding="SAME", use_bias=self.use_bias)
        norm = FakeBatchNorm if self.fake_batch_norm else lambda: lambda x: x
        x = nn.Dropout(rate=0.1, deterministic=not self.has_rng("dropout"))(x)
        for channels in (self.early_channels, self.late_channels):
            x = conv(channels, (3, 3))(x)
            x = norm()(x)
            x = HPerturb(self.nonlin)(x)
            x = conv(channels, (3, 3))(x)
            x = norm()(x)
            x = HPerturb(self.nonlin)(x)
            x = pool(x)
            x = nn.Dropout(rate=0.1, deterministic=not self.has_rng("dropout"))(x)
        x = conv(self.late_channels, (3, 3))(x)
        x = norm()(x)
        x = HPerturb(self.nonlin)(x)
        x = conv(self.late_channels, (1, 1))(x)
        x = norm()(x)
        x = HPerturb(self.nonlin)(x)
        x = conv(self.out, (1, 1))(x)
        x = norm()(x)
        x = HPerturb(self.nonlin)(x)
        x = jnp.mean(x, axis=(-2, -3))
        return x


from jaxwt.conv_fwt_2d import wavedec2, waverec2


class WaveletConv(nn.Module):
    wavelet: str = "haar"

    @nn.compact
    def __call__(self, x):
        pass


class TreeDense(nn.Module):
    hidden: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.Dense, self.hidden, use_bias=self.use_bias)
        leaves, treedef = tree_flatten(x)
        leaves = list(dense()(leaf) for leaf in leaves)
        x = tree_unflatten(treedef, leaves)
        return x

class StackedTreeDense(nn.Module):
    hidden: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        leaves, treedef = tree_flatten(x)
        in_features = leaves[0].shape[-1]
        kernel_shape = (len(leaves), in_features, self.hidden)
        kernel = self.param('kernel', nn.initializers.lecun_normal(), kernel_shape)
        leaves = list(leaf @ kern for leaf, kern in zip(leaves, kernel))
        x = tree_unflatten(treedef, leaves)
        return x

class SeparableStackedTreeDense(nn.Module):
    hidden: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        leaves, treedef = tree_flatten(x)
        in_features = leaves[0].shape[-1]
        feature_shape = (in_features, self.hidden)
        spatial_shape = (len(leaves), self.hidden)
        fkernel = self.param('fkernel', nn.initializers.lecun_normal(), feature_shape)
        skernel = self.param('skernel', nn.initializers.lecun_normal(), spatial_shape)

        leaves = list((leaf @ fkernel) * kern for leaf, kern in zip(leaves, skernel))
        x = tree_unflatten(treedef, leaves)
        return x


class WaveletNonlin(nn.Module):
    nonlin: Callable = jax.nn.swish
    wavelet: str = "haar"

    @nn.compact
    def __call__(self, x):
        level


class WaveletNet(nn.Module):
    out: int
    hidden: int = 64
    depth: int = 3
    wavelet: str = "haar"
    nonlin: Callable = jax.nn.swish
    level: int = 3
    downsample: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        del train

        # wdec = jax.vmap(partial(wavedec2, wavelet=self.wavelet), -1, -1)
        def wdec(x, axes=(-2, -3)):
            x = jnp.swapaxes(x, -1, -3)
            x = wavedec2(x, wavelet=self.wavelet, level=self.level)
            x = tree_map(lambda x: jnp.swapaxes(x, -1, -3), x)
            return x

        def wrec(x, axes=(-2, -3)):
            x = tree_map(lambda x: jnp.swapaxes(x, -1, -3), x)
            if self.downsample:
                x = x[:-1]
            x = waverec2(x, wavelet=self.wavelet)
            x = jnp.swapaxes(x, -1, -3)
            return x

        # wdec = partial(wavedec2, wavelet=self.wavelet, axes=(-2, -3), level=self.level)
        # wrec = jax.vmap(partial(waverec2, wavelet=self.wavelet), -1, -1)
        # wrec = partial(waverec2, wavelet=self.wavelet, axes=(-2, -3))
        def nonlin(x):
            return wdec(HPerturb(self.nonlin)(wrec(x)))

        x = wdec(x)
        for i in range(self.depth):
            x = TreeDense(self.hidden, use_bias=False, name=f"Wvt_{i}")(x)
            x = nonlin(x)
        # select lowest frequency element in decomposition
        y = jnp.mean(x[0], axis=(-2, -3))
        y = nn.Dense(self.out, use_bias=False)(y)
        return y

class WaveConv(struct.PyTreeNode):
    hidden: int
    levels: int = 4

    def __call__(self, x):
        decompose = partial(wavelet.decompose_2D, levels=self.levels)
        out_shape = x.shape[:-1] + (self.hidden,)
        z = decompose(x)
        z = SeparableStackedTreeDense(self.hidden)(z)
        (x,) = jax.linear_transpose(decompose, jnp.zeros(out_shape))(z)
        return x

class FourConv(struct.PyTreeNode):
    hidden: int
    levels: int = None

    def __call__(self, x):
        z = jnp.fft.rfft2(x, axes=(-3, -2))
        print(z.shape)
        exit()

class WaveNet(nn.Module):
    out: int
    widths: Sequence[int]
    nonlin: Callable = jax.nn.swish
    levels: int = 4

    @nn.compact
    def __call__(self, x):
        pool = partial(nn.max_pool, strides=(2, 2), padding="SAME")
        for w in self.widths:
            min_width = min(x.shape[-2], x.shape[-3])
            max_levels = int(np.ceil(np.log2(min_width)))
            x = WaveConv(hidden=w, levels=min(max_levels, self.levels))(x)
            x = HPerturb(self.nonlin)(x)
            #x = pool(x, window_shape=(2,2))
        y = jnp.mean(x, axis=(-2, -3))
        y = nn.Dense(self.out, use_bias=False)(y)
        return y

    def maybe_add_width(self, x):
        raise NotImplementedError


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


def reinit_vars(
    module,
    where=None,
    init=nn.initializers.lecun_normal,
    collection="params",
    filt=lambda arr: True,
):
    def reinit(arr, where=where, init=init):
        if not filt(arr):
            return arr
        if not len(where.shape) <= len(arr.shape):
            return arr
        key = module.make_rng("init")
        new_vals = init(key, arr.shape, arr.dtype)
        new_arr = jnp.where(where, new_vals, arr)
        return new_arr

    for key, value in module.variables[collection].items():
        if where is None:
            where = module.variables["can_reinit"][key]
            new_vars = tree_map(reinit, value, where)
        else:
            new_vars = tree_map(reinit, value)
        module.put_variable(collection, key, new_vars)
        if collection == "params" and module.is_mutable_collection("was_reinitialized"):

            def fake_reinit(arr):
                arr = jnp.zeros_like(arr, dtype=jnp.bool_)
                return reinit(arr, init=nn.initializer.ones)

            was_reinitialized = tree_map(fake_reinit, value)
            if module.has_variable("was_reinitialized", key):
                old = module.get_variable("was_reinitialized", key)
                was_reinitialized = tree_map(jnp.logical_or, was_reinitialized, old)
            module.put_variable("was_reinitialized", key, was_reinitialized)
    return None


def reinit_outputs(module, init=nn.initializers.lecun_normal):
    def reinit(param, allowed):
        pass  # maybe pass new_params explicitly and just choose?


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


class DenseBlock(nn.Module):
    linear: Any
    depth: int = 12
    nonlin: Callable = jax.nn.swish

    @nn.compact
    def __call__(self, x):
        featmaps = (x,)
        for i in range(self.depth):
            y = DenseLayer(self.linear, self.nonlin)(featmaps)
            # y = nn.remat(DenseLayer)(self.linear, self.nonlin)(featmaps)
            featmaps = featmaps + (y,)
        return jnp.concatenate(featmaps, axis=-1)


class DenseNet(nn.Module):
    out: int
    growth: int = 12
    depth: int = 12
    blocks: int = 3
    input_width: int = 16
    nonlin: Callable = jax.nn.swish
    use_bias: bool = False
    init_conv: bool = True

    @nn.compact
    def __call__(self, x):
        conv = partial(
            nn.Conv,
            features=self.growth,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.use_bias,
        )
        pool = partial(nn.max_pool, strides=(2, 2), padding="SAME")

        if self.init_conv:
            x = conv(
                features=self.input_width,
                kernel_size=(7, 7),
                strides=(2, 2),
                name="conv_init",
            )(x)
            x = HPerturb(self.nonlin)(x)
            x = pool(x, (3, 3))
        for block in range(self.blocks):
            x = DenseBlock(conv, depth=self.depth, nonlin=self.nonlin)(x)
            x = pool(x, (2, 2))
        x = jnp.mean(x, axis=(-2, -3))
        x = nn.Dense(self.out, use_bias=self.use_bias)(x)
        return x


class BottleneckDense(nn.Module):
    out: int
    blocks: int = 3
    width: int = 64
    extra_depth: int = 0
    nonlin: Callable = jax.nn.swish
    use_bias: bool = False
    maybe_bud_width: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        conv = partial(
            nn.Conv,
            features=self.width,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=self.use_bias,
        )
        pool = lambda x: nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        if self.maybe_bud_width is None:
            maybe_bud = None
        else:
            maybe_bud = partial(
                DenseLayer,
                linear=partial(conv, features=self.maybe_bud_width),
                nonlin=self.nonlin,
                maybe_bud=None,
            )

        layer = partial(
            DenseLayer, linear=conv, nonlin=self.nonlin, maybe_bud=maybe_bud
        )

        for block in range(self.blocks):
            fmaps = (x,)
            for i in range(self.extra_depth):
                fmaps = fmaps + (layer()(fmaps),)
            x = layer()(fmaps)
            x = pool(x)
        x = jnp.mean(x, axis=(-2, -3))
        x = nn.Dense(self.out, use_bias=self.use_bias)(x)
        return x


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


def expandable_builder(widthss, maybe_bud_width=None, **kwargs):
    conv = expandable_conv
    if (mw := self.maybe_bud_width) is None:
        maybe_bud = None
    else:
        maybe_bud = partial(
            DenseLayer,
            linear=partial(conv, features=maybe_bud_width),
            maybe_bud=None,
            **kwargs,
        )

    def layer(w):
        return DenseLayer(
            linear=partial(conv, features=w), maybe_bud=maybe_bud, **kwargs
        )

    blocks = list(
        ExpandableBlock(list(layer(w) for w in widths), **kwargs) for widths in widthss
    )


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

    def copy_bud(self, idx, col):
        def name(idx):
            return f"layers_{idx}"

        for layer, prev in zip(self.layers[idx + 1 :], self.layers[idx:]):
            layer.put_variable(col, "main", prev.get_variable(col, "main"))
            layer.put_variable(col, "bud", prev.get_variable(col, "bud"))
        for i, layer in range(len(self.layers)):
            if i > idx:
                v = self.get_variable(col, name(i - 1))
                self.put_variable(col, name(i), v)
            if i == idx:
                raise NotImplementedError

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


class IdentityModule(nn.Module):
    def __call__(self, x):
        return x


class ExpandableDense_DISABLE(nn.Module):
    blocks: Sequence[XBlock]
    final: nn.Module

    @nn.nowrap
    @classmethod
    def build(cls, *, out, nonlin, widthss, maybe_bud_width=None):
        def layer(w):
            # norm = FakeBatchnorm
            norm = IdentityModule
            return XLayer(
                linear=expandable_conv(w), nonlin=HPerturb(nonlin), norm=norm()
            )

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
        return cursor(old_widths)[block_idx].set(widths), new_layer

    @nn.nowrap
    def insert_into_tree(self, tree, bidx, lidx, item):
        bname = lambda i: f"blocks_{i}"
        lname = lambda i: f"layers_{i}"
        new_block = {lname(lidx): item}
        old_block = tree[bname(bidx)]
        N = len(old_block.keys())
        for i in range(0, lidx):
            new_block = cursor(new_block)[lname(i)].set(tree[lname(i)])
        for i in range(lidx, N + 1):
            new_block = cursor(new_block)[lname(i + 1)].set(tree[lname(i)])
        return cursor(tree)[bname(bidx)].set(new_block)
