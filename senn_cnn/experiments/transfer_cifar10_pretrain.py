CHECKPOINT_DIR = "/senn/orbax/pretrained/final"

from collections.abc import Callable
from functools import partial
from compose import compose
from math import prod

import jax
import tensorflow as tf
from jax import numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax.tree_util import tree_map, tree_leaves
from flax import struct, linen as nn
from typing import Any
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
import sklearn
from tensorflow_probability.substrates import jax as tfp
import dm_pix as pix

from time import time
from tqdm import tqdm, trange
import os
import sys

import wandb
from rtpt import RTPT
import copy

import models

from flax.training import orbax_utils
import orbax.checkpoint
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

calib_err_bins: Any = jax.nn.sigmoid(-1.0 * jnp.arange(-2, 12))


class CalibCount(struct.PyTreeNode):
    err: float
    correct: int
    incorrect: int

    @classmethod
    def count(cls, logits, labels):
        active_idx = jax.vmap(jnp.argmax)(logits)
        active_err = jax.vmap(lambda x: 1.0 - jnp.max(jax.nn.softmax(x)))(logits)
        correct = active_idx == labels
        incorrect = active_idx != labels
        bin_idx = jnp.digitize(active_err, calib_err_bins)
        idxs = jnp.arange(len(calib_err_bins) + 1)

        def count_idx(idx):
            where = bin_idx == idx
            return cls(
                jnp.sum(active_err, where=where),
                jnp.sum(correct, where=where),
                jnp.sum(incorrect, where=where),
            )

        return jax.vmap(count_idx)(idxs)

    def plot(self, title="Calibration"):
        percentiles = [50, 5, 95]
        keys = list(map(lambda p: f"{p}%", percentiles))
        xs = list(map(float, self.err / (self.correct + self.incorrect)))

        def make_ys(percentile):
            return list(
                map(
                    float,
                    tfp.math.betaincinv(
                        1.0 * self.incorrect, 1.0 * self.correct, percentile / 100.0
                    ),
                )
            )

        ys = list(map(make_ys, percentiles))
        # print(len(xs))
        # print(xs)
        # print(len(ys))
        # print(ys)
        return wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=title, xname="err")


def label_filter(filt, *item):
    img, label = item
    return filt[label]


def ymetrics(task, ys):
    loss = jnp.sum(task.loss(ys))
    argmax = jax.vmap(jnp.argmax)(ys)
    ground_truth = jnp.mod(task.label, wandb.config.num_classes)
    correct = argmax == ground_truth
    acc = jnp.sum(correct)
    err = jnp.sum(~correct)
    return dict(
        loss=loss, accuracy=acc, err=err, calib=CalibCount.count(ys, task.label)
    )


def eval_batch(key, task, train_state):
    out = dict()
    ys = train_state.eval(task)
    out = dict(max_a_posteriori=ymetrics(task, ys))
    if wandb.config.only_max_a_posteriori_eval or wandb.config.fast_turbo:
        return out
    mys = train_state.eval_marginalized(task, key, wandb.config.eval_samples)
    rys = train_state.eval(task, key=key)
    return dict(**out, marginalized=ymetrics(task, mys), sampled=ymetrics(task, rys))
    #return dict(
    #    max_a_posteriori=ymetrics(task, ys),
    #    marginalized=ymetrics(task, mys),
    #    sampled=ymetrics(task, rys),
    #)


def eval_one_epoch(name, dataset, train_state):
    filters = wandb.config.eval_filters
    for fidx, filt in enumerate(filters):
        if filt is None:
            # fname, fdataset = name, dataset
            fname = name
        else:
            fname = f"{name}_F{int(fidx)}"
            # fdataset = dataset.filter(partial(label_filter, tf.convert_to_tensor(np.array(filt))))
        _eval_one_epoch(fname, dataset, train_state, filt=filt)


def filtered_and_size(dataset, filt=None):
    dataset_size = dataset.cardinality().numpy()
    if False and filt is not None:
        fdataset = dataset.filter(
            partial(label_filter, tf.convert_to_tensor(np.array(filt)))
        )
        fsize = dataset_size * np.sum(np.array(filt)) / len(filt)
    else:
        fdataset, fsize = dataset, dataset_size
    return fdataset, fsize


def _eval_one_epoch(name, dataset, train_state, filt=None):
    fdataset, dataset_size = filtered_and_size(dataset, filt)
    key = PRNGKey(0)
    batched = fdataset.batch(wandb.config.eval_batch_size, drop_remainder=True)
    tasks = as_task_iter(batched)
    metrics = None
    items_seen = 0
    for task in tasks:
        items_seen += len(task.label)
        new_metrics = eval_batch(key, task, train_state)
        metrics = (
            new_metrics if metrics is None else tree_map(jnp.add, metrics, new_metrics)
        )
    is_leaf = lambda x: isinstance(x, CalibCount)

    def finalize(leaf):
        if isinstance(leaf, CalibCount):
            return leaf.plot()
        else:
            return leaf / items_seen

    # metrics = tree_map(finalize, metrics, is_leaf=is_leaf)
    name_metrics = dict()
    for key, value in metrics.items():
        calib_count = value.pop("calib")
        name_metrics.update(
            {key: tree_map(lambda a: a / items_seen, copy.deepcopy(value))}
        )
        # wandb.log({name: {key: tree_map(lambda a: a/items_seen, copy.deepcopy(value))}}, commit=False)
        plot_name = f"{name}_{key}_calib"
        if wandb.config.log_calibration:
            wandb.log({plot_name: calib_count.plot(title=plot_name)}, commit=False)
    # metrics = tree_map(lambda a: a/items_seen, metrics)
    wandb.log({name: name_metrics}, commit=False)


def eval_final(name, dataset, train_state):
    eval_one_epoch(name, dataset, train_state)


def get_train_weight():
    if wandb.config.burn_in_period is None:
        w = 1.0 / wandb.config.temperature
    else:
        epoch = wandb.run.summary.get("epoch", 0)
        # total_epochs = wandb.config.epochs
        # centered = epoch - total_epochs/2

        period = wandb.config.burn_in_period
        initial = wandb.config.burn_in_initial_weight
        lninit = jnp.log(initial)
        t = jnp.minimum(epoch, period) / period
        lnw = lninit * (1.0 - t)
        w = jnp.exp(lnw) / wandb.config.temperature
        # scaled = 2 * width * centered / (total_epochs)
        # w = jax.nn.sigmoid(scaled) / wandb.config.temperature
        wandb.log(dict(train_weight=w), commit=False)
    return w


def augment_img(img, key):
    if wandb.config.timnet_aug:
        return tiny_imagenet_augment_img(img, key)
    flip_key, trans_key = jax.random.split(key)
    img = pix.random_flip_left_right(flip_key, img)
    trans = jax.random.randint(trans_key, shape=(2,), minval=-5, maxval=5)
    affine = jnp.identity(4)
    affine = affine.at[:2, -1].set(trans)
    # affine = jnp.concatenate((jnp.identity(2), trans[:,None]), axis=-1)
    img = pix.affine_transform(img, affine)
    return img

def tiny_imagenet_augment_img(img, key):
    initial_shape = img.shape
    flip_key, crop_key, rotate_key, scale_key, color_key, shortcut_key = jax.random.split(key, 6)
    img = pix.random_flip_left_right(flip_key, img)
    only_flipped = img
    padto = wandb.config.augment_timnet_padtowidth
    img = pix.pad_to_size(img, padto, padto)
    max_radians = 2*jnp.pi / 18
    if wandb.config.augment_timnet_rotate:
        angle_key, rbool_key = jax.random.split(rotate_key)
        angle = jax.random.uniform(angle_key, minval=-max_radians, maxval=max_radians)
        do_rotation = jax.random.bernoulli(rbool_key, p=wandb.config.augment_timnet_rotate_prob)
        img = jnp.where(do_rotation, pix.rotate(img, angle), img)
    if wandb.config.augment_timnet_scale:
        scale_key, sbool_key = jax.random.split(scale_key)
        scale = jnp.exp(jax.random.uniform(scale_key, minval=jnp.log(0.8), maxval=jnp.log(1.2)))
        do_scale = jax.random.bernoulli(sbool_key, p=wandb.config.augment_timnet_scale_prob)
        scaled = pix.affine_transform(img, jnp.array([scale, scale, 1.]))
        img = jnp.where(do_scale, scaled, img)
    img = pix.random_crop(crop_key, img, initial_shape)

    if wandb.config.augment_color:
        sat_key, bri_key, con_key, cbool_key = jax.random.split(color_key, 4)
        cimg = img
        cimg = pix.random_saturation(sat_key, cimg, 0.8, 1.3)
        cimg = pix.random_brightness(bri_key, cimg, 0.3)
        cimg = pix.random_contrast(con_key, cimg, 0.8, 1.3)
        do_color = jax.random.bernoulli(cbool_key, p=wandb.config.augment_timnet_color_prob)
        img = jnp.where(do_color, cimg, img)
    if (shortcut_prob := wandb.config.augment_timnet_shortcut_prob) != 0.:
        do_shortcut = jax.random.bernoulli(shortcut_key, p=shortcut_prob)
        img = jnp.where(do_shortcut, only_flipped, img)
    return img


@jax.jit
def augment_task(task, key):
    imgs = task.x
    imgs = jax.vmap(augment_img)(imgs, jax.random.split(key, len(imgs)))
    return task.replace(x=imgs)


def train_one_epoch(stepper, key, dataset, train_state, filt=None):
    fdataset, dataset_size = filtered_and_size(dataset, filt)
    tasks = as_task_iter(fdataset.batch(wandb.config.batch_size, drop_remainder=True))
    epoch_weight = get_train_weight()

    def check(tree):
        return jax.tree_util.tree_all(
            tree_map(lambda arr: jnp.isfinite(arr).all(), tree)
        )

    for step, task in tqdm(enumerate(tasks)):
        # if not check(train_state):
        #    exit()
        key, opt_key, augment_key = jax.random.split(key, 3)
        if wandb.config.augment_data:
            task = augment_task(task, augment_key)
        weights = jnp.ones(len(task.label)) / len(task.label)
        if wandb.config.batch_upweighting:
            weights = weights * dataset_size
        weights = weights * epoch_weight
        new_train_state = stepper(train_state, task, weights, opt_key)
        if (
            not wandb.config.frozen_burn_in
            or (wandb.config.burn_in_period is None) or wandb.run.summary.get("epoch", 0) > wandb.config.burn_in_period
        ):
            train_state = new_train_state
        else:
            train_state = new_train_state.replace(params=train_state.params)

        def active_inputs(p):
            return jnp.sum(jax.vmap(jnp.linalg.norm, in_axes=-2)(p) != 0.)
        def activity_status(param, state):
            actual = int(active_inputs(param))
            mask = int(jnp.sum(state.curv[0].mask))
            print(f"{param.shape}: mask {mask} actual {actual}")
        #tree_map(activity_status, train_state.params, train_state.opt_state)
    return train_state


def lossfn(label, y):
    label = jnp.mod(label, wandb.config.num_classes)
    lsm = jax.nn.log_softmax(y)
    label_log_prob = jnp.take_along_axis(lsm, label[..., None], axis=-1)[..., 0]
    soften = wandb.config.soften_lossfn
    label_log_prob = (1.0 - soften) * label_log_prob + soften * jnp.mean(lsm, axis=-1)
    return -label_log_prob


def norm_numpy_img(npimg):
    return (npimg - jnp.array(wandb.config.dataset_mean)) / jnp.array(
        wandb.config.dataset_std
    )


def as_task_iter(dataset):
    def as_task(item):
        xs, labels = item
        xs = norm_numpy_img(xs)
        return Task(xs, labels, lossfn)

    return map(as_task, tfds.as_numpy(dataset))


def init_prune(train_state):
    return train_state.init_prune()
    # params = train_state.tx.init_prune_params(train_state.params)
    # opt_state = train_state.tx.init_prune_opt_state(
    #    train_state.params, train_state.opt_state
    # )
    # return train_state.replace(params=params, opt_state=opt_state)

def reset_output_layer(train_state):
    params = train_state.params
    kernel = params["final"]["kernel"]
    new_shape = kernel.shape[:-1] + (wandb.config.num_classes,)
    new_kernel = jnp.zeros(new_shape, dtype=kernel.dtype)
    train_state = train_state.replace(
            params=params.copy(
                dict(final=params["final"].copy(
                    dict(kernel=new_kernel)
                ))
            )
        )
    return train_state.tx_reinit_changed_shapes()


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else None
    if exp_name is not None: exp_name = str(exp_name)
    wandb_project = "senn_transfer"
    if len(sys.argv) > 2 and sys.argv[2] == "resume":
        # interpret exp_name as a run id
        wandb.init(project=wandb_project, config={}, id=exp_name, resume="must")
    else:
        wandb.init(project=wandb_project, config={}, name=exp_name)

    wandb.config.epochs = 120
    #wandb.config.epoch_len = 1562
    #wandb.config.epoch_len = 93
    #wandb.config.epoch_len = 156
    #wandb.config.epoch_len = 98
    wandb.config.epoch_len = 781
    #wandb.config.dataset = "caltech_birds2011"
    #wandb.config.dataset = "TinyImagenetDataset"
    wandb.config.dataset = "cifar10"
    wandb.config.resize_to = [64, 64]
    wandb.config.dataset_mean = 0.5 * 255
    wandb.config.dataset_std = 0.25 * 255
    wandb.config.augment_data = True
    wandb.config.augment_color = True
    wandb.config.augment_timnet_shortcut_prob = 0.5
    wandb.config.augment_timnet_color_prob = 0.3
    wandb.config.augment_timnet_rotate = True
    wandb.config.augment_timnet_rotate_prob = 0.3
    wandb.config.augment_timnet_scale = True
    wandb.config.augment_timnet_scale_prob = 0.3
    wandb.config.augment_timnet_padtowidth = 128
    wandb.config.timnet_aug = True
    wandb.config.train_split = "train"
    all_pass = jnp.ones(shape=(10,), dtype=jnp.bool_)
    none_pass = ~all_pass
    label_filters = [
        none_pass.at[:2].set(True),
        none_pass.at[2:4].set(True),
        none_pass.at[4:6].set(True),
        none_pass.at[6:8].set(True),
        none_pass.at[8:].set(True),
    ]
    label_filters = [all_pass]
    wandb.config.train_filters = label_filters
    #wandb.config.eval_splits = ["train[:10%]", "validation"]
    wandb.config.eval_splits = ["train[:10%]", "test"]
    wandb.config.eval_filters = [None]
    #wandb.config.num_classes = 200
    wandb.config.num_classes = 10
    # wandb.config.model_type = "SmallConvNet"
    # wandb.config.model_kwargs = dict(hidden=64, depth=3, multiplicity=1, use_bias=False, dense_size=64)
    # wandb.config.model_type = "Perceptron"
    # wandb.config.model_kwargs = dict(hidden=32, depth=1, use_bias=False, flatten_last_n=3)
    # wandb.config.model_type = "BasicResNet"
    # wandb.config.model_kwargs = dict(stage_sizes=[2, 2, 2])
    # wandb.config.model_type = "AllCnnC"
    # wandb.config.model_kwargs = dict(early_channels=1*96, late_channels=1*192, fake_batch_norm=True)
    # wandb.config.model_notes = "deleted_conv_4"
    # wandb.config.model_type = "DenseNet"
    # wandb.config.model_kwargs = dict(depth=2, growth=2**7, init_conv=False)
    # wandb.config.model_type = "WaveletNet"
    # wandb.config.model_kwargs = dict(hidden=64, depth=3, level=1)
    # wandb.config.model_type = "BottleneckDense"
    # wandb.config.model_kwargs = dict(blocks=3, width=128, extra_depth=0, maybe_bud_width=16)
    # config is special cased for ExpandableDense
    wandb.config.model_type = "ExpandableDense"
    IW = 128
    W0 = 1
    WN = 1
    wandb.config.bud_width = None
    #wandb.config.model_kwargs = dict(widths=([IW]*W0,) + ([IW] * WN,) * 2, maybe_bud_width=wandb.config.bud_width)
    WLIST = [3, 3, 3, 3]
    widths = tuple([IW]*w for w in WLIST)
    #widths = ([459], [503, 120], [499, 28])
    wandb.config.model_kwargs = dict(widths=widths, maybe_bud_width=wandb.config.bud_width)
    #wandb.config.model_type = "WaveNet"
    #wandb.config.model_kwargs = dict(widths=[32]*3, levels=4)

    wandb.config.use_batchnorm = False
    wandb.config.use_dropout = False
    wandb.config.batch_size = 64
    #wandb.config.batch_size = 1024
    wandb.config.batch_upweighting = True
    wandb.config.eval_batch_size = 64
    #wandb.config.eval_batch_size = 1024
    wandb.config.eval_samples = 16
    wandb.config.lr = 1e-3
    wandb.config.add_unit_normal_curvature = True
    wandb.config.grad_update_as_curvature = False
    wandb.config.grad_as_curvature = False
    wandb.config.root_of_grad_for_curvature = False
    wandb.config.grad_curvature_mul = 1e-1
    wandb.config.model_seed = 0
    wandb.config.train_seed = 0
    wandb.config.varinf = False
    wandb.config.paired_varinf = False
    wandb.config.hess_only_varinf = True
    wandb.config.per_example_varinf_grad = False
    wandb.config.per_example_varinf = False
    wandb.config.varinf_sample_scale = 1e0
    wandb.config.model_nonlinearity = "swish"
    wandb.config.curvature = "UBAH"
    wandb.config.ubah_mul = 1e0
    wandb.config.temperature = 1e1**-0.0
    wandb.config.burn_in_period = None
    wandb.config.burn_in_initial_weight = 1e-3
    wandb.config.frozen_burn_in = True
    # wandb.config.whitener = "Masked"
    # wandb.config.whitener_diag_fraction = 0e-1
    # wandb.config.initial_precision = 1e1
    wandb.config.soften_lossfn = 0e-1
    # wandb.config.init_prior_precision = 1e1
    wandb.config.log_calibration = False

    wandb.config.taylor_order = None

    # OPTIMIZER
    wandb.config.peak_learning_rate = 3e-4
    wandb.config.noise_std = 0e-4
    wandb.config.weight_decay = 1e-2
    wandb.config.pct_start = 0.1
    wandb.config.linear_annealing = False
    wandb.config.num_cycles = 3

    # SIZE ADAPTION
    wandb.config.freeze_thaw_disable = True
    wandb.config.freeze_is_prune = True
    wandb.config.expansion_lower_bound = True
    wandb.config.freeze_thresh = 0e-2
    wandb.config.thaw_thresh = 0e-2
    wandb.config.freeze_thresh_rel = 3e-4
    wandb.config.thaw_thresh_rel = 3e-4
    wandb.config.thaw_prob_size_compensate = True
    wandb.config.minimum_width = 8
    wandb.config.maximum_width = 512
    wandb.config.ignore_width = wandb.config.bud_width if wandb.config.bud_width is not None else 0
    # wandb.config.bud_width = 16
    wandb.config.init_prune_to_min_width = not wandb.config.freeze_thaw_disable
    wandb.config.expansion_min_step = (
        0.00 * wandb.config.epochs * wandb.config.epoch_len
    )
    wandb.config.expansion_max_step = (
        1.0 * wandb.config.epochs * wandb.config.epoch_len
    )
    wandb.config.pruned_lr_rescale = False

    wandb.config.untouched_thresh = 10
    wandb.config.reinit_prob = 1e-1
    wandb.config.add_width_thresh = 0.5
    wandb.config.add_width_factor = 0.5

    wandb.config.enable_reinit = False
    wandb.config.enable_width_expansion = False
    wandb.config.enable_depth_expansion = False
    wandb.config.depth_score_max_k = 64
    wandb.config.min_epoch_for_depth_expansion = 0
    wandb.config.depth_score_add_to_current_score = 1e0
    #wandb.config.depth_score_addition_thresh = 1e1
    wandb.config.depth_score_abs_thresh = 0e1
    wandb.config.depth_score_rel_thresh = wandb.config.depth_score_max_k * 1e0 * wandb.config.thaw_thresh_rel
    wandb.config.block_size_hard_cap = 4

    wandb.config.use_global_expansion_score = True
    wandb.config.global_score_is_max_not_sum = False

    wandb.config.only_max_a_posteriori_eval = True
    wandb.config.iroot_error_warn = False
    # Minimum computation necessary for standard baseline training:
    wandb.config.fast = False
    if wandb.config.fast:
        assert wandb.config.curvature == "EMP_FISH"
        assert wandb.config.freeze_thaw_disable
        assert not wandb.config.enable_width_expansion
        assert not wandb.config.enable_depth_expansion
        assert wandb.config.bud_width is None
        assert not wandb.config.enable_reinit
    wandb.config.fast_turbo = False
    if wandb.config.fast_turbo:
        assert wandb.config.fast
        assert not wandb.config.varinf
        assert not wandb.config.log_calibration

    # Load previously saved 'final' as initial state
    #wandb.config.load_as_init = "/senn/orbax/cifar10_128_final"
    #wandb.config.load_as_init = "/senn/orbax/lif1r5r6/final"
    wandb.config.load_as_init = None

    rtpt_initials = os.environ.get("RTPT_INITIALS")
    assert rtpt_initials is not None
    total_epochs = wandb.config.epochs * len(wandb.config.train_filters)
    rtpt = RTPT(
        name_initials=rtpt_initials,
        experiment_name=wandb.run.name,
        max_iterations=total_epochs,
    )

    import config

    trainset = config.trainset
    evalset_dict = config.evalset_dict

    stepper = Stepper(
        wandb.config.varinf,
        wandb.config.paired_varinf,
        wandb.config.hess_only_varinf,
        wandb.config.per_example_varinf_grad,
        wandb.config.per_example_varinf,
        wandb.config.varinf_sample_scale,
        grad_hgrad=universal_grad_hgrad,
    )
    train_state = config.initial_train_state

    if wandb.config.init_prune_to_min_width:
        train_state = train_state.init_prune()
        # train_state = init_prune(train_state)

    run_id = wandb.run.id
    checkpoint_dir = f"/tmp/flax_ckpt/orbax/{run_id}/managed"
    final_checkpoint_dir = CHECKPOINT_DIR
    checkpointer = PyTreeCheckpointer()
    checkpoint_manager_opts = CheckpointManagerOptions(
            max_to_keep=2,
            create=True,
        )
    checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            checkpointer,
            checkpoint_manager_opts,
        )
    #ckpt_save_args = orbax_utils.save_args_from_target(train_state)
    def final_save(state):
        ckpt_save_args = orbax_utils.save_args_from_target(state)
        checkpointer.save(final_checkpoint_dir, state, force=True, save_args=ckpt_save_args)
    def epoch_save(epoch, state):
        ckpt_save_args = orbax_utils.save_args_from_target(state)
        checkpoint_manager.save(epoch, state, save_kwargs=dict(save_args=ckpt_save_args))

    if (load_dir := wandb.config.load_as_init) is not None:
        train_state = checkpointer.restore(load_dir, item=train_state)
        train_state = reset_output_layer(train_state)

    latest_ckpt_step = checkpoint_manager.latest_step()
    if latest_ckpt_step is not None:
        train_state = checkpoint_manager.restore(latest_ckpt_step, items=train_state)

    key = PRNGKey(wandb.config.train_seed)
    rtpt.start()
    def log_metrics(epoch):
        metrics = train_state.get_metrics()
        metrics = tree_map(np.array, metrics).unfreeze()
        wandb.log(metrics, commit=False)
        for name, evalset in evalset_dict.items():
            eval_one_epoch(name, evalset, train_state)
        wandb.log(dict(epoch=epoch), commit=True)
    if checkpoint_manager.latest_step() is None:
        log_metrics(-1)
    for fidx, filt in enumerate(wandb.config.train_filters):
        for epoch, key in enumerate(jax.random.split(key, wandb.config.epochs)):
            if (ckpt_step := checkpoint_manager.latest_step()) is not None:
                if epoch <= ckpt_step: continue
            reinit_key, width_key, key = jax.random.split(key, 3)
            #metrics = train_state.get_metrics()
            #metrics = tree_map(np.array, metrics).unfreeze()
            #wandb.log(metrics, commit=False)
            #for name, evalset in evalset_dict.items():
            #    eval_one_epoch(name, evalset, train_state)
            if wandb.config.enable_reinit:
                train_state = train_state.maybe_reinit(reinit_key)
            train_state = train_one_epoch(
                stepper, key, trainset, train_state, filt=filt
            )
            log_metrics(epoch)
            #wandb.log(dict(epoch=epoch), commit=True)
            rtpt.step()

            epoch_save(epoch, train_state)

            old_train_state = train_state
            if wandb.config.enable_width_expansion:
                train_state = train_state.maybe_expand_width(
                    key=width_key, builder=config.build_from_widths
                )
            width_was_added = tree_map(jnp.shape, train_state) != tree_map(jnp.shape, old_train_state)
            if wandb.config.enable_depth_expansion and not width_was_added:
                if epoch >= wandb.config.min_epoch_for_depth_expansion:
                    train_state = train_state.maybe_insert_layer(
                        builder=config.build_from_widths
                    )
        final_save(train_state)
        train_state = train_state.pin_prior()
        for name, evalset in evalset_dict.items():
            eval_final(name, evalset, train_state)
            eval_final("train_full", trainset, train_state)
        wandb.log(dict(epoch=epoch + 1), commit=True)


if __name__ == "__main__":
    main()
