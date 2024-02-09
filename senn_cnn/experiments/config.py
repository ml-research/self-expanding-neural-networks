import wandb
import tensorflow_datasets as tfds
import tensorflow.data
from jax.random import PRNGKey
import models
import os
import jax
from jax import numpy as jnp
from flax.core import frozen_dict
from flax.traverse_util import ModelParamTraversal
import resnet_models
import optax
from flax import linen as nn
from functools import partial

import senn.opt
from senn.opt import (
    TrainState,
    KronTracker,
    IRootTracker,
    InnerOpt,
    InnerState,
    InnerConfig,
    TreeOpt,
    FlattenOpt,
    DiagOpt,
)
from senn.linalg import IRootWhitener, DiagWhitener, HybridWhitener, MaskedWhitener
from senn.taylor import taylorify
from senn.models import ExpandableDense

from tiny_imagenet import TinyImagenetDataset
import tensorflow as tf


def basic_resnet(*, out, nonlin, **kwargs):
    return resnet_models.ResNet(
        num_classes=out, act=nonlin, block_cls=resnet_models.ResNetBlock, **kwargs
    )


def bottleneck_resnet(*, out, nonlin, **kwargs):
    return resnet_models.ResNet(
        num_classes=out,
        act=nonlin,
        block_cls=resnet_models.BottleneckResNetBlock,
        **kwargs,
    )


dataset_kwargs = dict(
    name=wandb.config.dataset,
    download=True,
    as_supervised=True,
    data_dir=os.environ["DATASETS_ROOT_DIR"],
)


def get_trainset(info=False):
    return tfds.load(
        **dataset_kwargs,
        split=wandb.config.train_split,
        with_info=info,
        shuffle_files=True,
    )


trainset, info = get_trainset(info=True)
evalset_kwargs = dict()


def get_evalset_dict():
    evalset_dict = {
        split: tfds.load(**dataset_kwargs, **evalset_kwargs, split=split)
        for split in wandb.config.eval_splits
    }
    return evalset_dict


evalset_dict = get_evalset_dict()
if "num_classes" not in wandb.config:
    wandb.config.num_classes = info.features["label"].num_classes

def resize_dataset(img, label):
    img = tf.image.resize_with_pad(img, *wandb.config.resize_to)
    return img, label
if wandb.config.resize_to is not None:
    trainset = trainset.map(resize_dataset)
    evalset_dict = {key: val.map(resize_dataset) for key, val in evalset_dict.items()}

nonlin = {"tanh": jnp.tanh, "swish": jax.nn.swish}[wandb.config.model_nonlinearity]
model_types = dict(
    Perceptron=models.Perceptron,
    SmallConvNet=models.SmallConvNet,
    BasicResNet=basic_resnet,
    BottleneckResNet=bottleneck_resnet,
    AllCnnA=models.AllCnnA,
    AllCnnC=models.AllCnnC,
    DenseNet=models.DenseNet,
    WaveletNet=models.WaveletNet,
    BottleneckDense=models.BottleneckDense,
    ExpandableDense=senn.models.ExpandableDense,
    WaveNet=models.WaveNet,
)
module_def = model_types[wandb.config.model_type]


def blockify_kwargs(widths, **kwargs):
    blocks = list(models.ExpandableBlock(widths=ws, **kwargs) for ws in widths)
    return dict(blocks=blocks)


model_kwargs = wandb.config.model_kwargs
if wandb.config.model_type == "ExpandableDense":

    def build_from_widths(widths=None):
        kwargs = frozen_dict.freeze(wandb.config.model_kwargs)
        kwargs, init_widths = kwargs.pop("widths")
        widths = init_widths if widths is None else widths
        return senn.models.ExpandableDense.build(
            widthss=widths, out=wandb.config.num_classes, nonlin=nonlin, **kwargs
        )

    model = build_from_widths()
    print(model)
else:
    model = module_def(out=wandb.config.num_classes, **model_kwargs, nonlin=nonlin)
(example, _) = next(trainset.take(1).as_numpy_iterator())
example = jnp.array(example, dtype=jnp.float32)
print(model.tabulate(PRNGKey(0), example))
variables = model.init(PRNGKey(wandb.config.model_seed), example)


def has_kernel(pytree):
    return isinstance(pytree, frozen_dict.FrozenDict) and "kernel" in pytree.keys()


optimizer = senn.opt.SimpleOpt()
SCHEDULE_REPEAT = wandb.config.num_cycles
PHASE_LEN = (wandb.config.epochs * wandb.config.epoch_len) // SCHEDULE_REPEAT
schedule_kwargs = dict(
    transition_steps=PHASE_LEN,
    peak_value=wandb.config.peak_learning_rate,
    pct_start=wandb.config.pct_start,
)
schedule_fn = (
    optax.linear_onecycle_schedule
    if wandb.config.linear_annealing
    else optax.cosine_onecycle_schedule
)
schedules = [schedule_fn(**schedule_kwargs) for i in range(SCHEDULE_REPEAT)]
boundaries = [i * PHASE_LEN for i in range(SCHEDULE_REPEAT)][1:]
schedule = optax.join_schedules(schedules, boundaries)
#schedule = schedule_fn(**schedule_kwargs)
# schedule = optax.cosine_onecycle_schedule(transition_steps=wandb.config.epochs*781, peak_value=wandb.config.peak_learning_rate, pct_start=wandb.config.pct_start)
# schedule = optax.linear_onecycle_schedule(transition_steps=wandb.config.epochs*781, peak_value=wandb.config.peak_learning_rate, pct_start=wandb.config.pct_start)
# first_order = optax.adam(learning_rate=schedule, b2=0.99)
first_order = senn.opt.MyAdam(
    lr=schedule,
    mom1=1e-1,
    mom2=1e-2,
    weight_decay=wandb.config.weight_decay,
    noise_std=wandb.config.noise_std,
    order=0,
)
if wandb.config.fast_turbo:
    first_order = optax.adamw(learning_rate=schedule, b2=0.99, weight_decay=1e-2)
optimizer = senn.opt.WrappedFirstOrder(tx=first_order)

apply_fn = model.apply
add_width_fn = partial(model.apply, method=model.maybe_add_width, mutable=True)
init_variables = variables
if (order := wandb.config.taylor_order) is not None:
    # apply_fn = taylorify(apply_fn, basepoint=variables, order=order)
    def apply_fn(variables, *args, **kwargs):
        def inner(params, variables, *args, **kwargs):
            variables = frozen_dict.copy(variables, dict(params=params))
            return model.apply(variables, *args, **kwargs)

        tinner = taylorify(inner, basepoint=init_variables["params"], order=order)
        params = variables["params"]
        params = frozen_dict.freeze(params)
        return tinner(params, variables, *args, **kwargs)


initial_train_state = TrainState.create(
    optimizer,
    variables["params"],
    variables.get("probes", {}),
    apply_fn,
    batch_stats=variables.get("batch_stats", {}),
    dummy_input=example,
    # add_width_fn=add_width_fn,
    model=model,
    # traversal=ModelParamTraversal(lambda s: "/bud/" not in s),
    path_pred=lambda path: "bud" not in path,
)
