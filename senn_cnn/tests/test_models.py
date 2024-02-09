import pytest
from senn import models
from senn.opt import TrainState
import senn.opt

import jax
import optax
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.tree_util import tree_map, tree_reduce, tree_all
import numpy as np
import wandb

import flax
from flax.traverse_util import flatten_dict, unflatten_dict

widthss = ([4, 6],)
num_channels = 3
img_size = 32
example = jnp.zeros((1, img_size, img_size, num_channels))
num_classes = 2
bud_size = 8

wandb.init(mode="disabled")
wandb.config.ubah_mul = 1.0
wandb.config.add_unit_normal_curvature = True
wandb.config.grad_update_as_curvature = True
wandb.config.curvature = "UBAH"
wandb.config.grad_curvature_mul = 1.0
wandb.config.use_dropout = False
wandb.config.pruned_lr_rescale = True
wandb.config.freeze_thaw_disable = False
wandb.config.expansion_cutoff_step = 1000000
wandb.config.freeze_thresh = 1e-2
wandb.config.thaw_thresh = 1e-2
wandb.config.freeze_is_prune = True
wandb.config.freeze_thresh_rel = 1e-2
wandb.config.thaw_thresh_rel = 1e-2
wandb.config.thaw_prob_size_compensate = True
wandb.config.ignore_width = bud_size

wandb.config.minimum_width = num_classes * 2
wandb.config.num_classes = num_classes
wandb.config.untouched_thresh = 1


def builder(widthss):
    return models.ExpandableDense.build(
        widthss=widthss, out=num_classes, nonlin=jax.nn.swish, maybe_bud_width=bud_size
    )


def make_model():
    # model_kwargs = dict(widthss=widthss, out=num_classes, nonlin=jax.nn.swish, maybe_bud_width=bud_size)
    # model = models.ExpandableDense.build(**model_kwargs)
    model = builder(widthss)
    variables = model.init(PRNGKey(0), example)
    return model, variables


def make_schedule():
    return optax.cosine_onecycle_schedule(
        transition_steps=100 * 781,
        peak_value=1e-3,
        pct_start=0.1,
    )


def make_optimizer():
    schedule = make_schedule()
    first_order = senn.opt.MyAdam(
        lr=schedule, mom1=1e-1, mom2=1e-2, weight_decay=1e-2, noise_std=0.0, order=0
    )
    optimizer = senn.opt.WrappedFirstOrder(tx=first_order)
    return optimizer


def test_init():
    model, variables = make_model()
    assert set(variables) == {"params", "probes"}


def make_trainstate():
    model, variables = make_model()
    optimizer = make_optimizer()
    ts = TrainState.create(
        optimizer,
        variables["params"],
        variables.get("probes", {}),
        model.apply,
        batch_stats=variables.get("batch_stats", {}),
        dummy_input=example,
        model=model,
        path_pred=lambda path: "bud" not in path,
    )
    ts = ts.init_prune()
    return ts


def make_stepper():
    return senn.opt.Stepper(varinf=True, paired_varinf=True, hess_only_varinf=True)


def lossfn(label, y):
    lsm = jax.nn.log_softmax(y)
    label_log_prob = jnp.take_along_axis(lsm, label[..., None], axis=-1)[..., 0]
    return -label_log_prob


def make_tasks(batch_size=64, key=PRNGKey(0)):
    xkey, lkey = jax.random.split(key)
    shape = (batch_size,) + example.shape[1:]
    xs = jax.random.normal(xkey, shape=shape)
    logits = jnp.zeros((num_classes,))
    labels = jax.random.categorical(lkey, logits, shape=(batch_size,))
    return senn.opt.Task(xs, labels, lossfn)


def test_trainstate_create():
    train_state = make_trainstate()
    call_stepper(train_state)


def call_stepper(train_state):
    stepper = make_stepper()
    key = PRNGKey(0)
    tasks = make_tasks()
    weights = jnp.ones(len(tasks.label))
    new_train_state = stepper(train_state, tasks, weights, key)
    return new_train_state


def test_get_metrics():
    train_state = make_trainstate()
    train_state.get_metrics()


def test_pruned_init():
    train_state = make_trainstate()
    for key in flatten_dict(train_state.opt_state):
        assert "bud" not in key
    for key, value in flatten_dict(train_state.params).items():
        if "main" in key:
            assert jnp.all(value[..., -bud_size:, :] == 0.0)


def test_maybe_add_width():
    train_state = make_trainstate()
    add_width = tree_map(lambda a: 7, train_state.subparams)
    key = PRNGKey(0)
    train_state = train_state.maybe_expand_width(
        key=key, builder=builder, add_width=add_width
    )
    call_stepper(train_state)


def test_insert_layer():
    train_state = make_trainstate()
    train_state = call_stepper(train_state)
    bidx, lidx = 0, 1
    train_state = train_state.insert_layer(builder, bidx, lidx)
    print(tree_map(lambda arr: arr.shape, train_state.params))
    print(tree_map(lambda arr: arr.shape, train_state.probes))
    train_state = call_stepper(train_state)


def test_inserted_same_function():
    train_state = make_trainstate()
    print(train_state.params)
    tasks = make_tasks()
    ys0 = jax.vmap(train_state.eval)(tasks)
    print(ys0)
    bidx, lidx = 0, 1
    train_state = train_state.insert_layer(builder, bidx, lidx)
    print(train_state.params)
    ys1 = jax.vmap(train_state.eval)(tasks)
    print(ys1)
    assert pytest.approx(ys0) == ys1
