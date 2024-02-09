from functools import partial
from itertools import islice
from typing import Any, Callable, Sequence, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

import jax
jax.config.update("jax_enable_x64", False)
from jax import lax, random, numpy as jnp
from jax.numpy.linalg import norm as jnorm
from jax.tree_util import tree_map as jtm, tree_reduce as jtr, Partial
import flax
from flax import linen as nn
from flax.training import checkpoints
from tensorflow import summary
from tensorflow import convert_to_tensor

import nets
from nets import DDense, Rational1D, DConv, Layer, Layers
from nets import tree_length, pad_target
from optim import SimpleGradient, KrylovNG, CGNG

from data import smallnist, smallfnist, cfg_tranches, tranch_cat
from jaxutils import key_iter
import experiment_utils

from langevin import mala_step, mala_steps

import os
import confuse
import argparse
from datetime import datetime

from rtpt.rtpt import RTPT
# import gpustat

from tqdm import tqdm

def tree_dot(a, b):
    return jtr(lambda c, d: c + d, jtm(lambda c, d: jnp.sum(c * d), a, b))

def insert_params(state, params):
    return state.copy({'params': params})

def validation_loss(model, loss_function, dataset, labels, params):
    Y = jax.vmap(lambda x: model.apply(params, x))(dataset)
    # Ym = Y.mean(axis=(-2,-3))
    Ym = Y
    L = jax.vmap(loss_function)(labels, Y)
    correct =  Ym.argmax(axis=-1) == labels
    pseudo_loss = jnp.log10(jnp.sum(correct) + 1) - jnp.log10(jnp.sum(~correct) + 1)
    return L.mean(), correct.mean(), pseudo_loss

def print_tree_type(tree):
    print(jtm(lambda x: x.shape, tree))

def print_tree_mags(tree):
    print(jtm(lambda arr: (arr**2).mean(), tree))

@dataclass
class ModelTemplate:
    capacities: Sequence[int]
    contents: Sequence[Optional[int]]
    rational: bool = False

    def layer(self, features, final=False):
        if self.rational:
            nonlin = partial(Rational1D, residual=True, init_identity=False)
        else:
            nonlin = lambda: jax.nn.silu
        return Layer(features, [], DDense, [] if final else [nonlin])

    def build(self):
        hidden_layers = [self.layer(f) for f in self.capacities[:-1]]
        final_layer = self.layer(self.capacities[-1], final=True)
        return Layers(hidden_layers + [final_layer])

    def enabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is not None])

    def disabled_layers(self):
        return list([i for i, d in enumerate(self.contents) if d is None])

    def in_out_indices(self, layer_index):
        conarr = np.array(self.contents)
        split_at = layer_index + 1
        (preceding_enabled,) = np.nonzero(conarr[:split_at][:-1] != None)
        in_index = 0 if len(preceding_enabled) == 0 else preceding_enabled[-1] + 1
        # in_index = layer_index
        (subsequent_enabled,) = np.nonzero(conarr[split_at:] != None)
        out_index = split_at + subsequent_enabled[0]
        return in_index, out_index

class Task(ABC):

    @abstractmethod
    def get_data(cfg, test=False):
        pass
        return dataset, labels

    @abstractmethod
    def loss_function(label, output):
        pass

class Regression(Task):

    @staticmethod
    def loss_function(label, output):
        return ((output - label)**2).mean()

class Classification(Task):

    @staticmethod
    def loss_function(label, output):
        return jax.nn.logsumexp(output) - output[label]

class SameFamilyRegression(Regression):

    def __init__(self, cfg, key):
        self.out_size = cfg['task']['out_size'].get()
        capacities = cfg['task']['hidden'].get() + [self.out_size]
        self.template = ModelTemplate(capacities, capacities, rational=True)
        self.model = self.template.build()
        self.state = self.model.init(key, jnp.zeros((self.out_size,)))
        self.target_func = jax.jit(Partial(self.model.apply, self.state))

        self.N = cfg['data']['N'].get()
        self.TN = cfg['data']['TN'].get(self.N)

    def get_data(self, cfg, test=False, key=None):
        assert key is not None
        N = self.TN if test else self.N
        xs = jax.random.normal(key, (N, self.out_size))
        ys = jax.vmap(self.target_func)(xs)
        return xs, ys

class ImgVecClass(Classification):

    def __init__(self, cfg):
        self.cfg = cfg
        self.tranches = cfg_tranches(cfg['data'], cfg['task']['resolution'].get())

    def get_data(self, _, test=False, index=0):
        dataset, labels = tranch_cat(self.tranches, index, train=not test)
        return jax.vmap(jnp.ravel)(dataset), labels


class Solver:
    """probably should be further factored"""

    def __init__(self, cfg, template, task, key, example):
        self.cfg = cfg['opt']
        self.template = template
        self.task = task
        self.model = self.template.build()
        self.state = self.model.init(key, example)
        print(f"contents: {self.template.contents}")
        def nonincreasing(seq):
            return all(x >= y for x, y in zip(seq, seq[1:]))
        assert nonincreasing([c for c in self.template.contents if c is not None]), \
            "model.restrict_params is broken for increasing feature sizes"
        self.state = self.model.apply(self.state, 
                                      self.template.contents,
                                      method=self.model.restrict_params)
        self.optimizer, self.opt_state = self.make_opt(self.state)
        self.lr = cfg['opt']['lr'].get()
        self.recompile()
        self.last_natlen = None
        self.weight_decay = cfg['opt']['weight_decay'].get(0.)

    def item_loss(self, state, datum, label):
        y = self.model.apply(state, datum)
        loss = self.task.loss_function(label, y)
        return loss

    def item_correct(self, state, datum, label):
        y = self.model.apply(state, datum)
        return jnp.argmax(y) == label

    def _batch_loss(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_loss, state))(data, labels).mean()

    def _batch_acc(self, state, data, labels):
        """NOTE: This will not account for a changing model -
        if model changes this must be re-jitted"""
        return jax.vmap(partial(self.item_correct, state))(data, labels).mean()

    def make_opt(self, state):
        opt = CGNG(self.cfg, state['params'])
        opt_state = opt.init(flax.core.frozen_dict.freeze({}))
        return opt, opt_state

    def restrict_grad(self, grad):
        variables = {"params": grad}
        return self.model.apply(variables, self.template.contents, method=self.model.restrict_grad)['params']

    def recompile(self):
        # self.restrict_grad = lambda grad: self.model.apply(
        #     {'params': grad}, self.template.contents, method=self.model.restrict_grad)['params']
        func = lambda params, x: self.model.apply(self.state.copy({'params': params}), x)
        self.observe = jax.jit(partial(self.optimizer.observe, 
                                       func, 
                                       self.task.loss_function, 
                                       self.restrict_grad))
        self.apply_model = jax.jit(self.model.apply)
        self.batch_loss = jax.jit(self._batch_loss)
        self.batch_acc = jax.jit(self._batch_acc)

    def update_params(self, mul, tan, weight_decay=0.):
        # apply weight decay to tangent
        tan = jtm(lambda t, s: 1/(1+weight_decay)*t + weight_decay/(1+weight_decay)*s, tan, self.state['params'])
        tan = self.restrict_grad(tan)
        # apply tangent to state with learning rate
        new = jtm(lambda t, s: s + -mul*self.lr*t, tan, self.state['params'])
        self.state = self.state.copy({'params': new})

    def train_batch(self, batch, observe_only=False, loud=False):
        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)
        if loud:
            print(f"loss: {loss:.3E}")
        summary.scalar("loss", loss)
        summary.scalar("features", sum([c for c in self.template.contents if c is not None]))
        for i, f in enumerate(self.template.contents):
            summary.scalar(f"features_{i}", f if f is not None else 0)
        self.opt_state = self.observe(data, labels, self.state['params'], self.opt_state)
        
        grad = self.optimizer.SG.read(self.opt_state)
        ngrad = self.optimizer.read(self.opt_state)
        nat_len = tree_dot(ngrad, grad)
        self.last_natlen = nat_len
        summary.scalar("baseline", nat_len)
        summary.scalar("normed_baseline", nat_len/loss)

        summary.scalar("param_Fnorm", self.optimizer.param_Fnorm.read(self.opt_state))
        summary.scalar("param_l2norm", tree_dot(self.state['params'], self.state['params']))

        # summary.scalar("fresh_baseline", self.eval_feature_proposal(batch, self.state))

        if not observe_only:
            # self.update_params(loss/nat_len, ngrad)
            self.update_params(1., ngrad, weight_decay=self.weight_decay)
        if loud:
            return loss

    def eval_feature_proposal(self, batch, fstate):
        assert self.cfg['tau'].get() is None
        assert False, "nat_len calculation is inconsistent with that of the training update"
        data, labels = batch
        _, fopt_state = self.make_opt(fstate)
        # fopt_state = self.observe(data, labels, fstate['params'], fopt_state)
        fopt_state = self.observe(data, labels, fstate['params'], self.opt_state)
        
        grad = self.optimizer.SG.read(fopt_state)
        ngrad = self.optimizer.read(fopt_state)
        nat_len = tree_dot(ngrad, grad)
        return nat_len
            
    def test_batch(self, batch):
        # print("test batch...")
        data, labels = batch
        loss = self.batch_loss(self.state, data, labels)
        # print(f"validation loss: {loss:.3E}")
        summary.scalar("validation loss", loss)

    def test_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("validation accuracy", acc)

    def train_acc(self, batch):
        data, labels = batch
        acc = self.batch_acc(self.state, data, labels)
        summary.scalar("training accuracy", acc)


def kfac_observe(model, loss_function, state, tangent, pair):
    # y, dL, Jt, aux, grad, res
    datum, label = pair
    y, Jt, aux = jax.jvp(
        fun=lambda state: model.apply(state, datum, mutable='intermediates'),
        primals=(state,),
        tangents=(tangent,),
        has_aux=True
        )
    dloss = jax.grad(partial(loss_function, label))(y)
    loss_sqnorm = lax.pmean(jnp.sum(dloss**2), 'batch')
    # dloss = jtm(lambda arr: arr.astype('float64'), dloss32)

    # print(aux)
    _, backward = jax.vjp(
        lambda state: model.apply(state, datum),
        state,
        )
    # loss_rescale = 1/(jnp.sqrt(jnp.sum(dloss**2))+1e-16)
    (grad,) = backward(dloss/jnp.sqrt(jnp.sum(dloss**2)))
    # (grad,) = backward(dloss)
    # grad = jtm(lambda arr: arr*loss_rescale, grad)
    # grad = jax.grad(lambda state: loss_function(label, model.apply(state, datum)))(state)
    Jt_rescale = lax.psum(jnp.sum(Jt*dloss), 'batch')/(lax.psum(jnp.sum(Jt*Jt), 'batch')+1e-10)
    # Jt_rescale = 1.
    lres = dloss - Jt*Jt_rescale
    # lres = dloss
    # lres = Jt*Jt_rescale
    (state_residual,) = backward(lres)

    acts = model.apply(aux, method=model.extract_activations)
    out_grads = model.apply(grad, method=model.extract_out_grads)
    out_ress = model.apply(state_residual, method=model.extract_out_grads)

    As = [lax.pmean(a[:,None]*a[None,:], 'batch') for (a,) in acts]
    Gs = [lax.pmean(g[:,None]*g[None,:], 'batch') for g in out_grads]

    # normed_out_ress = [res/loss_norm for res in out_ress]
    return As, Gs, acts, out_ress, out_grads, loss_sqnorm

def kfac_single_eval(fmodel, act, Ginv, res, feature):
    fval = fmodel.apply(feature, act)
    A = lax.pmean(fval[...,:,None]*fval[...,None,:], 'batch')
    Ainv = jnp.linalg.pinv(A)
    corr = lax.pmean(fval[:,None]*res[None,:], 'batch')
    # corr = lax.pmean(res[:,None]*res[None,:], 'batch')
    # return jnp.sum(corr*corr)
    # normed_corr = Ainv@corr
    normed_corr = Ainv@corr@Ginv
    rscore = jnp.sum(corr*normed_corr)
    return rscore

def kfac_single_sgd(fmodel, act, Ginv, res, feature, eps, atikh=0.):
    fval, backward = jax.vjp(
        lambda theta: fmodel.apply(theta, act),
        feature
    )
    A = lax.pmean(fval[...,:,None]*fval[...,None,:], 'batch')
    Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
    corr = lax.pmean(fval[:,None]*res[None,:], 'batch')

    normed_corr = Ainv@corr@Ginv
    rscore = jnp.sum(corr*normed_corr)

    resres = res - corr.T@Ainv@fval
    fgrad = normed_corr@resres
    (theta_grad,) = jtm(lambda arr: lax.pmean(arr, 'batch'), backward(fgrad))

    def ldet(feat):
        L = feat["params"]["linear"]["kernel"]
        sign, logabs = jnp.linalg.slogdet(L.T@L)
        return logabs

    REGULARISE = 1e-2
    lnmag, lnmag_grad = jax.value_and_grad(ldet)(feature)

    new_feature = jtm(lambda theta, g, lgrad: theta + eps*g - REGULARISE*lnmag*lgrad, feature, theta_grad, lnmag_grad)
    return rscore, new_feature

def get_prior_var(feature):
    # return jtm(lambda f: jnp.ones_like(f)/jnp.size(f, axis=0), feature)
    return jtm(lambda f: jnp.ones_like(f), feature)

def kfac_mala_burst(fmodel, act, Ginv, res, feature, key, lr, temp=1e0, atikh=0., steps=10):
    priorvar = get_prior_var(feature)
    def score_cotan(fval):
        A = lax.pmean(fval[..., :, None] * fval[..., None, :], 'batch')
        Ainv = jnp.linalg.pinv(tikhonov(A, atikh))
        corr = lax.pmean(fval[:, None] * res[None, :], 'batch')

        normed_corr = Ainv @ corr @ Ginv
        rscore = jnp.sum(corr * normed_corr)

        resres = res - corr.T @ Ainv @ fval
        fgrad = normed_corr @ resres
        return rscore, fgrad
    def loss_grad(feature):
        fval, backward = jax.vjp(
            lambda theta: fmodel.apply(theta, act),
            feature
        )
        rscore, cotangent = score_cotan(fval)
        (theta_grad,) = jtm(lambda arr: lax.pmean(arr, 'batch'), backward(cotangent))
        return -rscore, theta_grad

    feature, accept_rate = mala_steps(loss_grad, priorvar, feature, key, lr, steps, temp=temp, legacy=True)
    rscore, _ = score_cotan(fmodel.apply(feature, act))
    return rscore, feature, accept_rate

def kfac_direct_mala(model, loss_function, fmodel, state, tangent, full_grad, in_index, out_index, feature, pair, lr, steps, key, temp=1e0):
    As, Gs, acts, resids, grads, loss_sqnorm = kfac_observe(model, loss_function, state, tangent, pair)
    A, G, (act_in,), (act_out,) = As[out_index], Gs[out_index], acts[in_index], acts[out_index]
    res, grad = resids[out_index], grads[out_index]
    Ginv = jnp.linalg.pinv(tikhonov(G, 1e-1*meandiag(G)))
    atikh = 0e-1*meandiag(A)
    res = layer_residual(A, res, act_out)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)
    @flax.struct.dataclass
    class State:
        lr: float
        rscore: float
        feature: Any
    init_state = State(lr, 0., feature)
    def fun(state, key):
        new_rscore, new_feature, accept_rate = kfac_mala_burst(
            fmodel,
            act_in,
            Ginv,
            res,
            state.feature,
            key,
            state.lr,
            temp=temp
        )
        TARGET = 0.6
        changed_lr = jnp.where(accept_rate > TARGET, state.lr*1.3, state.lr/1.3)
        new_lr = jnp.where(jnp.abs(accept_rate-TARGET) > 0.3, changed_lr, state.lr)
        return State(new_lr, new_rscore, new_feature), accept_rate

    final_state, accepts = jax.lax.scan(fun, init_state, jax.random.split(key, steps))
    return final_state.rscore, final_state.feature, layer_score, loss_sqnorm

def layer_residual(A, res, act):
    Ainv = jnp.linalg.pinv(A)
    Cra = lax.pmean(res[:,None]*act[None,:], 'batch')
    lres = res - Cra@Ainv@act
    return lres

def kfac_direct_eval(model, loss_function, fmodel, state, tangent, full_grad, in_index, out_index, feature, pair):
    As, Gs, acts, resids, _, loss_sqnorm = kfac_observe(model, loss_function, state, tangent, pair)
    A, G, (act_in,), (act_out,), res = As[out_index], Gs[out_index], acts[in_index], acts[out_index], resids[out_index]
    Ginv = jnp.linalg.pinv(G)
    res = layer_residual(A, res, act_out)
    rscore = kfac_single_eval(fmodel, act_in, Ginv, res, feature)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)
    return rscore, layer_score, loss_sqnorm

def meandiag(matrix):
    assert len(matrix.shape) == 2
    N = matrix.shape[0]
    return jnp.trace(matrix)/N

def tikhonov(matrix, strength):
    assert len(matrix.shape) == 2
    N = matrix.shape[0]
    diag = jnp.eye(N)*strength
    return matrix + diag

def layer_baseline(A, G, lin_grad):
    """lin_grad: best estimate of true gradient on layer linear kernel"""
    # corr = lax.pmean(act[:, None]*grad[None, :], 'batch')
    corr = lin_grad
    Ainv = jnp.linalg.pinv(A)
    Ginv = jnp.linalg.pinv(G)
    normed_corr = Ainv@corr@Ginv
    score = jnp.sum(corr*normed_corr)
    return score

def kfac_direct_sgd(model, loss_function, fmodel, state, tangent, full_grad, in_index, out_index, feature, pair, eps, steps):
    As, Gs, acts, resids, grads, loss_sqnorm = kfac_observe(model, loss_function, state, tangent, pair)
    A, G, (act_in,), (act_out,) = As[out_index], Gs[out_index], acts[in_index], acts[out_index]
    res, grad = resids[out_index], grads[out_index]
    Ginv = jnp.linalg.pinv(tikhonov(G, 1e-1*meandiag(G)))
    atikh = 0e-1*meandiag(A)
    res = layer_residual(A, res, act_out)
    lin_grad = full_grad["params"][f"layers_{out_index}"]["linear"]["kernel"]
    layer_score = layer_baseline(A, G, lin_grad)
    @flax.struct.dataclass
    class State:
        eps: float
        rscore: float
        feature: Any
    init_state = State(eps, 0., feature)
    def body_fun(_, state):
        new_rscore, new_feature = kfac_single_sgd(fmodel, act_in, Ginv, res, state.feature, state.eps, atikh)
        # reduce learning rate if loss increased
        rscore_increased = new_rscore/state.rscore > 1.
        return State(
            jnp.where(rscore_increased, eps, eps/3.),
            new_rscore,
            new_feature
        )
    final_state = jax.lax.fori_loop(0, steps, body_fun, init_state)
    return final_state.rscore, final_state.feature, layer_score, loss_sqnorm

class Proposer:

    def __init__(self, template, model, key_iter):
        self.template = template
        self.model = model
        self.key_iter = key_iter

    def new_feature(self, null, input_size, output_size=1, key=None):
        # print(f"sum nonnull: {int(jnp.sum(~null))}")
        if key is None:
            key = next(self.key_iter)
        active = int(sum(~null))
        total = len(null)
        #Modify according to kernel size
        input_shape = (3,3,active)
        layer = self.template.layer(output_size)
        feature = layer.init(key, jnp.zeros(input_shape))
        assert input_size >= active
        feature = layer.apply(feature, input_size-active, method=layer.pad_inputs)
        return feature

    def embed_feature(self, state, feature, layer_index):
        lift = partial(nn.apply(Layers.lift, self.model), state, layer_index)
        assert not lift(Layer.dormant)
        null = lift(Layer.null)
        assert null.any()
        idx = null.argmax()
        state = state.unfreeze()
        name = f'layers_{layer_index}'
        state['params'][name] = lift(Layer.insert_feature, feature.unfreeze(), idx)['params']
        return flax.core.frozen_dict.freeze(state)

    def embed_layer(self, state, feature, layer_index):
        unpadded_basis = feature["params"]["linear"]["kernel"]
        unpadded_shift = feature["params"]["linear"]["bias"]

        pad_amount = self.template.capacities[layer_index] - unpadded_shift.shape[-1]
        assert pad_amount >= 0
        padded_basis = nets.pad_axis(unpadded_basis, pad_amount, axis=-1)
        padded_shift = nets.pad_axis(unpadded_shift, pad_amount, axis=-1)

        new_state = self.model.apply(
            state,
            dims=self.template.contents,
            index=layer_index,
            basis=padded_basis,
            shift=padded_shift,
            method=self.model.activate_layer
        )
        return new_state

    def get_input_size(self, state, layer_index):
        return self.model.apply(
            state,
            index=layer_index,
            func=Layer.input_size,
            method=self.model.lift
        )

    def get_output_size(self, state, layer_index):
        return self.model.apply(
            state,
            index=layer_index,
            func=Layer.output_size,
            method=self.model.lift
        )

    def get_input_null(self, state, layer_index):
        # print(f"null of: {layer_index}")
        if layer_index == 0:
            # if first layer, assume all inputs are nonnull
            input_size = self.get_input_size(state, 0)
            return jnp.zeros((input_size,), dtype=jnp.bool_)
        else:
            return self.model.apply(
                state,
                index=layer_index - 1,
                func=Layer.null,
                method=self.model.lift
            )

    # def make_invertible(self, L):
    #     Q, R = jnp.linalg.qr(L)
    #     Rdiag = jnp.diag(R)
    #     mag = 0.1*jnp.mean(jnp.abs(Rdiag))
    #     newRdiag = Rdiag.at[jnp.abs(Rdiag)<mag].set(mag)
    #     R = R - Rdiag + newRdiag
    #     return Q@R

    def generate_feature(self, state, layer_index, key=None, kill_lin=False, in_index=None, output_size=1):
        if in_index is None:
            in_index = layer_index
        feature = self.new_feature(
            null=self.get_input_null(state, in_index),
            input_size=self.get_input_size(state, layer_index),
            output_size=output_size,
            key=key
        )
        if kill_lin:
            feature = feature.copy({"params": feature["params"].copy({
                "equivariant_0": feature["params"]["equivariant_0"].copy({
                    "w_lin": jnp.array([0.])
                })
            })})
        # if invertible:
        #     L = feature["params"]["linear"]["kernel"]
        #     L = self.make_invertible(L)
        #     feature = feature.copy({"params": feature["params"].copy({
        #         "linear": feature["params"]["linear"].copy({
        #             "kernel": L
        #         })
        #     })})
        return feature

    def propose_feature(self, state, layer_index):
        raise NotImplementedError
        if layer_index == 0:
            input_size = self.model.apply(
                state, 
                index=0, 
                func=Layer.input_size, 
                method=self.model.lift
            )
            input_null = jnp.zeros((input_size,), dtype=jnp.bool_)
        else:
            input_null = self.model.apply(
                state,
                index=layer_index - 1,
                func=Layer.null,
                method=self.model.lift
            )
        input_null = self.get_input_null(state, layer_index)
        feature = self.new_feature(input_null)
        fstate = self.embed_feature(state, feature, layer_index)
        return feature, fstate

    def verify_state(self, state):
        nulls = self.model.apply(
            state,
            method=self.model.nulls
        )
        for i, (c, n) in enumerate(zip(self.template.contents, nulls)):
            if c is not None:
                sum_not_null = jnp.sum(~n)
                assert c == sum_not_null, f"content-null mismatch at layer {i} with content {c} and sum(~null) {sum_not_null}"
        

@dataclass
class Sampler:
    key_iter: Any
    dataset: Any
    labels: Any
    batch_size: Optional[int]

    def num_batches(self):
        if self.batch_size is None:
            return 1
        DLEN = self.dataset.shape[0]
        num_batches = DLEN // self.batch_size
        return num_batches
        

    def batches(self):
        if self.batch_size is None:
            yield self.dataset, self.labels
            return
        DLEN = self.dataset.shape[0]
        pindices = jax.random.permutation(next(self.key_iter), DLEN)
        num_batches = DLEN // self.batch_size
        for batch_num in range(num_batches):
            bindices = pindices[batch_num*self.batch_size:][:self.batch_size]
            bdataset = self.dataset[bindices, ...]
            blabels = self.labels[bindices, ...]
            yield bdataset, blabels

def process_task(cfg):
    task_type = cfg['task']['type'].get('regression')
    if task_type == 'regression':
        seed = cfg['task']['seed'].get(cfg['meta']['seed'].get())
        key = key_iter(seed)
        task = SameFamilyRegression(cfg, next(key))
        train = task.get_data(cfg, test=False, key=next(key))
        test = task.get_data(cfg, test=True, key=next(key))
        out_size = task.out_size
        return task, train, test, out_size
    elif task_type == 'classification':
        task = ImgVecClass(cfg)
        train = task.get_data(cfg, test=False)
        test = task.get_data(cfg, test=True)
        out_size = test[1].max() + 1
        return task, train, test, out_size

    else:
        raise ValueError(f"unrecognised task type: {task_type}")

@flax.struct.dataclass
class TrainState:
    epoch: int
    contents: Sequence[Optional[int]]
    solver_state: Any

def main():
    cfg = experiment_utils.get_cfg('experiment4')
    writer = experiment_utils.set_writer(cfg)

    key = key_iter(cfg['meta']['seed'].get())
    # batch_key = key_iter(cfg['meta']['batchseed'].get())

    task, (dataset, labels), (testset, testlabels), out_size = process_task(cfg)

    # task = SameFamilyRegression(cfg, next(key))
    # dataset, labels = task.get_data(cfg, False, next(key))
    train_sampler = Sampler(
        key_iter=key_iter(cfg['meta']['seed'].get()),
        dataset=dataset,
        labels=labels,
        batch_size = cfg['opt']['batch_size'].get()
    )

    # addition_batch, addition_validation_batch = islice(train_sampler.batches(), 2)
    (addition_batch,) = islice(train_sampler.batches(), 1)
    addition_validation_batch = (dataset, labels)
    # testset, testlabels = task.get_data(cfg, True, next(key))
    example = dataset[0]

    # dataset, labels = task.get_data(cfg, test=False, key=next(key))
    # testset, testlabels = task.get_data(cfg, test=True, key=next(key))

    capacities = tuple(cfg['net']['capacities'].get() + [int(out_size)])
    initial_contents = list(cfg['net']['contents'].get() + [int(out_size)])
    use_rationals = cfg['net']['rational'].get(False)
    template = ModelTemplate(capacities, initial_contents, rational=use_rationals)
    solver = Solver(cfg, template, task, next(key), example)
    proposer = Proposer(solver.template, solver.model, key_iter(cfg['meta']['propseed'].get()))

    initial_train_state = TrainState(0, initial_contents, solver.state)
    if cfg['checkpointing']['enable'].get(False) and cfg['checkpointing']['restore'].get():
        ckpt_dir = f"{cfg['checkpointing']['directory'].get()}/{cfg['meta']['name'].get()}"
        initial_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=initial_train_state
        )
        if initial_train_state.contents[-1] != initial_train_state.contents[-1]:
            print("output size changed")
            exit()
        solver.state = initial_train_state.solver_state
        template.contents = list(initial_train_state.contents)
    print(initial_train_state)

    def baseline_eval():
        return solver.last_natlen
        # return solver.eval_feature_proposal((dataset, labels), solver.state)
    
    print(f"enabled indices: {template.enabled_layers()}")
    print(f"disabled indices: {template.disabled_layers()}")

    print("Initialising...")
    print_tree_type(solver.state['params'])
    
    max_epochs = cfg['opt']['max_epochs'].get()
    rtpt = experiment_utils.get_rtpt(f"XNNs: {cfg['meta']['name'].get('untitled')}", max_iter=max_epochs)

    @dataclass
    class Proposal:
        feature: Any
        scale: float
        location: float
        theta: float

    def get_evaluator(layer_index, size=None):
        in_index, out_index = solver.template.in_out_indices(layer_index)
        # return jax.experimental.maps.xmap(
        #     lambda state, ngrad, grad, feature, pair, eps, key: kfac_direct_sgd(
        #         solver.model,
        #         solver.task.loss_function,
        #         proposer.template.layer(1 if size is None else size),
        #         state,
        #         ngrad,
        #         grad,
        #         # layer_index,
        #         # layer_index+1 if out_index is None else out_index,
        #         layer_index,
        #         out_index,
        #         feature,
        #         pair,
        #         eps,
        #         300
        #     ),
        #     ([...], [...], [...], ['features', ...], ['batch', ...], ['features', ...], ['features', ...]),
        #     (['features', ...], ['features', ...], [...], [...])
        # )
        return jax.experimental.maps.xmap(
            lambda state, ngrad, grad, feature, pair, eps, key: kfac_direct_mala(
                solver.model,
                solver.task.loss_function,
                proposer.template.layer(1 if size is None else size),
                state,
                ngrad,
                grad,
                # layer_index,
                # layer_index+1 if out_index is None else out_index,
                layer_index,
                out_index,
                feature,
                pair,
                eps,
                300,
                key,
                temp=1e1
            ),
            ([...], [...], [...], ['features', ...], ['batch', ...], ['features', ...], ['features', ...]),
            (['features', ...], ['features', ...], [...], [...])
        )

    def get_validator(layer_index, size=None):
        in_index, out_index = solver.template.in_out_indices(layer_index)
        return jax.experimental.maps.xmap(
            lambda state, ngrad, grad, feature, pair: kfac_direct_eval(
                solver.model,
                solver.task.loss_function,
                proposer.template.layer(1 if size is None else size),
                state,
                ngrad,
                grad,
                # layer_index,
                # layer_index+1 if out_index is None else out_index,
                layer_index,
                out_index,
                feature,
                pair
            ),
            ([...], [...], [...], [...], ['batch', ...]),
            ([...], [...], [...])
        )

    @dataclass
    class WidthModification:
        ratio: float
        layer_index: int
        new_state: Any

        def apply(self):
            assert solver.template.contents[self.layer_index] is not None
            prev_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
            solver.state = self.new_state
            new_size = solver.template.contents[self.layer_index] + 1
            solver.template.contents[self.layer_index] = new_size
            refresh_evaluators(layer_only=True)
            print(f"new size for layer {self.layer_index}: {new_size}")
            new_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
            assert new_loss/prev_loss < 1.001, "adding width made loss worse"

    @dataclass
    class DepthModification:
        ratio: float
        layer_index: int
        new_state: Any

        def apply(self):
            # raise NotImplementedError
            assert solver.template.contents[self.layer_index] is None
            prev_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
            old_state = solver.state
            solver.state = self.new_state
            assert self.layer_index > 0, "cannot guess new size because there is no preceding layer"
            new_size = solver.template.contents[self.layer_index-1]
            solver.template.contents[self.layer_index] = new_size
            refresh_evaluators()
            print(f"Created new layer at {self.layer_index} with size: {new_size}")
            new_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
            if new_loss/prev_loss >= 1.2:
                print(f"old state: {old_state}")
                print(f"new state: {self.new_state}")
                assert new_loss/prev_loss < 1.2, "adding layer made loss worse"

    def consider_adding_width(evaluator, validator, layer_index):
        baseline = baseline_eval()
        num_props = cfg['evo']['proposals_per_layer'].get(10)

        in_index, out_index = solver.template.in_out_indices(layer_index)
        print(f"in and out for {layer_index}: {in_index}, {out_index}")
        featstack = jax.vmap(
            lambda key: proposer.generate_feature(solver.state, layer_index, key, in_index=in_index)
        )(jax.random.split(next(proposer.key_iter), num_props))

        # def get_feature(i):
        #     return jtm(lambda arr: arr[i], featstack)
        # print(get_feature(0))

        tangent = insert_params(solver.state, solver.optimizer.read(solver.opt_state))
        full_grad = insert_params(solver.state, solver.optimizer.raw_grad(solver.opt_state))
        if cfg['evo']['pure_kfac'].get(False):
            tangent = jtm(jnp.zeros_like, tangent)
        @flax.struct.dataclass
        class OptState:
            rmetrics: Any
            epss: Any
            featstack: Any

            def get_feature(self, i):
                return jtm(lambda arr: arr[i], self.featstack)

        init_eps = 3e-1
        opt_state = OptState(
            jnp.zeros((num_props,)),
            jnp.ones((num_props,))*init_eps,
            featstack
        )
        def step(state):
            rmetrics, featstack, layer_score, loss_sqnorm = evaluator(
                solver.state,
                tangent,
                full_grad,
                state.featstack,
                addition_batch,
                state.epss,
                jax.random.split(next(proposer.key_iter), num_props)
            )
            inc = rmetrics > state.rmetrics
            # new_epss = state.epss.at[increased].set(state.epss[increased]/3.)
            return OptState(
                rmetrics,
                jnp.where(inc, state.epss, state.epss/3.),
                featstack
            ), layer_score, loss_sqnorm
        for i in range(1):
            opt_state, layer_score, loss_sqnorm = step(opt_state)
            # best = jnp.argmax(opt_state.rmetrics)
            best = jax.random.categorical(next(proposer.key_iter), logits=opt_state.rmetrics/1e1, shape=())
            # best_raw = opt_state.rmetrics[best]
            # best_raw = jax.scipy.special.logsumexp(opt_state.rmetrics) - jnp.log(jnp.size(opt_state.rmetrics))
            best_raw = jnp.sum(jax.nn.softmax(opt_state.rmetrics/1e1)*opt_state.rmetrics)
            best_ratio = 1. + best_raw/baseline
            # print(f"layer {layer_index} best_ratio: {best_ratio:.5f}")
        # def step(eps):
        #     nonlocal featstack
        #     rmetrics, featstack, loss_sqnorm = evaluator(
        #         solver.state,
        #         tangent,
        #         featstack,
        #         addition_batch,
        #         jnp.ones((num_props,))*eps
        #     )
        #     best = jnp.argmax(rmetrics)
        #     best_raw = rmetrics[best]
        #     best_ratio = 1. + best_raw/baseline
        #     print(f"layer {layer_index} best_ratio: {best_ratio:.5f}")
        #     return best, best_raw, best_ratio, loss_sqnorm
        # best, best_raw, best_ratio, loss_sqnorm = step(3e-1)
        # for i in range(300):
        #     best, best_raw, best_ratio, loss_sqnorm = step(3e-1)
        # for i in range(400):
        #     best, best_raw, best_ratio, loss_sqnorm = step(1e-1)
        # for i in range(300):
        #     best, best_raw, best_ratio, loss_sqnorm = step(3e-2)
        best_local_ratio = best_raw/layer_score
        summary.scalar(f"local baseline {layer_index}", layer_score)
        summary.scalar(f"best local ratio {layer_index}", best_local_ratio)
        summary.scalar(f"best proposal {layer_index}", best_ratio)
        summary.scalar(f"best raw rmetric {layer_index}", best_raw)
        summary.scalar(f"normed rmetric {layer_index}", best_raw/loss_sqnorm)
        summary.scalar(f"normed baseline", baseline/loss_sqnorm)
        summary.scalar(f"loss sqnorm", loss_sqnorm)

        vmetric, vlayer_score, _ = validator(
            solver.state,
            tangent,
            full_grad,
            opt_state.get_feature(best),
            addition_validation_batch
        )

        # print(f"layer {layer_index} best_vratio: {vmetric/vlayer_score:.5f}")
        print(f"layer {layer_index} best_ratio: {best_raw/layer_score:.5f} ({best_raw:.5f}/{layer_score:.5f})")

        summary.scalar(f"local validation baseline {layer_index}", vlayer_score)
        summary.scalar(f"best local validation ratio {layer_index}", vmetric/vlayer_score)
        summary.scalar(f"best raw validation metric {layer_index}", vmetric)

        # exit()

        if best_local_ratio > cfg['evo']['thresh'].get() and best_raw/loss_sqnorm > cfg['evo']['abs_thresh'].get():
            new_state = proposer.embed_feature(solver.state, opt_state.get_feature(best), layer_index)
            return WidthModification(ratio=best_ratio, layer_index=layer_index, new_state=new_state)
        else:
            return None

        #
        #     assert solver.template.contents[layer_index] is not None
        #     prev_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
        #     solver.state = proposer.embed_feature(solver.state, get_feature(best), layer_index)
        #     new_size = solver.template.contents[layer_index] + 1
        #     solver.template.contents[layer_index] = new_size
        #     refresh_evaluators(layer_only=True)
        #     print(f"new size for layer {layer_index}: {new_size}")
        #     new_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
        #     assert new_loss/prev_loss < 1.001, "adding width made loss worse"
        #     return True
        # return False

    def consider_inserting_layer(evaluator, validator, layer_index):
        assert solver.template.contents[layer_index] is None
        baseline = baseline_eval()
        num_props = cfg['evo']['layer_proposals_per_layer'].get()

        in_index, out_index = solver.template.in_out_indices(layer_index)
        new_layer_size = jnp.sum(~proposer.get_input_null(solver.state, in_index))
        print(f"new layer size: {new_layer_size}")
        featstack = jax.vmap(
            lambda key: proposer.generate_feature(
                state=solver.state,
                layer_index=layer_index,
                key=key,
                in_index=in_index,
                output_size=new_layer_size
            )
        )(jax.random.split(next(proposer.key_iter), num_props))
        # print_tree_type(featstack)

        def get_feature(i):
            return jtm(lambda arr: arr[i], featstack)
        # print(get_feature(0))

        tangent = insert_params(solver.state, solver.optimizer.read(solver.opt_state))
        full_grad = insert_params(solver.state, solver.optimizer.raw_grad(solver.opt_state))
        if cfg['evo']['pure_kfac'].get(False):
            tangent = jtm(jnp.zeros_like, tangent)
        @flax.struct.dataclass
        class OptState:
            rmetrics: Any
            epss: Any
            featstack: Any

        init_eps = 3e-1
        opt_state = OptState(
            jnp.zeros((num_props,)),
            jnp.ones((num_props,))*init_eps,
            featstack
        )
        def step(state):
            rmetrics, featstack, layer_score, loss_sqnorm = evaluator(
                solver.state,
                tangent,
                full_grad,
                state.featstack,
                addition_batch,
                state.epss,
                jax.random.split(next(proposer.key_iter), num_props)
            )
            inc = rmetrics > state.rmetrics
            return OptState(
                rmetrics,
                jnp.where(inc, state.epss, state.epss/3.),
                featstack
            ), layer_score, loss_sqnorm
        for i in range(1):
            opt_state, layer_score, loss_sqnorm = step(opt_state)
            # best = jnp.argmax(opt_state.rmetrics)
            best = jax.random.categorical(next(proposer.key_iter), logits=opt_state.rmetrics/1e1, shape=())
            # best_raw = opt_state.rmetrics[best]
            best_raw = jnp.sum(jax.nn.softmax(opt_state.rmetrics/1e1)*opt_state.rmetrics)
            best_ratio = 1. + best_raw/baseline
            # print(f"layer {layer_index} best_ratio: {best_ratio:.5f}")

        best_local_ratio = best_raw/layer_score
        summary.scalar(f"local baseline {layer_index}", layer_score)
        summary.scalar(f"best layer local ratio {layer_index}", best_local_ratio)
        summary.scalar(f"best layer proposal {layer_index}", best_ratio)
        summary.scalar(f"best layer raw metric {layer_index}", best_raw)
        summary.scalar(f"normed layer rmetric {layer_index}", best_raw/loss_sqnorm)
        summary.scalar(f"normed layer baseline", baseline/loss_sqnorm)
        summary.scalar(f"layer loss sqnorm", loss_sqnorm)

        def make_invertible(feat):
            L = feat["params"]["linear"]["kernel"]
            # print(L)
            u, s, vh = jnp.linalg.svd(L, full_matrices=False)
            # print(s)
            s = jnp.clip(s, 1e-3*jnp.mean(s), None)
            # print(u.shape)
            # print(s.shape)
            # print(vh.shape)
            L = u@jnp.diag(s)@vh
            # print(L)
            return feat.copy({"params": feat["params"].copy({
                "linear": feat["params"]["linear"].copy({
                    "kernel": L
                })
            })})

        print("making invertible...")
        best_feature = make_invertible(get_feature(best))

        vmetric, vlayer_score, _ = validator(
            solver.state,
            tangent,
            full_grad,
            best_feature,
            addition_validation_batch
        )

        summary.scalar(f"local validation baseline {layer_index}", vlayer_score)
        summary.scalar(f"best layer local validation ratio {layer_index}", vmetric / vlayer_score)
        summary.scalar(f"best layer raw validation metric {layer_index}", vmetric)

        cost_mul = cfg['evo']['layer_cost_mul'].get()
        if cfg['evo']['size_costing'].get():
            cost_mul *= new_layer_size
        adjusted_metric = best_raw / cost_mul
        print(f"layer {layer_index} adjusted ratio: {adjusted_metric/vlayer_score:.5f} ({adjusted_metric:.5f}/{vlayer_score:.5f})")

        if adjusted_metric/vlayer_score > cfg['evo']['thresh'].get() and adjusted_metric/loss_sqnorm > cfg['evo']['layer_abs_thresh'].get():
            # print("Reached layer addition, terminating.")
            # exit()
            new_state = proposer.embed_layer(solver.state, best_feature, layer_index)
            # print("Created new state")
            # exit()
            adjusted_ratio = 1. + adjusted_metric/baseline
            return DepthModification(ratio=adjusted_ratio, layer_index=layer_index, new_state=new_state)
        else:
            return None

        # rmetrics = evaluator(
        #     solver.state,
        #     tangent,
        #     featstack,
        #     (dataset, labels)
        # )
        # best = jnp.argmax(rmetrics)
        # best_raw = rmetrics[best]
        # best_ratio = 1. + best_raw/baseline
        # print(f"layer proposal {layer_index} best_ratio: {best_ratio:.5f}")
        # summary.scalar(f"best layer proposal {layer_index}", best_ratio - 1.)
        # summary.scalar(f"best layer raw rmetric {layer_index}", best_raw)
        # thresh = cfg['evo']['layer_thresh'].get(cfg['evo']['thresh'].get())
        # abs_thresh = cfg['evo']['layer_abs_thresh'].get(cfg['evo']['abs_thresh'].get())
        # if best_ratio > thresh and best_raw > abs_thresh:
        #     prev_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
        #     assert solver.template.contents[layer_index] is None
        #     # raise NotImplementedError("need to implement insertion of layer into state and recompilation")
        #     best_feature = get_feature(best)
        #     fmodule = proposer.template.layer(new_layer_size)
        #     input_size = proposer.get_input_size(solver.state, layer_index)
        #     output_size = proposer.get_output_size(solver.state, layer_index)
        #     fmodule.init(next(proposer.key_iter), jnp.zeros((3, 3, input_size)))
        #     padded_feature = fmodule.apply(best_feature, output_size - new_layer_size, method=Layer.pad_features)
        #     if output_size > input_size:
        #         padded_feature = fmodule.apply(padded_feature, output_size - input_size, method=Layer.pad_inputs)
        #     # print(f"padded feature:\n{padded_feature}")
        #     basis = padded_feature["params"]["linear"]["kernel"]
        #     shift = padded_feature["params"]["linear"]["bias"]
        #     print(f"old state:\n{solver.state}")
        #     solver.state = solver.model.apply(
        #         solver.state,
        #         dims=solver.template.contents,
        #         index=layer_index,
        #         basis=basis,
        #         shift=shift,
        #         method=solver.model.activate_layer
        #     )
        #     solver.template.contents[layer_index] = new_layer_size
        #     # solver.state = solver.model.apply(
        #     #     solver.state,
        #     #     dims=solver.template.contents,
        #     #     method=Layers.restrict_params
        #     # )
        #     print(f"new state:\n{solver.state}")
        #     # exit()
        #     refresh_evaluators()
        #     print(f"size of new layer at {layer_index}: {new_layer_size}")
        #     new_loss = solver.train_batch((dataset, labels), observe_only=True, loud=True)
        #     assert new_loss/prev_loss < 1.1, "Introducing new layer changed overall loss"
        #     return True
        # return False


    def diagnose():
        print("---- DIAGNOSIS ----:")
        solver_grad = insert_params(solver.state, solver.optimizer.SG.read(solver.opt_state))
        print(f"solver_grad: {solver_grad}")
        solver_ngrad = insert_params(solver.state, solver.optimizer.read(solver.opt_state))
        print(f"solver_ngrad: {solver_ngrad}")
        def eval_model(state):
            return jax.vmap(partial(solver.model.apply, state))(dataset)
        Y, backward = jax.vjp(eval_model, solver.state)
        # Y = jax.vmap(partial(solver.model.apply, solver.state))(dataset)
        def item_grad(l, y):
            return jax.grad(partial(solver.task.loss_function, l))(y)
        raw_dloss = jax.vmap(item_grad)(labels, Y)
        print(f"raw_dloss: {raw_dloss}")
        raw_grad = backward(raw_dloss)
        print(f"raw_grad: {raw_grad}")
        def forward(tan_state):
            Y, out = jax.jvp(eval_model, (solver.state,), (tan_state,))
            return out
        Jngrad = forward(solver_ngrad)
        print(f"Jngrad: {Jngrad}")
        out_res = raw_dloss - Jngrad
        print(f"out_res: {out_res}")
        raw_res = backward(out_res)
        print(f"raw_res: {raw_res}")
        Fngrad = backward(Jngrad)
        print(f"Fngrad: {Fngrad}")

    evaluators = [get_evaluator(i) for i in range(len(solver.template.contents[:-1]))]
    validators = [get_validator(i) for i in range(len(solver.template.contents[:-1]))]
    def refresh_evaluators(layer_only=False):
        for layer_index, con in enumerate(solver.template.contents[:-1]):
            if con is None:
                in_index, out_index = solver.template.in_out_indices(layer_index)
                new_layer_size = jnp.sum(~proposer.get_input_null(solver.state, in_index))
                evaluators[layer_index] = get_evaluator(layer_index, size=new_layer_size)
                validators[layer_index] = get_validator(layer_index, size=new_layer_size)
            elif not layer_only:
                evaluators[layer_index] = get_evaluator(layer_index)
                validators[layer_index] = get_validator(layer_index)
    refresh_evaluators()

    layer_cooldown = 0
    initial_epoch = initial_train_state.epoch
    for epoch in range(initial_epoch, max_epochs):

        for bidx, batch in enumerate(train_sampler.batches()):
            # print("hi")
            writer.set_as_default(step=bidx + train_sampler.num_batches()*epoch)
            bdata, blabels = batch
            solver.train_batch(batch)

        solver.test_batch((testset, testlabels))
        solver.test_acc((testset, testlabels))
        solver.train_acc((dataset, labels))
        solver.train_batch((dataset, labels), observe_only=True)

        proposer.verify_state(solver.state)

        if epoch % cfg['evo']['cooldown'].get(20) == 0 :
            summary.scalar(f"outer_baseline", baseline_eval())
            conts = solver.template.contents
            caps = solver.template.capacities
            def consider(stuff):
                layeridx, (cont, cap) = stuff
                if cont is not None:
                    if cont < cap - 1:
                        return consider_adding_width(evaluators[layeridx], validators[layeridx], layeridx)
                    else:
                        return None
                else:
                    if layeridx == 0 or conts[layeridx-1] is not None and layer_cooldown <= 0:
                        return consider_inserting_layer(evaluators[layeridx], validators[layeridx], layeridx)
                    else:
                        return None

            best_props = list(map(consider, enumerate(zip(conts, caps))))
            best_props = [p for p in best_props if p is not None]
            while len(best_props) > 0:
                best = max(best_props, key=lambda p: p.ratio if p is not None else 0.)
                best.apply()
                if isinstance(best, DepthModification):
                    print("recompiling...")
                    solver.recompile()
                    print(solver.state)
                    layer_cooldown = cfg['evo']['layer_cooldown'].get(0)
                else:
                    print("recompiling...")
                    solver.recompile()
                if not cfg['evo']['recursive'].get(False):
                    break
                best_props = list(map(consider, enumerate(zip(conts, caps))))
                best_props = [p for p in best_props if p is not None]

            if layer_cooldown > 0:
                layer_cooldown -= 1

            # for layeridx, (cont, cap) in enumerate(zip(conts, caps)):
            #     # print(layeridx)
            #     # print(cont)
            #     if cont is not None:
            #         if cont < cap - 1:
            #             if consider_adding_width(evaluators[layeridx], layeridx):
            #                 break
            #     else:
            #         if consider_inserting_layer(evaluators[layeridx], layeridx):
            #             break

        if cfg['checkpointing']['enable'].get(False) and epoch % cfg['checkpointing']['cooldown'].get() == 0:
            #save checkpoint
            ckpt_dir = f"{cfg['checkpointing']['directory'].get()}/{cfg['meta']['name'].get()}"
            # print(f"creating checkpoint in {ckpt_dir}")
            checkpoint_data = TrainState(epoch+1, tuple(template.contents), solver.state)
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=checkpoint_data,
                step = epoch
            )
            pass

        rtpt.step()
    exit()

if __name__ == '__main__':
    main()
