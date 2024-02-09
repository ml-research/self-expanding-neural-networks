from functools import partial
from compose import compose
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from flax import struct

import wandb


def direct_update(M, v, multiplier=1.0):
    return M + jnp.outer(v, v) * multiplier


def inv_update(M, v, multiplier=1.0, soln=None):
    EPS = 1e-12
    Mv = M @ v
    denom = 1 + jnp.inner(v, Mv.conj()) * multiplier
    new_mult = -multiplier / (denom + EPS)
    if soln is None:
        return direct_update(M, Mv, multiplier=new_mult)
    else:
        delta_soln = Mv @ soln
        delta_soln = jnp.outer(Mv, delta_soln) * new_mult
        new_soln = soln + delta_soln
        return direct_update(M, Mv, multiplier=new_mult), new_soln


def chol_update(M, v, multiplier=1.0):
    return tfp.math.cholesky_update(M, v, multiplier=multiplier)


def ichol_update(M, v, multiplier=1.0):
    MTv = M.T @ v
    denom = 1 + jnp.inner(MTv, MTv.conj()) * multiplier
    new_mult = -multiplier / denom
    return chol_update(M, M @ MTv, multiplier=new_mult)


class SecondMoment(struct.PyTreeNode):
    direct: jax.Array
    inv: jax.Array
    chol: jax.Array
    ichol: jax.Array

    def scale_by(self, scale):
        return self.replace(
            direct=self.direct * scale,
            inv=self.inv / scale,
            chol=self.chol * jnp.sqrt(scale),
            ichol=self.ichol / jnp.sqrt(scale),
        )

    def rank_one_update(self, v, multiplier=1.0, decay=None, soln=None):
        scale = 1.0 if decay is None else decay
        multiplier = multiplier if decay is None else (1.0 - decay) * multiplier
        inv, soln = inv_update(self.inv / scale, v, multiplier=multiplier, soln=soln)
        newmom = SecondMoment(
            direct_update(self.direct * scale, v, multiplier=multiplier),
            inv,
            chol_update(self.chol * jnp.sqrt(scale), v, multiplier=multiplier),
            ichol_update(self.ichol / jnp.sqrt(scale), v, multiplier=multiplier),
        )
        return newmom if soln is None else newmom, soln

    def init_identity(size):
        return SecondMoment(*(jnp.identity(size),) * 4)


class Whitener(struct.PyTreeNode):
    iroot: Any
    eps: float = 1e-12
    mag_cap: float = 1e3

    # for compatibility with senn.opt.HessTracker
    def init(self, x, initial_precision=1.0):
        return self.init_identity(x.size).rescale(initial_precision**2)

    @classmethod
    def init_identity(cls, size, **kwargs):
        return cls(iroot=jnp.identity(size), **kwargs)

    def rescale(self, factor):
        # trace = jnp.sum(jnp.square(jnp.abs(self.iroot)), axis=-2)
        new_iroot = self.iroot * jnp.reciprocal(jnp.sqrt(factor))
        # mag_cap = wandb.config.get("inv_root_curvature_diag_elem_cap", self.mag_cap)
        # new_iroot = jnp.where(trace[None,:] < self.mag_cap, self.iroot, new_iroot)
        return self.replace(iroot=new_iroot)
        # return self.replace(iroot=self.iroot*jnp.reciprocal(jnp.sqrt(factor)))

    def trace_inv(self):
        return jnp.sum(jnp.square(jnp.abs(self.iroot)))

    def diag_inv(self):
        return jnp.sum(jnp.square(jnp.abs(self.iroot)), axis=-1)

    def iroot_mul(self, tangents):
        return tangents @ self.iroot.T

    def whiten(self, tangents):
        return tangents @ self.iroot

    def solve(self, tangents):
        return self.iroot_mul(self.whiten(tangents))

    def w_solve(self, whites):
        return self.iroot_mul(whites)

    def w_rank_n_update(self, whites):
        def update(carry, white):
            return carry.w_rank_one_update(white), None

        out, _ = jax.lax.scan(update, self, whites)
        return out

    def rank_n_update(self, vecs):
        return self.w_rank_n_update(self.whiten(vecs))

    @abstractmethod
    def w_rank_one_update(self, white):
        raise NotImplementedError

    def rank_one_update(self, vec):
        return self.w_rank_one_update(self.whiten(vec))


class LinearWhitener(Whitener):
    def multiplier(self, white_sqnorm):
        x = -white_sqnorm * jnp.reciprocal(1.0 + white_sqnorm)
        x = jnp.expm1(0.5 * jnp.log1p(jnp.maximum(-1.0, x)))
        x = x * jnp.reciprocal(self.eps + white_sqnorm)
        jax.lax.cond(
            jnp.isfinite(x).all(),
            lambda a: None,
            lambda a: jax.debug.print("multiplier error with sqnorm {}", white_sqnorm),
            x,
        )
        return x

    @abstractmethod
    def _get_factor(self, whites):
        pass

    def _iroot_update(self, factor):
        identity = jnp.identity(self.iroot.shape[-1])
        # new_iroot = self.iroot @ (identity + factor)
        new_iroot = self.iroot + self.iroot @ factor
        return self.replace(iroot=new_iroot)

    def w_rank_n_update(self, whites):
        return self._iroot_update(self._get_factor(whites))

    def w_rank_one_update(self, white):
        return self.w_rank_n_update(white[None, ...])

    def w_rank_n_inv_update(self, whites):
        wmag = jnp.sum(jnp.square(jnp.abs(whites)))
        # jax.debug.print("inv_update wmag: {}", wmag)
        # jax.lax.cond(
        #        jnp.isfinite(wmag),
        #        lambda x: None,
        #        lambda x: jax.debug.breakpoint(),
        #        None)

        mul = jnp.expm1(0.5 * jnp.log1p(wmag))
        mul = mul * jnp.reciprocal(wmag + self.eps)
        factor = mul * whites.T @ whites
        # jax.lax.cond(jnp.isfinite(factor).all(), lambda x: None, lambda x: jax.debug.breakpoint(), 0)
        return self._iroot_update(factor)


class IRootWhitener(LinearWhitener):
    def _get_factor(self, whites):
        wmag = jnp.sum(jnp.inner(whites, whites))
        return self.multiplier(wmag) * whites.T @ whites


class DiagWhitener(LinearWhitener):
    def _get_factor(self, whites):
        mags = jax.vmap(lambda w: jnp.sum(jnp.inner(w, w)), in_axes=-1, out_axes=-1)(
            whites
        )
        return jnp.diag(self.multiplier(mags) * mags)


class HybridWhitener(IRootWhitener, DiagWhitener):
    diag_fraction: float = 0.8

    def check_iroot_finite(self, note):
        jax.lax.cond(
            jnp.isfinite(self.iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print(note),
            0,
        )

    def w_rank_n_update(self, whites):
        # FOR DEBUGGING:
        # return self._iroot_update(IRootWhitener._get_factor(self, whites))
        # END DEBUGGING

        out = self
        diag_whites = jnp.sqrt(self.diag_fraction) * whites
        diag_factor = DiagWhitener._get_factor(self, diag_whites)
        out = out._iroot_update(diag_factor)
        out.check_iroot_finite("nonfinite iroot after diag update")
        identity = jnp.identity(self.iroot.shape[-1])
        # iroot_whites = jnp.sqrt(1. - self.diag_fraction)*whites@(identity + diag_factor)
        iroot_whites = jnp.sqrt(1.0 - self.diag_fraction) * whites
        iroot_whites = iroot_whites + iroot_whites @ diag_factor
        iroot_factor = IRootWhitener._get_factor(self, iroot_whites)
        out = out._iroot_update(iroot_factor)
        out.check_iroot_finite("nonfinite iroot after non-diag update")
        return out


class MaskedWhitener(HybridWhitener):
    """Important: do not call w_rank_one_update or w_rank_n_update since this class needs access to unwhitened updates."""

    direct: Any = None
    mask: Any = None

    def rescale(self, factor):
        out = super().rescale(factor)
        return out.replace(direct=out.direct * factor)

    def _direct_from_iroot(self):
        return jnp.linalg.inv(self.iroot @ self.iroot.T)

    def _recompute_direct(self):
        return self.replace(direct=self._direct_from_iroot())

    def init(self, x, *args, **kwargs):
        out = super().init(x, *args, **kwargs)
        out = out.replace(mask=jnp.ones(x.size, dtype=jnp.bool_))
        return out._recompute_direct()

    @classmethod
    def init_identity(cls, size, **kwargs):
        kwargs = {"mask": jnp.ones(size, dtype=jnp.bool_), **kwargs}
        out = super(MaskedWhitener, cls).init_identity(size, **kwargs)
        return out._recompute_direct()

    def rank_one_update(self, vec):
        return self.rank_n_update(vec[None, ...])

    def rank_n_update(self, vecs):
        out = super().rank_n_update(vecs)
        delta_direct = vecs.T @ vecs
        # adjust for diag_fraction to make updates consistent with direct
        delta_direct = (
            delta_direct * (1.0 - self.diag_fraction)
            + jnp.diag(jnp.diag(delta_direct)) * self.diag_fraction
        )
        out = out.replace(direct=out.direct + delta_direct)
        jax.lax.cond(
            jnp.isfinite(self.iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print("nonfinite iroot"),
            0,
        )
        return out.maybe_reset_iroot("rank_n_update")

    def direct_mul(self, vecs):
        return vecs @ self.direct

    def reset_iroot(self):
        D = len(self.direct)
        mdirect = jnp.where(
            self.mask[:, None] & self.mask, self.direct, jnp.identity(D)
        )
        chol = jnp.linalg.cholesky(mdirect)
        chol = jnp.where(self.mask[:, None] & self.mask, chol, jnp.zeros_like(chol))
        iroot = jnp.linalg.pinv(chol.T)
        # pinv = jnp.linalg.pinv(self.direct)
        # iroot = jnp.linalg.cholesky(pinv)
        return self.replace(iroot=iroot)

    def maybe_reset_iroot(self, where):
        D = len(self.iroot)
        error = self.iroot.T @ self.direct @ self.iroot - jnp.identity(D)
        error_norm = jnp.max(jnp.abs(error))
        should_reset = (error_norm > 1e0) | (~jnp.isfinite(error_norm).all())

        def do_reset(error_norm):
            if wandb.config.iroot_error_warn:
                jax.debug.print(
                    "iroot error norm reached {} in " + where + ", recalculating...",
                    error_norm,
                )
            # jax.debug.print("iroot finite: {}", jnp.isfinite(self.iroot).all())
            # jax.debug.print("direct finite: {}", jnp.isfinite(self.direct).all())
            # jax.debug.print("error finite: {}", jnp.isfinite(error).all())
            # jax.debug.print("direct: {}", self.direct)
            return self.reset_iroot()

        def do_nothing(error_norm):
            return self

        return jax.lax.cond(should_reset, do_reset, do_nothing, error_norm)

    def kill_latent(self, idx):
        # jax.debug.print("iroot sandwich {}",self.iroot.T @ self.direct @ self.iroot)
        # remove redundant column from iroot at idx
        true_at_idx = jnp.arange(len(self.mask)) == idx
        ir_col = jax.lax.dynamic_index_in_dim(self.iroot, idx, axis=-1)
        # could substitute lines below with a preconditioned solve?
        Dmul = self.direct @ ir_col
        Dmul = jnp.where(self.mask[:, None], Dmul, 0.0)
        mag = jnp.sum(ir_col.T @ Dmul)
        scale = jnp.reciprocal(jnp.maximum(1e-3, 1.0 - mag))
        # scale = 1e-6
        new_iroot = jnp.where(true_at_idx[None, :], 0.0, self.iroot)
        # new_var = new_iroot.T @ Dmul * scale
        precon = new_iroot.T @ (self.direct + Dmul @ Dmul.T * scale)
        # new_var = jnp.linalg.lstsq(new_iroot, ir_col)[0]
        new_var = jax.scipy.sparse.linalg.gmres(
            new_iroot,
            ir_col,
            M=precon,
            maxiter=1,
            restart=20,
            solve_method="incremental",
        )[0]
        return self.replace(iroot=new_iroot).w_rank_n_inv_update(new_var.T)

    def freeze(self, idx):
        # jax.debug.print("pre-freeze iroot sandwich {}",jnp.diag(self.iroot.T @ self.direct @ self.iroot))
        ir_row = jax.lax.dynamic_index_in_dim(self.iroot, idx)  # [1, N]
        ir_row_mag = jnp.sum(jnp.square(jnp.abs(ir_row)))
        normed = ir_row / (jnp.sqrt(ir_row_mag + self.eps))
        delta = (self.iroot @ normed.T) @ normed
        true_at_idx = jnp.arange(len(self.mask)) == idx
        new_iroot = self.iroot - delta
        new_iroot = jnp.where(true_at_idx[:, None], 0.0, new_iroot)
        # STILL NEED TO REINSERT VARIANCE FROM FOLLOWING LINE
        # extra_var = jnp.where(true_at_idx, 0., ir_row)
        # THIS PROBABLY IS WRONG:
        # extra_wvar = jnp.where(true_at_idx, 0., self.direct @ ir_row)
        # new_iroot = jnp.where(true_at_idx[None, :], 0., new_iroot)
        new_mask = self.mask & ~true_at_idx
        new_whitener = self.replace(iroot=new_iroot, mask=new_mask)
        jax.lax.cond(
            jnp.isfinite(new_iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print("freeze error"),
            0,
        )
        # remove redundant column of self.iroot
        new_whitener = new_whitener.kill_latent(idx)
        jax.lax.cond(
            jnp.isfinite(new_whitener.iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print("kill latent error"),
            0,
        )
        return new_whitener.maybe_reset_iroot("freeze")

    def freeze_many(self, where):
        cond_fn = lambda tup: jnp.any(tup[0])

        def body_fn(tup):
            where, out = tup
            idx = jnp.argmax(where)
            out = out.freeze(idx)
            where = where & ~(jnp.arange(len(where)) == idx)
            return where, out

        where, out = jax.lax.while_loop(cond_fn, body_fn, (where, self))
        return out

    def thaw(self, idx):
        dir_row = jax.lax.dynamic_index_in_dim(self.direct, idx)
        true_at_idx = jnp.arange(len(self.mask)) == idx
        new_col = -self.iroot @ (self.iroot.T @ dir_row.T)
        new_col = jnp.where(true_at_idx[:, None], 1.0, new_col)
        dir_elem = jax.lax.dynamic_index_in_dim(dir_row, idx, axis=1)

        dir_elem = jnp.maximum(dir_elem + dir_row @ new_col, 1e-3 * dir_elem)
        dir_elem = jnp.maximum(dir_elem, self.eps)

        col_scale = jnp.reciprocal(jnp.sqrt(jnp.abs(dir_elem)))
        new_iroot = jnp.where(true_at_idx[None, :], new_col * col_scale, self.iroot)

        # iroot_row = jnp.reciprocal(jnp.sqrt(jnp.abs(dir_row)) + self.eps)
        # only_diag = jnp.where(true_at_idx[None, :], iroot_row, jnp.zeros_like(iroot_row))
        # new_iroot = jnp.where(true_at_idx[:, None], only_diag, self.iroot)
        new_mask = self.mask | true_at_idx
        # jax.lax.cond(jnp.isfinite(new_iroot).all(), lambda x: None, lambda x: jax.debug.breakpoint(), 0)
        jax.lax.cond(
            jnp.isfinite(new_iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print("thaw error"),
            0,
        )
        health = jnp.max(jnp.diag(new_iroot.T @ self.direct @ new_iroot))
        jax.lax.cond(
            health < 1e3,
            lambda x: None,
            lambda x: jax.debug.print("WARN: thaw health"),
            0,
        )
        return self.replace(iroot=new_iroot, mask=new_mask).maybe_reset_iroot("thaw")

    def thaw_many(self, where):
        cond_fn = lambda tup: jnp.any(tup[0])

        def body_fn(tup):
            where, out = tup
            idx = jnp.argmax(where)
            out = out.thaw(idx)
            where = where & ~(jnp.arange(len(where)) == idx)
            return where, out

        where, out = jax.lax.while_loop(cond_fn, body_fn, (where, self))
        return out

    @jax.jit
    def gmres_solve(self, vecs):
        subvecs = jnp.where(self.mask, vecs, 0.0)
        subdirect = jnp.where(self.mask[:, None] & self.mask, self.direct, 0.0)
        precon = self.iroot @ self.iroot.T
        solns = jax.scipy.sparse.linalg.gmres(
            subdirect,
            subvecs.T,
            M=precon,
            maxiter=1,
            restart=20,
            solve_method="batched",
        )[0].T
        return solns

    @jax.jit
    def cg_solve(self, vecs):
        subvecs = jnp.where(self.mask, vecs, 0.0)
        subdirect = jnp.where(self.mask[:, None] & self.mask, self.direct, 0.0)
        precon = self.iroot @ self.iroot.T
        solns = jax.scipy.sparse.linalg.cg(
            subdirect,
            subvecs.T,
            M=precon,
            maxiter=20,
        )[0].T
        return solns

    @jax.jit
    def cg_project(self, vecs):
        vecs = vecs @ self.direct
        return self.cg_solve(vecs)

    def freeze_prune_thaw_scores(self, grads, params, ngrad=None):
        if ngrad is None:
            ngrad = grads @ self.iroot @ self.iroot.T
        frz_scaling = jnp.reciprocal(
            jnp.sqrt(jnp.maximum(self.eps, jnp.abs(self.diag_inv())))
        )
        if wandb.config.expansion_lower_bound:
            frz_scaling = jnp.sqrt(jnp.diag(self.direct))
        tha_center = jnp.abs(jnp.diag(self.direct))
        if not wandb.config.expansion_lower_bound:
            tha_center_ = jnp.abs(
                tha_center - tha_center * self.diag_inv() * tha_center
            )
            tha_center = jnp.maximum(0e-2 * tha_center, tha_center_)
        tha_scaling = jnp.reciprocal(jnp.sqrt(jnp.maximum(self.eps, tha_center)))
        root_scores = (
            ngrad * frz_scaling,
            (ngrad - params) * frz_scaling,
            (grads - ngrad @ self.direct) * tha_scaling,
        )
        scores = tuple(jnp.sum(jnp.square(jnp.abs(rs)), axis=0) for rs in root_scores)
        return scores

    def thaw(self, idx):
        dir_row = jax.lax.dynamic_index_in_dim(self.direct, idx)
        true_at_idx = jnp.arange(len(self.mask)) == idx
        new_col = -self.iroot @ (self.iroot.T @ dir_row.T)
        new_col = jnp.where(true_at_idx[:, None], 1.0, new_col)
        dir_elem = jax.lax.dynamic_index_in_dim(dir_row, idx, axis=1)

        dir_elem = jnp.maximum(dir_elem + dir_row @ new_col, 1e-3 * dir_elem)
        dir_elem = jnp.maximum(dir_elem, self.eps)

        col_scale = jnp.reciprocal(jnp.sqrt(jnp.abs(dir_elem)))
        new_iroot = jnp.where(true_at_idx[None, :], new_col * col_scale, self.iroot)

        # iroot_row = jnp.reciprocal(jnp.sqrt(jnp.abs(dir_row)) + self.eps)
        # only_diag = jnp.where(true_at_idx[None, :], iroot_row, jnp.zeros_like(iroot_row))
        # new_iroot = jnp.where(true_at_idx[:, None], only_diag, self.iroot)
        new_mask = self.mask | true_at_idx
        # jax.lax.cond(jnp.isfinite(new_iroot).all(), lambda x: None, lambda x: jax.debug.breakpoint(), 0)
        jax.lax.cond(
            jnp.isfinite(new_iroot).all(),
            lambda x: None,
            lambda x: jax.debug.print("thaw error"),
            0,
        )
        health = jnp.max(jnp.diag(new_iroot.T @ self.direct @ new_iroot))
        jax.lax.cond(
            health < 1e3,
            lambda x: None,
            lambda x: jax.debug.print("WARN: thaw health"),
            0,
        )
        return self.replace(iroot=new_iroot, mask=new_mask).maybe_reset_iroot("thaw")

    def thaw_many(self, where):
        cond_fn = lambda tup: jnp.any(tup[0])

        def body_fn(tup):
            where, out = tup
            idx = jnp.argmax(where)
            out = out.thaw(idx)
            where = where & ~(jnp.arange(len(where)) == idx)
            return where, out

        where, out = jax.lax.while_loop(cond_fn, body_fn, (where, self))
        return out

    @jax.jit
    def gmres_solve(self, vecs):
        subvecs = jnp.where(self.mask, vecs, 0.0)
        subdirect = jnp.where(self.mask[:, None] & self.mask, self.direct, 0.0)
        precon = self.iroot @ self.iroot.T
        solns = jax.scipy.sparse.linalg.gmres(
            subdirect,
            subvecs.T,
            M=precon,
            maxiter=1,
            restart=20,
            solve_method="batched",
        )[0].T
        return solns

    @jax.jit
    def cg_solve(self, vecs):
        subvecs = jnp.where(self.mask, vecs, 0.0)
        subdirect = jnp.where(self.mask[:, None] & self.mask, self.direct, 0.0)
        precon = self.iroot @ self.iroot.T
        solns = jax.scipy.sparse.linalg.cg(
            subdirect,
            subvecs.T,
            M=precon,
            maxiter=20,
        )[0].T
        return solns

    def freeze_prune_thaw_scores(self, grads, params, ngrad=None):
        if ngrad is None:
            ngrad = grads @ self.iroot @ self.iroot.T
        frz_scaling = jnp.reciprocal(
            jnp.sqrt(jnp.maximum(self.eps, jnp.abs(self.diag_inv())))
        )
        tha_center = jnp.abs(jnp.diag(self.direct))
        tha_center_ = jnp.abs(tha_center - tha_center * self.diag_inv() * tha_center)
        tha_center = jnp.maximum(0e-2 * tha_center, tha_center_)
        tha_scaling = jnp.reciprocal(jnp.sqrt(jnp.maximum(self.eps, tha_center)))
        root_scores = (
            ngrad * frz_scaling,
            (ngrad - params) * frz_scaling,
            (grads - ngrad @ self.direct) * tha_scaling,
        )
        scores = tuple(jnp.sum(jnp.square(jnp.abs(rs)), axis=0) for rs in root_scores)
        return scores
