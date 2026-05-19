from functools import partial

import jax.numpy as jnp
from jax import grad, jit, lax, vmap

from couplednet_minimal.core import ForwardIVP
from couplednet_minimal.utils import ntk_fn


def high_frequency_solution(x, omega=30.0):
    return jnp.sin(omega * 2.0 * jnp.pi * x[0] ** 2 + omega / 2.0 * jnp.pi * x[1] ** 2)


def high_dynamic_range_solution(x):
    return jnp.exp(5.0 * jnp.sin(2.0 * jnp.pi * jnp.sin(jnp.pi * (x[0] ** 2 + 2.0 * x[1] ** 2))))


class FirstOrderPDE(ForwardIVP):
    """First-order PDE on [-1, 1]^2 with exact boundary values and gradient residual."""

    def __init__(self, config, solution_fn, x_star, y_star):
        super().__init__(config)
        self.solution = solution_fn
        self.u_target = vmap(solution_fn)
        self.du_target = vmap(grad(solution_fn))
        self.x_star = x_star
        self.y_star = y_star
        self.u_pred_grid = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))

    def u_ref_grid(self, x, y):
        return vmap(vmap(lambda a, b: self.solution(jnp.stack([a, b])), (None, 0)), (0, None))(x, y)

    def u_net(self, params, x, y):
        z = jnp.stack([x, y])
        return self.state.apply_fn(params, z)[0]

    def r_net(self, params, x, y):
        u_x = grad(self.u_net, argnums=1)(params, x, y)
        u_y = grad(self.u_net, argnums=2)(params, x, y)
        return jnp.stack([u_x, u_y])

    def r_net_sum(self, params, x, y):
        r = self.r_net(params, x, y)
        return r[0] + r[1]

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        x0 = jnp.stack([batch[:, 0], -jnp.ones_like(batch[:, 0])], axis=-1)
        x1 = jnp.stack([batch[:, 0], jnp.ones_like(batch[:, 0])], axis=-1)
        x2 = jnp.stack([-jnp.ones_like(batch[:, 1]), batch[:, 1]], axis=-1)
        x3 = jnp.stack([jnp.ones_like(batch[:, 1]), batch[:, 1]], axis=-1)
        boundary = jnp.concatenate([x0, x1, x2, x3], axis=0)

        u_pred = vmap(self.u_net, (None, 0, 0))(params, boundary[:, 0], boundary[:, 1])
        u_true = self.u_target(boundary)
        ics_loss = jnp.mean((u_pred - u_true) ** 2)

        if self.config.weighting.use_causal:
            d = jnp.min(jnp.ones_like(batch) - jnp.abs(batch), axis=-1)
            batch = batch[jnp.argsort(d, axis=0)]
            residual_chunks, causal_weights = self.res_and_w(params, batch)
            res_loss = jnp.mean(residual_chunks * causal_weights)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
            r_true = self.du_target(batch)
            res_loss = jnp.mean((r_pred - r_true) ** 2)

        return {"ics": ics_loss, "res": res_loss}

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, x, y):
        u_pred = self.u_pred_grid(params, x, y)
        u_true = self.u_ref_grid(x, y)
        return jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        r_pred = r_pred.reshape(self.num_chunks, -1)
        losses = jnp.mean(r_pred**2, axis=1)
        weights = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ losses)))
        return losses, weights

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        x0 = jnp.stack([batch[:, 0], -jnp.ones_like(batch[:, 0])], axis=-1)
        x1 = jnp.stack([batch[:, 0], jnp.ones_like(batch[:, 0])], axis=-1)
        x2 = jnp.stack([-jnp.ones_like(batch[:, 1]), batch[:, 1]], axis=-1)
        x3 = jnp.stack([jnp.ones_like(batch[:, 1]), batch[:, 1]], axis=-1)
        boundary = jnp.concatenate([x0, x1, x2, x3], axis=0)
        ics_ntk = vmap(ntk_fn, (None, None, 0, 0))(self.u_net, params, boundary[:, 0], boundary[:, 1])

        if self.config.weighting.use_causal:
            d = jnp.min(jnp.ones_like(batch) - jnp.abs(batch), axis=-1)
            batch = batch[jnp.argsort(d, axis=0)]
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(self.r_net_sum, params, batch[:, 0], batch[:, 1])
            res_ntk = jnp.mean(res_ntk.reshape(self.num_chunks, -1), axis=1)
            _, causal_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * causal_weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(self.r_net_sum, params, batch[:, 0], batch[:, 1])

        return {"ics": ics_ntk, "res": res_ntk}
