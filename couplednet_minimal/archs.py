from typing import Callable, Dict, Optional, Tuple, Union

from flax import linen as nn
from flax.core.frozen_dict import freeze
import jax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import constant, glorot_normal, normal, zeros


activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(name):
    if name not in activation_fn:
        raise NotImplementedError(f"Activation {name} not supported.")
    return activation_fn[name]


def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = jnp.exp(mean + normal(stddev)(key2, (shape[-1],)))
        return g, w / g

    return init


class PeriodEmbs(nn.Module):
    period: Tuple[float]
    axis: Tuple[int]
    trainable: Tuple[bool]

    def setup(self):
        params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                params[f"period_{idx}"] = self.param(f"period_{idx}", constant(self.period[idx]), ())
            else:
                params[f"period_{idx}"] = self.period[idx]
        self.period_params = freeze(params)

    @nn.compact
    def __call__(self, x):
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)
        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2))
        return jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)


class Embedding(nn.Module):
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)
        return x


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(self.kernel_init, self.reparam["mean"], self.reparam["stddev"]),
                (x.shape[-1], self.features),
            )
            kernel = g * v
        else:
            raise NotImplementedError(f"Unknown reparam type: {self.reparam['type']}")
        bias = self.param("bias", self.bias_init, (self.features,))
        return jnp.dot(x, kernel) + bias


class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None
    nonlinearity: Union[None, float] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        for _ in range(self.num_layers):
            x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        return Dense(self.out_dim, reparam=self.reparam)(x)


class Bottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        identity = x
        x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        x = Dense(self.output_dim, reparam=self.reparam)(x)
        return self.activation_fn(x + identity)


class ResNet(nn.Module):
    arch_name: Optional[str] = "ResNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None
    nonlinearity: Union[None, float] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        for _ in range(self.num_layers):
            x = Bottleneck(self.hidden_dim, x.shape[-1], self.activation, self.reparam)(x)
        return Dense(self.out_dim, reparam=self.reparam)(x)


class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, u, v):
        identity = x
        x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        x = x * u + (1.0 - x) * v
        x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        x = x * u + (1.0 - x) * v
        x = self.activation_fn(Dense(self.output_dim, reparam=self.reparam)(x))
        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        return alpha * x + (1.0 - alpha) * identity


class PirateNet(nn.Module):
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        u = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        v = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                self.hidden_dim, x.shape[-1], self.activation, self.nonlinearity, self.reparam
            )(x, u, v)
        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            return jnp.dot(x, kernel)
        return Dense(self.out_dim, reparam=self.reparam)(x)


class ModifiedMlp(nn.Module):
    arch_name: Optional[str] = "ModifiedMlp"
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, Dict] = None
    nonlinearity: Union[None, float] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        x = Embedding(self.periodicity, self.fourier_emb)(x)
        u = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        v = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
        for _ in range(self.num_layers):
            x = self.activation_fn(Dense(self.hidden_dim, reparam=self.reparam)(x))
            x = x * u + (1.0 - x) * v
        return Dense(self.out_dim, reparam=self.reparam)(x)


class CoupledMlp(nn.Module):
    arch_name: Optional[str] = "CoupledMlp"
    permute: bool = True
    num_layers: int = 4
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    nonlinearity: float = 0.0
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)
        if self.fourier_emb:
            x = Dense(self.hidden_dim, reparam=self.reparam)(x)
            x = x + FourierEmbs(**self.fourier_emb)(x)
        else:
            x = Dense(self.hidden_dim, reparam=self.reparam)(x)

        if self.hidden_dim % 2 != 0:
            raise ValueError("CoupledMlp requires an even hidden_dim.")

        for i in range(self.num_layers):
            if self.permute:
                x = random.permutation(random.PRNGKey(i), x, axis=-1)
            x0, x1 = jnp.split(x, 2, axis=-1)
            sx = self.activation_fn(Dense(self.hidden_dim // 2, reparam=self.reparam)(x0))
            sx = self.activation_fn(Dense(self.hidden_dim // 2, reparam=self.reparam)(sx))
            sx = nn.LayerNorm(use_scale=True, use_bias=False)(sx) * jnp.log(2.0) / jnp.sqrt(self.num_layers) / 2.3263
            sx = jnp.exp(sx)

            tx = self.activation_fn(Dense(self.hidden_dim // 2, reparam=self.reparam)(x0))
            tx = Dense(self.hidden_dim // 2, reparam=self.reparam)(tx)
            x = jnp.concatenate([x0, sx * x1 + tx], axis=-1)

        return Dense(self.out_dim, reparam=self.reparam)(x)
