import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.scipy.stats import multivariate_normal


class ARMA(nn.Module):
    ar_lag: int
    ma_lag: int
    dim: int

    def setup(self):
        self._ar_mats = [nn.Dense(self.dim, use_bias=False) for _ in range(self.ar_lag)]
        self._ma_mats = [nn.Dense(self.dim, use_bias=False) for _ in range(self.ma_lag)]
        error_covar = self.param("error_covar", nn.initializers.normal(), (self.dim * (self.dim + 1) / 2,))
        error_cholesky_mat = (
            jnp.zeros((self.dim, self.dim))
            .at[jnp.triu_indices(self.dim)]
            .set(error_covar)
            .at[jnp.diag_indices(self.dim)]
            .apply(nn.softplus)
        )
        self._error_covariance = error_cholesky_mat @ jnp.transpose(error_cholesky_mat)

    def __call__(self, x):
        T, N = x.shape
        assert N == self.dim
        assert T > max(self.ar_lag, self.ma_lag)
        initial_outputs = x[:self.ar_lag, :]
        initial_residuals = jnp.zeros((self.ma_lag, N))
        intial_carry = (initial_outputs, initial_residuals)

        return jax.lax.scan(self.log_likelihood, intial_carry, x[self.ar_lag:, :])

    def log_likelihood(self, carry, nxt):
        outputs, residuals = carry
        assert (self.ar_lag, self.dim) == outputs.shape
        assert (self.ma_lag, self.dim) == residuals.shape

        mean_est = jnp.zeros(self.dim)

        for i in range(self.ar_lag):
            mean_est += self._ar_mats[i](outputs[i, :])

        for i in range(self.ma_lag):
            mean_est += self._ma_mats[i](residuals[i, :])

        residual = nxt - mean_est
        res = multivariate_normal.logpdf(nxt, mean_est, self._error_covariance)

        new_outputs = jnp.concatenate([outputs[1:, :], jnp.expand_dims(nxt, 0)])
        new_residuals = jnp.concatenate([residuals[1:, :], jnp.expand_dims(residual, 0)])

        new_carry = (new_outputs, new_residuals)

        return new_carry, res
        



