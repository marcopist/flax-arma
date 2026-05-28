import flax.linen as nn
import jax.numpy as jnp
import jax
from jax.scipy.stats import multivariate_normal
from jax import Array


class ARMA(nn.Module):
    ar_lag: int
    ma_lag: int
    dim: int

    _ar_mats: list[nn.Dense]
    _ma_mats: list[nn.Dense]
    _error_covariance: Array

    def setup(self):
        self._ar_mats = [nn.Dense(self.dim, use_bias=False) for _ in range(self.ar_lag)]
        self._ma_mats = [nn.Dense(self.dim, use_bias=False) for _ in range(self.ma_lag)]

        # NOTE: we need to estimate a covariance matrix for the gaussian error in the ARMA process
        # This matrix needs to be positive definite. The minimum number of parameters needed to construct
        # this matrix is d * (d + 1) / 2. So we:
        # - construct a vector of this ^ number of parameters
        # - reshape it into an upper triangular matrix
        # - apply softplus to the diagonal entries (to guarantee positive definiteness)
        # - multiply this matrix by its transpose (Cholesky)
        # and we finally obtain the covariance matrix we need.

        error_covar = self.param("error_covar", nn.initializers.normal(), (self.dim * (self.dim + 1) // 2,))
        error_cholesky_mat = (
            jnp.zeros((self.dim, self.dim))
            .at[jnp.triu_indices(self.dim)]
            .set(error_covar)
            .at[jnp.diag_indices(self.dim)]
            .apply(nn.softplus)
        )
        self._error_covariance = error_cholesky_mat @ jnp.transpose(error_cholesky_mat)

    def __call__(self, x: Array, initial_residuals: Array | None = None) -> Array:
        """Compute log likelihood of every observation after the `ar_lag`-th.

        Args:
            - x: Array[T, N], the N-dimensional, T-length time series

        Returns:
            Array[T - ar_lag,], the log-likelihood of each observation following the `ar_lag`.

        """
        T, N = x.shape
        assert N == self.dim
        assert T > self.ar_lag
        initial_outputs = x[:self.ar_lag, :]
        if initial_residuals is None:
            # When initial residuals are not passed, they are assumed to be 0.
            _initial_residuals = jnp.zeros((self.ma_lag, self.dim))
        else:
            assert initial_residuals.shape == (self.ma_lag, self.dim)
            _initial_residuals = initial_residuals
        intial_carry = (initial_outputs, _initial_residuals)

        _final_carry, scores = jax.lax.scan(self.log_likelihood, intial_carry, x[self.ar_lag:, :])
        return scores

    def log_likelihood(self, carry: tuple[Array, Array], nxt: Array) -> tuple[tuple[Array, Array], Array]:
        """Compute the log likelihood of an individual observation given the priors.

        Args:
            - carry: tuple[Array[ar_lag, N], Array[ma_lag, N]], the sliding window of
                accumulated observations and initial_residuals
            - nxt: Array[N,], a single observation whose log-likelihood needs calculating

        Returns:
            tuple[tuple[Array[ar_lag, N], Array[ma_lag, N]], Array[,]], updated carry and scalar log-likelihood
                of the individual observation
        """
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
        



