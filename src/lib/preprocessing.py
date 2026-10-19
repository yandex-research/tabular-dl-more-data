from typing import Self

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.utils.validation

from lib.types import PartKey


def _check_X(X):
    if not isinstance(X, np.ndarray | pd.DataFrame):
        raise TypeError('X must be either a NumPy array or a Pandas dataframe')
    return X


def _check_X_numpy(X):
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be a NumPy array')
    return X


def _check_X_1d(X):
    if X.ndim != 1:
        raise TypeError('X must be one-dimensional')
    return X


class DropConstantFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None) -> Self:
        X = _check_X(X)
        del y

        is_numpy = isinstance(X, np.ndarray)
        self.keep_idx_ = np.array(
            [
                i
                for i in range(X.shape[1])
                if len(np.unique(X[:, i]) if is_numpy else X.iloc[:, i].unique()) > 1  # pyright: ignore[reportAttributeAccessIssue]
            ]
        )
        return self

    def transform(self, X, y=None):
        X = _check_X(X)
        del y
        sklearn.utils.validation.check_is_fitted(self)

        return (
            X[:, self.keep_idx_]
            if isinstance(X, np.ndarray)
            else X.iloc[:, self.keep_idx_]
        )


class NoisyQuantileTransformer(sklearn.preprocessing.QuantileTransformer):
    def fit(self, X, y=None) -> Self:
        X = _check_X(X)

        random_state = sklearn.utils.validation.check_random_state(self.random_state)  # pyright: ignore[reportAttributeAccessIssue]
        noise_scale = 1e-5
        noise = random_state.normal(0.0, noise_scale, X.shape)
        noise = (
            noise.astype(X.dtype)
            if isinstance(X, np.ndarray)
            else pd.DataFrame(noise).astype(X.dtypes)
        )
        return super().fit(X + noise, y)


class OrdinalEncoderWithAdaptiveUnknowns(sklearn.preprocessing.OrdinalEncoder):
    def __init__(self, *, dtype: np.typing.DTypeLike = np.float64):
        dtype_name = str(np.dtype(dtype))
        if np.issubdtype(dtype, np.integer):
            unknown_value = np.iinfo(dtype_name).max
        elif np.issubdtype(dtype, np.floating):
            unknown_value = np.finfo(dtype_name).max
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')

        super().__init__(
            dtype=dtype,
            handle_unknown='use_encoded_value',
            unknown_value=unknown_value,  # pyright: ignore[reportArgumentType]
        )

    def fit(self, X, y=None):
        X = _check_X(X)
        output = super().fit(X, y)

        self.adaptive_unknown_values_ = None
        max_values = self.transform(X).max(axis=0)
        self.adaptive_unknown_values_ = 1 + (
            max_values
            if isinstance(max_values, np.ndarray)
            else max_values.to_numpy(self.dtype)  # pyright: ignore[reportAttributeAccessIssue]
        )

        return output

    def transform(self, X):
        X = _check_X(X)
        sklearn.utils.validation.check_is_fitted(self)

        X = super().transform(X)
        if self.adaptive_unknown_values_ is not None:
            unknown_value = self.unknown_value  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(X, np.ndarray):
                for i in range(X.shape[1]):
                    X[X[:, i] == unknown_value, i] = self.adaptive_unknown_values_[i]
            else:
                for i, column in enumerate(X.columns):
                    X.loc[X[:, column] == unknown_value, column] = (
                        self.adaptive_unknown_values_[i]
                    )
        return X


class StandardScaler1d(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    @staticmethod
    def _check_input(X):
        X = _check_X_numpy(X)
        X = _check_X_1d(X)
        return X

    def fit(self, X, y=None):
        X = self._check_input(X)
        del y

        # The attribute names follow sklearn.preprocessing.StandardScaler.
        self.mean_ = X.mean().item()
        self.var_ = X.var().item()
        self.scale_ = self.var_**0.5
        return self

    def transform(self, X):
        X = self._check_input(X)
        sklearn.utils.validation.check_is_fitted(self)

        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        X = self._check_input(X)

        return (X * self.scale_) + self.mean_


def fit_transform_dict(
    transformer, x: dict[PartKey, np.ndarray]
) -> dict[PartKey, np.ndarray]:
    transformer.fit(x['train'])
    return {k: transformer.transform(v) for k, v in x.items()}
