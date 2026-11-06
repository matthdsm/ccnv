# Math & ML Standard Library Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-rolled mathematical, statistical, and machine learning routines across `ccnv` with battle-tested implementations from `numpy`, `scipy`, and `scikit-learn` while preserving public function signatures and germline bit-for-bit invariance.

**Architecture:** Delegate 1D Gaussian mixture modeling, non-negative Lasso, truncated PCA with sign canonicalization, linear regression, Huber M-estimation, mean imputation, K-fold cross-validation, pairwise squared Euclidean distances, polynomial Vandermonde basis construction, Z-score standardization, and median absolute deviation (MAD) to `scikit-learn`, `scipy.stats`, `scipy.optimize`, and `numpy.polynomial`. Retain custom implementations only where library equivalents do not exist (`WeightedL2Cost` per-bin precision-weighted segmentation cost in `segment.py` and two-layer Sigmoid-output MLP forward pass optimized via `scipy.optimize.minimize` L-BFGS-B).

**Tech Stack:** Python 3.12+, `numpy`, `scipy` (`scipy.stats`, `scipy.optimize`, `scipy.special`), `scikit-learn` (`sklearn.mixture`, `sklearn.linear_model`, `sklearn.decomposition`, `sklearn.impute`, `sklearn.model_selection`, `sklearn.metrics.pairwise`, `sklearn.utils.extmath`).

**Spec:** `docs/designs/prenatal-cnv-integration.md`

---

## Audit Matrix: Hand-Rolled Math/ML vs. Library Replacements

| Location | Current Hand-Rolled Implementation | Lines | Replacement in `numpy` / `scipy` / `sklearn` | Action |
| :--- | :--- | :--- | :--- | :--- |
| `src/ccnv/prenatal_math.py` `infer_sex_gmm_1d` | Manual 100-iter 1D 2-GMM EM loop + manual quadratic root-finder | ~120 | `sklearn.mixture.GaussianMixture` + `scipy.optimize.brentq` | **Replace** |
| `src/ccnv/prenatal_math.py` `fit_lasso_fista` | Manual FISTA proximal gradient descent + SVD Lipschitz step | ~53 | `sklearn.linear_model.Lasso(positive=True, fit_intercept=False)` | **Replace** |
| `src/ccnv/prenatal_math.py` `pca_irlba_numpy` | Manual `svds` / `eigh` / `svd` dispatch + column sign-flip loop | ~58 | `sklearn.decomposition.PCA` + `sklearn.utils.extmath.svd_flip` | **Replace** |
| `src/ccnv/prenatal_math.py` `fit_rprop_mlp` | Manual 4-array Rprop- step-size state machine (`delta_*`, `prev_g*`) | ~116 | `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True)` *(Note: `MLPRegressor` forces identity output activation; PREFACE requires Sigmoid output)* | **Replace optimizer** |
| `src/ccnv/prenatal_math.py` `train_olm` | Manual `np.ones` column stack + `np.linalg.lstsq` | ~17 | `sklearn.linear_model.LinearRegression` | **Replace** |
| `src/ccnv/prenatal_math.py` `ols_fit` | Manual `np.ones` column stack + `np.linalg.lstsq` for 1D vectors | ~13 | `scipy.stats.linregress` | **Replace** |
| `src/ccnv/prenatal_math.py` `rlm_huber` | Manual 50-iter IRLS loop + manual MAD scale (`1.4826 * mad`) | ~42 | `sklearn.linear_model.HuberRegressor(alpha=0.0)` | **Replace** |
| `src/ccnv/prenatal.py` `fit_prenatal_models` | Manual column loop for NaN mean imputation + manual `permutation`/`array_split`/`setdiff1d` K-fold CV + manual MAD | ~35 | `sklearn.impute.SimpleImputer`, `sklearn.model_selection.KFold`, `scipy.stats.median_abs_deviation` | **Replace** |
| `src/ccnv/covariates.py` `_build_1d_basis`, `_fit_sample_trend`, `covariate_features` | Manual Z-score, manual polynomial power loop, manual MAD | ~40 | `scipy.stats.zscore`, `np.polynomial.polynomial.polyvander`, `scipy.stats.median_abs_deviation(scale="normal")` | **Replace** |
| `src/ccnv/reference.py` `_median_abs_deviation` | Manual `np.median(np.abs(values - np.median(values)))` helper | ~5 | `scipy.stats.median_abs_deviation` | **Replace** |
| `src/ccnv/reference_compute.py` `_chunk_nearest_worker` | Manual `query_norm + _w_norm - 2.0 * (batch @ _w_data.T)` + `np.maximum` | ~6 | `sklearn.metrics.pairwise.euclidean_distances(..., squared=True)` | **Replace** |
| `src/ccnv/segment.py` `WeightedL2Cost` | Prefix-sum $O(1)$ per-bin precision-weighted L2 cost for `ruptures` | ~45 | None (`ruptures.costs.CostL2` lacks sample weights) | **Keep** |

---

## Global Constraints

- All source files in `src/ccnv/` must remain strictly under 1,000 lines.
- No new third-party dependencies beyond existing `pyproject.toml` (`numpy`, `scipy`, `scikit-learn`).
- All public function names, parameter signatures, and return types in `src/ccnv/prenatal_math.py`, `src/ccnv/prenatal.py`, `src/ccnv/covariates.py`, `src/ccnv/reference.py`, and `src/ccnv/reference_compute.py` must remain unchanged.
- Germline mode (`--mode germline`) numerical outputs and test assertions must remain bit-for-bit identical.
- Suppress `sklearn.exceptions.ConvergenceWarning` on degenerate/tiny synthetic inputs where `max_iter` is intentionally small, rather than leaking warnings to test output.

## Review Focus

1. **Constant / Zero-Variance Inputs (`np.var(x) < 1e-12`)**: `infer_sex_gmm_1d`, `fit_lasso_fista`, `ols_fit`, `rlm_huber`, and `pca_irlba_numpy` must return deterministic fallback values without raising `ValueError` or `LinAlgError` when all inputs are identical.
2. **All-Zero Target Vector in `fit_lasso_fista`**: `sklearn.linear_model.Lasso` emits a `UserWarning` / `ConvergenceWarning` when `y` is all zeros; `fit_lasso_fista` must short-circuit to `np.zeros(n_features)` when `np.allclose(y, 0.0)` or `alpha >= np.max(np.abs(X.T @ y)) / n_samples`.
3. **All-NaN Columns in `pca_irlba_numpy` and `SimpleImputer`**: When a genomic bin is `NaN` across 100% of samples, mean imputation must fill `0.0` (`keep_empty_features=True` or `np.nan_to_num`) without dropping columns or changing feature matrix width.
4. **Rank-Deficient / Collinear Predictors in `rlm_huber`**: If `HuberRegressor` fails to converge on collinear or extreme synthetic inputs, it must fall back cleanly to `ols_fit(x, y)`.
5. **Single-Component Collapse in `infer_sex_gmm_1d`**: When the two fitted GMM means are identical or posterior difference does not bracket zero on `[mu_low, mu_high]`, threshold calculation must fall back to the midpoint `0.5 * (mu_low + mu_high)` without `brentq` raising `ValueError`.

---

## File Structure

- Modify: `src/ccnv/prenatal_math.py` — Replace all 7 hand-rolled math/ML implementations with `sklearn` / `scipy` / `numpy` calls (~465 lines -> ~230 lines).
- Modify: `src/ccnv/prenatal.py` — Replace manual column imputation loop with `SimpleImputer`, manual K-fold index arithmetic with `KFold`, and manual MAD with `scipy.stats.median_abs_deviation`.
- Modify: `src/ccnv/covariates.py` — Replace manual polynomial loop with `np.polynomial.polynomial.polyvander`, manual Z-score with `scipy.stats.zscore`, and manual MAD with `scipy.stats.median_abs_deviation`.
- Modify: `src/ccnv/reference.py` & `src/ccnv/reference_compute.py` — Replace `_median_abs_deviation` with `scipy.stats.median_abs_deviation` and manual squared distance expansion with `sklearn.metrics.pairwise.euclidean_distances`.
- Modify: `tests/test_prenatal_math.py` — Add edge-case tests for zero-target Lasso, all-NaN PCA columns, and collinear Huber fallback.

---

### Task 1: Replace Linear, Robust, and Lasso Solvers in `src/ccnv/prenatal_math.py`

**Files:**
- Modify: `src/ccnv/prenatal_math.py:156-209, 390-465`
- Test: `tests/test_prenatal_math.py`

**Interfaces:**
- Produces (unchanged signatures):
  - `fit_lasso_fista(X: NDArray[np.floating] | np.ndarray, y: NDArray[np.floating] | np.ndarray, alpha: float, max_iter: int = 2000, tol: float = 1e-6) -> NDArray[np.float64]`
  - `train_olm(X: NDArray[np.floating] | np.ndarray, y: NDArray[np.floating] | np.ndarray) -> tuple[NDArray[np.float64], float]`
  - `ols_fit(x: NDArray[np.floating] | np.ndarray, y: NDArray[np.floating] | np.ndarray) -> tuple[float, float]`
  - `rlm_huber(x: NDArray[np.floating] | np.ndarray, y: NDArray[np.floating] | np.ndarray, k: float = 1.345, max_iter: int = 50, tol: float = 1e-6) -> tuple[float, float]`

- [ ] **Step 1: Add edge-case unit tests in `tests/test_prenatal_math.py`**

```python
def test_fit_lasso_fista_all_zero_target_no_warning(self) -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(20, 10))
    y = np.zeros(20, dtype=np.float64)
    w = fit_lasso_fista(X, y, alpha=1e-4)
    np.testing.assert_allclose(w, 0.0, atol=1e-12)


def test_rlm_huber_matches_sklearn_huber_on_clean_and_outlier_data(self) -> None:
    x = np.linspace(0.0, 10.0, 50)
    y = 1.5 - 0.5 * x
    y[0] = 50.0
    intercept, slope = rlm_huber(x, y)
    self.assertAlmostEqual(intercept, 1.5, delta=0.1)
    self.assertAlmostEqual(slope, -0.5, delta=0.05)
```

- [ ] **Step 2: Run tests to verify baseline**

Run: `PIXI_HOME=/tmp/.pixi pixi run test -k "TestLasso or TestLinearModels"`
Expected: PASS

- [ ] **Step 3: Replace `fit_lasso_fista`, `train_olm`, `ols_fit`, and `rlm_huber` in `src/ccnv/prenatal_math.py`**

Use:
- `fit_lasso_fista`: Short-circuit to `np.zeros(n_features)` if `n_samples == 0`, `n_features == 0`, `np.allclose(x_arr, 0.0)`, or `np.allclose(y_arr, 0.0)`. Otherwise fit `sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=False, positive=True, max_iter=max_iter, tol=tol)` inside `warnings.catch_warnings()` ignoring `ConvergenceWarning`, returning `model.coef_.astype(np.float64, copy=False)`.
- `train_olm`: Fit `sklearn.linear_model.LinearRegression().fit(x_arr, y_arr)` and return `(reg.coef_.astype(np.float64, copy=False), float(reg.intercept_))`.
- `ols_fit`: Use `res = scipy.stats.linregress(xv, yv)` and return `(float(res.intercept), float(res.slope))` after the existing `xv.size < 2 or float(np.var(xv)) < 1e-12` guard.
- `rlm_huber`: Fit `sklearn.linear_model.HuberRegressor(epsilon=k, max_iter=max_iter, tol=tol, alpha=0.0).fit(xv.reshape(-1, 1), yv)` and return `(float(reg.intercept_), float(reg.coef_[0]))`, falling back to `ols_fit(xv, yv)` on `ValueError` or non-finite coefficients.

- [ ] **Step 4: Run tests to verify pass**

Run: `PIXI_HOME=/tmp/.pixi pixi run test tests/test_prenatal_math.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ccnv/prenatal_math.py tests/test_prenatal_math.py
git commit -m "refactor(prenatal): replace hand-rolled Lasso, OLS, and Huber solvers with sklearn and scipy.stats"
```

---

### Task 2: Replace 1D GMM, Truncated PCA, and Rprop Optimizer in `src/ccnv/prenatal_math.py`

**Files:**
- Modify: `src/ccnv/prenatal_math.py:33-155, 211-388`
- Test: `tests/test_prenatal_math.py`

**Interfaces:**
- Produces (unchanged signatures):
  - `infer_sex_gmm_1d(yfracs: NDArray[np.floating] | np.ndarray, seed: int = 42) -> tuple[NDArray[np.bool_], float]`
  - `pca_irlba_numpy(X: NDArray[np.floating] | np.ndarray, n_components: int, seed: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`
  - `fit_rprop_mlp(X: NDArray[np.floating] | np.ndarray, y: NDArray[np.floating] | np.ndarray, hidden_dim: int = 2, n_inits: int = 5, max_epochs: int = 1500, seed: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`

- [ ] **Step 1: Add unit test for all-NaN column handling in `pca_irlba_numpy` in `tests/test_prenatal_math.py`**

```python
    def test_pca_irlba_numpy_all_nan_column_imputed_to_zero(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(15, 8))
        X[:, 3] = np.nan
        scores, center, rotation = pca_irlba_numpy(X, n_components=3, seed=42)
        self.assertEqual(scores.shape, (15, 3))
        self.assertEqual(center[3], 0.0)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(rotation)))
```

- [ ] **Step 2: Run test to verify baseline**

Run: `PIXI_HOME=/tmp/.pixi pixi run test tests/test_prenatal_math.py`
Expected: PASS

- [ ] **Step 3: Refactor `infer_sex_gmm_1d`, `pca_irlba_numpy`, and `fit_rprop_mlp` in `src/ccnv/prenatal_math.py`**

- `infer_sex_gmm_1d`:
  - Keep `n == 0`, `n == 1`, and `abs(mu1 - mu0) < 1e-8 or sample_var < 1e-8` guards.
  - Fit `gmm = GaussianMixture(n_components=2, covariance_type="full", tol=1e-6, reg_covar=1e-8, max_iter=100, means_init=[[mu0], [mu1]], weights_init=[0.5, 0.5], random_state=seed).fit(y.reshape(-1, 1))`.
  - Identify `male_idx = int(np.argmax(gmm.means_.ravel()))`, `female_idx = 1 - male_idx`, `mu_low = float(gmm.means_.ravel()[female_idx])`, `mu_high = float(gmm.means_.ravel()[male_idx])`.
  - Find the `P(male | val) = 0.5` crossing on `[mu_low, mu_high]` using `scipy.optimize.brentq(lambda v: float(gmm.predict_proba(np.array([[v]]))[0, male_idx] - 0.5), mu_low, mu_high)` when signs differ at endpoints, else `0.5 * (mu_low + mu_high)`.
- `pca_irlba_numpy`:
  - Impute column means with `SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(x_arr)` (with `center` computed via `np.nan_to_num(np.nanmean(x_arr, axis=0), nan=0.0)`).
  - Fit `pca = PCA(n_components=k, svd_solver="arpack" if k < min(n_samples, n_features) - 1 else "full", random_state=seed)` on `x_centered`.
  - Apply `_, components = svd_flip(np.zeros((n_samples, k)), pca.components_, u_based_decision=False)` so the largest-magnitude element per component is positive; set `rotation = components.T` and `scores = x_centered @ rotation`.
- `fit_rprop_mlp`:
  - Replace the manual 4-parameter Rprop state arrays (`delta_w1..b2`, `prev_gw1..b2`) and update loop with `scipy.optimize.minimize(_loss_and_grad, theta0, method="L-BFGS-B", jac=True, options={"maxiter": max_epochs, "gtol": 1e-4})` over the flattened parameter vector `theta = [w1, b1, w2, b2]`.

- [ ] **Step 4: Run prenatal math and prenatal pipeline tests**

Run: `PIXI_HOME=/tmp/.pixi pixi run test tests/test_prenatal_math.py tests/test_prenatal.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ccnv/prenatal_math.py tests/test_prenatal_math.py
git commit -m "refactor(prenatal): use GaussianMixture, sklearn PCA, and scipy L-BFGS-B in prenatal_math"
```

---

### Task 3: Replace Manual Imputation, K-Fold Splitting, and MAD in `src/ccnv/prenatal.py`

**Files:**
- Modify: `src/ccnv/prenatal.py:350-425`
- Test: `tests/test_prenatal.py`

**Interfaces:**
- Consumes: `sklearn.impute.SimpleImputer`, `sklearn.model_selection.KFold`
- Produces: `fit_prenatal_models`, `predict_prenatal`, `classify_sca`, `annotate_prenatal_segments` (all signatures unchanged)

- [ ] **Step 1: Replace manual column imputation loop and K-fold splitting in `src/ccnv/prenatal.py`**

- In `fit_prenatal_models`:
  - Replace the `for j in range(n_auto_100k):` NaN-mean computation and `np.where` imputation (lines 351-361) with `imputer = SimpleImputer(strategy="mean", keep_empty_features=True)` on `np.where(np.isfinite(log2_ratios_100k), log2_ratios_100k, np.nan)`, extracting `X_autosomal = imputer.fit_transform(...)` and `mean_features = np.nan_to_num(imputer.statistics_, nan=0.0)`.
  - Replace manual `rng.permutation` + `np.array_split` + `np.setdiff1d` K-fold splitting (lines 396-403) with `kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)` iterating `for fold_idx, (train_part, test_part) in enumerate(kf.split(X_labeled)):`.

- [ ] **Step 2: Run prenatal test suite**

Run: `PIXI_HOME=/tmp/.pixi pixi run test tests/test_prenatal.py tests/test_reference.py tests/test_predict.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/ccnv/prenatal.py
git commit -m "refactor(prenatal): use SimpleImputer and KFold in fit_prenatal_models"
```

---

### Task 4: Replace Manual Polynomial Basis, Z-Score, MAD, and Pairwise Distance Kernels in `covariates.py`, `reference.py`, and `reference_compute.py`

**Files:**
- Modify: `src/ccnv/covariates.py:503-563, 647-672`
- Modify: `src/ccnv/reference.py:305-310, 963-965`
- Modify: `src/ccnv/reference_compute.py:61-67`
- Test: `tests/test_covariates.py`, `tests/test_reference.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `np.polynomial.polynomial.polyvander`, `scipy.stats.zscore`, `scipy.stats.median_abs_deviation`, `sklearn.metrics.pairwise.euclidean_distances`
- Produces: All existing functions in `covariates.py`, `reference.py`, and `reference_compute.py` with identical signatures and bit-for-bit germline outputs.

- [ ] **Step 1: Refactor `src/ccnv/covariates.py`**

- In `_build_1d_basis(values, degree)`:
  - Compute standardized `z = np.clip(scipy.stats.zscore(values, ddof=0), -2.5, 2.5)` and return `np.polynomial.polynomial.polyvander(z, max_cols - 1)` instead of the manual `for _ in range(1, max_cols):` loop.
- In `_fit_sample_trend(design, y)`:
  - Replace `med_res = float(np.median(residuals)); mad = float(np.median(np.abs(residuals - med_res))); delta = max(_OUTLIER_SIGMA_CUTOFF * _MAD_SIGMA_SCALE * mad, 0.08)` with `med_res = float(np.median(residuals))` and `sigma = float(scipy.stats.median_abs_deviation(residuals, scale="normal"))` (`delta = max(_OUTLIER_SIGMA_CUTOFF * sigma, 0.08)`).
- In `covariate_features(usable, gc, mappability)`:
  - Use `scipy.stats.zscore(sub, ddof=0)` when `float(np.std(sub)) > _MIN_VARIANCE_STD`.

- [ ] **Step 2: Refactor `src/ccnv/reference.py` and `src/ccnv/reference_compute.py`**

- In `src/ccnv/reference.py`:
  - Replace `_median_abs_deviation(values)` body with `return float(scipy.stats.median_abs_deviation(values, scale=1.0))`.
- In `src/ccnv/reference_compute.py`:
  - In `_chunk_nearest_worker`: replace manual `query_norm = np.sum(batch * batch, axis=1, keepdims=True); dist = query_norm + _w_norm - 2.0 * (batch @ _w_data.T); np.maximum(dist, 0.0, out=dist)` with `dist = euclidean_distances(batch, _w_data, squared=True, Y_norm_squared=_w_norm)`.

- [ ] **Step 3: Run full test suite, linter, and hooks to verify zero regression**

Run: `PIXI_HOME=/tmp/.pixi pixi run lint && PIXI_HOME=/tmp/.pixi pixi run hooks && PIXI_HOME=/tmp/.pixi pixi run test`
Expected: 100% PASS (including `tests/test_baseline.py` bit-for-bit regression checks).

- [ ] **Step 4: Commit**

```bash
git add src/ccnv/covariates.py src/ccnv/reference.py src/ccnv/reference_compute.py
git commit -m "refactor(math): use polyvander, scipy.stats.zscore, median_abs_deviation, and sklearn euclidean_distances"
```
