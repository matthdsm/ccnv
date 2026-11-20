use std::iter::once;

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::invalid_input_error;

#[inline]
fn neumaier_add(sum: f64, comp: f64, x: f64) -> (f64, f64) {
    let next = sum + x;
    if !next.is_finite() {
        return (next, 0.0);
    }
    let delta = if sum.abs() >= x.abs() {
        (sum - next) + x
    } else {
        (x - next) + sum
    };
    (next, comp + delta)
}

/// Mean-shifted prefix-sum weighted L2 PELT segmentation kernel.
///
/// Mirrors `WeightedL2Cost` + `ruptures.Pelt._seg` exact DP recursion and
/// first-minimum tie-breaking while returning `[]` for `n == 0` and `[n]` for
/// `0 < n < 2 * min_size` after input validation.
pub(crate) fn pelt_weighted_l2_core(
    values: &[f64],
    weights: &[f64],
    penalty: f64,
    min_size: usize,
    jump: usize,
) -> Result<Vec<usize>, String> {
    if values.len() != weights.len() {
        return Err("values and weights must have equal length".to_owned());
    }
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err("penalty must be finite and positive".to_owned());
    }
    if min_size < 1 {
        return Err("min_size must be at least 1".to_owned());
    }
    if jump < 1 {
        return Err("jump must be at least 1".to_owned());
    }
    if weights.iter().any(|&w| w.is_finite() && w < 0.0) {
        return Err("weights must be non-negative".to_owned());
    }

    let n = values.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n < min_size.saturating_mul(2) {
        return Ok(vec![n]);
    }

    let (mut total_w_sum, mut total_w_comp) = (0.0_f64, 0.0_f64);
    let (mut total_wv_sum, mut total_wv_comp) = (0.0_f64, 0.0_f64);
    for (&v, &w) in values.iter().zip(weights.iter()) {
        if v.is_finite() && w.is_finite() && w > 0.0 {
            (total_w_sum, total_w_comp) = neumaier_add(total_w_sum, total_w_comp, w);
            (total_wv_sum, total_wv_comp) = neumaier_add(total_wv_sum, total_wv_comp, w * v);
        }
    }
    let total_w = total_w_sum + total_w_comp;
    let mu_0 = if total_w > 0.0 {
        (total_wv_sum + total_wv_comp) / total_w
    } else {
        0.0
    };

    let mut w_sum = Vec::with_capacity(n + 1);
    let mut wv_sum = Vec::with_capacity(n + 1);
    let mut wsq_sum = Vec::with_capacity(n + 1);
    w_sum.push(0.0_f64);
    wv_sum.push(0.0_f64);
    wsq_sum.push(0.0_f64);

    let mut acc_w = 0.0_f64;
    let mut acc_wv = 0.0_f64;
    let mut acc_wsq = 0.0_f64;
    for (&v, &w) in values.iter().zip(weights.iter()) {
        if v.is_finite() && w.is_finite() && w > 0.0 {
            let shifted = v - mu_0;
            acc_w += w;
            acc_wv += w * shifted;
            acc_wsq += w * (shifted * shifted);
        }
        w_sum.push(acc_w);
        wv_sum.push(acc_wv);
        wsq_sum.push(acc_wsq);
    }

    let seg_error = |start: usize, end: usize| -> f64 {
        let weight = w_sum[end] - w_sum[start];
        if weight <= 0.0 {
            return 0.0;
        }
        let weighted_val = wv_sum[end] - wv_sum[start];
        let weighted_sq = wsq_sum[end] - wsq_sum[start];
        let rss = weighted_sq - (weighted_val * weighted_val) / weight;
        rss.max(0.0)
    };

    let mut best_sum = vec![f64::INFINITY; n + 1];
    let mut best_comp = vec![0.0_f64; n + 1];
    let mut prev_bkp = vec![0usize; n + 1];
    best_sum[0] = 0.0;

    let mut admissible: Vec<usize> = Vec::new();
    let mut candidates: Vec<(usize, f64)> = Vec::new();

    for bkp in (0..n)
        .step_by(jump)
        .filter(|&k| k >= min_size)
        .chain(once(n))
    {
        let new_adm_pt = ((bkp - min_size) / jump) * jump;
        if admissible.last() != Some(&new_adm_pt) {
            admissible.push(new_adm_pt);
        }

        candidates.clear();
        let mut min_cost = f64::INFINITY;
        let mut best_t = 0usize;
        let mut best_s = f64::INFINITY;
        let mut best_c = 0.0_f64;
        let mut found_any = false;

        for &t in &admissible {
            let left_sum = best_sum[t];
            if !left_sum.is_finite() {
                continue;
            }
            let seg_pen = seg_error(t, bkp) + penalty;
            let (next_sum, next_comp) = neumaier_add(left_sum, best_comp[t], seg_pen);
            let cand_cost = next_sum + next_comp;
            candidates.push((t, cand_cost));
            if !found_any || cand_cost < min_cost {
                min_cost = cand_cost;
                best_t = t;
                best_s = next_sum;
                best_c = next_comp;
                found_any = true;
            }
        }

        best_sum[bkp] = best_s;
        best_comp[bkp] = best_c;
        prev_bkp[bkp] = best_t;

        admissible.clear();
        let (pl_s, pl_c) = neumaier_add(best_s, best_c, penalty);
        let prune_limit = pl_s + pl_c;
        for &(t, cand_cost) in &candidates {
            if cand_cost <= prune_limit {
                admissible.push(t);
            }
        }
    }

    let mut bkps = Vec::new();
    let mut cur = n;
    while cur > 0 {
        bkps.push(cur);
        cur = prev_bkp[cur];
    }
    bkps.reverse();
    Ok(bkps)
}

#[pyfunction]
#[pyo3(signature = (signal, penalty, min_size = 2, jump = 1))]
pub(crate) fn pelt_weighted_l2(
    py: Python<'_>,
    signal: PyReadonlyArray2<'_, f64>,
    penalty: f64,
    min_size: isize,
    jump: isize,
) -> PyResult<Vec<usize>> {
    let shape = signal.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(invalid_input_error(
            "signal must be a 2D array of shape (N, 2) containing (values, weights)",
        ));
    }
    if min_size < 1 {
        return Err(invalid_input_error("min_size must be at least 1"));
    }
    if jump < 1 {
        return Err(invalid_input_error("jump must be at least 1"));
    }
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err(invalid_input_error("penalty must be finite and positive"));
    }

    let n = shape[0];
    let mut values = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    if signal.is_c_contiguous()
        && let Ok(slice) = signal.as_slice()
    {
        for pair in slice.chunks_exact(2) {
            values.push(pair[0]);
            weights.push(pair[1]);
        }
    } else {
        let view = signal.as_array();
        for row in view.rows() {
            values.push(row[0]);
            weights.push(row[1]);
        }
    }

    let min_size_usize = min_size as usize;
    let jump_usize = jump as usize;
    py.detach(|| pelt_weighted_l2_core(&values, &weights, penalty, min_size_usize, jump_usize))
        .map_err(invalid_input_error)
}

#[cfg(test)]
mod tests {
    use super::pelt_weighted_l2_core;

    #[test]
    fn detects_step_change_and_is_scale_invariant() {
        let mut values = vec![0.0; 12];
        values.extend(vec![2.0; 12]);
        let weights: Vec<f64> = (0..24)
            .map(|i| if i % 2 == 0 { 1.0 } else { 10.0 })
            .collect();
        let tiny_weights: Vec<f64> = weights.iter().map(|w| w * 1e-9).collect();
        let huge_weights: Vec<f64> = weights.iter().map(|w| w * 1e6).collect();

        let bkps_base = pelt_weighted_l2_core(&values, &weights, 15.0, 2, 1).unwrap();
        let bkps_tiny = pelt_weighted_l2_core(&values, &tiny_weights, 15.0 * 1e-9, 2, 1).unwrap();
        let bkps_huge = pelt_weighted_l2_core(&values, &huge_weights, 15.0 * 1e6, 2, 1).unwrap();

        assert_eq!(bkps_base, vec![12, 24]);
        assert_eq!(bkps_tiny, bkps_base);
        assert_eq!(bkps_huge, bkps_base);
    }

    #[test]
    fn handles_zero_and_non_finite_entries_like_masked_bins() {
        let values = vec![
            0.0,
            f64::NAN,
            0.0,
            0.0,
            f64::INFINITY,
            2.0,
            2.0,
            2.0,
            f64::NEG_INFINITY,
            2.0,
        ];
        let weights = vec![
            1.0,
            5.0,
            0.0,
            1.0,
            1.0,
            f64::NAN,
            2.0,
            2.0,
            f64::INFINITY,
            2.0,
        ];
        let bkps = pelt_weighted_l2_core(&values, &weights, 0.5, 2, 1).unwrap();
        assert_eq!(bkps, vec![4, 10]);

        let all_zero_w = vec![0.0; 10];
        assert_eq!(
            pelt_weighted_l2_core(&values, &all_zero_w, 0.5, 2, 1).unwrap(),
            vec![10]
        );
    }

    #[test]
    fn handles_short_empty_jump_and_tie_breaking() {
        assert_eq!(
            pelt_weighted_l2_core(&[], &[], 1.0, 2, 1).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            pelt_weighted_l2_core(&[0.0, 1.0, 2.0], &[1.0, 1.0, 1.0], 1.0, 2, 1).unwrap(),
            vec![3]
        );

        let jump_vals = vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5];
        let jump_w = vec![4.0; 12];
        let jump_bkps = pelt_weighted_l2_core(&jump_vals, &jump_w, 2.0, 3, 3).unwrap();
        assert_eq!(jump_bkps, vec![3, 6, 12]);
        let high_pen_jump = pelt_weighted_l2_core(&jump_vals, &jump_w, 5.0, 3, 3).unwrap();
        assert_eq!(high_pen_jump, vec![6, 12]);

        let tie_vals = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let tie_w = vec![1.0; 6];
        let tie_bkps = pelt_weighted_l2_core(&tie_vals, &tie_w, 0.5, 2, 1).unwrap();
        assert_eq!(tie_bkps, vec![2, 4, 6]);
        let single_split_tie = pelt_weighted_l2_core(&tie_vals, &tie_w, 1.1, 2, 1).unwrap();
        assert_eq!(single_split_tie, vec![6]);

        let extreme_pen = pelt_weighted_l2_core(&tie_vals, &tie_w, 1e308, 2, 1).unwrap();
        assert_eq!(extreme_pen, vec![6]);
        let dup_adm = pelt_weighted_l2_core(&[0.0; 11], &[1.0; 11], 1.0, 2, 5).unwrap();
        assert_eq!(dup_adm, vec![11]);
    }

    #[test]
    fn rejects_invalid_inputs_including_negative_weight_on_short_inputs() {
        assert!(pelt_weighted_l2_core(&[1.0], &[1.0, 2.0], 1.0, 1, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, -0.1], 1.0, 5, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], 0.0, 1, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], -1.0, 1, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], f64::NAN, 1, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], f64::INFINITY, 1, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], 1.0, 0, 1).is_err());
        assert!(pelt_weighted_l2_core(&[1.0, 2.0], &[1.0, 1.0], 1.0, 1, 0).is_err());
    }
}
