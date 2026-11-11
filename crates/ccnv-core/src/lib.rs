mod alignments;
mod bgzf_index;
mod tracks;

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub(crate) const FRAG_BAND_EDGES_ARR: [i32; 25] = [
    30, 50, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 300, 350, 400, 601,
];

pub(crate) const GC_QUANTILE_EDGES_ARR: [f64; 9] =
    [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 1.0];

pub(crate) fn alignment_error(msg: impl Into<String>) -> PyErr {
    let message = msg.into();
    Python::attach(|py| {
        if let Ok(module) = py.import("ccnv.models.exceptions")
            && let Ok(cls) = module.getattr("AlignmentError")
            && let Ok(instance) = cls.call1((message.as_str(),))
        {
            return PyErr::from_value(instance);
        }
        PyRuntimeError::new_err(message)
    })
}

pub(crate) fn covariate_error(msg: impl Into<String>) -> PyErr {
    let message = msg.into();
    Python::attach(|py| {
        if let Ok(module) = py.import("ccnv.models.exceptions")
            && let Ok(cls) = module.getattr("CovariateError")
            && let Ok(instance) = cls.call1((message.as_str(),))
        {
            return PyErr::from_value(instance);
        }
        PyRuntimeError::new_err(message)
    })
}

pub(crate) fn normalize_chr_name(name: &str) -> String {
    let bare = if name.len() >= 3 && name.as_bytes()[..3].eq_ignore_ascii_case(b"chr") {
        &name[3..]
    } else {
        name
    };
    if bare.eq_ignore_ascii_case("x") {
        "X".to_owned()
    } else if bare.eq_ignore_ascii_case("y") {
        "Y".to_owned()
    } else {
        bare.to_owned()
    }
}

pub(crate) fn resolve_contig_index(target: &str, available: &[String]) -> Option<usize> {
    if let Some(idx) = available.iter().position(|c| c == target) {
        return Some(idx);
    }
    let stripped = normalize_chr_name(target);
    if let Some(idx) = available.iter().position(|c| c == &stripped) {
        return Some(idx);
    }
    let prefixed = format!("chr{stripped}");
    if let Some(idx) = available.iter().position(|c| c == &prefixed) {
        return Some(idx);
    }
    let mut lower_map: HashMap<String, usize> = HashMap::with_capacity(available.len());
    for (idx, key) in available.iter().enumerate() {
        lower_map.insert(normalize_chr_name(key).to_ascii_lowercase(), idx);
    }
    lower_map.get(&stripped.to_ascii_lowercase()).copied()
}

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let frag_edges = PyTuple::new(py, FRAG_BAND_EDGES_ARR)?;
    let gc_edges = PyTuple::new(py, GC_QUANTILE_EDGES_ARR)?;
    m.add("FRAG_BAND_EDGES", frag_edges)?;
    m.add("GC_QUANTILE_EDGES", gc_edges)?;

    m.add_function(wrap_pyfunction!(alignments::read_alignment_header, m)?)?;
    m.add_function(wrap_pyfunction!(alignments::scan_alignment_bins, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::read_fasta_contigs_core, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::scan_fasta_gc_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::scan_bigwig_means, m)?)?;
    m.add_function(wrap_pyfunction!(bgzf_index::write_bgzf_indexed, m)?)?;
    Ok(())
}
