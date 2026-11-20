mod alignments;
mod bgzf_index;
mod pelt;
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

fn py_domain_error(cls_name: &str, msg: impl Into<String>) -> PyErr {
    let message = msg.into();
    Python::attach(|py| {
        if let Ok(module) = py.import("ccnv.models.exceptions")
            && let Ok(cls) = module.getattr(cls_name)
            && let Ok(instance) = cls.call1((message.as_str(),))
        {
            return PyErr::from_value(instance);
        }
        PyRuntimeError::new_err(message)
    })
}

pub(crate) fn alignment_error(msg: impl Into<String>) -> PyErr {
    py_domain_error("AlignmentError", msg)
}

pub(crate) fn covariate_error(msg: impl Into<String>) -> PyErr {
    py_domain_error("CovariateError", msg)
}

pub(crate) fn invalid_input_error(msg: impl Into<String>) -> PyErr {
    py_domain_error("InvalidInputError", msg)
}

pub(crate) fn normalize_chr_name(name: &str) -> &str {
    let bare = if name.len() >= 3 && name.as_bytes()[..3].eq_ignore_ascii_case(b"chr") {
        &name[3..]
    } else {
        name
    };
    if bare.eq_ignore_ascii_case("x") {
        "X"
    } else if bare.eq_ignore_ascii_case("y") {
        "Y"
    } else {
        bare
    }
}

pub(crate) struct ContigLookup {
    exact: HashMap<String, usize>,
    lower_norm: HashMap<String, usize>,
}

impl ContigLookup {
    pub(crate) fn new(available: &[String]) -> Self {
        let mut exact = HashMap::with_capacity(available.len());
        let mut lower_norm = HashMap::with_capacity(available.len());
        for (idx, key) in available.iter().enumerate() {
            exact.entry(key.clone()).or_insert(idx);
            lower_norm
                .entry(normalize_chr_name(key).to_ascii_lowercase())
                .or_insert(idx);
        }
        Self { exact, lower_norm }
    }

    pub(crate) fn resolve(&self, target: &str) -> Option<usize> {
        if let Some(&idx) = self.exact.get(target) {
            return Some(idx);
        }
        let stripped = normalize_chr_name(target);
        if let Some(&idx) = self.exact.get(stripped) {
            return Some(idx);
        }
        let prefixed = format!("chr{stripped}");
        if let Some(&idx) = self.exact.get(prefixed.as_str()) {
            return Some(idx);
        }
        self.lower_norm.get(&stripped.to_ascii_lowercase()).copied()
    }
}

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let frag_edges = PyTuple::new(py, FRAG_BAND_EDGES_ARR)?;
    let gc_edges = PyTuple::new(py, GC_QUANTILE_EDGES_ARR)?;
    m.add("FRAG_BAND_EDGES", frag_edges)?;
    m.add("GC_QUANTILE_EDGES", gc_edges)?;

    m.add_function(wrap_pyfunction!(alignments::read_alignment_header, m)?)?;
    m.add_function(wrap_pyfunction!(alignments::scan_alignment_bins, m)?)?;
    m.add_function(wrap_pyfunction!(pelt::pelt_weighted_l2, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::read_fasta_contigs_core, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::scan_fasta_gc_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(tracks::scan_bigwig_means, m)?)?;
    m.add_function(wrap_pyfunction!(bgzf_index::write_bgzf_indexed, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ContigLookup;

    #[test]
    fn contig_lookup_priority_and_tie_breaking() {
        let avail = vec![
            "CHR1".to_owned(),
            "chr1".to_owned(),
            "1".to_owned(),
            "chr1".to_owned(),
            "ChrX".to_owned(),
            "CHR_X_ALT".to_owned(),
            "CHR2".to_owned(),
            "Chr2".to_owned(),
        ];
        let lookup = ContigLookup::new(&avail);
        assert_eq!(lookup.resolve("1"), Some(2));
        assert_eq!(lookup.resolve("chr1"), Some(1));
        assert_eq!(lookup.resolve("CHR1"), Some(0));
        assert_eq!(lookup.resolve("Chr1"), Some(2));
        assert_eq!(lookup.resolve("x"), Some(4));
        assert_eq!(lookup.resolve("2"), Some(6));
        assert_eq!(lookup.resolve("missing"), None);
    }
}
