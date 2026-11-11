use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use bigtools::BigWigRead;
use noodles::fasta;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::{covariate_error, resolve_contig_index};

const NUM_GC_QUANTILES: usize = 8;
const FASTA_WINDOW_BYTES: usize = 16 * 1024 * 1024;
const SLIDING_GC_WINDOW_BP: usize = 200;

const FASTA_BASE_LUT: [u8; 256] = build_fasta_base_lut();

const fn build_fasta_base_lut() -> [u8; 256] {
    // 0: whitespace / newline (ignored in sequence coordinates)
    // 1: G/C (valid=1, gc=1)
    // 2: A/T/U (valid=1, gc=0)
    // 3: N / IUPAC ambiguity (valid=0, gc=0, advances coordinate)
    let mut lut = [3u8; 256];
    lut[b'\n' as usize] = 0;
    lut[b'\r' as usize] = 0;
    lut[b' ' as usize] = 0;
    lut[b'\t' as usize] = 0;
    lut[b'G' as usize] = 1;
    lut[b'g' as usize] = 1;
    lut[b'C' as usize] = 1;
    lut[b'c' as usize] = 1;
    lut[b'A' as usize] = 2;
    lut[b'a' as usize] = 2;
    lut[b'T' as usize] = 2;
    lut[b't' as usize] = 2;
    lut[b'U' as usize] = 2;
    lut[b'u' as usize] = 2;
    lut
}

#[inline]
fn gc_quantile_bin(gc: f64) -> usize {
    if gc < 0.30 {
        0
    } else if gc < 0.35 {
        1
    } else if gc < 0.40 {
        2
    } else if gc < 0.45 {
        3
    } else if gc < 0.50 {
        4
    } else if gc < 0.55 {
        5
    } else if gc < 0.65 {
        6
    } else {
        7
    }
}

struct ContigBinScanner {
    contig_len: u64,
    bin_size: u64,
    n_bins: usize,
    pos: u64,
    cur_bin: usize,
    cur_bin_end: u64,
    cur_eff_win: usize,
    bin_gc: u64,
    bin_valid: u64,
    ring_gc: [u8; SLIDING_GC_WINDOW_BP],
    ring_valid: [u8; SLIDING_GC_WINDOW_BP],
    ring_idx: usize,
    ring_filled: usize,
    win_gc: u16,
    win_valid: u16,
    win_counts: [u64; NUM_GC_QUANTILES],
    gc_mean: Vec<f32>,
    n_frac: Vec<f32>,
    gc_quantiles: Vec<f32>,
}

impl ContigBinScanner {
    fn new(contig_len: u64, bin_size: u64) -> Self {
        let n_bins = contig_len.div_ceil(bin_size) as usize;
        let first_end = bin_size.min(contig_len);
        let first_eff_win = (first_end as usize).clamp(1, SLIDING_GC_WINDOW_BP);
        Self {
            contig_len,
            bin_size,
            n_bins,
            pos: 0,
            cur_bin: 0,
            cur_bin_end: first_end,
            cur_eff_win: first_eff_win,
            bin_gc: 0,
            bin_valid: 0,
            ring_gc: [0u8; SLIDING_GC_WINDOW_BP],
            ring_valid: [0u8; SLIDING_GC_WINDOW_BP],
            ring_idx: 0,
            ring_filled: 0,
            win_gc: 0,
            win_valid: 0,
            win_counts: [0u64; NUM_GC_QUANTILES],
            gc_mean: vec![f32::NAN; n_bins],
            n_frac: vec![1.0f32; n_bins],
            gc_quantiles: vec![0.0f32; n_bins * NUM_GC_QUANTILES],
        }
    }

    #[inline]
    fn flush_current_bin(&mut self) {
        if self.cur_bin >= self.n_bins {
            return;
        }
        let b_start = (self.cur_bin as u64) * self.bin_size;
        let b_len = self.cur_bin_end.saturating_sub(b_start).max(1);
        let n_count = b_len.saturating_sub(self.bin_valid);
        self.n_frac[self.cur_bin] = (n_count as f64 / b_len as f64) as f32;
        self.gc_mean[self.cur_bin] = if self.bin_valid > 0 {
            (self.bin_gc as f64 / self.bin_valid as f64) as f32
        } else {
            f32::NAN
        };

        let total_wins: u64 = self.win_counts.iter().sum();
        if total_wins > 0 {
            let base = self.cur_bin * NUM_GC_QUANTILES;
            let denom = total_wins as f64;
            for q in 0..NUM_GC_QUANTILES {
                self.gc_quantiles[base + q] = (self.win_counts[q] as f64 / denom) as f32;
            }
        }

        self.cur_bin += 1;
        if self.cur_bin < self.n_bins {
            let next_start = (self.cur_bin as u64) * self.bin_size;
            self.cur_bin_end = (next_start + self.bin_size).min(self.contig_len);
            self.cur_eff_win =
                ((self.cur_bin_end - next_start) as usize).clamp(1, SLIDING_GC_WINDOW_BP);
            self.bin_gc = 0;
            self.bin_valid = 0;
            self.ring_idx = 0;
            self.ring_filled = 0;
            self.win_gc = 0;
            self.win_valid = 0;
            self.win_counts = [0u64; NUM_GC_QUANTILES];
        }
    }

    fn feed_bytes(&mut self, buf: &[u8]) {
        for &byte in buf {
            if self.pos >= self.contig_len {
                break;
            }
            let code = FASTA_BASE_LUT[usize::from(byte)];
            if code == 0 {
                continue;
            }
            let is_gc = u8::from(code == 1);
            let is_val = u8::from(code == 1 || code == 2);

            self.bin_gc += u64::from(is_gc);
            self.bin_valid += u64::from(is_val);

            let eff_w = self.cur_eff_win;
            if self.ring_filled == eff_w {
                self.win_gc -= u16::from(self.ring_gc[self.ring_idx]);
                self.win_valid -= u16::from(self.ring_valid[self.ring_idx]);
            } else {
                self.ring_filled += 1;
            }
            self.ring_gc[self.ring_idx] = is_gc;
            self.ring_valid[self.ring_idx] = is_val;
            self.win_gc += u16::from(is_gc);
            self.win_valid += u16::from(is_val);
            self.ring_idx += 1;
            if self.ring_idx == eff_w {
                self.ring_idx = 0;
            }

            if self.ring_filled == eff_w && usize::from(self.win_valid) * 2 >= eff_w {
                let g = f64::from(self.win_gc) / f64::from(self.win_valid);
                self.win_counts[gc_quantile_bin(g)] += 1;
            }

            self.pos += 1;
            if self.pos == self.cur_bin_end {
                self.flush_current_bin();
            }
        }
    }

    fn finish(mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if self.cur_bin < self.n_bins && self.pos > (self.cur_bin as u64) * self.bin_size {
            self.flush_current_bin();
        }
        (self.gc_mean, self.n_frac, self.gc_quantiles)
    }
}

fn parse_fasta_header_name(header_line: &[u8]) -> String {
    let trimmed = header_line
        .strip_prefix(b">")
        .unwrap_or(header_line)
        .trim_ascii_start();
    let end = trimmed
        .iter()
        .position(|b| b.is_ascii_whitespace())
        .unwrap_or(trimmed.len());
    String::from_utf8_lossy(&trimmed[..end]).into_owned()
}

fn scan_fasta_contigs_streaming(path: &Path) -> Result<Vec<(String, usize)>, String> {
    let file =
        File::open(path).map_err(|e| format!("failed to open FASTA {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(FASTA_WINDOW_BYTES, file);
    let mut contigs: Vec<(String, usize)> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_len: u64 = 0;
    let mut at_line_start = true;
    let mut header_buf: Vec<u8> = Vec::new();
    let mut in_header = false;

    loop {
        let buf = reader
            .fill_buf()
            .map_err(|e| format!("FASTA read error in {}: {e}", path.display()))?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len();
        let mut i = 0usize;
        while i < n {
            if in_header {
                if let Some(nl) = buf[i..].iter().position(|&b| b == b'\n') {
                    header_buf.extend_from_slice(&buf[i..i + nl]);
                    if let Some(prev) = current_name.take() {
                        contigs.push((prev, current_len as usize));
                    }
                    current_name = Some(parse_fasta_header_name(&header_buf));
                    current_len = 0;
                    header_buf.clear();
                    in_header = false;
                    at_line_start = true;
                    i += nl + 1;
                } else {
                    header_buf.extend_from_slice(&buf[i..]);
                    i = n;
                }
            } else if at_line_start && buf[i] == b'>' {
                in_header = true;
                at_line_start = false;
            } else {
                let b = buf[i];
                if b == b'\n' {
                    at_line_start = true;
                } else {
                    at_line_start = false;
                    if FASTA_BASE_LUT[usize::from(b)] != 0 {
                        current_len += 1;
                    }
                }
                i += 1;
            }
        }
        reader.consume(n);
    }

    if in_header && !header_buf.is_empty() {
        if let Some(prev) = current_name.take() {
            contigs.push((prev, current_len as usize));
        }
        current_name = Some(parse_fasta_header_name(&header_buf));
        current_len = 0;
    }
    if let Some(last) = current_name {
        contigs.push((last, current_len as usize));
    }
    Ok(contigs)
}

type ContigGcArrays = (Vec<f32>, Vec<f32>, Vec<f32>);

fn scan_indexed_fasta_contig(
    file: &mut File,
    fai_rec: &fasta::fai::Record,
    target_len: u64,
    bin_size: u64,
    chunk_buf: &mut [u8],
) -> Result<ContigGcArrays, String> {
    let seq_len = fai_rec.length().min(target_len);
    let line_bases = fai_rec.line_base_count().get();
    let line_width = fai_rec.line_width().get();
    if line_width < line_bases {
        return Err("invalid FASTA .fai record line dimensions".to_owned());
    }

    let mut scanner = ContigBinScanner::new(target_len, bin_size);
    if seq_len == 0 {
        return Ok(scanner.finish());
    }

    let full_lines = seq_len / line_bases;
    let rem_bases = seq_len % line_bases;
    let total_file_bytes = full_lines * line_width + rem_bases;

    file.seek(SeekFrom::Start(fai_rec.position()))
        .map_err(|e| format!("FASTA seek error: {e}"))?;

    let mut remaining = total_file_bytes;
    while remaining > 0 {
        let to_read = (remaining as usize).min(chunk_buf.len());
        let n = file
            .read(&mut chunk_buf[..to_read])
            .map_err(|e| format!("FASTA read error: {e}"))?;
        if n == 0 {
            break;
        }
        scanner.feed_bytes(&chunk_buf[..n]);
        remaining -= n as u64;
    }

    Ok(scanner.finish())
}

type FastaGcResult = (Vec<f32>, Vec<f32>, Vec<f32>, usize);

fn scan_unindexed_fasta(
    path: &Path,
    contigs: &[(String, usize)],
    bin_size: u64,
) -> Result<FastaGcResult, String> {
    let fa_contigs = scan_fasta_contigs_streaming(path)?;
    let fa_names: Vec<String> = fa_contigs.iter().map(|(n, _)| n.clone()).collect();

    let mut fa_idx_to_targets: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut matched_any = false;
    let mut per_target_nbins: Vec<usize> = Vec::with_capacity(contigs.len());
    let mut total_bins = 0usize;

    for (t_idx, (cname, clen)) in contigs.iter().enumerate() {
        let n_bins = (*clen as u64).div_ceil(bin_size) as usize;
        per_target_nbins.push(n_bins);
        total_bins += n_bins;
        if let Some(fa_idx) = resolve_contig_index(cname, &fa_names) {
            matched_any = true;
            fa_idx_to_targets.entry(fa_idx).or_default().push(t_idx);
        }
    }
    if !matched_any && !contigs.is_empty() {
        return Err(format!(
            "FASTA {} contains none of the target contigs",
            path.display()
        ));
    }

    let mut results: Vec<Option<ContigGcArrays>> = vec![None; contigs.len()];
    let file =
        File::open(path).map_err(|e| format!("failed to open FASTA {}: {e}", path.display()))?;
    let mut reader = BufReader::with_capacity(FASTA_WINDOW_BYTES, file);

    let mut cur_fa_idx: Option<usize> = None;
    let mut active_scanners: Vec<(usize, ContigBinScanner)> = Vec::new();
    let mut at_line_start = true;
    let mut in_header = false;

    let finalize_active = |scanners: &mut Vec<(usize, ContigBinScanner)>,
                           out: &mut [Option<ContigGcArrays>]| {
        for (t_idx, sc) in scanners.drain(..) {
            out[t_idx] = Some(sc.finish());
        }
    };

    loop {
        let buf = reader
            .fill_buf()
            .map_err(|e| format!("FASTA read error in {}: {e}", path.display()))?;
        if buf.is_empty() {
            break;
        }
        let n = buf.len();
        let mut i = 0usize;
        while i < n {
            if in_header {
                if let Some(nl) = buf[i..].iter().position(|&b| b == b'\n') {
                    in_header = false;
                    at_line_start = true;
                    i += nl + 1;
                } else {
                    i = n;
                }
            } else if at_line_start && buf[i] == b'>' {
                finalize_active(&mut active_scanners, &mut results);
                let next_idx = cur_fa_idx.map_or(0, |k| k + 1);
                cur_fa_idx = Some(next_idx);
                if let Some(t_indices) = fa_idx_to_targets.get(&next_idx) {
                    for &t_idx in t_indices {
                        active_scanners.push((
                            t_idx,
                            ContigBinScanner::new(contigs[t_idx].1 as u64, bin_size),
                        ));
                    }
                }
                in_header = true;
                at_line_start = false;
            } else {
                let start = i;
                while i < n && !(at_line_start && buf[i] == b'>') {
                    at_line_start = buf[i] == b'\n';
                    i += 1;
                }
                if !active_scanners.is_empty() {
                    let slice = &buf[start..i];
                    for (_, sc) in &mut active_scanners {
                        sc.feed_bytes(slice);
                    }
                }
            }
        }
        reader.consume(n);
    }
    finalize_active(&mut active_scanners, &mut results);

    let mut gc_all = Vec::with_capacity(total_bins);
    let mut nf_all = Vec::with_capacity(total_bins);
    let mut gq_all = Vec::with_capacity(total_bins * NUM_GC_QUANTILES);
    for (t_idx, res_opt) in results.into_iter().enumerate() {
        let nb = per_target_nbins[t_idx];
        if let Some((gc, nf, gq)) = res_opt {
            gc_all.extend_from_slice(&gc);
            nf_all.extend_from_slice(&nf);
            gq_all.extend_from_slice(&gq);
        } else {
            gc_all.resize(gc_all.len() + nb, f32::NAN);
            nf_all.resize(nf_all.len() + nb, 1.0f32);
            gq_all.resize(gq_all.len() + nb * NUM_GC_QUANTILES, 0.0f32);
        }
    }
    Ok((gc_all, nf_all, gq_all, total_bins))
}

#[pyfunction]
pub(crate) fn read_fasta_contigs_core(
    py: Python<'_>,
    fasta_path: &str,
) -> PyResult<Vec<(String, usize)>> {
    let p = PathBuf::from(fasta_path);
    let res = py.detach(|| -> Result<Vec<(String, usize)>, String> {
        if !p.is_file() {
            return Err(format!("FASTA not found: {}", p.display()));
        }
        let fai_path = PathBuf::from(format!("{}.fai", p.display()));
        if fai_path.is_file() {
            let fai = fasta::fai::fs::read(&fai_path)
                .map_err(|e| format!("failed to read FASTA index {}: {e}", fai_path.display()))?;
            Ok(fai
                .as_ref()
                .iter()
                .map(|r| {
                    (
                        String::from_utf8_lossy(r.name()).into_owned(),
                        r.length() as usize,
                    )
                })
                .collect())
        } else {
            scan_fasta_contigs_streaming(&p)
        }
    });
    res.map_err(covariate_error)
}

#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn scan_fasta_gc_quantiles<'py>(
    py: Python<'py>,
    fasta_path: &str,
    contigs: Vec<(String, usize)>,
    bin_size: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    if bin_size == 0 {
        return Err(covariate_error("bin_size must be a positive integer"));
    }
    let p = PathBuf::from(fasta_path);
    let bsize_u64 = bin_size as u64;

    let (gc_vec, nf_vec, gq_vec, total_bins) = py
        .detach(|| -> Result<FastaGcResult, String> {
            if !p.is_file() {
                return Err(format!("FASTA not found: {}", p.display()));
            }
            for (cname, clen) in &contigs {
                if *clen == 0 {
                    return Err(format!("contig {cname} length must be positive"));
                }
            }
            let fai_path = PathBuf::from(format!("{}.fai", p.display()));
            if !fai_path.is_file() {
                return scan_unindexed_fasta(&p, &contigs, bsize_u64);
            }

            let fai = fasta::fai::fs::read(&fai_path)
                .map_err(|e| format!("failed to read FASTA index {}: {e}", fai_path.display()))?;
            let fai_records = fai.as_ref();
            let fa_names: Vec<String> = fai_records
                .iter()
                .map(|r| String::from_utf8_lossy(r.name()).into_owned())
                .collect();

            let mut matched_any = false;
            let mut total_bins = 0usize;
            for (_, clen) in &contigs {
                total_bins += (*clen as u64).div_ceil(bsize_u64) as usize;
            }

            let mut gc_all = Vec::with_capacity(total_bins);
            let mut nf_all = Vec::with_capacity(total_bins);
            let mut gq_all = Vec::with_capacity(total_bins * NUM_GC_QUANTILES);
            let mut file =
                File::open(&p).map_err(|e| format!("failed to open FASTA {}: {e}", p.display()))?;
            let mut chunk_buf = vec![0u8; FASTA_WINDOW_BYTES];

            for (cname, clen) in &contigs {
                let nb = (*clen as u64).div_ceil(bsize_u64) as usize;
                if let Some(fa_idx) = resolve_contig_index(cname, &fa_names) {
                    matched_any = true;
                    let (gc, nf, gq) = scan_indexed_fasta_contig(
                        &mut file,
                        &fai_records[fa_idx],
                        *clen as u64,
                        bsize_u64,
                        &mut chunk_buf,
                    )?;
                    gc_all.extend_from_slice(&gc);
                    nf_all.extend_from_slice(&nf);
                    gq_all.extend_from_slice(&gq);
                } else {
                    gc_all.resize(gc_all.len() + nb, f32::NAN);
                    nf_all.resize(nf_all.len() + nb, 1.0f32);
                    gq_all.resize(gq_all.len() + nb * NUM_GC_QUANTILES, 0.0f32);
                }
            }

            if !matched_any && !contigs.is_empty() {
                return Err(format!(
                    "FASTA {} contains none of the target contigs",
                    p.display()
                ));
            }
            Ok((gc_all, nf_all, gq_all, total_bins))
        })
        .map_err(covariate_error)?;

    let gc_arr = PyArray1::from_vec(py, gc_vec);
    let nf_arr = PyArray1::from_vec(py, nf_vec);
    let gq_arr = PyArray1::from_vec(py, gq_vec).reshape([total_bins, NUM_GC_QUANTILES])?;
    Ok((gc_arr, nf_arr, gq_arr))
}

#[pyfunction]
#[pyo3(signature = (bw_path, contigs, bin_size, full_bin_denominator=false))]
pub(crate) fn scan_bigwig_means<'py>(
    py: Python<'py>,
    bw_path: &str,
    contigs: Vec<(String, usize)>,
    bin_size: usize,
    full_bin_denominator: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if bin_size == 0 {
        return Err(covariate_error("bin_size must be a positive integer"));
    }
    let p = PathBuf::from(bw_path);
    let bsize_u64 = bin_size as u64;

    let out = py
        .detach(|| -> Result<Vec<f32>, String> {
            if !p.is_file() {
                return Err(format!("BigWig file not found: {}", p.display()));
            }
            let mut bw = BigWigRead::open_file(&p)
                .map_err(|e| format!("failed to open BigWig {}: {e}", p.display()))?;
            let chroms: Vec<(String, u32)> = bw
                .chroms()
                .iter()
                .map(|c| (c.name.clone(), c.length))
                .collect();
            let bw_names: Vec<String> = chroms.iter().map(|(n, _)| n.clone()).collect();

            let default_val = if full_bin_denominator {
                0.0f32
            } else {
                f32::NAN
            };
            let mut total_bins = 0usize;
            for (cname, clen) in &contigs {
                if *clen == 0 {
                    return Err(format!("contig {cname} length must be positive"));
                }
                total_bins += (*clen as u64).div_ceil(bsize_u64) as usize;
            }
            if contigs.is_empty() {
                return Ok(Vec::new());
            }

            let mut out_vals = vec![default_val; total_bins];
            let mut matched_any = false;
            let mut bin_offset = 0usize;

            for (cname, clen) in &contigs {
                let target_len = *clen as u64;
                let n_bins = target_len.div_ceil(bsize_u64) as usize;
                if let Some(bw_idx) = resolve_contig_index(cname, &bw_names) {
                    matched_any = true;
                    let (ref bw_contig, bw_len) = chroms[bw_idx];
                    let query_end = (target_len.min(u64::from(bw_len))) as u32;
                    if query_end > 0 {
                        let mut weighted_sum = vec![0.0f64; n_bins];
                        let mut covered_len = vec![0.0f64; n_bins];
                        let iter = bw.get_interval(bw_contig, 0, query_end).map_err(|e| {
                            format!(
                                "BigWig interval error on {bw_contig} in {}: {e}",
                                p.display()
                            )
                        })?;
                        for iv_res in iter {
                            let iv = iv_res.map_err(|e| {
                                format!("BigWig decode error in {}: {e}", p.display())
                            })?;
                            let val = f64::from(iv.value);
                            if !val.is_finite() || iv.end <= iv.start {
                                continue;
                            }
                            let iv_s = u64::from(iv.start);
                            let iv_e = u64::from(iv.end).min(target_len);
                            if iv_e <= iv_s {
                                continue;
                            }
                            let first_b = (iv_s / bsize_u64) as usize;
                            let last_b = ((iv_e - 1) / bsize_u64) as usize;
                            for b in first_b..=last_b.min(n_bins - 1) {
                                let b_start = (b as u64) * bsize_u64;
                                let b_end = (b_start + bsize_u64).min(target_len);
                                let ov_s = iv_s.max(b_start);
                                let ov_e = iv_e.min(b_end);
                                if ov_e > ov_s {
                                    let w = (ov_e - ov_s) as f64;
                                    weighted_sum[b] += w * val;
                                    covered_len[b] += w;
                                }
                            }
                        }

                        for b in 0..n_bins {
                            if covered_len[b] > 0.0 {
                                let denom = if full_bin_denominator {
                                    let b_start = (b as u64) * bsize_u64;
                                    let b_end = (b_start + bsize_u64).min(target_len);
                                    (b_end - b_start).max(1) as f64
                                } else {
                                    covered_len[b]
                                };
                                out_vals[bin_offset + b] = (weighted_sum[b] / denom) as f32;
                            }
                        }
                    }
                }
                bin_offset += n_bins;
            }

            if !matched_any {
                return Err(format!(
                    "BigWig {} contains none of the target contigs",
                    p.display()
                ));
            }
            Ok(out_vals)
        })
        .map_err(covariate_error)?;

    Ok(PyArray1::from_vec(py, out))
}
