mod accumulator;
mod record;

use std::fs::File;
use std::io::{BufReader, SeekFrom};
use std::path::{Path, PathBuf};

use noodles::bam;
use noodles::bgzf;
use noodles::core::Position;
use noodles::core::region::Interval;
use noodles::cram;
use noodles::csi::{self, BinningIndex};
use noodles::fasta;
use noodles::sam;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use self::accumulator::{
    FilterSpec, GlobalScanOutput, NUM_FRAG_BANDS, TargetContig, TileAccumulator, build_lpt_tiles,
    classify_contig_role, reduce_tile_accumulators,
};
use self::record::{process_bam_record, process_record_buf};
use crate::{alignment_error, resolve_contig_index};

#[derive(Clone, Copy, PartialEq, Eq)]
enum AlignmentKind {
    Bam,
    Cram,
}

fn detect_alignment_kind(path: &Path) -> Result<AlignmentKind, String> {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("bam") => Ok(AlignmentKind::Bam),
        Some("cram") => Ok(AlignmentKind::Cram),
        other => Err(format!(
            "{} must be BAM or CRAM (found suffix {:?})",
            path.display(),
            other.unwrap_or("")
        )),
    }
}

fn validate_cram_fasta_path<'a>(
    cram_path: &Path,
    fasta_path: Option<&'a Path>,
) -> Result<&'a Path, String> {
    let fa = fasta_path.ok_or_else(|| {
        format!(
            "{} is CRAM and requires a reference FASTA; pass --fasta /path/to/reference.fa",
            cram_path.display()
        )
    })?;
    if !fa.is_file() {
        return Err(format!("CRAM reference FASTA not found: {}", fa.display()));
    }
    File::open(fa).map_err(|e| format!("failed to open FASTA {}: {e}", fa.display()))?;
    Ok(fa)
}

fn build_fasta_repository(fasta_path: &Path) -> Result<fasta::Repository, String> {
    if !fasta_path.is_file() {
        return Err(format!(
            "CRAM reference FASTA not found: {}",
            fasta_path.display()
        ));
    }
    let fai_path = PathBuf::from(format!("{}.fai", fasta_path.display()));
    if fai_path.is_file() {
        let reader = fasta::io::indexed_reader::Builder::default()
            .build_from_path(fasta_path)
            .map_err(|e| format!("failed to open indexed FASTA {}: {e}", fasta_path.display()))?;
        let adapter = fasta::repository::adapters::IndexedReader::new(reader);
        Ok(fasta::Repository::new(adapter))
    } else {
        let file = File::open(fasta_path)
            .map_err(|e| format!("failed to open FASTA {}: {e}", fasta_path.display()))?;
        let mut reader = fasta::io::Reader::new(BufReader::new(file));
        let records: Vec<fasta::Record> = reader
            .records()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("failed to read FASTA {}: {e}", fasta_path.display()))?;
        Ok(fasta::Repository::new(records))
    }
}

fn read_sam_header(
    path: &Path,
    kind: AlignmentKind,
    fasta_path: Option<&Path>,
) -> Result<sam::Header, String> {
    if !path.is_file() {
        return Err(format!("alignment not found: {}", path.display()));
    }
    match kind {
        AlignmentKind::Bam => {
            let mut reader = bam::io::reader::Builder
                .build_from_path(path)
                .map_err(|e| format!("failed to open BAM {}: {e}", path.display()))?;
            let header = reader
                .read_header()
                .map_err(|e| format!("failed to read BAM header {}: {e}", path.display()))?;
            Ok(header)
        }
        AlignmentKind::Cram => {
            let _fa = validate_cram_fasta_path(path, fasta_path)?;
            let mut reader = cram::io::reader::Builder::default()
                .build_from_path(path)
                .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
            let header = reader
                .read_header()
                .map_err(|e| format!("failed to read CRAM header {}: {e}", path.display()))?;
            Ok(header)
        }
    }
}

enum BamIndexData {
    Bai(bam::bai::Index),
    Csi(csi::Index),
}

impl BamIndexData {
    fn query(
        &self,
        ref_id: usize,
        interval: Interval,
    ) -> Result<Vec<csi::binning_index::index::reference_sequence::bin::Chunk>, String> {
        match self {
            Self::Bai(idx) => idx
                .query(ref_id, interval)
                .map_err(|e| format!("BAI query error: {e}")),
            Self::Csi(idx) => idx
                .query(ref_id, interval)
                .map_err(|e| format!("CSI query error: {e}")),
        }
    }
}

fn load_bam_index(path: &Path, threads: usize) -> Result<BamIndexData, String> {
    let p_str = path.display().to_string();
    let stem = path.with_extension("");
    let bai_candidates = [
        PathBuf::from(format!("{p_str}.bai")),
        stem.with_extension("bai"),
    ];
    for cand in &bai_candidates {
        if cand.is_file() {
            return bam::bai::fs::read(cand)
                .map(BamIndexData::Bai)
                .map_err(|e| format!("failed to read BAM index {}: {e}", cand.display()));
        }
    }
    let csi_candidates = [
        PathBuf::from(format!("{p_str}.csi")),
        stem.with_extension("csi"),
    ];
    for cand in &csi_candidates {
        if cand.is_file() {
            return csi::fs::read(cand)
                .map(BamIndexData::Csi)
                .map_err(|e| format!("failed to read CSI index {}: {e}", cand.display()));
        }
    }
    Err(format!(
        "multi-threaded BAM scan (threads={threads}) requires an existing .bai or .csi index for {}; \
         index the alignment first (e.g. samtools index) or pass --threads 1",
        path.display()
    ))
}

fn load_cram_index(path: &Path, threads: usize) -> Result<cram::crai::Index, String> {
    let p_str = path.display().to_string();
    let stem = path.with_extension("");
    let candidates = [
        PathBuf::from(format!("{p_str}.crai")),
        stem.with_extension("crai"),
    ];
    for cand in &candidates {
        if cand.is_file() {
            return cram::crai::fs::read(cand)
                .map_err(|e| format!("failed to read CRAM index {}: {e}", cand.display()));
        }
    }
    Err(format!(
        "multi-threaded CRAM scan (threads={threads}) requires an existing .crai index for {}; \
         index the alignment first (e.g. samtools index) or pass --threads 1",
        path.display()
    ))
}

fn scan_sequential_bam(
    path: &Path,
    ref_to_target: &[Option<usize>],
    targets: &[TargetContig],
    total_bins: usize,
    bin_size: usize,
    filters: FilterSpec,
) -> Result<GlobalScanOutput, String> {
    let mut reader = bam::io::reader::Builder
        .build_from_path(path)
        .map_err(|e| format!("failed to open BAM {}: {e}", path.display()))?;
    let _header = reader
        .read_header()
        .map_err(|e| format!("failed to read BAM header {}: {e}", path.display()))?;

    let mut acc = TileAccumulator::new(0, 0, total_bins, filters.collect_bands);
    let mut record = bam::Record::default();
    loop {
        let bytes_read = reader
            .read_record(&mut record)
            .map_err(|e| format!("BAM read error in {}: {e}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        let Some(ref_id_res) = record.reference_sequence_id() else {
            continue;
        };
        let ref_id = ref_id_res.map_err(|e| format!("invalid BAM reference id: {e}"))?;
        let Some(Some(t_idx)) = ref_to_target.get(ref_id) else {
            continue;
        };
        process_bam_record(&record, &targets[*t_idx], bin_size, filters, None, &mut acc)?;
    }
    Ok(reduce_tile_accumulators(
        vec![acc],
        total_bins,
        filters.collect_bands,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scan_sequential_cram(
    path: &Path,
    header: &sam::Header,
    repo: fasta::Repository,
    ref_to_target: &[Option<usize>],
    targets: &[TargetContig],
    total_bins: usize,
    bin_size: usize,
    filters: FilterSpec,
) -> Result<GlobalScanOutput, String> {
    let inner = cram::io::reader::Builder::default()
        .set_reference_sequence_repository(repo)
        .build_from_path(path)
        .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
    let mut reader = cram::io::BufReader::new(inner);
    reader
        .get_mut()
        .read_header()
        .map_err(|e| format!("failed to read CRAM header {}: {e}", path.display()))?;

    let mut acc = TileAccumulator::new(0, 0, total_bins, filters.collect_bands);
    let mut record = sam::alignment::RecordBuf::default();
    loop {
        let n = reader
            .read_record_buf(header, &mut record)
            .map_err(|e| format!("CRAM read error in {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        let Some(ref_id) = record.reference_sequence_id() else {
            continue;
        };
        let Some(Some(t_idx)) = ref_to_target.get(ref_id) else {
            continue;
        };
        process_record_buf(&record, &targets[*t_idx], bin_size, filters, None, &mut acc)?;
    }
    Ok(reduce_tile_accumulators(
        vec![acc],
        total_bins,
        filters.collect_bands,
    ))
}

fn scan_tiled_bam(
    path: &Path,
    targets: &[TargetContig],
    total_bins: usize,
    bin_size: usize,
    threads: usize,
    filters: FilterSpec,
) -> Result<GlobalScanOutput, String> {
    let index = load_bam_index(path, threads)?;
    let tiles = build_lpt_tiles(targets, bin_size);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("failed to build rayon thread pool: {e}"))?;

    let tile_results: Result<Vec<TileAccumulator>, String> = pool.install(|| {
        tiles
            .par_iter()
            .map_init(
                || -> Option<(bam::io::Reader<bgzf::io::Reader<File>>, bam::Record)> { None },
                |state, tile| {
                    if state.is_none() {
                        let mut r = bam::io::reader::Builder
                            .build_from_path(path)
                            .map_err(|e| format!("failed to open BAM {}: {e}", path.display()))?;
                        r.read_header().map_err(|e| {
                            format!("failed to read BAM header {}: {e}", path.display())
                        })?;
                        *state = Some((r, bam::Record::default()));
                    }
                    let (reader, record) = state.as_mut().unwrap();
                    let target = &targets[tile.target_idx];
                    let start_pos = Position::try_from(tile.tile_start + 1)
                        .map_err(|e| format!("invalid tile start position: {e}"))?;
                    let end_pos = Position::try_from(tile.tile_end)
                        .map_err(|e| format!("invalid tile end position: {e}"))?;
                    let interval = Interval::from(start_pos..=end_pos);
                    let chunks = index.query(target.header_ref_id, interval)?;

                    let mut acc = TileAccumulator::new(
                        tile.tile_idx,
                        tile.local_bin_start,
                        tile.local_n_bins,
                        filters.collect_bands,
                    );
                    for chunk in chunks {
                        reader
                            .get_mut()
                            .seek(chunk.start())
                            .map_err(|e| format!("BAM BGZF seek error: {e}"))?;
                        while reader.get_ref().virtual_position() < chunk.end() {
                            let n = reader
                                .read_record(record)
                                .map_err(|e| format!("BAM tile read error: {e}"))?;
                            if n == 0 {
                                break;
                            }
                            let Some(ref_res) = record.reference_sequence_id() else {
                                continue;
                            };
                            let ref_id =
                                ref_res.map_err(|e| format!("invalid BAM reference id: {e}"))?;
                            if ref_id != target.header_ref_id {
                                continue;
                            }
                            process_bam_record(
                                record,
                                target,
                                bin_size,
                                filters,
                                Some((tile.tile_start, tile.tile_end)),
                                &mut acc,
                            )?;
                        }
                    }
                    Ok(acc)
                },
            )
            .collect()
    });

    Ok(reduce_tile_accumulators(
        tile_results?,
        total_bins,
        filters.collect_bands,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scan_tiled_cram(
    path: &Path,
    header: &sam::Header,
    repo: fasta::Repository,
    targets: &[TargetContig],
    total_bins: usize,
    bin_size: usize,
    threads: usize,
    filters: FilterSpec,
) -> Result<GlobalScanOutput, String> {
    let crai = load_cram_index(path, threads)?;
    let tiles = build_lpt_tiles(targets, bin_size);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| format!("failed to build rayon thread pool: {e}"))?;

    let tile_results: Result<Vec<TileAccumulator>, String> = pool.install(|| {
        tiles
            .par_iter()
            .map_init(
                || -> Option<cram::io::Reader<File>> { None },
                |state, tile| {
                    if state.is_none() {
                        let mut r = cram::io::reader::Builder::default()
                            .set_reference_sequence_repository(repo.clone())
                            .build_from_path(path)
                            .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
                        r.read_header().map_err(|e| {
                            format!("failed to read CRAM header {}: {e}", path.display())
                        })?;
                        *state = Some(r);
                    }
                    let reader = state.as_mut().unwrap();
                    let target = &targets[tile.target_idx];

                    let mut container_offsets: Vec<u64> = Vec::new();
                    for entry in &crai {
                        let matches_tile = match entry.reference_sequence_id() {
                            Some(rid) if rid == target.header_ref_id => {
                                if let Some(s_pos) = entry.alignment_start() {
                                    let s0 = usize::from(s_pos) - 1;
                                    let span = entry.alignment_span().max(1);
                                    let e0 = s0.saturating_add(span);
                                    s0 < tile.tile_end && e0 > tile.tile_start
                                } else {
                                    tile.tile_start == 0
                                }
                            }
                            _ => false,
                        };
                        if matches_tile && container_offsets.last() != Some(&entry.offset()) {
                            container_offsets.push(entry.offset());
                        }
                    }
                    container_offsets.sort_unstable();
                    container_offsets.dedup();

                    let mut acc = TileAccumulator::new(
                        tile.tile_idx,
                        tile.local_bin_start,
                        tile.local_n_bins,
                        filters.collect_bands,
                    );
                    let mut container = cram::io::reader::Container::default();
                    let mut rec_buf = sam::alignment::RecordBuf::default();
                    for offset in container_offsets {
                        reader
                            .seek(SeekFrom::Start(offset))
                            .map_err(|e| format!("CRAM seek error: {e}"))?;
                        let bytes = reader
                            .read_container(&mut container)
                            .map_err(|e| format!("CRAM container read error: {e}"))?;
                        if bytes == 0 {
                            continue;
                        }
                        let compression_header = container
                            .compression_header()
                            .map_err(|e| format!("CRAM compression header error: {e}"))?;
                        for slice_res in container.slices() {
                            let slice = slice_res.map_err(|e| format!("CRAM slice error: {e}"))?;
                            let (core_data_src, external_data_srcs) = slice
                                .decode_blocks()
                                .map_err(|e| format!("CRAM slice block decode error: {e}"))?;
                            let records = slice
                                .records(
                                    repo.clone(),
                                    header,
                                    &compression_header,
                                    &core_data_src,
                                    &external_data_srcs,
                                )
                                .map_err(|e| format!("CRAM slice decode error: {e}"))?;
                            for rec in records {
                                rec_buf
                                    .try_clone_from_alignment_record(header, &rec)
                                    .map_err(|e| format!("CRAM record conversion error: {e}"))?;
                                if rec_buf.reference_sequence_id() != Some(target.header_ref_id) {
                                    continue;
                                }
                                process_record_buf(
                                    &rec_buf,
                                    target,
                                    bin_size,
                                    filters,
                                    Some((tile.tile_start, tile.tile_end)),
                                    &mut acc,
                                )?;
                            }
                        }
                    }
                    Ok(acc)
                },
            )
            .collect()
    });

    Ok(reduce_tile_accumulators(
        tile_results?,
        total_bins,
        filters.collect_bands,
    ))
}

#[pyfunction]
#[pyo3(signature = (path, fasta_path=None))]
pub(crate) fn read_alignment_header(
    py: Python<'_>,
    path: &str,
    fasta_path: Option<&str>,
) -> PyResult<Vec<(String, usize)>> {
    let p = PathBuf::from(path);
    let fa = fasta_path.map(PathBuf::from);
    let result = py.detach(|| -> Result<Vec<(String, usize)>, String> {
        let kind = detect_alignment_kind(&p)?;
        let header = read_sam_header(&p, kind, fa.as_deref())?;
        let mut contigs = Vec::with_capacity(header.reference_sequences().len());
        for (name, map) in header.reference_sequences() {
            contigs.push((name.to_string(), usize::from(map.length())));
        }
        Ok(contigs)
    });
    result.map_err(alignment_error)
}

#[pyfunction]
#[pyo3(signature = (
    path,
    contigs,
    bin_size,
    min_mapq=1,
    require_proper_pair=true,
    threads=1,
    fasta_path=None,
    collect_bands=true,
    se_fragment_length=0,
    *,
    exclude_duplicates=true,
    exclude_secondary=true,
    exclude_supplementary=true,
    exclude_unmapped=true,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_alignment_bins<'py>(
    py: Python<'py>,
    path: &str,
    contigs: Vec<(String, usize)>,
    bin_size: usize,
    min_mapq: i64,
    require_proper_pair: bool,
    threads: usize,
    fasta_path: Option<&str>,
    collect_bands: bool,
    se_fragment_length: usize,
    exclude_duplicates: bool,
    exclude_secondary: bool,
    exclude_supplementary: bool,
    exclude_unmapped: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if bin_size == 0 {
        return Err(alignment_error("bin_size must be a positive integer"));
    }
    if threads == 0 {
        return Err(alignment_error("threads must be a positive integer"));
    }
    let clamped_mapq = min_mapq.clamp(0, 255) as u8;
    let p = PathBuf::from(path);
    let fa = fasta_path.map(PathBuf::from);

    let filters = FilterSpec {
        min_mapq: clamped_mapq,
        require_proper_pair,
        exclude_duplicates,
        exclude_secondary,
        exclude_supplementary,
        exclude_unmapped,
        collect_bands,
        se_fragment_length,
    };

    let output = py
        .detach(|| -> Result<GlobalScanOutput, String> {
            let kind = detect_alignment_kind(&p)?;
            let header = read_sam_header(&p, kind, fa.as_deref())?;
            let header_names: Vec<String> = header
                .reference_sequences()
                .keys()
                .map(|k| k.to_string())
                .collect();

            let mut targets = Vec::with_capacity(contigs.len());
            let mut ref_to_target: Vec<Option<usize>> = vec![None; header_names.len()];
            let mut total_bins = 0usize;

            for (cname, clen) in &contigs {
                if *clen == 0 {
                    return Err(format!("contig {cname} length must be positive"));
                }
                let Some(ref_id) = resolve_contig_index(cname, &header_names) else {
                    return Err(format!(
                        "contig {cname} not found in alignment header of {}",
                        p.display()
                    ));
                };
                if ref_to_target[ref_id].is_some() {
                    return Err(format!(
                        "duplicate target contig mapping for {cname} in {}",
                        p.display()
                    ));
                }
                let n_bins = clen.div_ceil(bin_size);
                let target_idx = targets.len();
                ref_to_target[ref_id] = Some(target_idx);
                targets.push(TargetContig {
                    header_ref_id: ref_id,
                    length: *clen,
                    bin_offset: total_bins,
                    role: classify_contig_role(cname),
                });
                total_bins += n_bins;
            }

            match kind {
                AlignmentKind::Bam => {
                    if threads == 1 {
                        scan_sequential_bam(
                            &p,
                            &ref_to_target,
                            &targets,
                            total_bins,
                            bin_size,
                            filters,
                        )
                    } else {
                        scan_tiled_bam(&p, &targets, total_bins, bin_size, threads, filters)
                    }
                }
                AlignmentKind::Cram => {
                    let fa_path = validate_cram_fasta_path(&p, fa.as_deref())?;
                    let repo = build_fasta_repository(fa_path)?;
                    if threads == 1 {
                        scan_sequential_cram(
                            &p,
                            &header,
                            repo,
                            &ref_to_target,
                            &targets,
                            total_bins,
                            bin_size,
                            filters,
                        )
                    } else {
                        scan_tiled_cram(
                            &p, &header, repo, &targets, total_bins, bin_size, threads, filters,
                        )
                    }
                }
            }
        })
        .map_err(alignment_error)?;

    let total_bins = output.count.len();
    let dict = PyDict::new(py);
    dict.set_item("count", PyArray1::from_vec(py, output.count))?;
    dict.set_item("paired_count", PyArray1::from_vec(py, output.paired_count))?;
    dict.set_item("frag_len_sum", PyArray1::from_vec(py, output.frag_len_sum))?;
    dict.set_item(
        "short_frag_count",
        PyArray1::from_vec(py, output.short_frag_count),
    )?;
    dict.set_item(
        "endpoint_count",
        PyArray1::from_vec(py, output.endpoint_count),
    )?;
    dict.set_item("frag_gc_sum", PyArray1::from_vec(py, output.frag_gc_sum))?;

    if let Some(bands_flat) = output.frag_bands {
        let arr1 = PyArray1::from_vec(py, bands_flat);
        let arr2: Bound<'py, PyArray2<u32>> = arr1.reshape([total_bins, NUM_FRAG_BANDS])?;
        dict.set_item("frag_bands", arr2)?;
    } else {
        dict.set_item("frag_bands", py.None())?;
    }

    dict.set_item(
        "hist_autosomal",
        PyArray1::from_slice(py, &*output.hist_autosomal),
    )?;
    dict.set_item("hist_chrx", PyArray1::from_slice(py, &*output.hist_chrx))?;
    dict.set_item("hist_chry", PyArray1::from_slice(py, &*output.hist_chry))?;
    dict.set_item(
        "motif_counts",
        PyArray1::from_slice(py, &*output.motif_counts),
    )?;
    dict.set_item("paired_end", output.paired_end)?;

    let qc_dict = PyDict::new(py);
    qc_dict.set_item("reads_seen", output.qc.reads_seen)?;
    qc_dict.set_item("reads_kept", output.qc.reads_kept)?;
    qc_dict.set_item("mate_dedup_skipped", output.qc.mate_dedup_skipped)?;
    qc_dict.set_item("reads_filtered", output.qc.reads_filtered)?;
    qc_dict.set_item("autosomal_reads_kept", output.qc.autosomal_reads_kept)?;
    qc_dict.set_item("sex_reads_kept", output.qc.sex_reads_kept)?;
    qc_dict.set_item("filter_unmapped", output.qc.filter_unmapped)?;
    qc_dict.set_item("filter_no_start", output.qc.filter_no_start)?;
    qc_dict.set_item("filter_secondary", output.qc.filter_secondary)?;
    qc_dict.set_item("filter_supplementary", output.qc.filter_supplementary)?;
    qc_dict.set_item("filter_duplicate", output.qc.filter_duplicate)?;
    qc_dict.set_item("filter_improper_pair", output.qc.filter_improper_pair)?;
    qc_dict.set_item("filter_mapq", output.qc.filter_mapq)?;
    dict.set_item("qc_counts", qc_dict)?;

    Ok(dict)
}
