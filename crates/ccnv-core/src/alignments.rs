mod accumulator;
mod record;

use std::collections::HashMap;
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
use self::record::{AlignmentRecord, process_record};
use crate::{ContigLookup, alignment_error};

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

fn open_bam_reader(
    path: &Path,
) -> Result<(bam::io::Reader<bgzf::io::Reader<File>>, sam::Header), String> {
    let mut reader = bam::io::reader::Builder
        .build_from_path(path)
        .map_err(|e| format!("failed to open BAM {}: {e}", path.display()))?;
    let header = reader
        .read_header()
        .map_err(|e| format!("failed to read BAM header {}: {e}", path.display()))?;
    Ok((reader, header))
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
        AlignmentKind::Bam => Ok(open_bam_reader(path)?.1),
        AlignmentKind::Cram => {
            let _fa = validate_cram_fasta_path(path, fasta_path)?;
            let mut reader = cram::io::reader::Builder::default()
                .build_from_path(path)
                .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
            reader
                .read_header()
                .map_err(|e| format!("failed to read CRAM header {}: {e}", path.display()))
        }
    }
}

fn find_index_candidate(path: &Path, ext: &str) -> Option<PathBuf> {
    let appended = PathBuf::from(format!("{}.{ext}", path.display()));
    if appended.is_file() {
        return Some(appended);
    }
    let replaced = path.with_extension("").with_extension(ext);
    replaced.is_file().then_some(replaced)
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

struct CraiLandmark {
    span: Option<(usize, usize)>,
    offset: u64,
}

struct CramIndexData {
    by_ref: HashMap<usize, Vec<CraiLandmark>>,
}

impl CramIndexData {
    fn from_crai(crai: &[cram::crai::Record]) -> Self {
        let mut by_ref: HashMap<usize, Vec<CraiLandmark>> = HashMap::new();
        for entry in crai {
            let Some(rid) = entry.reference_sequence_id() else {
                continue;
            };
            let span = entry.alignment_start().map(|s_pos| {
                let s0 = usize::from(s_pos) - 1;
                (s0, s0.saturating_add(entry.alignment_span().max(1)))
            });
            by_ref.entry(rid).or_default().push(CraiLandmark {
                span,
                offset: entry.offset(),
            });
        }
        Self { by_ref }
    }

    fn container_offsets_for_tile(
        &self,
        ref_id: usize,
        tile_start: usize,
        tile_end: usize,
    ) -> Vec<u64> {
        let mut offsets: Vec<u64> = self
            .by_ref
            .get(&ref_id)
            .into_iter()
            .flatten()
            .filter(|lm| match lm.span {
                Some((s0, e0)) => s0 < tile_end && e0 > tile_start,
                None => tile_start == 0,
            })
            .map(|lm| lm.offset)
            .collect();
        offsets.sort_unstable();
        offsets.dedup();
        offsets
    }
}

trait AlignmentBackend: Sync {
    type Record: AlignmentRecord;
    type TileState;
    type Index: Sync;

    fn for_each_sequential(
        &self,
        path: &Path,
        on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String>;

    fn load_index(&self, path: &Path, threads: usize) -> Result<Self::Index, String>;

    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String>;

    fn for_each_in_tile(
        &self,
        state: &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile_start: usize,
        tile_end: usize,
        on_record: impl FnMut(&Self::Record) -> Result<(), String>,
    ) -> Result<(), String>;
}

#[inline(always)]
fn bam_ref_id(record: &bam::Record) -> Result<Option<usize>, String> {
    record
        .reference_sequence_id()
        .transpose()
        .map_err(|e| format!("invalid BAM reference id: {e}"))
}

struct BamBackend;

impl AlignmentBackend for BamBackend {
    type Record = bam::Record;
    type TileState = (bam::io::Reader<bgzf::io::Reader<File>>, bam::Record);
    type Index = BamIndexData;

    #[inline(always)]
    fn for_each_sequential(
        &self,
        path: &Path,
        mut on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let (mut reader, _) = open_bam_reader(path)?;
        let mut record = bam::Record::default();
        loop {
            let n = reader
                .read_record(&mut record)
                .map_err(|e| format!("BAM read error in {}: {e}", path.display()))?;
            if n == 0 {
                return Ok(());
            }
            if let Some(ref_id) = bam_ref_id(&record)? {
                on_record(ref_id, &record)?;
            }
        }
    }

    fn load_index(&self, path: &Path, threads: usize) -> Result<Self::Index, String> {
        if let Some(cand) = find_index_candidate(path, "bai") {
            return bam::bai::fs::read(&cand)
                .map(BamIndexData::Bai)
                .map_err(|e| format!("failed to read BAM index {}: {e}", cand.display()));
        }
        if let Some(cand) = find_index_candidate(path, "csi") {
            return csi::fs::read(&cand)
                .map(BamIndexData::Csi)
                .map_err(|e| format!("failed to read CSI index {}: {e}", cand.display()));
        }
        Err(format!(
            "multi-threaded BAM scan (threads={threads}) requires an existing .bai or .csi index for {}; \
             index the alignment first (e.g. samtools index) or pass --threads 1",
            path.display()
        ))
    }

    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String> {
        let (reader, _) = open_bam_reader(path)?;
        Ok((reader, bam::Record::default()))
    }

    #[inline(always)]
    fn for_each_in_tile(
        &self,
        (reader, record): &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile_start: usize,
        tile_end: usize,
        mut on_record: impl FnMut(&Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let start_pos = Position::try_from(tile_start + 1)
            .map_err(|e| format!("invalid tile start position: {e}"))?;
        let end_pos =
            Position::try_from(tile_end).map_err(|e| format!("invalid tile end position: {e}"))?;
        for chunk in index.query(ref_id, Interval::from(start_pos..=end_pos))? {
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
                if bam_ref_id(record)? == Some(ref_id) {
                    on_record(record)?;
                }
            }
        }
        Ok(())
    }
}

struct CramBackend<'a> {
    header: &'a sam::Header,
    repo: fasta::Repository,
}

impl CramBackend<'_> {
    fn open_cram_reader(&self, path: &Path) -> Result<cram::io::Reader<File>, String> {
        let mut reader = cram::io::reader::Builder::default()
            .set_reference_sequence_repository(self.repo.clone())
            .build_from_path(path)
            .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
        reader
            .read_header()
            .map_err(|e| format!("failed to read CRAM header {}: {e}", path.display()))?;
        Ok(reader)
    }
}

impl AlignmentBackend for CramBackend<'_> {
    type Record = sam::alignment::RecordBuf;
    type TileState = (
        cram::io::Reader<File>,
        cram::io::reader::Container,
        sam::alignment::RecordBuf,
    );
    type Index = CramIndexData;

    #[inline(always)]
    fn for_each_sequential(
        &self,
        path: &Path,
        mut on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut reader = cram::io::BufReader::new(self.open_cram_reader(path)?);
        let mut record = sam::alignment::RecordBuf::default();
        loop {
            let n = reader
                .read_record_buf(self.header, &mut record)
                .map_err(|e| format!("CRAM read error in {}: {e}", path.display()))?;
            if n == 0 {
                return Ok(());
            }
            if let Some(ref_id) = record.reference_sequence_id() {
                on_record(ref_id, &record)?;
            }
        }
    }

    fn load_index(&self, path: &Path, threads: usize) -> Result<Self::Index, String> {
        if let Some(cand) = find_index_candidate(path, "crai") {
            let crai = cram::crai::fs::read(&cand)
                .map_err(|e| format!("failed to read CRAM index {}: {e}", cand.display()))?;
            return Ok(CramIndexData::from_crai(&crai));
        }
        Err(format!(
            "multi-threaded CRAM scan (threads={threads}) requires an existing .crai index for {}; \
             index the alignment first (e.g. samtools index) or pass --threads 1",
            path.display()
        ))
    }

    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String> {
        Ok((
            self.open_cram_reader(path)?,
            cram::io::reader::Container::default(),
            sam::alignment::RecordBuf::default(),
        ))
    }

    #[inline(always)]
    fn for_each_in_tile(
        &self,
        (reader, container, rec_buf): &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile_start: usize,
        tile_end: usize,
        mut on_record: impl FnMut(&Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        for offset in index.container_offsets_for_tile(ref_id, tile_start, tile_end) {
            reader
                .seek(SeekFrom::Start(offset))
                .map_err(|e| format!("CRAM seek error: {e}"))?;
            if reader
                .read_container(container)
                .map_err(|e| format!("CRAM container read error: {e}"))?
                == 0
            {
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
                        self.repo.clone(),
                        self.header,
                        &compression_header,
                        &core_data_src,
                        &external_data_srcs,
                    )
                    .map_err(|e| format!("CRAM slice decode error: {e}"))?;
                for rec in records {
                    rec_buf
                        .try_clone_from_alignment_record(self.header, &rec)
                        .map_err(|e| format!("CRAM record conversion error: {e}"))?;
                    if rec_buf.reference_sequence_id() == Some(ref_id) {
                        on_record(rec_buf)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_backend<B: AlignmentBackend>(
    backend: &B,
    path: &Path,
    ref_to_target: &[Option<usize>],
    targets: &[TargetContig],
    total_bins: usize,
    bin_size: usize,
    threads: usize,
    filters: FilterSpec,
) -> Result<GlobalScanOutput, String> {
    if threads == 1 {
        let mut acc = TileAccumulator::new(0, 0, total_bins, filters.collect_bands);
        backend.for_each_sequential(path, |ref_id, record| {
            if let Some(&Some(t_idx)) = ref_to_target.get(ref_id) {
                process_record(record, &targets[t_idx], bin_size, filters, None, &mut acc)?;
            }
            Ok(())
        })?;
        Ok(reduce_tile_accumulators(
            vec![acc],
            total_bins,
            filters.collect_bands,
        ))
    } else {
        let index = backend.load_index(path, threads)?;
        let tiles = build_lpt_tiles(targets, bin_size);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| format!("failed to build rayon thread pool: {e}"))?;

        let tile_results: Result<Vec<TileAccumulator>, String> = pool.install(|| {
            tiles
                .par_iter()
                .map_init(
                    || -> Option<B::TileState> { None },
                    |state_opt, tile| {
                        let state = match state_opt {
                            Some(s) => s,
                            None => state_opt.insert(backend.open_tile_state(path)?),
                        };
                        let target = &targets[tile.target_idx];
                        let bounds = Some((tile.tile_start, tile.tile_end));
                        let mut acc = TileAccumulator::new(
                            tile.tile_idx,
                            tile.local_bin_start,
                            tile.local_n_bins,
                            filters.collect_bands,
                        );
                        backend.for_each_in_tile(
                            state,
                            &index,
                            target.header_ref_id,
                            tile.tile_start,
                            tile.tile_end,
                            |record| {
                                process_record(record, target, bin_size, filters, bounds, &mut acc)
                            },
                        )?;
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
            let lookup = ContigLookup::new(&header_names);

            let mut targets = Vec::with_capacity(contigs.len());
            let mut ref_to_target: Vec<Option<usize>> = vec![None; header_names.len()];
            let mut total_bins = 0usize;

            for (cname, clen) in &contigs {
                if *clen == 0 {
                    return Err(format!("contig {cname} length must be positive"));
                }
                let Some(ref_id) = lookup.resolve(cname) else {
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
                AlignmentKind::Bam => run_backend(
                    &BamBackend,
                    &p,
                    &ref_to_target,
                    &targets,
                    total_bins,
                    bin_size,
                    threads,
                    filters,
                ),
                AlignmentKind::Cram => {
                    let fa_path = validate_cram_fasta_path(&p, fa.as_deref())?;
                    let repo = build_fasta_repository(fa_path)?;
                    let backend = CramBackend {
                        header: &header,
                        repo,
                    };
                    run_backend(
                        &backend,
                        &p,
                        &ref_to_target,
                        &targets,
                        total_bins,
                        bin_size,
                        threads,
                        filters,
                    )
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

#[cfg(test)]
mod tests {
    use noodles::core::Position;
    use noodles::cram::crai;

    use super::CramIndexData;

    #[test]
    fn crai_pregrouping_preserves_overlap_and_dedup_semantics() {
        let records = vec![
            crai::Record::new(Some(0), Position::new(101), 50, 300, 10, 100),
            crai::Record::new(Some(0), Position::new(140), 0, 200, 10, 100),
            crai::Record::new(Some(0), None, 0, 150, 10, 100),
            crai::Record::new(Some(0), Position::new(101), 50, 300, 20, 100),
            crai::Record::new(Some(1), Position::new(1), 500, 900, 10, 100),
            crai::Record::new(None, None, 0, 999, 10, 100),
        ];
        let idx = CramIndexData::from_crai(&records);

        assert_eq!(idx.container_offsets_for_tile(0, 0, 100), vec![150]);
        assert_eq!(idx.container_offsets_for_tile(0, 100, 140), vec![200, 300]);
        assert_eq!(idx.container_offsets_for_tile(0, 140, 200), vec![300]);
        assert!(idx.container_offsets_for_tile(0, 200, 300).is_empty());
        assert_eq!(idx.container_offsets_for_tile(1, 0, 100), vec![900]);
        assert!(idx.container_offsets_for_tile(2, 0, 100).is_empty());
    }
}
