mod accumulator;
mod fasta;
mod record;

use std::collections::HashMap;
use std::fs::File;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};

use noodles::core::Position;
use noodles::core::region::Interval;
use noodles::csi::{self, BinningIndex};
use noodles::{bam, bgzf, cram, fasta as noodles_fasta, sam};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use self::accumulator::{
    FilterSpec, GlobalScanOutput, NUM_FRAG_BANDS, TargetContig, TileAccumulator, TileSpec,
    build_lpt_tiles, classify_contig_role, reduce_tile_accumulators,
};
use self::fasta::{FastaSource, TileRepoGuard, evict_other_contigs_in_seq_repo};
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
        .map(str::to_ascii_lowercase)
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

type BamReader = bam::io::Reader<bgzf::io::Reader<File>>;

fn open_bam_reader(path: &Path) -> Result<(BamReader, sam::Header), String> {
    let mut r = bam::io::reader::Builder
        .build_from_path(path)
        .map_err(|e| format!("failed to open BAM {}: {e}", path.display()))?;
    let h = r
        .read_header()
        .map_err(|e| format!("failed to read BAM header {}: {e}", path.display()))?;
    Ok((r, h))
}

fn open_cram_reader(path: &Path) -> Result<(cram::io::Reader<File>, sam::Header), String> {
    let mut r = cram::io::reader::Builder::default()
        .build_from_path(path)
        .map_err(|e| format!("failed to open CRAM {}: {e}", path.display()))?;
    let h = r
        .read_header()
        .map_err(|e| format!("failed to read CRAM header {}: {e}", path.display()))?;
    Ok((r, h))
}

enum OpenedReader {
    Bam(BamReader),
    Cram(cram::io::Reader<File>),
}

fn open_alignment(
    path: &Path,
    kind: AlignmentKind,
    fasta_path: Option<&Path>,
) -> Result<(OpenedReader, sam::Header), String> {
    if !path.is_file() {
        return Err(format!("alignment not found: {}", path.display()));
    }
    if kind == AlignmentKind::Cram && fasta_path.is_none() {
        return Err(format!(
            "{} is CRAM and requires a reference FASTA; pass --fasta /path/to/reference.fa",
            path.display()
        ));
    }
    if let Some(fa) = fasta_path {
        if !fa.is_file() {
            let tag = if kind == AlignmentKind::Cram {
                "CRAM reference FASTA"
            } else {
                "reference FASTA"
            };
            return Err(format!("{tag} not found: {}", fa.display()));
        }
        File::open(fa).map_err(|e| format!("failed to open FASTA {}: {e}", fa.display()))?;
    }
    match kind {
        AlignmentKind::Bam => open_bam_reader(path).map(|(r, h)| (OpenedReader::Bam(r), h)),
        AlignmentKind::Cram => open_cram_reader(path).map(|(r, h)| (OpenedReader::Cram(r), h)),
    }
}

fn find_index_candidate(path: &Path, ext: &str) -> Option<PathBuf> {
    let appended = PathBuf::from(format!("{}.{ext}", path.display()));
    if appended.is_file() {
        return Some(appended);
    }
    let replaced = path.with_extension(ext);
    (replaced != path && replaced.is_file()).then_some(replaced)
}

fn missing_index_err(fmt: &str, exts: &str, threads: usize, path: &Path) -> String {
    format!(
        "multi-threaded {fmt} scan (threads={threads}) requires an existing {exts} index for {}; \
         index the alignment first (e.g. samtools index) or pass --threads 1",
        path.display()
    )
}

enum BamIndexData {
    Bai(bam::bai::Index),
    Csi(csi::Index),
}

impl BamIndexData {
    fn query(
        &self,
        ref_id: usize,
        iv: Interval,
    ) -> Result<Vec<csi::binning_index::index::reference_sequence::bin::Chunk>, String> {
        match self {
            Self::Bai(i) => i.query(ref_id, iv).map_err(|e| format!("BAI error: {e}")),
            Self::Csi(i) => i.query(ref_id, iv).map_err(|e| format!("CSI error: {e}")),
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
            let offset = entry.offset();
            by_ref
                .entry(rid)
                .or_default()
                .push(CraiLandmark { span, offset });
        }
        Self { by_ref }
    }

    fn container_offsets_for_tile(&self, ref_id: usize, start: usize, end: usize) -> Vec<u64> {
        let mut offsets: Vec<u64> = self
            .by_ref
            .get(&ref_id)
            .into_iter()
            .flatten()
            .filter(|lm| {
                lm.span
                    .map_or(start == 0, |(s0, e0)| s0 < end && e0 > start)
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
    type SeqReader;
    type TileState;
    type Index: Sync;

    fn for_each_sequential(
        &self,
        reader: Self::SeqReader,
        path: &Path,
        on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String>;

    fn prepare_tiles(&self, _tiles: &[TileSpec], _targets: &[TargetContig]) {}
    fn load_index(&self, path: &Path, threads: usize) -> Result<Self::Index, String>;
    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String>;
    fn for_each_in_tile(
        &self,
        state: &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile: &TileSpec,
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
    type SeqReader = BamReader;
    type TileState = (BamReader, bam::Record);
    type Index = BamIndexData;

    #[inline(always)]
    fn for_each_sequential(
        &self,
        mut reader: Self::SeqReader,
        path: &Path,
        mut on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut record = bam::Record::default();
        while reader
            .read_record(&mut record)
            .map_err(|e| format!("BAM read error in {}: {e}", path.display()))?
            != 0
        {
            if let Some(ref_id) = bam_ref_id(&record)? {
                on_record(ref_id, &record)?;
            }
        }
        Ok(())
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
        Err(missing_index_err("BAM", ".bai or .csi", threads, path))
    }

    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String> {
        open_bam_reader(path).map(|(r, _)| (r, bam::Record::default()))
    }

    #[inline(always)]
    fn for_each_in_tile(
        &self,
        (reader, record): &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile: &TileSpec,
        mut on_record: impl FnMut(&Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let s = Position::try_from(tile.tile_start + 1).map_err(|e| e.to_string())?;
        let e = Position::try_from(tile.tile_end).map_err(|e| e.to_string())?;
        for chunk in index.query(ref_id, Interval::from(s..=e))? {
            reader
                .get_mut()
                .seek(chunk.start())
                .map_err(|e| format!("BAM BGZF seek error: {e}"))?;
            while reader.get_ref().virtual_position() < chunk.end() {
                if reader
                    .read_record(record)
                    .map_err(|e| format!("BAM tile read error: {e}"))?
                    == 0
                {
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
    header_names: &'a [String],
    fasta: FastaSource,
}

impl CramBackend<'_> {
    fn acquire_tile_repo(&self, ref_id: usize) -> Result<TileRepoGuard<'_>, String> {
        self.fasta
            .acquire_contig_tile(self.header_names[ref_id].as_bytes())
    }

    fn decode_container(
        &self,
        container: &cram::io::reader::Container,
        repo: &noodles_fasta::Repository,
        rec_buf: &mut sam::alignment::RecordBuf,
        mut after_slice: impl FnMut(bool, Option<usize>),
        mut on_record: impl FnMut(&sam::alignment::RecordBuf) -> Result<(), String>,
    ) -> Result<(), String> {
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
                    self.header,
                    &compression_header,
                    &core_data_src,
                    &external_data_srcs,
                )
                .map_err(|e| format!("CRAM slice decode error: {e}"))?;
            let mut saw_any = false;
            let mut last_ref_id: Option<usize> = None;
            for rec in records {
                rec_buf
                    .try_clone_from_alignment_record(self.header, &rec)
                    .map_err(|e| format!("CRAM record conversion error: {e}"))?;
                saw_any = true;
                if let Some(rid) = rec_buf.reference_sequence_id() {
                    last_ref_id = Some(rid);
                }
                on_record(rec_buf)?;
            }
            after_slice(saw_any, last_ref_id);
        }
        Ok(())
    }
}

impl AlignmentBackend for CramBackend<'_> {
    type Record = sam::alignment::RecordBuf;
    type SeqReader = cram::io::Reader<File>;
    type TileState = (
        cram::io::Reader<File>,
        cram::io::reader::Container,
        sam::alignment::RecordBuf,
    );
    type Index = CramIndexData;

    #[inline(always)]
    fn for_each_sequential(
        &self,
        mut reader: Self::SeqReader,
        path: &Path,
        mut on_record: impl FnMut(usize, &Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let (seq_repo, handoff) = self.fasta.open_sequential();
        let mut container = cram::io::reader::Container::default();
        let mut rec_buf = sam::alignment::RecordBuf::default();
        while reader
            .read_container(&mut container)
            .map_err(|e| format!("CRAM read error in {}: {e}", path.display()))?
            != 0
        {
            self.decode_container(
                &container,
                &seq_repo,
                &mut rec_buf,
                |saw_any, last_rid| {
                    let Some(h) = handoff.as_ref() else { return };
                    if saw_any && last_rid.is_none() {
                        seq_repo.clear();
                    } else if let Some(rid) = last_rid
                        && seq_repo.len() > 1
                        && let Some(keep_name) = self.header_names.get(rid)
                    {
                        evict_other_contigs_in_seq_repo(&seq_repo, h, keep_name);
                    }
                },
                |rec| match rec.reference_sequence_id() {
                    Some(ref_id) => on_record(ref_id, rec),
                    None => Ok(()),
                },
            )?;
        }
        Ok(())
    }

    fn prepare_tiles(&self, tiles: &[TileSpec], targets: &[TargetContig]) {
        let mut counts: HashMap<&[u8], usize> = HashMap::with_capacity(targets.len());
        for tile in tiles {
            let rid = targets[tile.target_idx].header_ref_id;
            *counts.entry(self.header_names[rid].as_bytes()).or_insert(0) += 1;
        }
        self.fasta.prepare_contig_tile_counts(&counts);
    }

    fn load_index(&self, path: &Path, threads: usize) -> Result<Self::Index, String> {
        if let Some(cand) = find_index_candidate(path, "crai") {
            let crai = cram::crai::fs::read(&cand)
                .map_err(|e| format!("failed to read CRAM index {}: {e}", cand.display()))?;
            return Ok(CramIndexData::from_crai(&crai));
        }
        Err(missing_index_err("CRAM", ".crai", threads, path))
    }

    fn open_tile_state(&self, path: &Path) -> Result<Self::TileState, String> {
        let (r, _) = open_cram_reader(path)?;
        Ok((r, Default::default(), Default::default()))
    }

    #[inline(always)]
    fn for_each_in_tile(
        &self,
        (reader, container, rec_buf): &mut Self::TileState,
        index: &Self::Index,
        ref_id: usize,
        tile: &TileSpec,
        mut on_record: impl FnMut(&Self::Record) -> Result<(), String>,
    ) -> Result<(), String> {
        let tile_guard = self.acquire_tile_repo(ref_id)?;
        for offset in index.container_offsets_for_tile(ref_id, tile.tile_start, tile.tile_end) {
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
            self.decode_container(
                container,
                &tile_guard.repo,
                rec_buf,
                |_, _| {},
                |rec| {
                    if rec.reference_sequence_id() == Some(ref_id) {
                        on_record(rec)?;
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }
}

struct ScanPlan<'a> {
    path: &'a Path,
    ref_to_target: &'a [Option<usize>],
    targets: &'a [TargetContig],
    total_bins: usize,
    bin_size: usize,
    threads: usize,
    filters: FilterSpec,
}

fn run_backend<B: AlignmentBackend>(
    backend: &B,
    seq_reader: B::SeqReader,
    p: &ScanPlan<'_>,
) -> Result<GlobalScanOutput, String> {
    if p.threads == 1 {
        let mut acc = TileAccumulator::new(0, 0, p.total_bins, p.filters.collect_bands);
        backend.for_each_sequential(seq_reader, p.path, |rid, rec| {
            if let Some(&Some(ti)) = p.ref_to_target.get(rid) {
                process_record(rec, &p.targets[ti], p.bin_size, p.filters, None, &mut acc)?;
            }
            Ok(())
        })?;
        return Ok(reduce_tile_accumulators(
            vec![acc],
            p.total_bins,
            p.filters.collect_bands,
        ));
    }
    drop(seq_reader);
    let index = backend.load_index(p.path, p.threads)?;
    let tiles = build_lpt_tiles(p.targets, p.bin_size);
    backend.prepare_tiles(&tiles, p.targets);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(p.threads)
        .build()
        .map_err(|e| format!("failed to build rayon thread pool: {e}"))?;
    let tile_results: Result<Vec<TileAccumulator>, String> = pool.install(|| {
        tiles
            .par_iter()
            .map_init(
                || None::<B::TileState>,
                |state_opt, t| {
                    let state = match state_opt {
                        Some(s) => s,
                        None => state_opt.insert(backend.open_tile_state(p.path)?),
                    };
                    let tgt = &p.targets[t.target_idx];
                    let bounds = Some((t.tile_start, t.tile_end));
                    let mut acc = TileAccumulator::new(
                        t.tile_idx,
                        t.local_bin_start,
                        t.local_n_bins,
                        p.filters.collect_bands,
                    );
                    backend.for_each_in_tile(state, &index, tgt.header_ref_id, t, |rec| {
                        process_record(rec, tgt, p.bin_size, p.filters, bounds, &mut acc)
                    })?;
                    Ok(acc)
                },
            )
            .collect()
    });
    Ok(reduce_tile_accumulators(
        tile_results?,
        p.total_bins,
        p.filters.collect_bands,
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
    py.detach(|| -> Result<Vec<(String, usize)>, String> {
        let kind = detect_alignment_kind(&p)?;
        let (_, header) = open_alignment(&p, kind, fa.as_deref())?;
        Ok(header
            .reference_sequences()
            .iter()
            .map(|(n, m)| (n.to_string(), usize::from(m.length())))
            .collect())
    })
    .map_err(alignment_error)
}

#[pyfunction]
#[pyo3(signature = (path, contigs, bin_size, min_mapq=1, require_proper_pair=true, threads=1, fasta_path=None, collect_bands=true, se_fragment_length=0, *, exclude_duplicates=true, exclude_secondary=true, exclude_supplementary=true, exclude_unmapped=true))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_alignment_bins<'py>(
    py: Python<'py>,
    path: &str,
    contigs: Vec<(String, isize)>,
    bin_size: isize,
    min_mapq: i64,
    require_proper_pair: bool,
    threads: isize,
    fasta_path: Option<&str>,
    collect_bands: bool,
    se_fragment_length: isize,
    exclude_duplicates: bool,
    exclude_secondary: bool,
    exclude_supplementary: bool,
    exclude_unmapped: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if bin_size <= 0 {
        return Err(alignment_error("bin_size must be a positive integer"));
    }
    if threads <= 0 {
        return Err(alignment_error("threads must be a positive integer"));
    }
    if se_fragment_length < 0 {
        return Err(alignment_error("se_fragment_length must be non-negative"));
    }
    let bin_size = bin_size as usize;
    let threads = threads as usize;
    let se_fragment_length = se_fragment_length as usize;
    let p = PathBuf::from(path);
    let fa = fasta_path.map(PathBuf::from);
    let filters = FilterSpec {
        min_mapq: min_mapq.clamp(0, 255) as u8,
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
            let (opened_reader, header) = open_alignment(&p, kind, fa.as_deref())?;
            let header_names: Vec<String> = header
                .reference_sequences()
                .keys()
                .map(ToString::to_string)
                .collect();
            let lookup = ContigLookup::new(&header_names);

            let mut targets = Vec::with_capacity(contigs.len());
            let mut ref_to_target: Vec<Option<usize>> = vec![None; header_names.len()];
            let mut total_bins = 0usize;
            for (cname, clen) in &contigs {
                if *clen <= 0 {
                    return Err(format!("contig {cname} length must be positive"));
                }
                let clen_u = *clen as usize;
                let ref_id = lookup.resolve(cname).ok_or_else(|| {
                    format!(
                        "contig {cname} not found in alignment header of {}",
                        p.display()
                    )
                })?;
                if ref_to_target[ref_id].is_some() {
                    return Err(format!(
                        "duplicate target contig mapping for {cname} in {}",
                        p.display()
                    ));
                }
                ref_to_target[ref_id] = Some(targets.len());
                targets.push(TargetContig {
                    header_ref_id: ref_id,
                    length: clen_u,
                    bin_offset: total_bins,
                    role: classify_contig_role(cname),
                });
                total_bins += clen_u.div_ceil(bin_size);
            }

            let plan = ScanPlan {
                path: &p,
                ref_to_target: &ref_to_target,
                targets: &targets,
                total_bins,
                bin_size,
                threads,
                filters,
            };
            match opened_reader {
                OpenedReader::Bam(reader) => run_backend(&BamBackend, reader, &plan),
                OpenedReader::Cram(reader) => {
                    let fa_path = fa
                        .as_deref()
                        .ok_or_else(|| "missing CRAM FASTA path".to_owned())?;
                    let fasta = FastaSource::open(fa_path)?;
                    let backend = CramBackend {
                        header: &header,
                        header_names: &header_names,
                        fasta,
                    };
                    run_backend(&backend, reader, &plan)
                }
            }
        })
        .map_err(alignment_error)?;

    let total_bins = output.count.len();
    let dict = PyDict::new(py);
    for (k, v) in [
        ("count", output.count),
        ("paired_count", output.paired_count),
        ("short_frag_count", output.short_frag_count),
        ("endpoint_count", output.endpoint_count),
    ] {
        dict.set_item(k, PyArray1::from_vec(py, v))?;
    }
    dict.set_item("frag_len_sum", PyArray1::from_vec(py, output.frag_len_sum))?;
    dict.set_item("frag_gc_sum", PyArray1::from_vec(py, output.frag_gc_sum))?;
    match output.frag_bands {
        Some(bands_flat) => {
            let arr2: Bound<'py, PyArray2<u32>> =
                PyArray1::from_vec(py, bands_flat).reshape([total_bins, NUM_FRAG_BANDS])?;
            dict.set_item("frag_bands", arr2)?;
        }
        None => dict.set_item("frag_bands", py.None())?,
    }
    for (k, s) in [
        ("hist_autosomal", &output.hist_autosomal[..]),
        ("hist_chrx", &output.hist_chrx[..]),
        ("hist_chry", &output.hist_chry[..]),
        ("motif_counts", &output.motif_counts[..]),
    ] {
        dict.set_item(k, PyArray1::from_slice(py, s))?;
    }
    dict.set_item("paired_end", output.paired_end)?;
    let qc = &output.qc;
    dict.set_item(
        "qc_counts",
        PyDict::from_sequence(
            &[
                ("reads_seen", qc.reads_seen),
                ("reads_kept", qc.reads_kept),
                ("mate_dedup_skipped", qc.mate_dedup_skipped),
                ("reads_filtered", qc.reads_filtered),
                ("autosomal_reads_kept", qc.autosomal_reads_kept),
                ("sex_reads_kept", qc.sex_reads_kept),
                ("filter_unmapped", qc.filter_unmapped),
                ("filter_no_start", qc.filter_no_start),
                ("filter_secondary", qc.filter_secondary),
                ("filter_supplementary", qc.filter_supplementary),
                ("filter_duplicate", qc.filter_duplicate),
                ("filter_improper_pair", qc.filter_improper_pair),
                ("filter_mapq", qc.filter_mapq),
            ]
            .into_pyobject(py)?,
        )?,
    )?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use noodles::{core::Position, cram::crai};

    use super::{CramIndexData, find_index_candidate};

    #[test]
    fn find_index_candidate_resolves_multi_dot_picard_index_not_sibling_stem() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("ccnv_idx_test_{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");

        let bam = dir.join("sample.sorted.bam");
        let decoy_bai = dir.join("sample.bai");
        fs::write(&bam, b"").expect("write bam");
        fs::write(&decoy_bai, b"decoy").expect("write decoy bai");

        assert_eq!(find_index_candidate(&bam, "bai"), None);

        let picard_bai = dir.join("sample.sorted.bai");
        fs::write(&picard_bai, b"picard").expect("write picard bai");
        assert_eq!(find_index_candidate(&bam, "bai"), Some(picard_bai));

        let samtools_bai = dir.join("sample.sorted.bam.bai");
        fs::write(&samtools_bai, b"samtools").expect("write samtools bai");
        assert_eq!(find_index_candidate(&bam, "bai"), Some(samtools_bai));

        let _ = fs::remove_dir_all(&dir);
    }

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
