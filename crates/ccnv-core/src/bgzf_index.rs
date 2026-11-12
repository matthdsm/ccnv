use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use noodles::bgzf;
use noodles::bgzf::VirtualPosition;
use noodles::core::Position;
use noodles::csi;
use noodles::csi::binning_index::Indexer as CsiIndexer;
use noodles::csi::binning_index::index::header::{
    Builder as HeaderBuilder, ReferenceSequenceNames,
};
use noodles::csi::binning_index::index::reference_sequence::bin::Chunk;
use noodles::csi::binning_index::index::reference_sequence::index::BinnedIndex;
use noodles::tabix;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

const TBI_MAX_CONTIG_LEN: usize = 536_870_912;
const BGZF_MAX_PAYLOAD: usize = 65_280;
const PARALLEL_BATCH_BLOCKS: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexPreset {
    Bed,
    Vcf,
}

impl IndexPreset {
    fn parse(raw: &str) -> PyResult<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "bed" => Ok(Self::Bed),
            "vcf" => Ok(Self::Vcf),
            _ => Err(PyValueError::new_err(format!(
                "unsupported index preset {raw:?}; expected 'bed' or 'vcf'"
            ))),
        }
    }
    fn header_builder(preset: Self) -> HeaderBuilder {
        match preset {
            Self::Bed => HeaderBuilder::bed(),
            Self::Vcf => HeaderBuilder::vcf(),
        }
    }
}

struct OutputCleanupGuard {
    out_path: PathBuf,
    idx_path: PathBuf,
    armed: bool,
}

impl OutputCleanupGuard {
    fn new(out_path: PathBuf, idx_path: PathBuf) -> Self {
        Self {
            out_path,
            idx_path,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OutputCleanupGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.out_path);
            let _ = std::fs::remove_file(&self.idx_path);
        }
    }
}

#[derive(Default)]
struct ChromInterner {
    by_name: HashMap<String, u32>,
    names: Vec<String>,
}

impl ChromInterner {
    fn intern_str(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        let owned = name.to_owned();
        self.names.push(owned.clone());
        self.by_name.insert(owned, id);
        id
    }
}

enum IndexBackend {
    Tabix(tabix::index::Indexer),
    Csi {
        indexer: CsiIndexer<BinnedIndex>,
        ref_names: ReferenceSequenceNames,
    },
}

struct IndexSink {
    backend: IndexBackend,
    interner: ChromInterner,
}

impl IndexSink {
    fn new(preset: IndexPreset, contig_names: &[String], max_contig_len: usize) -> PyResult<Self> {
        if max_contig_len <= TBI_MAX_CONTIG_LEN {
            let mut indexer = tabix::index::Indexer::default();
            indexer.set_header(IndexPreset::header_builder(preset).build());
            Ok(Self {
                backend: IndexBackend::Tabix(indexer),
                interner: ChromInterner::default(),
            })
        } else {
            let mut depth: u8 = 5;
            while (1u64 << (14 + 3 * u32::from(depth))) < (max_contig_len as u64).saturating_add(1)
                && depth < 10
            {
                depth += 1;
            }
            let indexer = CsiIndexer::<BinnedIndex>::new(14, depth)
                .ok_or_else(|| PyValueError::new_err("invalid CSI min_shift/depth parameters"))?;
            let mut ref_names = ReferenceSequenceNames::new();
            let mut interner = ChromInterner::default();
            for name in contig_names {
                ref_names.insert(name.as_bytes().into());
                interner.intern_str(name);
            }
            Ok(Self {
                backend: IndexBackend::Csi { indexer, ref_names },
                interner,
            })
        }
    }

    fn intern_chrom(&mut self, chrom: &[u8]) -> PyResult<u32> {
        let chrom_str = std::str::from_utf8(chrom)
            .map_err(|e| PyValueError::new_err(format!("invalid UTF-8 contig: {e}")))?;
        if let Some(&id) = self.interner.by_name.get(chrom_str) {
            return Ok(id);
        }
        if let IndexBackend::Csi { ref_names, .. } = &mut self.backend {
            let (ref_id, _) = ref_names.insert_full(chrom.into());
            let id = self.interner.intern_str(chrom_str);
            debug_assert_eq!(id as usize, ref_id);
            Ok(id)
        } else {
            Ok(self.interner.intern_str(chrom_str))
        }
    }

    fn add_parsed(
        &mut self,
        chrom_id: u32,
        start_pos: Position,
        end_pos: Position,
        chunk: Chunk,
    ) -> PyResult<()> {
        match &mut self.backend {
            IndexBackend::Tabix(indexer) => {
                let chrom_str = &self.interner.names[chrom_id as usize];
                indexer
                    .add_record(chrom_str, start_pos, end_pos, chunk)
                    .map_err(|e| PyValueError::new_err(format!("tabix index error: {e}")))?;
            }
            IndexBackend::Csi { indexer, .. } => {
                indexer
                    .add_record(Some((chrom_id as usize, start_pos, end_pos, true)), chunk)
                    .map_err(|e| PyValueError::new_err(format!("CSI index error: {e}")))?;
            }
        }
        Ok(())
    }

    fn add_line(
        &mut self,
        preset: IndexPreset,
        line_without_nl: &[u8],
        chunk: Chunk,
    ) -> PyResult<()> {
        let trimmed = line_without_nl
            .strip_suffix(b"\r")
            .unwrap_or(line_without_nl);
        if trimmed.is_empty() || trimmed.starts_with(b"#") {
            return Ok(());
        }
        let (chrom, start_pos, end_pos) = parse_interval(preset, trimmed)?;
        let chrom_id = self.intern_chrom(chrom)?;
        self.add_parsed(chrom_id, start_pos, end_pos, chunk)
    }

    fn write(self, idx_path: &str, preset: IndexPreset) -> PyResult<()> {
        match self.backend {
            IndexBackend::Tabix(indexer) => {
                let index = indexer.build();
                tabix::fs::write(idx_path, &index).map_err(|e| {
                    PyIOError::new_err(format!("failed to write tabix index {idx_path}: {e}"))
                })?;
                Ok(())
            }
            IndexBackend::Csi { indexer, ref_names } => {
                let n_refs = ref_names.len();
                let header = IndexPreset::header_builder(preset)
                    .set_reference_sequence_names(ref_names)
                    .build();
                let index = indexer.set_header(header).build(n_refs);
                csi::fs::write(idx_path, &index).map_err(|e| {
                    PyIOError::new_err(format!("failed to write CSI index {idx_path}: {e}"))
                })?;
                Ok(())
            }
        }
    }
}

fn parse_u64_ascii(field: &[u8]) -> PyResult<u64> {
    let s = std::str::from_utf8(field)
        .map_err(|_| PyValueError::new_err("non-UTF8 coordinate field"))?;
    s.parse::<u64>()
        .map_err(|e| PyValueError::new_err(format!("invalid integer coordinate {s:?}: {e}")))
}

fn parse_vcf_info_end(info: &[u8]) -> PyResult<Option<u64>> {
    for token in info.split(|&b| b == b';') {
        if let Some(val) = token.strip_prefix(b"END=")
            && val != b"."
        {
            return Ok(Some(parse_u64_ascii(val)?));
        }
    }
    Ok(None)
}

fn parse_interval(preset: IndexPreset, line: &[u8]) -> PyResult<(&[u8], Position, Position)> {
    let mut fields = line.split(|&b| b == b'\t');
    let chrom = fields
        .next()
        .ok_or_else(|| PyValueError::new_err("missing contig column"))?;
    match preset {
        IndexPreset::Bed => {
            let start_s = fields
                .next()
                .ok_or_else(|| PyValueError::new_err("missing BED start column"))?;
            let end_s = fields
                .next()
                .ok_or_else(|| PyValueError::new_err("missing BED end column"))?;
            let start_0 = parse_u64_ascii(start_s)?;
            let end_0 = parse_u64_ascii(end_s)?;
            let start_1_u64 = start_0
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("BED start coordinate overflow"))?;
            let start_1 = usize::try_from(start_1_u64)
                .map_err(|_| PyValueError::new_err("BED start coordinate overflow"))?;
            let end_1 = usize::try_from(end_0.max(start_1_u64))
                .map_err(|_| PyValueError::new_err("BED end coordinate overflow"))?;
            let s_pos = Position::try_from(start_1)
                .map_err(|e| PyValueError::new_err(format!("invalid BED start: {e}")))?;
            let e_pos = Position::try_from(end_1)
                .map_err(|e| PyValueError::new_err(format!("invalid BED end: {e}")))?;
            Ok((chrom, s_pos, e_pos))
        }
        IndexPreset::Vcf => {
            let pos_s = fields
                .next()
                .ok_or_else(|| PyValueError::new_err("missing VCF POS column"))?;
            let _id = fields.next();
            let ref_allele = fields
                .next()
                .ok_or_else(|| PyValueError::new_err("missing VCF REF column"))?;
            let _alt = fields.next();
            let _qual = fields.next();
            let _filter = fields.next();
            let info_opt = fields.next();

            let start_1_u64 = parse_u64_ascii(pos_s)?;
            if start_1_u64 == 0 {
                return Err(PyValueError::new_err("VCF POS must be >= 1"));
            }
            let end_1_u64 = if let Some(info) = info_opt
                && let Some(info_end) = parse_vcf_info_end(info)?
                && info_end >= start_1_u64
            {
                info_end
            } else {
                let ref_len = if ref_allele.starts_with(b"<") || ref_allele == b"." {
                    1u64
                } else {
                    (ref_allele.len() as u64).max(1)
                };
                start_1_u64
                    .checked_add(ref_len - 1)
                    .ok_or_else(|| PyValueError::new_err("VCF end coordinate overflow"))?
            };

            let s_pos = Position::try_from(
                usize::try_from(start_1_u64)
                    .map_err(|_| PyValueError::new_err("VCF start coordinate overflow"))?,
            )
            .map_err(|e| PyValueError::new_err(format!("invalid VCF start: {e}")))?;
            let e_pos = Position::try_from(
                usize::try_from(end_1_u64)
                    .map_err(|_| PyValueError::new_err("VCF end coordinate overflow"))?,
            )
            .map_err(|e| PyValueError::new_err(format!("invalid VCF end: {e}")))?;
            Ok((chrom, s_pos, e_pos))
        }
    }
}

#[derive(Clone, Copy)]
struct PendingLine {
    chrom_id: u32,
    start_pos: Position,
    end_pos: Position,
    start_block: usize,
    start_uncompressed: u16,
    end_block: usize,
    end_uncompressed: u16,
}

struct ParallelBgzfBatcher {
    pool: rayon::ThreadPool,
    out: BufWriter<File>,
    file_offset: u64,
    global_block_idx: usize,
    cur_block: Vec<u8>,
    ready_blocks: Vec<Vec<u8>>,
    block_offsets: Vec<u64>,
    pending_lines: Vec<PendingLine>,
}

impl ParallelBgzfBatcher {
    fn new(out: BufWriter<File>, threads: usize) -> PyResult<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyIOError::new_err(format!("failed to create BGZF thread pool: {e}")))?;
        Ok(Self {
            pool,
            out,
            file_offset: 0,
            global_block_idx: 0,
            cur_block: Vec::with_capacity(BGZF_MAX_PAYLOAD),
            ready_blocks: Vec::with_capacity(PARALLEL_BATCH_BLOCKS),
            block_offsets: vec![0],
            pending_lines: Vec::new(),
        })
    }

    #[inline]
    fn current_pos(&self) -> (usize, u16) {
        (self.global_block_idx, self.cur_block.len() as u16)
    }

    fn append_bytes(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let space = BGZF_MAX_PAYLOAD - self.cur_block.len();
            let take = data.len().min(space);
            self.cur_block.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.cur_block.len() == BGZF_MAX_PAYLOAD {
                let finished =
                    std::mem::replace(&mut self.cur_block, Vec::with_capacity(BGZF_MAX_PAYLOAD));
                self.ready_blocks.push(finished);
                self.global_block_idx += 1;
            }
        }
    }

    fn write_line(
        &mut self,
        preset: IndexPreset,
        sink: &mut IndexSink,
        line_without_nl: &[u8],
    ) -> PyResult<()> {
        let (start_block, start_uncompressed) = self.current_pos();
        self.append_bytes(line_without_nl);
        self.append_bytes(b"\n");
        let (end_block, end_uncompressed) = self.current_pos();

        let trimmed = line_without_nl
            .strip_suffix(b"\r")
            .unwrap_or(line_without_nl);
        if trimmed.is_empty() || trimmed.starts_with(b"#") {
            return Ok(());
        }
        let (chrom, start_pos, end_pos) = parse_interval(preset, trimmed)?;
        let chrom_id = sink.intern_chrom(chrom)?;
        self.pending_lines.push(PendingLine {
            chrom_id,
            start_pos,
            end_pos,
            start_block,
            start_uncompressed,
            end_block,
            end_uncompressed,
        });
        Ok(())
    }

    fn maybe_flush_batch(&mut self, py: Python<'_>, sink: &mut IndexSink) -> PyResult<()> {
        if self.ready_blocks.len() >= PARALLEL_BATCH_BLOCKS {
            self.flush_ready_blocks(py, sink, false)?;
        }
        Ok(())
    }

    fn flush_ready_blocks(
        &mut self,
        py: Python<'_>,
        sink: &mut IndexSink,
        final_flush: bool,
    ) -> PyResult<()> {
        if final_flush && !self.cur_block.is_empty() {
            let tail = std::mem::take(&mut self.cur_block);
            self.ready_blocks.push(tail);
            self.global_block_idx += 1;
        }
        if self.ready_blocks.is_empty() {
            return Ok(());
        }

        let blocks_to_compress = std::mem::replace(
            &mut self.ready_blocks,
            Vec::with_capacity(PARALLEL_BATCH_BLOCKS),
        );
        let compressed_blocks: Result<Vec<Vec<u8>>, String> = py.detach(|| {
            self.pool.install(|| {
                blocks_to_compress
                    .par_iter()
                    .map(|payload| {
                        let mut w = bgzf::io::Writer::new(Vec::with_capacity(payload.len() + 64));
                        w.write_all(payload)
                            .map_err(|e| format!("BGZF compress error: {e}"))?;
                        w.flush().map_err(|e| format!("BGZF flush error: {e}"))?;
                        Ok(w.into_inner())
                    })
                    .collect()
            })
        });
        let compressed = compressed_blocks.map_err(PyIOError::new_err)?;

        for blk in &compressed {
            self.out
                .write_all(blk)
                .map_err(|e| PyIOError::new_err(format!("BGZF disk write error: {e}")))?;
            self.file_offset += blk.len() as u64;
            self.block_offsets.push(self.file_offset);
        }

        let max_flushed_block = self.block_offsets.len() - 1;
        let mut remaining_lines = Vec::new();
        for line in self.pending_lines.drain(..) {
            let end_ready = line.end_block < max_flushed_block
                || (line.end_block == max_flushed_block && line.end_uncompressed == 0);
            if end_ready {
                let vp_start = VirtualPosition::new(
                    self.block_offsets[line.start_block],
                    line.start_uncompressed,
                )
                .ok_or_else(|| PyValueError::new_err("BGZF virtual position overflow"))?;
                let vp_end =
                    VirtualPosition::new(self.block_offsets[line.end_block], line.end_uncompressed)
                        .ok_or_else(|| PyValueError::new_err("BGZF virtual position overflow"))?;
                sink.add_parsed(
                    line.chrom_id,
                    line.start_pos,
                    line.end_pos,
                    Chunk::new(vp_start, vp_end),
                )?;
            } else {
                remaining_lines.push(line);
            }
        }
        self.pending_lines = remaining_lines;
        Ok(())
    }

    fn finish(mut self, py: Python<'_>, sink: &mut IndexSink) -> PyResult<()> {
        self.flush_ready_blocks(py, sink, true)?;
        {
            let mut eof_writer = bgzf::io::Writer::new(&mut self.out);
            eof_writer
                .try_finish()
                .map_err(|e| PyIOError::new_err(format!("BGZF EOF write error: {e}")))?;
        }
        self.out
            .flush()
            .map_err(|e| PyIOError::new_err(format!("BGZF file flush error: {e}")))?;
        Ok(())
    }
}

fn feed_chunk_lines<E>(
    mut slice: &[u8],
    line_rem: &mut Vec<u8>,
    on_line: &mut impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    while let Some(nl_pos) = slice.iter().position(|&b| b == b'\n') {
        if line_rem.is_empty() {
            on_line(&slice[..nl_pos])?;
        } else {
            line_rem.extend_from_slice(&slice[..nl_pos]);
            on_line(line_rem)?;
            line_rem.clear();
        }
        slice = &slice[nl_pos + 1..];
    }
    if !slice.is_empty() {
        line_rem.extend_from_slice(slice);
    }
    Ok(())
}

fn try_for_each_line(
    chunks: &Bound<'_, PyAny>,
    mut on_line: impl FnMut(&[u8]) -> PyResult<()>,
) -> PyResult<()> {
    let mut line_rem: Vec<u8> = Vec::new();
    for item_res in chunks.try_iter()? {
        let item = item_res?;
        let py_bytes = item.cast::<PyBytes>()?;
        feed_chunk_lines(py_bytes.as_bytes(), &mut line_rem, &mut on_line)?;
    }
    if !line_rem.is_empty() {
        on_line(&line_rem)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_bgzf_and_index_inner(
    py: Python<'_>,
    file: File,
    idx_path: &str,
    idx_preset: IndexPreset,
    contig_names: &[String],
    max_contig_len: usize,
    chunks: &Bound<'_, PyAny>,
    threads: usize,
) -> PyResult<()> {
    let buf_out = BufWriter::with_capacity(256 * 1024, file);
    let mut sink = IndexSink::new(idx_preset, contig_names, max_contig_len)?;

    if threads <= 1 {
        let mut writer = bgzf::io::Writer::new(buf_out);
        try_for_each_line(chunks, |raw_line| {
            let vp_start = writer.virtual_position();
            writer
                .write_all(raw_line)
                .and_then(|()| writer.write_all(b"\n"))
                .map_err(|e| PyIOError::new_err(format!("BGZF write error: {e}")))?;
            let vp_end = writer.virtual_position();
            sink.add_line(idx_preset, raw_line, Chunk::new(vp_start, vp_end))
        })?;
        let mut inner = writer
            .finish()
            .map_err(|e| PyIOError::new_err(format!("BGZF finish error: {e}")))?;
        inner
            .flush()
            .map_err(|e| PyIOError::new_err(format!("BGZF flush error: {e}")))?;
    } else {
        let mut batcher = ParallelBgzfBatcher::new(buf_out, threads)?;
        try_for_each_line(chunks, |raw_line| {
            batcher.write_line(idx_preset, &mut sink, raw_line)?;
            batcher.maybe_flush_batch(py, &mut sink)
        })?;
        batcher.finish(py, &mut sink)?;
    }

    sink.write(idx_path, idx_preset)
}

#[pyfunction]
#[pyo3(signature = (gz_path, preset, contig_names, max_contig_len, chunks, threads=1))]
pub(crate) fn write_bgzf_indexed(
    py: Python<'_>,
    gz_path: &str,
    preset: &str,
    contig_names: Vec<String>,
    max_contig_len: isize,
    chunks: &Bound<'_, PyAny>,
    threads: isize,
) -> PyResult<String> {
    if threads <= 0 {
        return Err(PyValueError::new_err("threads must be a positive integer"));
    }
    if max_contig_len < 0 {
        return Err(PyValueError::new_err("max_contig_len must be non-negative"));
    }
    let max_contig_len = max_contig_len as usize;
    let threads = threads as usize;
    let idx_preset = IndexPreset::parse(preset)?;
    let out_path = PathBuf::from(gz_path);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            PyIOError::new_err(format!(
                "failed to create directory {}: {e}",
                parent.display()
            ))
        })?;
    }
    let idx_path = if max_contig_len <= TBI_MAX_CONTIG_LEN {
        format!("{gz_path}.tbi")
    } else {
        format!("{gz_path}.csi")
    };
    let file = File::create(&out_path)
        .map_err(|e| PyIOError::new_err(format!("failed to create {gz_path}: {e}")))?;
    let mut cleanup = OutputCleanupGuard::new(out_path, PathBuf::from(&idx_path));
    write_bgzf_and_index_inner(
        py,
        file,
        &idx_path,
        idx_preset,
        &contig_names,
        max_contig_len,
        chunks,
        threads,
    )?;
    cleanup.disarm();
    Ok(idx_path)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufWriter;

    use super::{
        IndexPreset, IndexSink, OutputCleanupGuard, ParallelBgzfBatcher, feed_chunk_lines,
    };

    #[test]
    fn chunk_line_framing_handles_splits_and_trailing_bytes() {
        let chunks: &[&[u8]] = &[b"chr1\t0\t1", b"00\nchr1\t100\t200\n\nchr2", b"\t0\t50"];
        let mut rem = Vec::new();
        let mut lines: Vec<Vec<u8>> = Vec::new();
        for &c in chunks {
            feed_chunk_lines::<()>(c, &mut rem, &mut |line| {
                lines.push(line.to_vec());
                Ok(())
            })
            .unwrap();
        }
        if !rem.is_empty() {
            lines.push(rem);
        }
        assert_eq!(
            lines,
            vec![
                b"chr1\t0\t100".to_vec(),
                b"chr1\t100\t200".to_vec(),
                b"".to_vec(),
                b"chr2\t0\t50".to_vec(),
            ]
        );
    }

    #[test]
    fn cleanup_guard_unlinks_partial_outputs_unless_disarmed() {
        let dir = std::env::temp_dir();
        let gz_fail = dir.join(format!("ccnv_guard_fail_{}.bed.gz", std::process::id()));
        let idx_fail = dir.join(format!("ccnv_guard_fail_{}.bed.gz.tbi", std::process::id()));
        File::create(&gz_fail).unwrap();
        File::create(&idx_fail).unwrap();
        {
            let _guard = OutputCleanupGuard::new(gz_fail.clone(), idx_fail.clone());
        }
        assert!(!gz_fail.exists());
        assert!(!idx_fail.exists());

        let gz_ok = dir.join(format!("ccnv_guard_ok_{}.bed.gz", std::process::id()));
        let idx_ok = dir.join(format!("ccnv_guard_ok_{}.bed.gz.tbi", std::process::id()));
        File::create(&gz_ok).unwrap();
        File::create(&idx_ok).unwrap();
        {
            let mut guard = OutputCleanupGuard::new(gz_ok.clone(), idx_ok.clone());
            guard.disarm();
        }
        assert!(gz_ok.exists());
        assert!(idx_ok.exists());
        let _ = std::fs::remove_file(&gz_ok);
        let _ = std::fs::remove_file(&idx_ok);
    }

    #[test]
    fn parallel_batcher_skips_comments_and_empty_lines_in_pending_queue() {
        let out_path =
            std::env::temp_dir().join(format!("ccnv_batcher_{}.bed.gz", std::process::id()));
        let file = File::create(&out_path).unwrap();
        let mut batcher = ParallelBgzfBatcher::new(BufWriter::new(file), 2).unwrap();
        let mut sink = IndexSink::new(IndexPreset::Bed, &["chr1".to_owned()], 10_000).unwrap();

        batcher
            .write_line(IndexPreset::Bed, &mut sink, b"#chrom\tstart\tend")
            .unwrap();
        batcher
            .write_line(IndexPreset::Bed, &mut sink, b"")
            .unwrap();
        assert!(batcher.pending_lines.is_empty());

        batcher
            .write_line(IndexPreset::Bed, &mut sink, b"chr1\t0\t100")
            .unwrap();
        batcher
            .write_line(IndexPreset::Bed, &mut sink, b"chr1\t100\t200")
            .unwrap();
        assert_eq!(batcher.pending_lines.len(), 2);
        assert_eq!(batcher.pending_lines[0].chrom_id, 0);
        assert_eq!(batcher.pending_lines[1].chrom_id, 0);
        drop(batcher);
        let _ = std::fs::remove_file(&out_path);
    }
}
