use noodles::bam;
use noodles::sam;
use noodles::sam::alignment::record::Cigar as CigarTrait;
use noodles::sam::alignment::record::Flags;
use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;

use super::accumulator::{
    ContigRole, FRAG_MAX, FRAG_MIN, FilterSpec, HIST_LEN, NUM_FRAG_BANDS, QcCounters, TargetContig,
    TileAccumulator,
};
use crate::FRAG_BAND_EDGES_ARR;

const BAND_LUT: [u8; HIST_LEN] = build_band_lut();
const BAM_NIBBLE_GC_LUT: [(u8, u8); 256] = build_bam_nibble_gc_lut();
const ASCII_GC_LUT: [(u8, u8); 256] = build_ascii_gc_lut();
const ASCII_BASE2_LUT: [u8; 256] = build_ascii_base2_lut();
const ASCII_COMP2_LUT: [u8; 256] = build_ascii_comp2_lut();
const BAM_NIBBLE_BASE2_LUT: [u8; 16] = build_bam_nibble_base2_lut();
const BAM_NIBBLE_COMP2_LUT: [u8; 16] = build_bam_nibble_comp2_lut();

const fn build_band_lut() -> [u8; HIST_LEN] {
    let mut lut = [0u8; HIST_LEN];
    let mut i = 0;
    while i < HIST_LEN {
        let len = (i as i32) + FRAG_MIN;
        let mut b = 0usize;
        while b < NUM_FRAG_BANDS {
            if len >= FRAG_BAND_EDGES_ARR[b] && len < FRAG_BAND_EDGES_ARR[b + 1] {
                lut[i] = b as u8;
                break;
            }
            b += 1;
        }
        i += 1;
    }
    lut
}

const fn nibble_gc_valid(nib: u8) -> (u8, u8) {
    match nib {
        1 | 8 => (0, 1), // A, T
        2 | 4 => (1, 1), // C, G
        _ => (0, 0),
    }
}

const fn build_bam_nibble_gc_lut() -> [(u8, u8); 256] {
    let mut lut = [(0u8, 0u8); 256];
    let mut byte = 0usize;
    while byte < 256 {
        let (gc_hi, val_hi) = nibble_gc_valid((byte >> 4) as u8);
        let (gc_lo, val_lo) = nibble_gc_valid((byte & 0x0f) as u8);
        lut[byte] = (gc_hi + gc_lo, val_hi + val_lo);
        byte += 1;
    }
    lut
}

const fn build_ascii_gc_lut() -> [(u8, u8); 256] {
    let mut lut = [(0u8, 0u8); 256];
    let mut b = 0usize;
    while b < 256 {
        lut[b] = match b as u8 {
            b'G' | b'g' | b'C' | b'c' => (1, 1),
            b'A' | b'a' | b'T' | b't' | b'U' | b'u' => (0, 1),
            _ => (0, 0),
        };
        b += 1;
    }
    lut
}

const fn build_ascii_base2_lut() -> [u8; 256] {
    let mut lut = [0xffu8; 256];
    lut[b'A' as usize] = 0;
    lut[b'a' as usize] = 0;
    lut[b'C' as usize] = 1;
    lut[b'c' as usize] = 1;
    lut[b'G' as usize] = 2;
    lut[b'g' as usize] = 2;
    lut[b'T' as usize] = 3;
    lut[b't' as usize] = 3;
    lut
}

const fn build_ascii_comp2_lut() -> [u8; 256] {
    let mut lut = [0xffu8; 256];
    lut[b'A' as usize] = 3;
    lut[b'a' as usize] = 3;
    lut[b'C' as usize] = 2;
    lut[b'c' as usize] = 2;
    lut[b'G' as usize] = 1;
    lut[b'g' as usize] = 1;
    lut[b'T' as usize] = 0;
    lut[b't' as usize] = 0;
    lut
}

const fn build_bam_nibble_base2_lut() -> [u8; 16] {
    let mut lut = [0xffu8; 16];
    lut[1] = 0; // A
    lut[2] = 1; // C
    lut[4] = 2; // G
    lut[8] = 3; // T
    lut
}

const fn build_bam_nibble_comp2_lut() -> [u8; 16] {
    let mut lut = [0xffu8; 16];
    lut[1] = 3; // comp(A) = T
    lut[2] = 2; // comp(C) = G
    lut[4] = 1; // comp(G) = C
    lut[8] = 0; // comp(T) = A
    lut
}

#[inline]
fn is_match_kind(kind: CigarKind) -> bool {
    matches!(
        kind,
        CigarKind::Match | CigarKind::SequenceMatch | CigarKind::SequenceMismatch
    )
}

#[inline]
fn bam_gc_fraction(packed_seq: &[u8], seq_len: usize) -> f64 {
    if seq_len == 0 {
        return 0.0;
    }
    let full_bytes = seq_len / 2;
    let mut gc_total: u32 = 0;
    let mut valid_total: u32 = 0;
    for &byte in &packed_seq[..full_bytes] {
        let (gc, val) = BAM_NIBBLE_GC_LUT[usize::from(byte)];
        gc_total += u32::from(gc);
        valid_total += u32::from(val);
    }
    if seq_len % 2 == 1 {
        let (gc, val) = nibble_gc_valid(packed_seq[full_bytes] >> 4);
        gc_total += u32::from(gc);
        valid_total += u32::from(val);
    }
    if valid_total > 0 {
        f64::from(gc_total) / f64::from(valid_total)
    } else {
        0.0
    }
}

#[inline]
fn bam_get_nibble(packed_seq: &[u8], idx: usize) -> u8 {
    let byte = packed_seq[idx >> 1];
    if idx & 1 == 0 { byte >> 4 } else { byte & 0x0f }
}

#[inline]
fn bam_extract_motif(
    packed_seq: &[u8],
    seq_len: usize,
    is_reverse: bool,
    first_cigar_match: bool,
    last_cigar_match: bool,
) -> Option<u8> {
    if seq_len < 4 {
        return None;
    }
    if !is_reverse {
        if !first_cigar_match {
            return None;
        }
        let b0 = BAM_NIBBLE_BASE2_LUT[usize::from(bam_get_nibble(packed_seq, 0))];
        let b1 = BAM_NIBBLE_BASE2_LUT[usize::from(bam_get_nibble(packed_seq, 1))];
        let b2 = BAM_NIBBLE_BASE2_LUT[usize::from(bam_get_nibble(packed_seq, 2))];
        let b3 = BAM_NIBBLE_BASE2_LUT[usize::from(bam_get_nibble(packed_seq, 3))];
        if (b0 | b1 | b2 | b3) & 0x80 != 0 {
            return None;
        }
        Some((b0 << 6) | (b1 << 4) | (b2 << 2) | b3)
    } else {
        if !last_cigar_match {
            return None;
        }
        let b0 = BAM_NIBBLE_COMP2_LUT[usize::from(bam_get_nibble(packed_seq, seq_len - 1))];
        let b1 = BAM_NIBBLE_COMP2_LUT[usize::from(bam_get_nibble(packed_seq, seq_len - 2))];
        let b2 = BAM_NIBBLE_COMP2_LUT[usize::from(bam_get_nibble(packed_seq, seq_len - 3))];
        let b3 = BAM_NIBBLE_COMP2_LUT[usize::from(bam_get_nibble(packed_seq, seq_len - 4))];
        if (b0 | b1 | b2 | b3) & 0x80 != 0 {
            return None;
        }
        Some((b0 << 6) | (b1 << 4) | (b2 << 2) | b3)
    }
}

#[inline]
fn ascii_gc_fraction(seq: &[u8]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    let mut gc_total: u32 = 0;
    let mut valid_total: u32 = 0;
    for &ch in seq {
        let (gc, val) = ASCII_GC_LUT[usize::from(ch)];
        gc_total += u32::from(gc);
        valid_total += u32::from(val);
    }
    if valid_total > 0 {
        f64::from(gc_total) / f64::from(valid_total)
    } else {
        0.0
    }
}

#[inline]
fn ascii_extract_motif(
    seq: &[u8],
    is_reverse: bool,
    first_cigar_match: bool,
    last_cigar_match: bool,
) -> Option<u8> {
    let n = seq.len();
    if n < 4 {
        return None;
    }
    if !is_reverse {
        if !first_cigar_match {
            return None;
        }
        let b0 = ASCII_BASE2_LUT[usize::from(seq[0])];
        let b1 = ASCII_BASE2_LUT[usize::from(seq[1])];
        let b2 = ASCII_BASE2_LUT[usize::from(seq[2])];
        let b3 = ASCII_BASE2_LUT[usize::from(seq[3])];
        if (b0 | b1 | b2 | b3) & 0x80 != 0 {
            return None;
        }
        Some((b0 << 6) | (b1 << 4) | (b2 << 2) | b3)
    } else {
        if !last_cigar_match {
            return None;
        }
        let b0 = ASCII_COMP2_LUT[usize::from(seq[n - 1])];
        let b1 = ASCII_COMP2_LUT[usize::from(seq[n - 2])];
        let b2 = ASCII_COMP2_LUT[usize::from(seq[n - 3])];
        let b3 = ASCII_COMP2_LUT[usize::from(seq[n - 4])];
        if (b0 | b1 | b2 | b3) & 0x80 != 0 {
            return None;
        }
        Some((b0 << 6) | (b1 << 4) | (b2 << 2) | b3)
    }
}

#[inline]
fn check_filters(
    flags: Flags,
    start_0: Option<usize>,
    mapq: u8,
    contig_len: usize,
    filters: FilterSpec,
    qc: &mut QcCounters,
) -> Option<usize> {
    qc.reads_seen += 1;
    if filters.exclude_unmapped && flags.is_unmapped() {
        qc.filter_unmapped += 1;
        qc.reads_filtered += 1;
        return None;
    }
    let Some(s0) = start_0 else {
        qc.filter_no_start += 1;
        qc.reads_filtered += 1;
        return None;
    };
    if filters.exclude_secondary && flags.is_secondary() {
        qc.filter_secondary += 1;
        qc.reads_filtered += 1;
        return None;
    }
    if filters.exclude_supplementary && flags.is_supplementary() {
        qc.filter_supplementary += 1;
        qc.reads_filtered += 1;
        return None;
    }
    if filters.exclude_duplicates && flags.is_duplicate() {
        qc.filter_duplicate += 1;
        qc.reads_filtered += 1;
        return None;
    }
    if filters.require_proper_pair && flags.is_segmented() && !flags.is_properly_segmented() {
        qc.filter_improper_pair += 1;
        qc.reads_filtered += 1;
        return None;
    }
    if mapq < filters.min_mapq {
        qc.filter_mapq += 1;
        qc.reads_filtered += 1;
        return None;
    }
    if s0 >= contig_len {
        qc.filter_no_start += 1;
        qc.reads_filtered += 1;
        return None;
    }
    Some(s0)
}

#[inline]
fn cigar_info<C: CigarTrait>(cigar: &C) -> Result<(usize, bool, bool), String> {
    let mut ref_span = 0usize;
    let mut first_match = false;
    let mut last_match = false;
    let mut seen_any = false;
    for op_res in cigar.iter() {
        let op = op_res.map_err(|e| format!("invalid CIGAR op: {e}"))?;
        let is_m = is_match_kind(op.kind());
        if !seen_any {
            first_match = is_m;
            seen_any = true;
        }
        last_match = is_m;
        if op.kind().consumes_reference() {
            ref_span += op.len();
        }
    }
    Ok((ref_span, first_match, last_match))
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn record_kept_paired(
    start_0: usize,
    tlen: i32,
    gc_frac: f64,
    motif: Option<u8>,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    acc: &mut TileAccumulator,
) {
    acc.qc.reads_kept += 1;
    match target.role {
        ContigRole::Autosomal => acc.qc.autosomal_reads_kept += 1,
        ContigRole::ChrX | ContigRole::ChrY => acc.qc.sex_reads_kept += 1,
        ContigRole::Other => {}
    }

    let max_coord = target.length - 1;
    let tlen_usize = tlen as usize;
    let p = (start_0 + tlen_usize / 2).min(max_coord);
    let end_pos = start_0.max(start_0 + tlen_usize - 1).min(max_coord);

    let b_mid = target.bin_offset + (p / bin_size);
    let b_start = target.bin_offset + (start_0 / bin_size);
    let b_end = target.bin_offset + (end_pos / bin_size);

    acc.add_endpoint(b_start);
    acc.add_endpoint(b_end);

    let in_range = (FRAG_MIN..=FRAG_MAX).contains(&tlen);
    let mut band_idx = 0u8;
    let mut is_short = false;
    if in_range {
        let hist_idx = (tlen - FRAG_MIN) as usize;
        band_idx = BAND_LUT[hist_idx];
        is_short = (80..=150).contains(&tlen);
        match target.role {
            ContigRole::Autosomal => acc.hist_autosomal[hist_idx] += 1,
            ContigRole::ChrX => acc.hist_chrx[hist_idx] += 1,
            ContigRole::ChrY => acc.hist_chry[hist_idx] += 1,
            ContigRole::Other => {}
        }
    }

    acc.add_midpoint(
        b_mid,
        in_range,
        if in_range { tlen as u64 } else { 0 },
        is_short,
        band_idx,
        gc_frac,
        filters.collect_bands,
    );

    if let Some(m) = motif {
        acc.motif_counts[usize::from(m)] += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn record_kept_single_end(
    start_0: usize,
    read_end: usize,
    is_reverse: bool,
    gc_frac: f64,
    motif: Option<u8>,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    acc: &mut TileAccumulator,
) {
    acc.qc.reads_kept += 1;
    match target.role {
        ContigRole::Autosomal => acc.qc.autosomal_reads_kept += 1,
        ContigRole::ChrX | ContigRole::ChrY => acc.qc.sex_reads_kept += 1,
        ContigRole::Other => {}
    }

    let max_coord = target.length - 1;
    let p = if filters.se_fragment_length == 0 {
        start_0.min(max_coord)
    } else if !is_reverse {
        (start_0 + filters.se_fragment_length / 2).min(max_coord)
    } else {
        let half = (filters.se_fragment_length / 2) as isize;
        let shifted = (read_end as isize) - 1 - half;
        (shifted.max(0) as usize).min(max_coord)
    };

    let five_prime = if !is_reverse {
        start_0.min(max_coord)
    } else {
        read_end.saturating_sub(1).min(max_coord)
    };

    let b_mid = target.bin_offset + (p / bin_size);
    let b_ep = target.bin_offset + (five_prime / bin_size);
    acc.add_endpoint(b_ep);
    acc.add_midpoint(b_mid, false, 0, false, 0, gc_frac, filters.collect_bands);

    if let Some(m) = motif {
        acc.motif_counts[usize::from(m)] += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn ingest_filtered<C, FGc, FMotif>(
    flags: Flags,
    start_0: usize,
    tlen: i32,
    seq_len: usize,
    cigar: &C,
    compute_gc: FGc,
    extract_motif: FMotif,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    acc: &mut TileAccumulator,
) -> Result<(), String>
where
    C: CigarTrait,
    FGc: FnOnce() -> f64,
    FMotif: FnOnce(bool, bool, bool) -> Option<u8>,
{
    let is_rev = flags.is_reverse_complemented();
    if flags.is_segmented() {
        acc.paired_end = true;
        if tlen <= 0 {
            if tlen < 0 && flags.is_properly_segmented() {
                let (_, first_m, last_m) = cigar_info(cigar)?;
                if let Some(m) = extract_motif(is_rev, first_m, last_m) {
                    acc.motif_counts[usize::from(m)] += 1;
                }
            }
            acc.qc.mate_dedup_skipped += 1;
            return Ok(());
        }
        let (_, first_m, last_m) = cigar_info(cigar)?;
        record_kept_paired(
            start_0,
            tlen,
            compute_gc(),
            extract_motif(is_rev, first_m, last_m),
            target,
            bin_size,
            filters,
            acc,
        );
    } else {
        let (ref_span, first_m, last_m) = cigar_info(cigar)?;
        let span = if ref_span > 0 {
            ref_span
        } else {
            seq_len.max(1)
        };
        record_kept_single_end(
            start_0,
            start_0 + span,
            is_rev,
            compute_gc(),
            extract_motif(is_rev, first_m, last_m),
            target,
            bin_size,
            filters,
            acc,
        );
    }
    Ok(())
}

pub(super) fn process_bam_record(
    record: &bam::Record,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    tile_bounds: Option<(usize, usize)>,
    acc: &mut TileAccumulator,
) -> Result<(), String> {
    let start_opt = match record.alignment_start() {
        Some(Ok(pos)) => Some(usize::from(pos) - 1),
        Some(Err(e)) => return Err(format!("invalid BAM alignment start: {e}")),
        None => None,
    };
    if let Some((t_start, t_end)) = tile_bounds {
        match start_opt {
            Some(s0) if s0 >= t_start && s0 < t_end => {}
            None if t_start == 0 => {}
            _ => return Ok(()),
        }
    }

    let flags = record.flags();
    let mapq = record.mapping_quality().map_or(255, u8::from);
    let Some(start_0) = check_filters(flags, start_opt, mapq, target.length, filters, &mut acc.qc)
    else {
        return Ok(());
    };

    let seq = record.sequence();
    let seq_len = seq.len();
    let packed = seq.as_ref();
    let cigar = record.cigar();
    ingest_filtered(
        flags,
        start_0,
        record.template_length(),
        seq_len,
        &cigar,
        || bam_gc_fraction(packed, seq_len),
        |is_rev, first_m, last_m| bam_extract_motif(packed, seq_len, is_rev, first_m, last_m),
        target,
        bin_size,
        filters,
        acc,
    )
}

pub(super) fn process_record_buf(
    record: &sam::alignment::RecordBuf,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    tile_bounds: Option<(usize, usize)>,
    acc: &mut TileAccumulator,
) -> Result<(), String> {
    let start_opt = record.alignment_start().map(|pos| usize::from(pos) - 1);
    if let Some((t_start, t_end)) = tile_bounds {
        match start_opt {
            Some(s0) if s0 >= t_start && s0 < t_end => {}
            None if t_start == 0 => {}
            _ => return Ok(()),
        }
    }

    let flags = record.flags();
    let mapq = record.mapping_quality().map_or(255, u8::from);
    let Some(start_0) = check_filters(flags, start_opt, mapq, target.length, filters, &mut acc.qc)
    else {
        return Ok(());
    };

    let seq_bytes = record.sequence().as_ref();
    let cigar = record.cigar();
    ingest_filtered(
        flags,
        start_0,
        record.template_length(),
        seq_bytes.len(),
        cigar,
        || ascii_gc_fraction(seq_bytes),
        |is_rev, first_m, last_m| ascii_extract_motif(seq_bytes, is_rev, first_m, last_m),
        target,
        bin_size,
        filters,
        acc,
    )
}
