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
const BAM_NIBBLE_BASE2_LUT: [u8; 16] = build_bam_nibble_base2_lut();

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
        1 | 8 => (0, 1),
        2 | 4 => (1, 1),
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

const fn build_bam_nibble_base2_lut() -> [u8; 16] {
    let mut lut = [0xffu8; 16];
    lut[1] = 0;
    lut[2] = 1;
    lut[4] = 2;
    lut[8] = 3;
    lut
}

pub(super) trait SeqAccess {
    fn len(&self) -> usize;
    fn base2_at(&self, idx: usize) -> u8;
    fn gc_and_valid_counts(&self) -> (u32, u32);
}

pub(super) struct PackedNibbles<'a>(bam::record::Sequence<'a>);

impl SeqAccess for PackedNibbles<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn base2_at(&self, idx: usize) -> u8 {
        let byte = self.0.as_ref()[idx >> 1];
        let nib = if idx & 1 == 0 { byte >> 4 } else { byte & 0x0f };
        BAM_NIBBLE_BASE2_LUT[usize::from(nib)]
    }

    #[inline(always)]
    fn gc_and_valid_counts(&self) -> (u32, u32) {
        let packed = self.0.as_ref();
        let len = self.0.len();
        let full_bytes = len >> 1;
        let mut gc_total = 0u32;
        let mut valid_total = 0u32;
        for &byte in &packed[..full_bytes] {
            let (gc, val) = BAM_NIBBLE_GC_LUT[usize::from(byte)];
            gc_total += u32::from(gc);
            valid_total += u32::from(val);
        }
        if len & 1 == 1 {
            let (gc, val) = nibble_gc_valid(packed[full_bytes] >> 4);
            gc_total += u32::from(gc);
            valid_total += u32::from(val);
        }
        (gc_total, valid_total)
    }
}

pub(super) struct AsciiBases<'a>(&'a [u8]);

impl SeqAccess for AsciiBases<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn base2_at(&self, idx: usize) -> u8 {
        ASCII_BASE2_LUT[usize::from(self.0[idx])]
    }

    #[inline(always)]
    fn gc_and_valid_counts(&self) -> (u32, u32) {
        let mut gc_total = 0u32;
        let mut valid_total = 0u32;
        for &ch in self.0 {
            let (gc, val) = ASCII_GC_LUT[usize::from(ch)];
            gc_total += u32::from(gc);
            valid_total += u32::from(val);
        }
        (gc_total, valid_total)
    }
}

pub(super) trait AlignmentRecord {
    type Seq<'a>: SeqAccess
    where
        Self: 'a;

    fn alignment_start_0(&self) -> Result<Option<usize>, String>;
    fn flags(&self) -> Flags;
    fn mapq(&self) -> u8;
    fn template_length(&self) -> i32;
    fn cigar_summary(&self) -> Result<(usize, bool, bool), String>;
    fn seq(&self) -> Self::Seq<'_>;
}

impl AlignmentRecord for bam::Record {
    type Seq<'a> = PackedNibbles<'a>;

    #[inline(always)]
    fn alignment_start_0(&self) -> Result<Option<usize>, String> {
        self.alignment_start()
            .transpose()
            .map_err(|e| format!("invalid BAM alignment start: {e}"))
            .map(|opt| opt.map(|pos| usize::from(pos) - 1))
    }

    #[inline(always)]
    fn flags(&self) -> Flags {
        bam::Record::flags(self)
    }

    #[inline(always)]
    fn mapq(&self) -> u8 {
        self.mapping_quality().map_or(255, u8::from)
    }

    #[inline(always)]
    fn template_length(&self) -> i32 {
        bam::Record::template_length(self)
    }

    #[inline(always)]
    fn cigar_summary(&self) -> Result<(usize, bool, bool), String> {
        cigar_info(&self.cigar())
    }

    #[inline(always)]
    fn seq(&self) -> Self::Seq<'_> {
        PackedNibbles(self.sequence())
    }
}

impl AlignmentRecord for sam::alignment::RecordBuf {
    type Seq<'a> = AsciiBases<'a>;

    #[inline(always)]
    fn alignment_start_0(&self) -> Result<Option<usize>, String> {
        Ok(self.alignment_start().map(|pos| usize::from(pos) - 1))
    }

    #[inline(always)]
    fn flags(&self) -> Flags {
        sam::alignment::RecordBuf::flags(self)
    }

    #[inline(always)]
    fn mapq(&self) -> u8 {
        self.mapping_quality().map_or(255, u8::from)
    }

    #[inline(always)]
    fn template_length(&self) -> i32 {
        sam::alignment::RecordBuf::template_length(self)
    }

    #[inline(always)]
    fn cigar_summary(&self) -> Result<(usize, bool, bool), String> {
        cigar_info(self.cigar())
    }

    #[inline(always)]
    fn seq(&self) -> Self::Seq<'_> {
        AsciiBases(self.sequence().as_ref())
    }
}

#[inline(always)]
fn gc_fraction<S: SeqAccess>(seq: &S) -> f64 {
    if seq.len() == 0 {
        return 0.0;
    }
    let (gc_total, valid_total) = seq.gc_and_valid_counts();
    if valid_total > 0 {
        f64::from(gc_total) / f64::from(valid_total)
    } else {
        0.0
    }
}

#[inline(always)]
fn extract_motif<S: SeqAccess>(
    seq: &S,
    is_reverse: bool,
    first_cigar_match: bool,
    last_cigar_match: bool,
) -> Option<u8> {
    let n = seq.len();
    let valid_end = if is_reverse {
        last_cigar_match
    } else {
        first_cigar_match
    };
    if n < 4 || !valid_end {
        return None;
    }
    let (b0, b1, b2, b3) = if !is_reverse {
        (
            seq.base2_at(0),
            seq.base2_at(1),
            seq.base2_at(2),
            seq.base2_at(3),
        )
    } else {
        (
            seq.base2_at(n - 1),
            seq.base2_at(n - 2),
            seq.base2_at(n - 3),
            seq.base2_at(n - 4),
        )
    };
    if (b0 | b1 | b2 | b3) & 0x80 != 0 {
        return None;
    }
    let packed = (b0 << 6) | (b1 << 4) | (b2 << 2) | b3;
    Some(if is_reverse { packed ^ 0xff } else { packed })
}

#[inline]
fn is_match_kind(kind: CigarKind) -> bool {
    matches!(
        kind,
        CigarKind::Match | CigarKind::SequenceMatch | CigarKind::SequenceMismatch
    )
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
    let reject_counter = if filters.exclude_unmapped && flags.is_unmapped() {
        &mut qc.filter_unmapped
    } else if start_0.is_none() {
        &mut qc.filter_no_start
    } else if filters.exclude_secondary && flags.is_secondary() {
        &mut qc.filter_secondary
    } else if filters.exclude_supplementary && flags.is_supplementary() {
        &mut qc.filter_supplementary
    } else if filters.exclude_duplicates && flags.is_duplicate() {
        &mut qc.filter_duplicate
    } else if filters.require_proper_pair && flags.is_segmented() && !flags.is_properly_segmented()
    {
        &mut qc.filter_improper_pair
    } else if mapq < filters.min_mapq {
        &mut qc.filter_mapq
    } else if start_0.is_some_and(|s0| s0 >= contig_len) {
        &mut qc.filter_no_start
    } else {
        return start_0;
    };
    *reject_counter += 1;
    qc.reads_filtered += 1;
    None
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

#[inline(always)]
pub(super) fn process_record<R: AlignmentRecord>(
    record: &R,
    target: &TargetContig,
    bin_size: usize,
    filters: FilterSpec,
    tile_bounds: Option<(usize, usize)>,
    acc: &mut TileAccumulator,
) -> Result<(), String> {
    let start_opt = record.alignment_start_0()?;
    if let Some((t_start, t_end)) = tile_bounds {
        match start_opt {
            Some(s0) if s0 >= t_start && s0 < t_end => {}
            None if t_start == 0 => {}
            _ => return Ok(()),
        }
    }

    let flags = record.flags();
    let Some(start_0) = check_filters(
        flags,
        start_opt,
        record.mapq(),
        target.length,
        filters,
        &mut acc.qc,
    ) else {
        return Ok(());
    };

    let tlen = record.template_length();
    let seq = record.seq();
    let is_rev = flags.is_reverse_complemented();
    if flags.is_segmented() {
        acc.paired_end = true;
        if tlen <= 0 {
            if tlen < 0 {
                let (_, first_m, last_m) = record.cigar_summary()?;
                if let Some(m) = extract_motif(&seq, is_rev, first_m, last_m) {
                    acc.motif_counts[usize::from(m)] += 1;
                }
            }
            acc.qc.mate_dedup_skipped += 1;
            return Ok(());
        }
    }

    let (ref_span, first_m, last_m) = record.cigar_summary()?;
    acc.qc.reads_kept += 1;
    match target.role {
        ContigRole::Autosomal => acc.qc.autosomal_reads_kept += 1,
        ContigRole::ChrX | ContigRole::ChrY => acc.qc.sex_reads_kept += 1,
        ContigRole::Other => {}
    }

    let gc_frac = gc_fraction(&seq);
    let max_coord = target.length - 1;

    if flags.is_segmented() {
        let tlen_usize = tlen as usize;
        let p = start_0.saturating_add(tlen_usize / 2).min(max_coord);
        let end_pos = start_0.saturating_add(tlen_usize - 1).min(max_coord);

        acc.add_endpoint(target.bin_offset + (start_0 / bin_size));
        acc.add_endpoint(target.bin_offset + (end_pos / bin_size));

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
            target.bin_offset + (p / bin_size),
            in_range,
            if in_range { tlen as u64 } else { 0 },
            is_short,
            band_idx,
            gc_frac,
            filters.collect_bands,
        );
    } else {
        let span = if ref_span > 0 {
            ref_span
        } else {
            seq.len().max(1)
        };
        let read_end = start_0.saturating_add(span);
        let p = if filters.se_fragment_length == 0 {
            start_0.min(max_coord)
        } else if !is_rev {
            start_0
                .saturating_add(filters.se_fragment_length / 2)
                .min(max_coord)
        } else {
            read_end
                .saturating_sub(1)
                .saturating_sub(filters.se_fragment_length / 2)
                .min(max_coord)
        };
        let five_prime = if !is_rev {
            start_0.min(max_coord)
        } else {
            read_end.saturating_sub(1).min(max_coord)
        };

        acc.add_endpoint(target.bin_offset + (five_prime / bin_size));
        acc.add_midpoint(
            target.bin_offset + (p / bin_size),
            false,
            0,
            false,
            0,
            gc_frac,
            filters.collect_bands,
        );
    }

    if let Some(m) = extract_motif(&seq, is_rev, first_m, last_m) {
        acc.motif_counts[usize::from(m)] += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use noodles::bam;
    use noodles::core::Position;
    use noodles::sam;
    use noodles::sam::alignment::io::Write as SamAlignmentWrite;
    use noodles::sam::alignment::record::cigar::Op;
    use noodles::sam::alignment::record::cigar::op::Kind as CigarKind;
    use noodles::sam::alignment::record::{Flags, MappingQuality};
    use noodles::sam::alignment::record_buf::{Cigar, Sequence};
    use noodles::sam::header::record::value::Map;
    use noodles::sam::header::record::value::map::ReferenceSequence;

    use super::process_record;
    use crate::alignments::accumulator::{
        ContigRole, FilterSpec, TargetContig, TileAccumulator, reduce_tile_accumulators,
    };

    fn make_record(
        flags: Flags,
        start_1: Option<usize>,
        mapq: u8,
        tlen: i32,
        ops: &[(CigarKind, usize)],
        seq: &[u8],
    ) -> sam::alignment::RecordBuf {
        let cigar: Cigar = ops
            .iter()
            .map(|&(k, len)| Op::new(k, len))
            .collect::<Vec<_>>()
            .into_iter()
            .collect();
        let mut builder = sam::alignment::RecordBuf::builder()
            .set_flags(flags)
            .set_reference_sequence_id(0)
            .set_template_length(tlen)
            .set_cigar(cigar)
            .set_sequence(Sequence::from(seq.to_vec()));
        if let Some(pos) = start_1 {
            builder = builder.set_alignment_start(Position::try_from(pos).unwrap());
        }
        if let Some(mq) = MappingQuality::new(mapq) {
            builder = builder.set_mapping_quality(mq);
        }
        builder.build()
    }

    #[test]
    fn bam_nibble_and_ascii_record_paths_match_identically() {
        let len_1000 = NonZeroUsize::try_from(1000).unwrap();
        let header = sam::Header::builder()
            .add_reference_sequence("chr1", Map::<ReferenceSequence>::new(len_1000))
            .build();
        let proper_fwd = Flags::SEGMENTED | Flags::PROPERLY_SEGMENTED | Flags::FIRST_SEGMENT;
        let proper_rev = Flags::SEGMENTED
            | Flags::PROPERLY_SEGMENTED
            | Flags::LAST_SEGMENT
            | Flags::REVERSE_COMPLEMENTED;

        let m9 = [(CigarKind::Match, 9)];
        let m4 = [(CigarKind::Match, 4)];
        let sc_m = [(CigarKind::SoftClip, 2), (CigarKind::Match, 6)];
        let records = vec![
            make_record(proper_fwd, Some(10), 60, 140, &m9, b"CCGGATNCA"),
            make_record(proper_rev, Some(141), 60, -140, &m9, b"TGNATCCGG"),
            make_record(proper_fwd, Some(220), 30, 185, &sc_m, b"AACCGGTT"),
            make_record(
                Flags::empty(),
                Some(410),
                40,
                0,
                &[(CigarKind::Match, 7)],
                b"ACGTACG",
            ),
            make_record(
                Flags::REVERSE_COMPLEMENTED,
                Some(520),
                40,
                0,
                &[(CigarKind::Match, 8)],
                b"TTGCACGT",
            ),
            make_record(Flags::UNMAPPED, None, 0, 0, &[], b"ACGT"),
            make_record(Flags::SECONDARY, Some(50), 60, 0, &m4, b"ACGT"),
            make_record(Flags::DUPLICATE, Some(60), 60, 0, &m4, b"ACGT"),
            make_record(Flags::empty(), Some(70), 5, 0, &m4, b"ACGT"),
        ];

        let mut bam_bytes = Vec::new();
        {
            let mut writer = bam::io::Writer::new(&mut bam_bytes);
            writer.write_header(&header).unwrap();
            for rec in &records {
                writer.write_alignment_record(&header, rec).unwrap();
            }
            writer.try_finish().unwrap();
        }

        let target = TargetContig {
            header_ref_id: 0,
            length: 1000,
            bin_offset: 0,
            role: ContigRole::Autosomal,
        };
        let filters = FilterSpec {
            min_mapq: 10,
            require_proper_pair: true,
            exclude_duplicates: true,
            exclude_secondary: true,
            exclude_supplementary: true,
            exclude_unmapped: true,
            collect_bands: true,
            se_fragment_length: 120,
        };

        let mut acc_bam = TileAccumulator::new(0, 0, 10, true);
        let mut reader = bam::io::Reader::new(bam_bytes.as_slice());
        reader.read_header().unwrap();
        let mut bam_rec = bam::Record::default();
        while reader.read_record(&mut bam_rec).unwrap() != 0 {
            process_record(
                &bam_rec,
                &target,
                100,
                filters,
                Some((0, 600)),
                &mut acc_bam,
            )
            .unwrap();
        }

        let mut acc_ascii = TileAccumulator::new(0, 0, 10, true);
        for rec in &records {
            process_record(rec, &target, 100, filters, Some((0, 600)), &mut acc_ascii).unwrap();
        }

        let out_bam = reduce_tile_accumulators(vec![acc_bam], 10, true);
        let out_ascii = reduce_tile_accumulators(vec![acc_ascii], 10, true);

        assert_eq!(out_bam.count, out_ascii.count);
        assert_eq!(out_bam.paired_count, out_ascii.paired_count);
        assert_eq!(out_bam.frag_len_sum, out_ascii.frag_len_sum);
        assert_eq!(out_bam.short_frag_count, out_ascii.short_frag_count);
        assert_eq!(out_bam.endpoint_count, out_ascii.endpoint_count);
        assert_eq!(out_bam.frag_gc_sum, out_ascii.frag_gc_sum);
        assert_eq!(out_bam.frag_bands, out_ascii.frag_bands);
        assert_eq!(out_bam.hist_autosomal, out_ascii.hist_autosomal);
        assert_eq!(out_bam.motif_counts, out_ascii.motif_counts);
        assert_eq!(out_bam.paired_end, out_ascii.paired_end);
        assert_eq!(
            (
                out_bam.qc.reads_seen,
                out_bam.qc.reads_kept,
                out_bam.qc.mate_dedup_skipped,
                out_bam.qc.reads_filtered
            ),
            (9, 4, 1, 4)
        );
        assert_eq!(
            (
                out_bam.qc.reads_kept,
                out_bam.qc.reads_filtered,
                out_bam.qc.mate_dedup_skipped
            ),
            (
                out_ascii.qc.reads_kept,
                out_ascii.qc.reads_filtered,
                out_ascii.qc.mate_dedup_skipped
            )
        );
    }

    #[test]
    fn improper_pair_motif_gating_is_symmetric_and_saturates_large_tlen() {
        let target = TargetContig {
            header_ref_id: 0,
            length: 1000,
            bin_offset: 0,
            role: ContigRole::Autosomal,
        };
        let improper_fwd = Flags::SEGMENTED | Flags::FIRST_SEGMENT;
        let improper_rev = Flags::SEGMENTED | Flags::LAST_SEGMENT | Flags::REVERSE_COMPLEMENTED;
        let m8 = [(CigarKind::Match, 8)];
        let r_fwd = make_record(improper_fwd, Some(100), 60, i32::MAX, &m8, b"CCCCAAAA");
        let r_rev = make_record(improper_rev, Some(200), 60, -140, &m8, b"TTTTGGGG");

        let filters = FilterSpec {
            min_mapq: 1,
            require_proper_pair: false,
            exclude_duplicates: true,
            exclude_secondary: true,
            exclude_supplementary: true,
            exclude_unmapped: true,
            collect_bands: false,
            se_fragment_length: 0,
        };
        let mut acc = TileAccumulator::new(0, 0, 10, false);
        process_record(&r_fwd, &target, 100, filters, None, &mut acc).unwrap();
        process_record(&r_rev, &target, 100, filters, None, &mut acc).unwrap();
        let out = reduce_tile_accumulators(vec![acc], 10, false);

        // Both CCCC (0b01_01_01_01 = 85) on forward and revcomp(GGGG) = CCCC (85) on reverse
        // must be counted symmetrically when require_proper_pair is false.
        assert_eq!(out.motif_counts[85], 2);
        // Large i32::MAX tlen clamps midpoint and downstream endpoint to last bin without overflow.
        assert_eq!(out.count[9], 1);
        assert_eq!(out.endpoint_count[0], 1);
        assert_eq!(out.endpoint_count[9], 1);
    }
}
