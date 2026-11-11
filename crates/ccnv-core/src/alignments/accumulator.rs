use crate::normalize_chr_name;

pub(super) const NUM_FRAG_BANDS: usize = 24;
pub(super) const HIST_LEN: usize = 571;
pub(super) const FRAG_MIN: i32 = 30;
pub(super) const FRAG_MAX: i32 = 600;
const TILE_TARGET_BP: usize = 50_000_000;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ContigRole {
    Autosomal,
    ChrX,
    ChrY,
    Other,
}

#[derive(Clone)]
pub(super) struct TargetContig {
    pub(super) header_ref_id: usize,
    pub(super) length: usize,
    pub(super) bin_offset: usize,
    pub(super) role: ContigRole,
}

#[derive(Clone, Copy)]
pub(super) struct FilterSpec {
    pub(super) min_mapq: u8,
    pub(super) require_proper_pair: bool,
    pub(super) exclude_duplicates: bool,
    pub(super) exclude_secondary: bool,
    pub(super) exclude_supplementary: bool,
    pub(super) exclude_unmapped: bool,
    pub(super) collect_bands: bool,
    pub(super) se_fragment_length: usize,
}

#[derive(Default, Clone)]
pub(super) struct QcCounters {
    pub(super) reads_seen: u64,
    pub(super) reads_kept: u64,
    pub(super) mate_dedup_skipped: u64,
    pub(super) reads_filtered: u64,
    pub(super) autosomal_reads_kept: u64,
    pub(super) sex_reads_kept: u64,
    pub(super) filter_unmapped: u64,
    pub(super) filter_no_start: u64,
    pub(super) filter_secondary: u64,
    pub(super) filter_supplementary: u64,
    pub(super) filter_duplicate: u64,
    pub(super) filter_improper_pair: u64,
    pub(super) filter_mapq: u64,
}

impl QcCounters {
    pub(super) fn merge(&mut self, other: &Self) {
        self.reads_seen += other.reads_seen;
        self.reads_kept += other.reads_kept;
        self.mate_dedup_skipped += other.mate_dedup_skipped;
        self.reads_filtered += other.reads_filtered;
        self.autosomal_reads_kept += other.autosomal_reads_kept;
        self.sex_reads_kept += other.sex_reads_kept;
        self.filter_unmapped += other.filter_unmapped;
        self.filter_no_start += other.filter_no_start;
        self.filter_secondary += other.filter_secondary;
        self.filter_supplementary += other.filter_supplementary;
        self.filter_duplicate += other.filter_duplicate;
        self.filter_improper_pair += other.filter_improper_pair;
        self.filter_mapq += other.filter_mapq;
    }
}

pub(super) struct MidpointOverflow {
    pub(super) global_bin: usize,
    pub(super) paired_in_range: bool,
    pub(super) tlen: u64,
    pub(super) is_short: bool,
    pub(super) band_idx: u8,
    pub(super) gc_frac: f64,
}

pub(super) struct TileSpec {
    pub(super) tile_idx: usize,
    pub(super) target_idx: usize,
    pub(super) tile_start: usize,
    pub(super) tile_end: usize,
    pub(super) local_bin_start: usize,
    pub(super) local_n_bins: usize,
}

pub(super) struct TileAccumulator {
    pub(super) tile_idx: usize,
    pub(super) global_bin_start: usize,
    pub(super) count: Vec<u32>,
    pub(super) paired_count: Vec<u32>,
    pub(super) frag_len_sum: Vec<u64>,
    pub(super) short_frag_count: Vec<u32>,
    pub(super) endpoint_count: Vec<u32>,
    pub(super) frag_gc_sum: Vec<f64>,
    pub(super) frag_bands: Vec<u32>,
    pub(super) midpoint_overflows: Vec<MidpointOverflow>,
    pub(super) endpoint_overflows: Vec<usize>,
    pub(super) hist_autosomal: Box<[u64; HIST_LEN]>,
    pub(super) hist_chrx: Box<[u64; HIST_LEN]>,
    pub(super) hist_chry: Box<[u64; HIST_LEN]>,
    pub(super) motif_counts: Box<[u64; 256]>,
    pub(super) paired_end: bool,
    pub(super) qc: QcCounters,
}

impl TileAccumulator {
    pub(super) fn new(
        tile_idx: usize,
        global_bin_start: usize,
        n_bins: usize,
        collect_bands: bool,
    ) -> Self {
        Self {
            tile_idx,
            global_bin_start,
            count: vec![0u32; n_bins],
            paired_count: vec![0u32; n_bins],
            frag_len_sum: vec![0u64; n_bins],
            short_frag_count: vec![0u32; n_bins],
            endpoint_count: vec![0u32; n_bins],
            frag_gc_sum: vec![0.0f64; n_bins],
            frag_bands: if collect_bands {
                vec![0u32; n_bins * NUM_FRAG_BANDS]
            } else {
                Vec::new()
            },
            midpoint_overflows: Vec::new(),
            endpoint_overflows: Vec::new(),
            hist_autosomal: Box::new([0u64; HIST_LEN]),
            hist_chrx: Box::new([0u64; HIST_LEN]),
            hist_chry: Box::new([0u64; HIST_LEN]),
            motif_counts: Box::new([0u64; 256]),
            paired_end: false,
            qc: QcCounters::default(),
        }
    }

    #[inline]
    pub(super) fn add_endpoint(&mut self, global_bin: usize) {
        let end_bin = self.global_bin_start + self.endpoint_count.len();
        if global_bin >= self.global_bin_start && global_bin < end_bin {
            self.endpoint_count[global_bin - self.global_bin_start] += 1;
        } else {
            self.endpoint_overflows.push(global_bin);
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn add_midpoint(
        &mut self,
        global_bin: usize,
        paired_in_range: bool,
        tlen: u64,
        is_short: bool,
        band_idx: u8,
        gc_frac: f64,
        collect_bands: bool,
    ) {
        let end_bin = self.global_bin_start + self.count.len();
        if global_bin >= self.global_bin_start && global_bin < end_bin {
            let local = global_bin - self.global_bin_start;
            self.count[local] += 1;
            self.frag_gc_sum[local] += gc_frac;
            if paired_in_range {
                self.paired_count[local] += 1;
                self.frag_len_sum[local] += tlen;
                if is_short {
                    self.short_frag_count[local] += 1;
                }
                if collect_bands {
                    self.frag_bands[local * NUM_FRAG_BANDS + usize::from(band_idx)] += 1;
                }
            }
        } else {
            self.midpoint_overflows.push(MidpointOverflow {
                global_bin,
                paired_in_range,
                tlen,
                is_short,
                band_idx,
                gc_frac,
            });
        }
    }
}

pub(super) struct GlobalScanOutput {
    pub(super) count: Vec<u32>,
    pub(super) paired_count: Vec<u32>,
    pub(super) frag_len_sum: Vec<u64>,
    pub(super) short_frag_count: Vec<u32>,
    pub(super) endpoint_count: Vec<u32>,
    pub(super) frag_gc_sum: Vec<f32>,
    pub(super) frag_bands: Option<Vec<u32>>,
    pub(super) hist_autosomal: Box<[u64; HIST_LEN]>,
    pub(super) hist_chrx: Box<[u64; HIST_LEN]>,
    pub(super) hist_chry: Box<[u64; HIST_LEN]>,
    pub(super) motif_counts: Box<[u64; 256]>,
    pub(super) paired_end: bool,
    pub(super) qc: QcCounters,
}

pub(super) fn classify_contig_role(name: &str) -> ContigRole {
    let norm = normalize_chr_name(name);
    if norm == "X" {
        ContigRole::ChrX
    } else if norm == "Y" {
        ContigRole::ChrY
    } else if !norm.is_empty()
        && norm.bytes().all(|b| b.is_ascii_digit())
        && norm.parse::<u64>().is_ok_and(|n| n >= 1)
    {
        ContigRole::Autosomal
    } else {
        ContigRole::Other
    }
}

pub(super) fn build_lpt_tiles(targets: &[TargetContig], bin_size: usize) -> Vec<TileSpec> {
    let bins_per_tile = TILE_TARGET_BP.div_ceil(bin_size).max(1);
    let tile_bp = bins_per_tile * bin_size;
    let mut tiles = Vec::new();
    let mut tile_idx = 0usize;
    for (target_idx, target) in targets.iter().enumerate() {
        let mut t_start = 0usize;
        let mut local_bin = 0usize;
        while t_start < target.length {
            let t_end = (t_start + tile_bp).min(target.length);
            let n_bins = (t_end - t_start).div_ceil(bin_size);
            tiles.push(TileSpec {
                tile_idx,
                target_idx,
                tile_start: t_start,
                tile_end: t_end,
                local_bin_start: target.bin_offset + local_bin,
                local_n_bins: n_bins,
            });
            tile_idx += 1;
            t_start = t_end;
            local_bin += n_bins;
        }
    }
    tiles.sort_by(|a, b| {
        let len_a = a.tile_end - a.tile_start;
        let len_b = b.tile_end - b.tile_start;
        len_b.cmp(&len_a).then_with(|| a.tile_idx.cmp(&b.tile_idx))
    });
    tiles
}

pub(super) fn reduce_tile_accumulators(
    mut tiles: Vec<TileAccumulator>,
    total_bins: usize,
    collect_bands: bool,
) -> GlobalScanOutput {
    tiles.sort_by_key(|t| t.tile_idx);

    let mut count = vec![0u32; total_bins];
    let mut paired_count = vec![0u32; total_bins];
    let mut frag_len_sum = vec![0u64; total_bins];
    let mut short_frag_count = vec![0u32; total_bins];
    let mut endpoint_count = vec![0u32; total_bins];
    let mut frag_gc_sum_f64 = vec![0.0f64; total_bins];
    let mut frag_bands = if collect_bands {
        vec![0u32; total_bins * NUM_FRAG_BANDS]
    } else {
        Vec::new()
    };
    let mut hist_autosomal = Box::new([0u64; HIST_LEN]);
    let mut hist_chrx = Box::new([0u64; HIST_LEN]);
    let mut hist_chry = Box::new([0u64; HIST_LEN]);
    let mut motif_counts = Box::new([0u64; 256]);
    let mut paired_end = false;
    let mut qc = QcCounters::default();

    for t in tiles {
        let b0 = t.global_bin_start;
        let n = t.count.len();
        for i in 0..n {
            count[b0 + i] += t.count[i];
            paired_count[b0 + i] += t.paired_count[i];
            frag_len_sum[b0 + i] += t.frag_len_sum[i];
            short_frag_count[b0 + i] += t.short_frag_count[i];
            endpoint_count[b0 + i] += t.endpoint_count[i];
            frag_gc_sum_f64[b0 + i] += t.frag_gc_sum[i];
        }
        if collect_bands && n > 0 {
            let start_flat = b0 * NUM_FRAG_BANDS;
            for (dst, &src) in frag_bands[start_flat..start_flat + n * NUM_FRAG_BANDS]
                .iter_mut()
                .zip(&t.frag_bands)
            {
                *dst += src;
            }
        }
        for ov in t.midpoint_overflows {
            count[ov.global_bin] += 1;
            frag_gc_sum_f64[ov.global_bin] += ov.gc_frac;
            if ov.paired_in_range {
                paired_count[ov.global_bin] += 1;
                frag_len_sum[ov.global_bin] += ov.tlen;
                if ov.is_short {
                    short_frag_count[ov.global_bin] += 1;
                }
                if collect_bands {
                    frag_bands[ov.global_bin * NUM_FRAG_BANDS + usize::from(ov.band_idx)] += 1;
                }
            }
        }
        for ep_bin in t.endpoint_overflows {
            endpoint_count[ep_bin] += 1;
        }
        for i in 0..HIST_LEN {
            hist_autosomal[i] += t.hist_autosomal[i];
            hist_chrx[i] += t.hist_chrx[i];
            hist_chry[i] += t.hist_chry[i];
        }
        for i in 0..256 {
            motif_counts[i] += t.motif_counts[i];
        }
        paired_end |= t.paired_end;
        qc.merge(&t.qc);
    }

    let frag_gc_sum = frag_gc_sum_f64.into_iter().map(|v| v as f32).collect();
    GlobalScanOutput {
        count,
        paired_count,
        frag_len_sum,
        short_frag_count,
        endpoint_count,
        frag_gc_sum,
        frag_bands: if collect_bands {
            Some(frag_bands)
        } else {
            None
        },
        hist_autosomal,
        hist_chrx,
        hist_chry,
        motif_counts,
        paired_end,
        qc,
    }
}
