# Validation

## Conversion concordance and storage comparison against WisecondorX (v1.3.2)

Validation of the `convert` stage was conducted on a real-world NIPT sample binned at 5 kb across 24 chromosomes (1–22, X, Y), comprising 617,665 total bins and 66,629,087 raw reads.

### Read count and bin-level concordance

- **Total reads examined:** Exactly identical between both tools (66,629,087 reads).
- **Total reads kept:** 56,781,413 reads in `ccnv` versus 56,649,323 reads in WisecondorX (+132,090 reads, a 0.23% delta).
- **Correlation across all 617,665 bins:** Pearson $r = 0.9779$ ($R^2 = 0.9562$); Spearman rank correlation $\rho = 0.9995$.
- **Error distribution:** Median absolute difference is 0.0 reads/bin (mean absolute error: 0.522 reads/bin).
- **Bin agreement thresholds:**
  - Exact match: 66.69% of bins (411,926 / 617,665).
  - Within $\pm 1$ read: 91.70% of bins (566,413 / 617,665).
  - Within $\pm 2$ reads: 98.66% of bins (609,418 / 617,665).
  - Within $\pm 5$ reads: 99.96% of bins (617,438 / 617,665).

### Root cause of count discrepancies

Only 56 of 617,665 bins (< 0.01%) exhibit an absolute difference exceeding 50 reads. All high-difference bins locate to pericentromeric heterochromatin and satellite repeat regions (e.g., centromeres of chromosomes 1, 3, 16, 17, and 20).

1. **Duplicate filtering mechanism:** `ccnv` respects the standard SAM specification duplicate bit (`read.is_duplicate` / flag 0x400) populated by upstream duplicate markers. WisecondorX employs an ad-hoc single-read streaming check (`larp == read_start and larp2 == next_reference_start`). Because reads in coordinate-sorted alignments can interleave across strands and pairs, WisecondorX fails to identify duplicate pairs that are not strictly adjacent in the stream. In repeat-dense regions, thousands of duplicate reads bypass the WisecondorX filter (for example, missing 16,565 duplicate reads in a single chromosome 3 centromeric bin). Conversely, WisecondorX can mistakenly discard primary non-duplicate reads that stream immediately after an identically positioned duplicate.
2. **Filter precedence:** `ccnv` filters duplicates before validating proper pair status (`filter_duplicate`: 6,546,411; `filter_improper_pair`: 302,029). WisecondorX evaluates proper pair status prior to duplicate checking (`pair_fail`: 409,495; `rmdup`: 6,440,137), causing improper-pair duplicates to be classified as pair failures rather than PCR duplicates.
3. **Unmapped and secondary alignments:** `ccnv` explicitly drops unmapped reads (`filter_unmapped`: 64,265), secondary reads, and supplementary alignments. WisecondorX streams alignment records without inspecting unmapped or secondary flags directly, relying solely on MAPQ < 1.