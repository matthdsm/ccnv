# Extract / Reference Functional Fixtures

[`tests/published_fixtures.py`](../../published_fixtures.py) downloads these published `chr21` CRAM and FASTA fixtures on first run (skipping tests if offline) for [`test_cohort.py`](../../test_cohort.py), [`test_extract.py`](../../test_extract.py), [`test_predict.py`](../../test_predict.py), [`test_predict_bam.py`](../../test_predict_bam.py), and [`test_reference.py`](../../test_reference.py), caching extracted `.ccnv` sample tables under `.cache_extracted/`. Downloaded fixtures and cached `.ccnv` files are gitignored.

- CRAM: https://raw.githubusercontent.com/nf-cmgg/test-datasets/wisecondorx/data/genomics/homo_sapiens/illumina/cram/test.cram
- CRAM: https://raw.githubusercontent.com/nf-cmgg/test-datasets/wisecondorx/data/genomics/homo_sapiens/illumina/cram/test2.cram
- FASTA: https://raw.githubusercontent.com/nf-cmgg/test-datasets/wisecondorx/data/genomics/homo_sapiens/genome/seq/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set_chr21.fna
