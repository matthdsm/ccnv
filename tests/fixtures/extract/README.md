# Testdata Fixtures

[`tests/published_fixtures.py`](../../published_fixtures.py) resolves minimal deterministic fixtures from [`matthdsm/ccnv-testdata`](https://github.com/matthdsm/ccnv-testdata) for [`test_extract.py`](../../test_extract.py), [`test_features_cohort.py`](../../test_features_cohort.py), [`test_predict.py`](../../test_predict.py), [`test_pipeline.py`](../../test_pipeline.py), and [`test_testdata_regression.py`](../../test_testdata_regression.py).

- **Local checkout (preferred)**: Reads directly from `CCNV_TESTDATA_DIR` or a sibling `../copynumber-cnv-testdata` / `../ccnv-testdata` checkout without network access.
- **Remote fallback**: Downloads requested files from `https://raw.githubusercontent.com/matthdsm/ccnv-testdata/main/` into `tests/fixtures/extract/testdata/` (skipping tests if offline), caching extracted sample `.ccnv` files under `.cache_extracted/`. Downloaded fixtures and caches are gitignored.
