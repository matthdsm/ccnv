use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use noodles::core::Region;
use noodles::fasta;

pub(super) type HandoffRecord = Arc<Mutex<Option<(Vec<u8>, fasta::Record)>>>;

#[derive(Default)]
pub(super) struct ContigSlot {
    repo: Mutex<Option<fasta::Repository>>,
    remaining_tiles: AtomicUsize,
}

pub(super) struct BoundedIndexedFasta {
    fasta_path: PathBuf,
    fai_index: fasta::fai::Index,
    slots: HashMap<Vec<u8>, ContigSlot>,
    disk_loads: AtomicUsize,
}

impl BoundedIndexedFasta {
    pub(super) fn open(fasta_path: &Path, fai_path: &Path) -> Result<Arc<Self>, String> {
        let fai_index = fasta::fai::fs::read(fai_path)
            .map_err(|e| format!("failed to read FASTA index {}: {e}", fai_path.display()))?;
        let mut slots = HashMap::with_capacity(fai_index.as_ref().len());
        for rec in fai_index.as_ref() {
            slots.entry(rec.name().to_vec()).or_default();
        }
        Ok(Arc::new(Self {
            fasta_path: fasta_path.to_path_buf(),
            fai_index,
            slots,
            disk_loads: AtomicUsize::new(0),
        }))
    }

    pub(super) fn prepare_contig_tile_counts(&self, counts_by_name: &HashMap<&[u8], usize>) {
        for (name, slot) in &self.slots {
            let n = counts_by_name.get(name.as_slice()).copied().unwrap_or(0);
            slot.remaining_tiles.store(n, Ordering::Release);
            if n == 0
                && let Ok(mut guard) = slot.repo.lock()
                && let Some(evicted) = guard.take()
            {
                evicted.clear();
            }
        }
    }

    pub(super) fn acquire_contig_tile(
        self: &Arc<Self>,
        contig_name: &[u8],
    ) -> Result<TileRepoGuard<'_>, String> {
        let make_repo = || {
            fasta::Repository::new(IndexedFastaAdapter {
                shared: Arc::clone(self),
                handoff: Arc::new(Mutex::new(None)),
            })
        };
        let Some(slot) = self.slots.get(contig_name) else {
            return Ok(TileRepoGuard {
                repo: make_repo(),
                slot: None,
            });
        };
        let mut guard = slot
            .repo
            .lock()
            .map_err(|_| "poisoned FASTA contig cache lock".to_owned())?;
        let repo = if let Some(existing) = guard.as_ref() {
            existing.clone()
        } else {
            let fresh = make_repo();
            if let Some(Err(e)) = fresh.get(contig_name) {
                return Err(format!(
                    "failed to read contig {} from FASTA {}: {e}",
                    String::from_utf8_lossy(contig_name),
                    self.fasta_path.display()
                ));
            }
            *guard = Some(fresh.clone());
            fresh
        };
        Ok(TileRepoGuard {
            repo,
            slot: Some(slot),
        })
    }

    #[cfg(test)]
    fn disk_loads(&self) -> usize {
        self.disk_loads.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn cached_contig_count(&self) -> usize {
        self.slots
            .values()
            .filter(|s| s.repo.lock().unwrap().is_some())
            .count()
    }
}

pub(super) struct IndexedFastaAdapter {
    pub(super) shared: Arc<BoundedIndexedFasta>,
    pub(super) handoff: HandoffRecord,
}

impl fasta::repository::Adapter for IndexedFastaAdapter {
    fn get(&mut self, name: &[u8]) -> Option<io::Result<fasta::Record>> {
        if let Ok(mut guard) = self.handoff.lock()
            && guard.as_ref().is_some_and(|(n, _)| n.as_slice() == name)
        {
            return guard.take().map(|(_, rec)| Ok(rec));
        }
        if !self.shared.slots.contains_key(name) {
            return None;
        }
        self.shared.disk_loads.fetch_add(1, Ordering::Relaxed);
        let res = fasta::io::indexed_reader::Builder::default()
            .set_index(self.shared.fai_index.clone())
            .build_from_path(&self.shared.fasta_path)
            .and_then(|mut reader| reader.query(&Region::new(name, ..)));
        Some(res)
    }
}

pub(super) struct TileRepoGuard<'a> {
    pub(super) repo: fasta::Repository,
    slot: Option<&'a ContigSlot>,
}

impl Drop for TileRepoGuard<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot
            && slot.remaining_tiles.fetch_sub(1, Ordering::AcqRel) == 1
            && let Ok(mut guard) = slot.repo.lock()
            && let Some(evicted) = guard.take()
        {
            evicted.clear();
        }
    }
}

pub(super) fn evict_other_contigs_in_seq_repo(
    seq_repo: &fasta::Repository,
    handoff: &HandoffRecord,
    keep_name: &str,
) {
    let keep_bytes = keep_name.as_bytes();
    let kept_seq = seq_repo.get(keep_bytes).and_then(Result::ok);
    seq_repo.clear();
    if let Some(arc_seq) = kept_seq {
        let seq = Arc::try_unwrap(arc_seq).unwrap_or_else(|shared| (*shared).clone());
        let def = fasta::record::Definition::new(keep_name, None);
        if let Ok(mut guard) = handoff.lock() {
            *guard = Some((keep_bytes.to_vec(), fasta::Record::new(def, seq)));
        }
        let _ = seq_repo.get(keep_bytes);
    }
}

pub(super) enum FastaSource {
    InMemory(fasta::Repository),
    Indexed(Arc<BoundedIndexedFasta>),
}

impl FastaSource {
    pub(super) fn open(fasta_path: &Path) -> Result<Self, String> {
        let fai_path = PathBuf::from(format!("{}.fai", fasta_path.display()));
        if fai_path.is_file() {
            return BoundedIndexedFasta::open(fasta_path, &fai_path).map(Self::Indexed);
        }
        let file = File::open(fasta_path)
            .map_err(|e| format!("failed to open FASTA {}: {e}", fasta_path.display()))?;
        let records: Vec<fasta::Record> = fasta::io::Reader::new(BufReader::new(file))
            .records()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("failed to read FASTA {}: {e}", fasta_path.display()))?;
        Ok(Self::InMemory(fasta::Repository::new(records)))
    }

    pub(super) fn open_sequential(&self) -> (fasta::Repository, Option<HandoffRecord>) {
        match self {
            Self::InMemory(repo) => (repo.clone(), None),
            Self::Indexed(shared) => {
                let handoff: HandoffRecord = Arc::new(Mutex::new(None));
                let adapter = IndexedFastaAdapter {
                    shared: Arc::clone(shared),
                    handoff: Arc::clone(&handoff),
                };
                (fasta::Repository::new(adapter), Some(handoff))
            }
        }
    }

    pub(super) fn prepare_contig_tile_counts(&self, counts_by_name: &HashMap<&[u8], usize>) {
        if let Self::Indexed(shared) = self {
            shared.prepare_contig_tile_counts(counts_by_name);
        }
    }

    pub(super) fn acquire_contig_tile(
        &self,
        contig_name: &[u8],
    ) -> Result<TileRepoGuard<'_>, String> {
        match self {
            Self::InMemory(repo) => Ok(TileRepoGuard {
                repo: repo.clone(),
                slot: None,
            }),
            Self::Indexed(shared) => shared.acquire_contig_tile(contig_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Barrier, Mutex};

    use noodles::fasta;

    use super::{BoundedIndexedFasta, IndexedFastaAdapter, evict_other_contigs_in_seq_repo};

    #[test]
    fn bounded_indexed_fasta_prevents_stampede_and_evicts_finished_contigs() {
        let dir = std::env::temp_dir();
        let fa = dir.join(format!("ccnv_bounded_{}.fa", std::process::id()));
        let fai = dir.join(format!("ccnv_bounded_{}.fa.fai", std::process::id()));
        std::fs::write(&fa, b">chr1\nACGTACGT\n>chr2\nGGGGCCCC\n>chr3\nTTTTAAAA\n").unwrap();
        std::fs::write(
            &fai,
            b"chr1\t8\t6\t8\t9\nchr2\t8\t21\t8\t9\nchr3\t8\t36\t8\t9\n",
        )
        .unwrap();

        let shared = BoundedIndexedFasta::open(&fa, &fai).unwrap();
        shared.prepare_contig_tile_counts(&HashMap::from([
            (b"chr1".as_slice(), 4),
            (b"chr2".as_slice(), 2),
        ]));

        let barrier = Barrier::new(4);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    barrier.wait();
                    let g = shared.acquire_contig_tile(b"chr1").unwrap();
                    let seq = g.repo.get(b"chr1").unwrap().unwrap();
                    assert_eq!(seq.as_ref().as_ref(), b"ACGTACGT");
                });
            }
        });
        assert_eq!((shared.disk_loads(), shared.cached_contig_count()), (1, 0));

        let g1 = shared.acquire_contig_tile(b"chr2").unwrap();
        let g2 = shared.acquire_contig_tile(b"chr2").unwrap();
        assert_eq!((shared.disk_loads(), shared.cached_contig_count()), (2, 1));
        drop(g1);
        assert_eq!(shared.cached_contig_count(), 1);
        drop(g2);
        assert_eq!(shared.cached_contig_count(), 0);

        let handoff = Arc::new(Mutex::new(None));
        let seq_repo = fasta::Repository::new(IndexedFastaAdapter {
            shared: Arc::clone(&shared),
            handoff: Arc::clone(&handoff),
        });
        let _ = seq_repo.get(b"chr1").unwrap().unwrap();
        let _ = seq_repo.get(b"chr2").unwrap().unwrap();
        assert_eq!(seq_repo.len(), 2);
        evict_other_contigs_in_seq_repo(&seq_repo, &handoff, "chr2");
        assert_eq!(seq_repo.len(), 1);
        let _ = seq_repo.get(b"chr2").unwrap().unwrap();
        assert_eq!(shared.disk_loads(), 4);

        let _ = std::fs::remove_file(&fa);
        let _ = std::fs::remove_file(&fai);
    }
}
