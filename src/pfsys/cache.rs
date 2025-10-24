use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

use halo2_proofs::plonk::ProvingKey;
use halo2curves::bn256::G1Affine;
use once_cell::sync::OnceCell;

use super::PfsysError;

static KZG_PK_CACHE: OnceCell<RwLock<HashMap<String, Arc<ProvingKey<G1Affine>>>>> = OnceCell::new();

fn cache() -> &'static RwLock<HashMap<String, Arc<ProvingKey<G1Affine>>>> {
    KZG_PK_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn key(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

/// Returns true when the proving-key cache should be used.
/// Disable by setting `EZKL_DISABLE_PK_CACHE=1` (or `true`).
pub fn cache_enabled() -> bool {
    std::env::var("EZKL_DISABLE_PK_CACHE")
        .map(|v| {
            let v = v.trim().to_lowercase();
            !(v == "1" || v == "true" || v == "yes")
        })
        .unwrap_or(true)
}

/// Fetches a cached KZG proving key if present.
pub fn get_kzg_pk(path: &Path) -> Option<Arc<ProvingKey<G1Affine>>> {
    let guard = cache().read().ok()?;
    guard.get(&key(path)).cloned()
}

/// Stores a proving key in the in-memory cache.
pub fn store_kzg_pk(path: &Path, pk: Arc<ProvingKey<G1Affine>>) {
    if !cache_enabled() {
        return;
    }
    if let Ok(mut guard) = cache().write() {
        guard.insert(key(path), pk);
    }
}

/// Returns a cached proving key when available, otherwise loads it using `loader`
/// and caches the result (unless caching is disabled).
pub fn get_or_try_insert_kzg<F>(
    path: &Path,
    loader: F,
) -> Result<Arc<ProvingKey<G1Affine>>, PfsysError>
where
    F: FnOnce() -> Result<ProvingKey<G1Affine>, PfsysError>,
{
    if cache_enabled() {
        if let Some(pk) = get_kzg_pk(path) {
            return Ok(pk);
        }
    }

    let pk = loader()?;
    let pk = Arc::new(pk);
    store_kzg_pk(path, pk.clone());
    Ok(pk)
}
