use std::{
    collections::{HashMap, HashSet},
    env, fmt, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context, Result, anyhow, ensure};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use ezkl::{
    Commitments,
    circuit::{CheckMode, region::RegionSettings},
    graph::{
        GraphCircuit, GraphSettings,
        input::{DataSource, GraphData},
    },
    pfsys::{
        self, PrettyElements, ProofSplitCommit, ProofType, StrategyType, TranscriptType, cache,
        evm::aggregation_kzg::PoseidonTranscript, field_to_string, srs::load_srs_prover,
    },
};
use halo2_proofs::{
    plonk::{Circuit, ProvingKey},
    poly::commitment::Params as CommitmentParams,
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::{
            AccumulatorStrategy as KZGAccumulatorStrategy, SingleStrategy as KZGSingleStrategy,
        },
    },
};
use halo2curves::bn256::{Bn256, G1Affine};
use instant::Instant;
use log::{debug, info, warn};

#[cfg(all(feature = "icicle", not(target_arch = "wasm32")))]
use halo2_proofs::icicle::try_load_and_set_backend_device;
use serde::{
    Deserialize, Serialize,
    de::{Deserializer, Error as DeError},
};
use serde_json::Value;
use snark_verifier::{
    loader::native::NativeLoader,
    system::halo2::transcript::evm::EvmTranscript,
    system::halo2::{Config, compile},
};
use tokio::{net::TcpListener, sync::Semaphore};

#[cfg(feature = "nvtx")]
struct NvtxGuard {
    _guard: nvtx::RangeGuard,
}

#[cfg(feature = "nvtx")]
fn nvtx_guard(message: impl fmt::Display) -> NvtxGuard {
    NvtxGuard {
        _guard: nvtx::range!("{}", message),
    }
}

#[cfg(not(feature = "nvtx"))]
struct NvtxGuard;

#[cfg(not(feature = "nvtx"))]
fn nvtx_guard(_message: impl fmt::Display) -> NvtxGuard {
    NvtxGuard
}

#[derive(Parser, Debug)]
struct Args {
    /// Root directory containing model_* folders with compiled circuits/settings
    #[arg(long)]
    model_root: Option<PathBuf>,
    /// Optional directory containing proving artifacts (pk/srs/vk) organized by model_* folders
    #[arg(long)]
    artifact_root: Option<PathBuf>,
    /// Path to compiled circuit (model.compiled) for single-model mode
    #[arg(long)]
    compiled_circuit: Option<PathBuf>,
    /// Path to circuit settings (settings.json) for single-model mode
    #[arg(long)]
    settings: Option<PathBuf>,
    /// Path to model metadata (metadata.json) for single-model mode
    #[arg(long)]
    metadata: Option<PathBuf>,
    /// Path to proving key (pk.key) for single-model mode
    #[arg(long)]
    pk: Option<PathBuf>,
    /// Path to structured reference string (kzg.srs) for single-model mode
    #[arg(long)]
    srs: Option<PathBuf>,
    /// Optional path to verifying key (vk.key) for single-model mode
    #[arg(long)]
    vk: Option<PathBuf>,
    /// Explicit model identifier for single-model mode
    #[arg(long)]
    model_id: Option<String>,
    /// Proof type to generate: single | for-aggr
    #[arg(long, default_value = "single")]
    proof_type: String,
    /// Check mode: safe | unsafe
    #[arg(long, default_value = "unsafe")]
    check_mode: String,
    /// Address to listen on (e.g. 127.0.0.1:8000)
    #[arg(long, default_value = "127.0.0.1:8000")]
    listen: SocketAddr,
    /// Maximum number of concurrent proofs
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

#[derive(Clone)]
struct AppState {
    models: Arc<HashMap<String, Arc<ModelState>>>,
    proof_type: ProofType,
    check_mode: CheckMode,
    semaphore: Arc<Semaphore>,
}

struct ModelState {
    model_id: String,
    compiled_bytes: Arc<Vec<u8>>,
    settings: GraphSettings,
    #[allow(dead_code)]
    metadata: Option<Value>,
    pk: Arc<ProvingKey<G1Affine>>,
    params: Arc<ParamsKZG<Bn256>>,
    vk: Option<Arc<halo2_proofs::plonk::VerifyingKey<G1Affine>>>,
    commitment: Commitments,
}

fn load_kzg_params(srs_path: &Path, logrows: u32) -> Result<ParamsKZG<Bn256>> {
    let mut params = load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path.to_path_buf())
        .context("loading SRS file")?;
    if logrows < params.k() {
        params.downsize(logrows);
    }
    Ok(params)
}

fn derive_model_id(path: &Path) -> Option<String> {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .and_then(|name| name.strip_prefix("model_"))
        .map(|rest| rest.to_string())
}

fn resolve_srs_path(hint: Option<&Path>, logrows: u32) -> Result<PathBuf> {
    if let Some(path) = hint {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        warn!(
            "SRS hint {} missing; attempting to locate kzg{}.srs via fallbacks",
            path.display(),
            logrows
        );
    }

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let mut push_candidate = |path: PathBuf| {
        if seen.insert(path.clone()) {
            candidates.push(path);
        }
    };

    if let Some(path) = hint {
        if let Some(parent) = path.parent() {
            push_candidate(parent.join(format!("kzg{logrows}.srs")));
            push_candidate(parent.join(format!("kzg_{logrows}.srs")));
        }
        push_candidate(path.to_path_buf());
    }

    if let Ok(explicit) = env::var(SRS_ENV_VAR) {
        let candidate = PathBuf::from(explicit);
        if candidate.is_file() {
            push_candidate(candidate.clone());
        } else {
            push_candidate(candidate.join(format!("kzg{logrows}.srs")));
            push_candidate(candidate.join("kzg.srs"));
        }
    }

    if let Ok(home) = env::var("HOME") {
        let base = PathBuf::from(home).join(".ezkl/srs");
        push_candidate(base.join(format!("kzg{logrows}.srs")));
        push_candidate(base.join("kzg.srs"));
    }

    if let Ok(cwd) = env::current_dir() {
        push_candidate(cwd.join(format!("kzg{logrows}.srs")));
        push_candidate(cwd.join("kzg.srs"));
    }

    push_candidate(PathBuf::from(format!("kzg{logrows}.srs")));
    push_candidate(PathBuf::from("kzg.srs"));

    for candidate in candidates {
        if candidate.exists() {
            debug!(
                "resolved SRS for logrows {} to {}",
                logrows,
                candidate.display()
            );
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "unable to locate SRS file for logrows {}; set {}=</path/to/srs/dir> or place kzg{}.srs alongside the model artifacts",
        logrows,
        SRS_ENV_VAR,
        logrows
    ))
}

fn load_model_state(
    model_id: &str,
    compiled_path: &Path,
    settings_path: &Path,
    metadata_path: Option<&Path>,
    pk_path: &Path,
    srs_hint: &Path,
    vk_path: Option<&Path>,
) -> Result<Arc<ModelState>> {
    let _nvtx_model = nvtx_guard(format_args!("model.load_state {}", model_id));

    let compiled_bytes = {
        let _nvtx = nvtx_guard(format_args!("model.load_state.read_compiled {}", model_id));
        Arc::new(fs::read(compiled_path).with_context(|| {
            format!("reading compiled circuit from {}", compiled_path.display())
        })?)
    };

    let circuit = {
        let _nvtx = nvtx_guard(format_args!("model.load_state.load_circuit {}", model_id));
        GraphCircuit::load(compiled_path.to_path_buf())
            .with_context(|| format!("loading compiled circuit {}", compiled_path.display()))?
    };
    let circuit_params = circuit.params();

    let settings = {
        let _nvtx = nvtx_guard(format_args!("model.load_state.load_settings {}", model_id));
        GraphSettings::load(&settings_path.to_path_buf())
            .with_context(|| format!("loading settings from {}", settings_path.display()))?
    };
    let commitment: Commitments = settings.run_args.commitment.into();
    if commitment != Commitments::KZG {
        return Err(anyhow!(
            "model {} uses unsupported commitment {:?}",
            model_id,
            commitment
        ));
    }

    let metadata = if let Some(path) = metadata_path {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("reading metadata from {}", path.display()))?;
        let value: Value = serde_json::from_str(&raw)
            .with_context(|| format!("parsing metadata from {}", path.display()))?;
        Some(value)
    } else {
        None
    };

    let pk = {
        let _nvtx = nvtx_guard(format_args!("model.load_state.load_pk {}", model_id));
        pfsys::load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            pk_path.to_path_buf(),
            circuit_params,
        )
        .with_context(|| format!("loading proving key from {}", pk_path.display()))?
    };
    let pk = Arc::new(pk);
    cache::store_kzg_pk(pk_path, pk.clone());

    let srs_path = resolve_srs_path(Some(srs_hint), settings.run_args.logrows)
        .with_context(|| format!("locating SRS for model {}", model_id))?;
    let params = {
        let _nvtx = nvtx_guard(format_args!("model.load_state.load_srs {}", model_id));
        Arc::new(
            load_kzg_params(&srs_path, settings.run_args.logrows)
                .with_context(|| format!("loading SRS from {}", srs_path.display()))?,
        )
    };

    let vk = if let Some(path) = vk_path {
        if path.exists() {
            Some(Arc::new({
                let _nvtx = nvtx_guard(format_args!("model.load_state.load_vk {}", model_id));
                pfsys::load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
                    path.to_path_buf(),
                    settings.clone(),
                )
                .with_context(|| format!("loading verifying key from {}", path.display()))?
            }))
        } else {
            None
        }
    } else {
        None
    };

    Ok(Arc::new(ModelState {
        model_id: model_id.to_string(),
        compiled_bytes,
        settings,
        metadata,
        pk,
        params,
        vk,
        commitment,
    }))
}

fn load_models_from_root(
    model_root: &Path,
    artifact_root: Option<&Path>,
) -> Result<HashMap<String, Arc<ModelState>>> {
    let mut models = HashMap::new();
    for entry in fs::read_dir(model_root)
        .with_context(|| format!("reading model root {}", model_root.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) if s.starts_with("model_") => s,
            _ => continue,
        };

        let model_id = name_str.trim_start_matches("model_").to_string();
        let _nvtx_model = nvtx_guard(format_args!("models.load_root.{model_id}"));
        let compiled_path = entry.path().join("model.compiled");
        let settings_path = entry.path().join("settings.json");
        let metadata_path = entry.path().join("metadata.json");
        if !compiled_path.exists() || !settings_path.exists() {
            warn!(
                "skipping {}: missing compiled circuit or settings",
                entry.path().display()
            );
            continue;
        }

        let artifacts_dir = artifact_root
            .map(|root| root.join(name_str))
            .filter(|p| p.exists())
            .unwrap_or(entry.path());
        let pk_path = artifacts_dir.join("pk.key");
        let srs_path = artifacts_dir.join("kzg.srs");
        if !pk_path.exists() {
            warn!("skipping {}: missing proving key", artifacts_dir.display());
            continue;
        }

        let vk_path = artifacts_dir.join("vk.key");
        let vk_opt = if vk_path.exists() {
            Some(vk_path)
        } else {
            None
        };
        let metadata_opt = if metadata_path.exists() {
            Some(metadata_path)
        } else {
            None
        };

        let model_state = load_model_state(
            &model_id,
            &compiled_path,
            &settings_path,
            metadata_opt.as_deref(),
            &pk_path,
            &srs_path,
            vk_opt.as_deref(),
        )?;
        info!("cached model {}", model_id);
        models.insert(model_id, model_state);
    }

    Ok(models)
}

const STATIC_MODEL_IDS: &[&str] = &[
    "43ecaacaded5ed16c9e08bc054366e409c7925245eca547472b27f2a61469cc5",
    "31df94d233053d9648c3c57362d9aa8aaa0f77761ac520af672103dbb387a6a5",
    "1876cfa9fb3c418b2559f3f7074db20565b5ca7237efdd43b907d9d697a452c4",
];

const DEPLOYMENT_ENV_VAR: &str = "OMRON_MODEL_DEPLOYMENT_ROOT";
const ARTIFACT_ENV_VAR: &str = "OMRON_MODEL_ARTIFACT_ROOT";
const SRS_ENV_VAR: &str = "OMRON_SRS_ROOT";

fn find_deployment_root() -> Result<PathBuf> {
    if let Ok(path) = env::var(DEPLOYMENT_ENV_VAR) {
        let path = PathBuf::from(path);
        ensure!(
            path.exists(),
            "deployment root specified via {} does not exist: {}",
            DEPLOYMENT_ENV_VAR,
            path.display()
        );
        return Ok(path);
    }

    let mut candidates = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("neurons/deployment_layer"));
        candidates.push(cwd.join("../neurons/deployment_layer"));
        candidates.push(cwd.join("git/omron/neurons/deployment_layer"));
        candidates.push(cwd.join("../git/omron/neurons/deployment_layer"));
    }
    if let Ok(home) = env::var("HOME") {
        let home_path = PathBuf::from(home);
        candidates.push(home_path.join("git/omron/neurons/deployment_layer"));
        candidates.push(home_path.join("omron/neurons/deployment_layer"));
    }
    candidates.push(PathBuf::from("git/omron/neurons/deployment_layer"));
    candidates.push(PathBuf::from("../git/omron/neurons/deployment_layer"));

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "unable to locate deployment layer root; set {} to override",
        DEPLOYMENT_ENV_VAR
    ))
}

fn find_artifact_root() -> Result<PathBuf> {
    if let Ok(path) = env::var(ARTIFACT_ENV_VAR) {
        let path = PathBuf::from(path);
        ensure!(
            path.exists(),
            "artifact root specified via {} does not exist: {}",
            ARTIFACT_ENV_VAR,
            path.display()
        );
        return Ok(path);
    }

    let mut candidates = Vec::new();
    if let Ok(home) = env::var("HOME") {
        let home_path = PathBuf::from(home);
        candidates.push(home_path.join(".bittensor/omron/models"));
    }
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join(".bittensor/omron/models"));
        candidates.push(cwd.join("../.bittensor/omron/models"));
    }
    candidates.push(PathBuf::from(".bittensor/omron/models"));
    candidates.push(PathBuf::from("../.bittensor/omron/models"));

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "unable to locate proving artifact root; set {} to override",
        ARTIFACT_ENV_VAR
    ))
}

fn load_static_models() -> Result<HashMap<String, Arc<ModelState>>> {
    let deployment_root = find_deployment_root().context("locating deployment layer root")?;
    let artifact_root = find_artifact_root().context("locating proving artifact root")?;

    let mut models = HashMap::new();
    for model_id in STATIC_MODEL_IDS {
        let folder = format!("model_{}", model_id);
        let deploy_dir = deployment_root.join(&folder);
        let artifact_dir = artifact_root.join(&folder);

        ensure!(
            deploy_dir.exists(),
            "deployment folder for model {} missing at {}",
            model_id,
            deploy_dir.display()
        );
        ensure!(
            artifact_dir.exists(),
            "artifact folder for model {} missing at {}",
            model_id,
            artifact_dir.display()
        );

        let compiled_path = deploy_dir.join("model.compiled");
        let settings_path = deploy_dir.join("settings.json");
        let metadata_path = deploy_dir.join("metadata.json");
        let pk_path = artifact_dir.join("pk.key");
        let srs_path = artifact_dir.join("kzg.srs");
        let vk_path = deploy_dir.join("vk.key");

        ensure!(
            compiled_path.exists(),
            "missing compiled circuit for model {} at {}",
            model_id,
            compiled_path.display()
        );
        ensure!(
            settings_path.exists(),
            "missing settings for model {} at {}",
            model_id,
            settings_path.display()
        );
        ensure!(
            metadata_path.exists(),
            "missing metadata for model {} at {}",
            model_id,
            metadata_path.display()
        );
        ensure!(
            pk_path.exists(),
            "missing proving key for model {} at {}",
            model_id,
            pk_path.display()
        );
        if !srs_path.exists() {
            debug!(
                "kzg.srs not found for model {} at {}; relying on fallback discovery",
                model_id,
                srs_path.display()
            );
        }

        let model_state = load_model_state(
            model_id,
            &compiled_path,
            &settings_path,
            Some(metadata_path.as_path()),
            &pk_path,
            &srs_path,
            if vk_path.exists() {
                Some(vk_path.as_path())
            } else {
                None
            },
        )?;
        info!("cached preconfigured model {}", model_id);
        models.insert((*model_id).to_string(), model_state);
    }

    Ok(models)
}

const MODEL_31_INPUT_FIELDS: &[&str] = &["list_items"];
const MODEL_43_INPUT_FIELDS: &[&str] = &[
    "scores",
    "top_tier_pct",
    "next_tier_pct",
    "top_tier_weight",
    "next_tier_weight",
    "bottom_tier_weight",
    "nonce",
];
const MODEL_1876_INPUT_FIELDS: &[&str] = &[
    "challenge_attempts",
    "challenge_successes",
    "last_20_challenge_failed",
    "challenge_elapsed_time_avg",
    "last_20_difficulty_avg",
    "has_docker",
    "uid",
    "allocated_uids",
    "penalized_uids",
    "validator_uids",
    "success_weight",
    "difficulty_weight",
    "time_elapsed_weight",
    "failed_penalty_weight",
    "allocation_weight",
    "pow_timeout",
    "pow_min_difficulty",
    "pow_max_difficulty",
    "nonce",
];

fn model_input_fields(model_id: &str) -> Option<&'static [&'static str]> {
    match model_id {
        "31df94d233053d9648c3c57362d9aa8aaa0f77761ac520af672103dbb387a6a5" => {
            Some(MODEL_31_INPUT_FIELDS)
        }
        "43ecaacaded5ed16c9e08bc054366e409c7925245eca547472b27f2a61469cc5" => {
            Some(MODEL_43_INPUT_FIELDS)
        }
        "1876cfa9fb3c418b2559f3f7074db20565b5ca7237efdd43b907d9d697a452c4" => {
            Some(MODEL_1876_INPUT_FIELDS)
        }
        _ => None,
    }
}

fn map_inputs_to_data_source(
    model: &ModelState,
    mut inputs: HashMap<String, Value>,
) -> Result<DataSource> {
    let field_order = model_input_fields(&model.model_id).with_context(|| {
        format!(
            "model {} has no configured field order; send direct input_data payload",
            model.model_id
        )
    })?;

    let mut ordered = Vec::with_capacity(field_order.len());
    for field in field_order {
        let value = inputs
            .remove(*field)
            .with_context(|| format!("missing required input field '{}'", field))?;
        if !value.is_array() {
            return Err(anyhow!("input '{}' must be an array", field));
        }
        ordered.push(value);
    }

    if !inputs.is_empty() {
        let extras = inputs.keys().cloned().collect::<Vec<_>>();
        debug!(
            "ignoring extra input keys for model {}: {:?}",
            model.model_id, extras
        );
    }

    let encoded = Value::Array(ordered);
    let input_data: DataSource =
        serde_json::from_value(encoded).context("encoding mapped inputs into DataSource")?;

    Ok(input_data)
}

fn synapse_to_graph_data(model: &ModelState, envelope: SynapseEnvelope) -> Result<GraphData> {
    let SynapseEnvelope { kind, payload, .. } = envelope;
    debug!(
        "building proof data for model {} from synapse kind {}",
        model.model_id, kind
    );
    let input_map = payload
        .into_input_map()
        .with_context(|| format!("extracting inputs for model {}", model.model_id))?;
    let input_data = map_inputs_to_data_source(model, input_map)?;

    Ok(GraphData {
        input_data,
        output_data: None,
    })
}

#[derive(Serialize)]
struct ProofResponse {
    model_id: String,
    instances: Vec<Vec<String>>,
    proof: Vec<u8>,
    proof_hex: String,
    transcript_type: TranscriptType,
    witness_time_ms: f64,
    prove_time_ms: f64,
    total_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pretty_public_inputs: Option<PrettyElements>,
}

#[derive(Deserialize)]
struct DirectProofRequest {
    model_id: String,
    input_data: DataSource,
    #[serde(default)]
    output_data: Option<DataSource>,
}

#[derive(Deserialize)]
struct SynapseEnvelope {
    kind: String,
    #[serde(default)]
    _timestamp: Option<Value>,
    payload: SynapsePayload,
}

#[derive(Deserialize)]
struct SynapsePayload {
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    verification_key_hash: Option<String>,
    #[serde(default)]
    query_input: Option<SynapseQueryInput>,
    #[serde(default)]
    inputs: Option<HashMap<String, Value>>,
    #[serde(default)]
    _dendrite_hotkey: Option<String>,
}

impl SynapsePayload {
    fn model_identifier(&self) -> Result<&str> {
        if let Some(id) = self.model_id.as_deref() {
            Ok(id)
        } else if let Some(hash) = self.verification_key_hash.as_deref() {
            Ok(hash)
        } else {
            Err(anyhow!("synapse payload missing model identifier"))
        }
    }

    fn into_input_map(self) -> Result<HashMap<String, Value>> {
        if let Some(query) = self.query_input {
            if let Some(public_inputs) = query.public_inputs {
                return Ok(public_inputs);
            }
        }
        if let Some(inputs) = self.inputs {
            return Ok(inputs);
        }
        Err(anyhow!("synapse payload missing input data"))
    }
}

#[derive(Deserialize)]
struct SynapseQueryInput {
    #[serde(default)]
    public_inputs: Option<HashMap<String, Value>>,
    #[serde(default)]
    #[allow(dead_code)]
    _private_inputs: Option<HashMap<String, Value>>,
}

enum ProofRequest {
    Direct(DirectProofRequest),
    Synapse(SynapseEnvelope),
}

enum RequestVariant {
    Direct(DirectProofRequest),
    Synapse(SynapseEnvelope),
}

#[cfg(all(feature = "icicle", not(target_arch = "wasm32")))]
fn configure_backend_device() {
    let device = env::var("ICICLE_DEVICE_TYPE")
        .ok()
        .or_else(|| {
            env::var("ENABLE_ICICLE_GPU")
                .ok()
                .map(|_| "CUDA".to_string())
        })
        .unwrap_or_else(|| "CPU".to_string());

    info!("Selecting ICICLE backend device: {}", device);
    try_load_and_set_backend_device(&device);
}

#[cfg(not(all(feature = "icicle", not(target_arch = "wasm32"))))]
fn configure_backend_device() {}

impl<'de> Deserialize<'de> for ProofRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        if value.get("model_id").is_some() && value.get("input_data").is_some() {
            let direct: DirectProofRequest =
                serde_json::from_value(value.clone()).map_err(DeError::custom)?;
            return Ok(ProofRequest::Direct(direct));
        }

        if value.get("kind").is_some() && value.get("payload").is_some() {
            let envelope: SynapseEnvelope =
                serde_json::from_value(value).map_err(DeError::custom)?;
            return Ok(ProofRequest::Synapse(envelope));
        }

        Err(DeError::custom("unsupported request format"))
    }
}

#[derive(Debug)]
struct DaemonError(anyhow::Error);

impl IntoResponse for DaemonError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.0.to_string() });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _nvtx_main = nvtx_guard("daemon.main");
    ezkl::logger::init_logger();
    log::info!("logger initialized");
    configure_backend_device();

    let args = Args::parse();

    let proof_type = match args.proof_type.to_lowercase().as_str() {
        "single" => ProofType::Single,
        "for-aggr" | "for_aggr" => ProofType::ForAggr,
        other => return Err(anyhow!("unsupported proof type '{}'", other)),
    };

    let check_mode = match args.check_mode.to_lowercase().as_str() {
        "safe" => CheckMode::SAFE,
        "unsafe" => CheckMode::UNSAFE,
        other => return Err(anyhow!("unsupported check mode '{}'", other)),
    };

    let models_map = {
        let _nvtx_models = nvtx_guard("daemon.load_models");
        if let Some(model_root) = args.model_root.as_ref() {
            let artifact_root = args.artifact_root.as_deref().or(Some(model_root.as_path()));
            load_models_from_root(model_root.as_path(), artifact_root)?
        } else if let Some(compiled) = args.compiled_circuit.as_ref() {
            let settings_path = args
                .settings
                .as_ref()
                .context("--settings required when --model-root is not provided")?;
            let pk_path = args
                .pk
                .as_ref()
                .context("--pk required when --model-root is not provided")?;
            let srs_path = args
                .srs
                .as_ref()
                .context("--srs required when --model-root is not provided")?;
            let model_id = args
                .model_id
                .clone()
                .or_else(|| derive_model_id(compiled))
                .context("unable to determine model_id; pass --model-id")?;
            let metadata_path = args
                .metadata
                .clone()
                .or_else(|| compiled.parent().map(|p| p.join("metadata.json")))
                .filter(|p| p.exists());
            if metadata_path.is_none() {
                warn!(
                    "metadata.json not found for model {}; continuing without metadata",
                    model_id
                );
            }
            let model_state = load_model_state(
                &model_id,
                compiled.as_path(),
                settings_path.as_path(),
                metadata_path.as_deref(),
                pk_path.as_path(),
                srs_path.as_path(),
                args.vk.as_deref(),
            )?;
            let mut map = HashMap::new();
            map.insert(model_id, model_state);
            map
        } else {
            load_static_models()?
        }
    };

    if models_map.is_empty() {
        return Err(anyhow!("no models loaded"));
    }

    println!("Loaded models: {:?}", models_map.keys().collect::<Vec<_>>());

    let shared_state = Arc::new(AppState {
        models: Arc::new(models_map),
        proof_type,
        check_mode,
        semaphore: Arc::new(Semaphore::new(args.concurrency.max(1))),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/prove", post(prove_handler))
        .with_state(shared_state);

    let listener = TcpListener::bind(args.listen)
        .await
        .context("binding listener")?;
    let addr = listener.local_addr().unwrap_or(args.listen);
    println!("ezkl daemon listening on {}", addr);
    axum::serve(listener, app.into_make_service())
        .await
        .context("running daemon server")?;

    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn prove_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ProofRequest>,
) -> Result<Json<ProofResponse>, DaemonError> {
    let permit = state
        .semaphore
        .acquire()
        .await
        .map_err(|e| DaemonError(anyhow!("failed to acquire proving slot: {e}")))?;

    let (model_id, variant) = match request {
        ProofRequest::Direct(req) => {
            let model_id = req.model_id.clone();
            (model_id, RequestVariant::Direct(req))
        }
        ProofRequest::Synapse(env) => {
            let model_id = env
                .payload
                .model_identifier()
                .map_err(DaemonError)?
                .to_owned();
            (model_id, RequestVariant::Synapse(env))
        }
    };

    let _nvtx_handler = nvtx_guard(format_args!("daemon.prove_handler.{}", model_id));

    let model = state
        .models
        .get(&model_id)
        .cloned()
        .ok_or_else(|| DaemonError(anyhow!("unknown model_id '{}'", model_id)))?;

    let graph_data = match variant {
        RequestVariant::Direct(req) => GraphData {
            input_data: req.input_data,
            output_data: req.output_data,
        },
        RequestVariant::Synapse(env) => {
            synapse_to_graph_data(model.as_ref(), env).map_err(DaemonError)?
        }
    };

    let result = prove_internal(state.as_ref(), model, graph_data).await;
    drop(permit);
    result.map(Json).map_err(DaemonError)
}

async fn prove_internal(
    app_state: &AppState,
    model: Arc<ModelState>,
    graph_data: GraphData,
) -> Result<ProofResponse> {
    let _nvtx_total = nvtx_guard(format_args!("prove_internal.{}", model.model_id));
    let total_start = Instant::now();

    let mut circuit: GraphCircuit = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.deserialize_circuit.{}",
            model.model_id
        ));
        bincode::deserialize(model.compiled_bytes.as_slice()).context("decoding circuit")?
    };

    let mut tensors = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.load_graph_input.{}",
            model.model_id
        ));
        circuit
            .load_graph_input(&graph_data)
            .await
            .context("loading graph input")?
    };

    let region_settings = RegionSettings::all_true(
        model.settings.run_args.decomp_base,
        model.settings.run_args.decomp_legs,
    );

    let witness_start = Instant::now();
    let witness = {
        let _nvtx = nvtx_guard(format_args!("prove_internal.forward.{}", model.model_id));
        circuit.forward::<KZGCommitmentScheme<_>>(
            &mut tensors,
            model.vk.as_ref().map(|vk| vk.as_ref()),
            Some(model.params.as_ref()),
            region_settings,
        )?
    };
    let witness_time = witness_start.elapsed();

    {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.load_graph_witness.{}",
            model.model_id
        ));
        circuit
            .load_graph_witness(&witness)
            .context("loading graph witness into circuit")?;
    }

    let pretty_public_inputs = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.pretty_public_inputs.{}",
            model.model_id
        ));
        circuit.pretty_public_inputs(&witness)?
    };
    let public_inputs = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.prepare_public_inputs.{}",
            model.model_id
        ));
        circuit.prepare_public_inputs(&witness)?
    };
    let proof_split_commits: Option<ProofSplitCommit> = witness.into();

    let strategy: StrategyType = app_state.proof_type.into();
    let transcript: TranscriptType = app_state.proof_type.into();

    let prove_start = Instant::now();
    let mut snark = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.create_proof.{:?}.{}",
            strategy, model.model_id
        ));
        match model.commitment {
            Commitments::KZG => match strategy {
                StrategyType::Single => pfsys::create_proof_circuit::<
                    KZGCommitmentScheme<Bn256>,
                    _,
                    ProverSHPLONK<_>,
                    VerifierSHPLONK<_>,
                    KZGSingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    model.params.as_ref(),
                    model.pk.as_ref(),
                    app_state.check_mode,
                    model.commitment,
                    transcript,
                    proof_split_commits,
                    None,
                )?,
                StrategyType::Accum => {
                    let protocol = Some(compile(
                        model.params.as_ref(),
                        model.pk.as_ref().get_vk(),
                        Config::kzg().with_num_instance(vec![public_inputs.len()]),
                    ));

                    pfsys::create_proof_circuit::<
                        KZGCommitmentScheme<Bn256>,
                        _,
                        ProverSHPLONK<_>,
                        VerifierSHPLONK<_>,
                        KZGAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        model.params.as_ref(),
                        model.pk.as_ref(),
                        app_state.check_mode,
                        model.commitment,
                        transcript,
                        proof_split_commits,
                        protocol,
                    )?
                }
            },
            other => return Err(anyhow!("unsupported commitment {:?}", other)),
        }
    };
    let prove_time = prove_start.elapsed();

    snark.pretty_public_inputs = pretty_public_inputs.clone();

    let (proof_bytes, proof_hex, transcript_type, instances) = {
        let _nvtx = nvtx_guard(format_args!(
            "prove_internal.prepare_response.{}",
            model.model_id
        ));
        let proof_bytes = snark.proof.clone();
        let proof_hex = format!("0x{}", hex::encode(&proof_bytes));
        let transcript_type = snark.transcript_type;
        let instances = snark
            .instances
            .iter()
            .map(|row| row.iter().map(field_to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        (proof_bytes, proof_hex, transcript_type, instances)
    };

    let total_time = total_start.elapsed();

    Ok(ProofResponse {
        model_id: model.model_id.clone(),
        proof: proof_bytes,
        proof_hex,
        transcript_type,
        instances,
        witness_time_ms: witness_time.as_secs_f64() * 1000.0,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        total_time_ms: total_time.as_secs_f64() * 1000.0,
        pretty_public_inputs,
    })
}
