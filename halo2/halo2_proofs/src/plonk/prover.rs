use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::Curve;
use instant::Instant;
use rand_core::RngCore;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::iter;
use std::ops::RangeTo;
#[cfg(feature = "parallel-synthesis")]
use std::sync::{Arc, Mutex};

use super::{
    circuit::{
        sealed::{self, Phase},
        Advice, Any, Assignment, Challenge, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner,
        Instance, Selector,
    },
    permutation, shuffle, vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX,
    ChallengeY, Error, ProvingKey,
};
#[cfg(all(feature = "parallel-synthesis", not(feature = "mv-lookup")))]
use maybe_rayon::iter::IntoParallelIterator;
#[cfg(any(feature = "mv-lookup", feature = "parallel-synthesis"))]
use maybe_rayon::iter::ParallelIterator;
#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator};

#[cfg(not(feature = "mv-lookup"))]
use super::lookup;
#[cfg(feature = "mv-lookup")]
use super::mv_lookup as lookup;

use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    circuit::Value,
    plonk::evaluation::EvaluateHOutput,
    plonk::Assigned,
    poly::{
        commitment::{Blind, CommitmentScheme, Params, Prover},
        Basis, Coeff, LagrangeCoeff, Polynomial, ProverQuery,
    },
    util::timing::TimingScope,
};
use crate::{
    poly::batch_invert_assigned,
    transcript::{EncodedChallenge, TranscriptWrite},
};
use group::prime::PrimeCurveAffine;

#[cfg(feature = "parallel-synthesis")]
/// Helper trait alias capturing the additional bounds required when
/// parallel witness synthesis is enabled.
pub trait ProverCircuit<F>: Circuit<F> + Sync
where
    F: Field,
    Self::Config: Send + Sync,
{
}

#[cfg(feature = "parallel-synthesis")]
impl<F, C> ProverCircuit<F> for C
where
    F: Field,
    C: Circuit<F> + Sync,
    C::Config: Send + Sync,
{
}

#[cfg(not(feature = "parallel-synthesis"))]
/// Helper trait alias when witness synthesis runs serially.
pub trait ProverCircuit<F>: Circuit<F>
where
    F: Field,
{
}

#[cfg(not(feature = "parallel-synthesis"))]
impl<F, C> ProverCircuit<F> for C
where
    F: Field,
    C: Circuit<F>,
{
}

/// This creates a proof for the provided `circuit` when given the public
/// parameters `params` and the proving key [`ProvingKey`] that was
/// generated previously for the same circuit. The provided `instances`
/// are zero-padded internally.
pub fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    R: RngCore + Send + Sync,
    T: TranscriptWrite<Scheme::Curve, E>,
    ConcreteCircuit: ProverCircuit<Scheme::Scalar>,
>(
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    circuits: &[ConcreteCircuit],
    instances: &[&[&[Scheme::Scalar]]],
    mut rng: R,
    transcript: &mut T,
) -> Result<(), Error>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    Scheme::ParamsProver: Send + Sync,
    ConcreteCircuit::Config: Send + Sync,
    <Scheme::Scalar as PrimeField>::Repr: Clone + AsRef<[u8]> + Send + Sync,
{
    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;

        // reset counters at the beginning of the prove
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    if circuits.len() != instances.len() {
        return Err(Error::InvalidInstances);
    }

    for instance in instances.iter() {
        if instance.len() != pk.vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    let proof_scope = TimingScope::root("plonk::create_proof");

    let start = Instant::now();
    // Hash verification key into transcript
    pk.vk.hash_into(transcript)?;
    log::info!("Hashing verification key: {:?}", start.elapsed());

    let domain = &pk.vk.domain;
    let mut meta = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut meta, circuits[0].params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut meta);

    // Selector optimizations cannot be applied here; use the ConstraintSystem
    // from the verification key.
    let meta = &pk.vk.cs;

    #[cfg(feature = "parallel-synthesis")]
    let advice_pools: Arc<Vec<Mutex<Vec<Polynomial<Assigned<Scheme::Scalar>, LagrangeCoeff>>>>> =
        Arc::new(
            (0..circuits.len())
                .map(|_| Mutex::new(Vec::new()))
                .collect(),
        );
    #[cfg(not(feature = "parallel-synthesis"))]
    let mut advice_pools: Vec<Vec<Polynomial<Assigned<Scheme::Scalar>, LagrangeCoeff>>> =
        vec![Vec::new(); circuits.len()];

    struct InstanceSingle<C: CurveAffine> {
        pub instance_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
        pub instance_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    }

    let instance: Vec<InstanceSingle<Scheme::Curve>> = instances
        .iter()
        .map(|instance| -> Result<InstanceSingle<Scheme::Curve>, Error> {
            let instance_values = instance
                .iter()
                .map(|values| {
                    let mut poly = domain.empty_lagrange();
                    assert_eq!(poly.len(), params.n() as usize);
                    if values.len() > (poly.len() - (meta.blinding_factors() + 1)) {
                        return Err(Error::InstanceTooLarge);
                    }
                    for (poly, value) in poly.iter_mut().zip(values.iter()) {
                        if !P::QUERY_INSTANCE {
                            transcript.common_scalar(*value)?;
                        }
                        *poly = *value;
                    }
                    Ok(poly)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if P::QUERY_INSTANCE {
                let instance_commitments_projective: Vec<_> = instance_values
                    .iter()
                    .map(|poly| params.commit_lagrange(poly, Blind::default()))
                    .collect();
                let mut instance_commitments =
                    vec![Scheme::Curve::identity(); instance_commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &instance_commitments_projective,
                    &mut instance_commitments,
                );
                let instance_commitments = instance_commitments;
                drop(instance_commitments_projective);

                for commitment in &instance_commitments {
                    transcript.common_point(*commitment)?;
                }
            }

            let instance_polys: Vec<_> = instance_values
                .iter()
                .map(|poly| {
                    let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                    domain.lagrange_to_coeff(lagrange_vec)
                })
                .collect();

            Ok(InstanceSingle {
                instance_values,
                instance_polys,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[derive(Clone)]
    struct AdviceSingle<C: CurveAffine, B: Basis> {
        pub advice_polys: Vec<Polynomial<C::Scalar, B>>,
        pub advice_blinds: Vec<Blind<C::Scalar>>,
    }

    #[cfg(feature = "parallel-synthesis")]
    struct PhaseAssignment<F: Field> {
        circuit_index: usize,
        columns: Vec<usize>,
        advice_values: Vec<Polynomial<F, LagrangeCoeff>>,
    }

    struct WitnessCollection<'a, F: Field> {
        k: u32,
        current_phase: sealed::Phase,
        advice: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
        challenges: &'a HashMap<usize, F>,
        instances: &'a [&'a [F]],
        usable_rows: RangeTo<usize>,
        _marker: std::marker::PhantomData<F>,
    }

    impl<'a, F: Field> Assignment<F> for WitnessCollection<'a, F> {
        fn enter_region<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about regions in this context.
        }

        fn exit_region(&mut self) {
            // Do nothing; we don't care about regions in this context.
        }

        fn enable_selector<A, AR>(&mut self, _: A, _: &Selector, _: usize) -> Result<(), Error>
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Do nothing
        }

        fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            self.instances
                .get(column.index())
                .and_then(|column| column.get(row))
                .map(|v| Value::known(*v))
                .ok_or(Error::BoundsFailure)
        }

        fn assign_advice<V, VR, A, AR>(
            &mut self,
            _: A,
            column: Column<Advice>,
            row: usize,
            to: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // Ignore assignment of advice column in different phase than current one.
            if self.current_phase != column.column_type().phase {
                return Ok(());
            }

            if !self.usable_rows.contains(&row) {
                return Err(Error::not_enough_rows_available(self.k));
            }

            *self
                .advice
                .get_mut(column.index())
                .and_then(|v| v.get_mut(row))
                .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

            Ok(())
        }

        fn assign_fixed<V, VR, A, AR>(
            &mut self,
            _: A,
            _: Column<Fixed>,
            _: usize,
            _: V,
        ) -> Result<(), Error>
        where
            V: FnOnce() -> Value<VR>,
            VR: Into<Assigned<F>>,
            A: FnOnce() -> AR,
            AR: Into<String>,
        {
            // We only care about advice columns here

            Ok(())
        }

        fn copy(
            &mut self,
            _: Column<Any>,
            _: usize,
            _: Column<Any>,
            _: usize,
        ) -> Result<(), Error> {
            // We only care about advice columns here

            Ok(())
        }

        fn fill_from_row(
            &mut self,
            _: Column<Fixed>,
            _: usize,
            _: Value<Assigned<F>>,
        ) -> Result<(), Error> {
            Ok(())
        }

        fn get_challenge(&self, challenge: Challenge) -> Value<F> {
            self.challenges
                .get(&challenge.index())
                .cloned()
                .map(Value::known)
                .unwrap_or_else(Value::unknown)
        }

        fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
        {
            // Do nothing; we don't care about namespaces in this context.
        }

        fn pop_namespace(&mut self, _: Option<String>) {
            // Do nothing; we don't care about namespaces in this context.
        }
    }

    let (advice, challenges) = {
        let mut advice = vec![
            AdviceSingle::<Scheme::Curve, LagrangeCoeff> {
                advice_polys: vec![domain.empty_lagrange(); meta.num_advice_columns],
                advice_blinds: vec![Blind::default(); meta.num_advice_columns],
            };
            instances.len()
        ];
        let s = FxBuildHasher;
        let mut challenges =
            HashMap::<usize, Scheme::Scalar>::with_capacity_and_hasher(meta.num_challenges, s);
        let unusable_rows_start = params.n() as usize - (meta.blinding_factors() + 1);
        let phase_column_indices: Vec<(Phase, Vec<usize>)> = pk
            .vk
            .cs
            .phases()
            .into_iter()
            .map(|phase| {
                let mut cols: Vec<usize> = meta
                    .advice_column_phase
                    .iter()
                    .enumerate()
                    .filter_map(|(column_index, current)| {
                        if phase == *current {
                            Some(column_index)
                        } else {
                            None
                        }
                    })
                    .collect();
                cols.sort_unstable();
                (phase, cols)
            })
            .collect();

        for (current_phase, column_indices) in phase_column_indices.iter() {
            let phase_start = Instant::now();
            log::info!("advice phase {:?} started", current_phase);
            let _phase_scope = proof_scope.child(format!("advice phase {:?}", current_phase));
            let phase_value = *current_phase;
            let unblinded_advice_columns: HashSet<usize> =
                HashSet::from_iter(meta.unblinded_advice_columns.clone());

            let mut handle_assignment = |circuit_index: usize,
                                         advice_values: Vec<
                Polynomial<Scheme::Scalar, LagrangeCoeff>,
            >,
                                         columns: Vec<usize>|
             -> Result<(), Error> {
                if columns.is_empty() {
                    return Ok(());
                }

                let commit_start = Instant::now();
                let advice_entry = &mut advice[circuit_index];
                let mut commitments_projective = Vec::with_capacity(columns.len());
                let mut updates = Vec::with_capacity(columns.len());

                for (column_index, mut values) in columns.into_iter().zip(advice_values.into_iter())
                {
                    if !unblinded_advice_columns.contains(&column_index) {
                        for cell in &mut values[unusable_rows_start..] {
                            *cell = Scheme::Scalar::random(&mut rng);
                        }
                    } else {
                        for cell in &mut values[unusable_rows_start..] {
                            *cell = Blind::default().0;
                        }
                    }

                    let blind = if !unblinded_advice_columns.contains(&column_index) {
                        Blind(Scheme::Scalar::random(&mut rng))
                    } else {
                        Blind::default()
                    };

                    commitments_projective.push(params.commit_lagrange(&values, blind));
                    updates.push((column_index, values, blind));
                }

                if commitments_projective.is_empty() {
                    return Ok(());
                }

                let mut commitments = vec![Scheme::Curve::identity(); commitments_projective.len()];
                <Scheme::Curve as CurveAffine>::CurveExt::batch_normalize(
                    &commitments_projective,
                    &mut commitments,
                );

                for commitment in &commitments {
                    transcript.write_point(*commitment)?;
                }

                for (column_index, values, blind) in updates {
                    advice_entry.advice_polys[column_index] = values;
                    advice_entry.advice_blinds[column_index] = blind;
                }

                log::info!(
                    "advice commitment work for circuit {} phase {:?} took {:.3?}",
                    circuit_index,
                    phase_value,
                    commit_start.elapsed()
                );

                Ok(())
            };

            #[cfg(feature = "parallel-synthesis")]
            {
                let current_phase_val = phase_value;
                let phase_collection_start = Instant::now();
                let column_indices = column_indices.clone();
                let advice_pools = Arc::clone(&advice_pools);
                let config = config.clone();
                let challenges_ref = &challenges;
                let mut phase_assignments = (0..circuits.len())
                    .into_par_iter()
                    .map(move |circuit_index| -> Result<PhaseAssignment<Scheme::Scalar>, Error> {
                        let synth_start = Instant::now();
                        let mut advice_storage = {
                            let mut guard = advice_pools[circuit_index].lock().unwrap();
                            if guard.is_empty() {
                                guard.resize_with(meta.num_advice_columns, || {
                                    domain.empty_lagrange_assigned()
                                });
                            }
                            for poly in guard.iter_mut() {
                                for value in poly.iter_mut() {
                                    *value = Assigned::Zero;
                                }
                            }
                            let mut storage = Vec::new();
                            std::mem::swap(&mut *guard, &mut storage);
                            storage
                        };

                        let mut witness = WitnessCollection {
                            k: params.k(),
                            current_phase: current_phase_val,
                            advice: advice_storage,
                            instances: instances[circuit_index],
                            challenges: challenges_ref,
                            usable_rows: ..unusable_rows_start,
                            _marker: std::marker::PhantomData,
                        };

                        ConcreteCircuit::FloorPlanner::synthesize(
                            &mut witness,
                            &circuits[circuit_index],
                            config.clone(),
                            meta.constants.clone(),
                        )?;
                        log::info!(
                            "witness generation for phase {:?} circuit {} took {:.3?}",
                            current_phase_val,
                            circuit_index,
                            synth_start.elapsed()
                        );

                        let advice_storage = witness.advice;
                        let columns = column_indices.clone();
                        let invert_start = Instant::now();
                        let advice_values = batch_invert_assigned(
                            columns.iter().map(|&column_index| &advice_storage[column_index]),
                        );
                        log::info!(
                            "phase {:?} circuit {}: batch inversion took {:.3?}",
                            current_phase_val,
                            circuit_index,
                            invert_start.elapsed()
                        );

                        {
                            let mut guard = advice_pools[circuit_index].lock().unwrap();
                            *guard = advice_storage;
                        }

                        Ok(PhaseAssignment {
                            circuit_index,
                            columns,
                            advice_values,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                phase_assignments.sort_by_key(|assignment| assignment.circuit_index);

                for assignment in phase_assignments {
                    handle_assignment(
                        assignment.circuit_index,
                        assignment.advice_values,
                        assignment.columns,
                    )?;
                }
                log::info!(
                    "phase {:?}: parallel witness collection completed in {:.3?}",
                    phase_value,
                    phase_collection_start.elapsed()
                );
            }

            #[cfg(not(feature = "parallel-synthesis"))]
            {
                let serial_collection_start = Instant::now();
                for (circuit_index, circuit) in circuits.iter().enumerate() {
                    let synth_start = Instant::now();
                    let advice_buffer = &mut advice_pools[circuit_index];
                    if advice_buffer.is_empty() {
                        advice_buffer
                            .resize_with(meta.num_advice_columns, || domain.empty_lagrange_assigned());
                    }
                    for poly in advice_buffer.iter_mut() {
                        for value in poly.iter_mut() {
                            *value = Assigned::Zero;
                        }
                    }
                    let mut advice_storage = Vec::new();
                    std::mem::swap(advice_buffer, &mut advice_storage);

                    let mut witness = WitnessCollection {
                        k: params.k(),
                        current_phase: phase_value,
                        advice: advice_storage,
                        instances: instances[circuit_index],
                        challenges: &challenges,
                        usable_rows: ..unusable_rows_start,
                        _marker: std::marker::PhantomData,
                    };

                    ConcreteCircuit::FloorPlanner::synthesize(
                        &mut witness,
                        circuit,
                        config.clone(),
                        meta.constants.clone(),
                    )?;
                    log::info!(
                        "witness generation for phase {:?} circuit {} took {:.3?}",
                        phase_value,
                        circuit_index,
                        synth_start.elapsed()
                    );

                    let advice_storage = witness.advice;
                    let columns = column_indices.clone();
                    let invert_start = Instant::now();
                    let advice_values = batch_invert_assigned(
                        columns.iter().map(|&column_index| &advice_storage[column_index]),
                    );
                    log::info!(
                        "phase {:?} circuit {}: batch inversion took {:.3?}",
                        phase_value,
                        circuit_index,
                        invert_start.elapsed()
                    );

                    *advice_buffer = advice_storage;

                    handle_assignment(circuit_index, advice_values, columns)?;
                }
                log::info!(
                    "phase {:?}: serial witness collection completed in {:.3?}",
                    phase_value,
                    serial_collection_start.elapsed()
                );
            }

            for (index, phase) in meta.challenge_phase.iter().enumerate() {
                if phase_value == *phase {
                    let challenge_start = Instant::now();
                    let existing =
                        challenges.insert(index, *transcript.squeeze_challenge_scalar::<()>());
                    assert!(existing.is_none());
                    log::info!(
                        "challenge {:?} sampling took {:.3?}",
                        index,
                        challenge_start.elapsed()
                    );
                }
            }
            log::info!(
                "advice phase {:?} completed in {:.3?}",
                current_phase,
                phase_start.elapsed()
            );
        }

        assert_eq!(challenges.len(), meta.num_challenges);
        let challenges = (0..meta.num_challenges)
            .map(|index| challenges.remove(&index).unwrap())
            .collect::<Vec<_>>();

        (advice, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    #[cfg(feature = "mv-lookup")]
    let lookups: Vec<Vec<lookup::prover::Prepared<Scheme::Curve>>> = instance
        .par_iter()
        .zip(advice.par_iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .par_iter()
                .map(|lookup| {
                    lookup.prepare(
                        &pk.vk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    #[cfg(not(feature = "mv-lookup"))]
    let lookups: Vec<Vec<lookup::prover::Permuted<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, Error> {
            // Construct and commit to permuted values for each lookup
            pk.vk
                .cs
                .lookups
                .iter()
                .map(|lookup| {
                    lookup.commit_permuted(
                        pk,
                        params,
                        domain,
                        theta,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let permutations: Vec<permutation::prover::Committed<Scheme::Curve>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| {
            pk.vk.cs.permutation.commit(
                params,
                pk,
                &pk.permutation,
                &advice.advice_polys,
                &pk.fixed_values,
                &instance.instance_values,
                beta,
                gamma,
                &mut rng,
                transcript,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // preallocate the lookups

    #[cfg(feature = "mv-lookup")]
    let commit_stage_start = Instant::now();
    #[cfg(feature = "mv-lookup")]
    let phi_blind_start = Instant::now();
    #[cfg(feature = "mv-lookup")]
    let phi_blinds = (0..pk.vk.cs.blinding_factors())
        .map(|_| Scheme::Scalar::random(&mut rng))
        .collect::<Vec<_>>();
    #[cfg(feature = "mv-lookup")]
    log::info!(
        "mv_lookup::commit_grand_sum: sampled {} phi blinds in {:.3?}",
        phi_blinds.len(),
        phi_blind_start.elapsed()
    );

    #[cfg(feature = "mv-lookup")]
    let prepared_lookup_batches = lookups.len();
    #[cfg(feature = "mv-lookup")]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_par_iter()
            .enumerate()
            .map(|(batch_idx, lookups)| -> Result<Vec<_>, _> {
                log::info!(
                    "mv_lookup::commit_grand_sum: launching batch {} with {} lookups",
                    batch_idx,
                    lookups.len()
                );
                // Construct and commit to products for each lookup
                lookups
                    .into_par_iter()
                    .map(|lookup| lookup.commit_grand_sum(&pk.vk, params, beta, &phi_blinds))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
    };

    #[cfg(not(feature = "mv-lookup"))]
    let commit_lookups = || -> Result<Vec<Vec<lookup::prover::Committed<Scheme::Curve>>>, _> {
        lookups
            .into_iter()
            .map(|lookups| -> Result<Vec<_>, _> {
                // Construct and commit to products for each lookup
                lookups
                    .into_iter()
                    .map(|lookup| {
                        lookup.commit_product(pk, params, beta, gamma, &mut rng, transcript)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
    };
    #[cfg(feature = "mv-lookup")]
    let lookups = {
        let _lookup_scope = proof_scope.child("mv_lookup::commit_grand_sum");
        commit_lookups()?
    };
    #[cfg(not(feature = "mv-lookup"))]
    let lookups = commit_lookups()?;
    #[cfg(feature = "mv-lookup")]
    log::info!(
        "mv_lookup::commit_grand_sum: completed {} batches in {:.3?}",
        prepared_lookup_batches,
        commit_stage_start.elapsed()
    );

    #[cfg(feature = "mv-lookup")]
    {
        for lookups_ in &lookups {
            for lookup in lookups_.iter() {
                transcript.write_point(lookup.commitment)?;
            }
        }
    }

    let shuffles: Vec<Vec<shuffle::prover::Committed<Scheme::Curve>>> = instance
        .iter()
        .zip(advice.iter())
        .map(|(instance, advice)| -> Result<Vec<_>, _> {
            // Compress expressions for each shuffle
            pk.vk
                .cs
                .shuffles
                .iter()
                .map(|shuffle| {
                    shuffle.commit_product(
                        pk,
                        params,
                        domain,
                        theta,
                        gamma,
                        &advice.advice_polys,
                        &pk.fixed_values,
                        &instance.instance_values,
                        &challenges,
                        &mut rng,
                        transcript,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Commit to the vanishing argument's random polynomial for blinding h(x_3)
    let vanishing = vanishing::Argument::commit(params, domain, &mut rng, transcript)?;

    // Obtain challenge for keeping all separate gates linearly independent
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    let advice: Vec<AdviceSingle<Scheme::Curve, Coeff>> = advice
        .into_iter()
        .map(
            |AdviceSingle {
                 advice_polys,
                 advice_blinds,
             }| {
                AdviceSingle {
                    advice_polys: advice_polys
                        .into_iter()
                        .map(|poly| domain.lagrange_to_coeff(poly))
                        .collect::<Vec<_>>(),
                    advice_blinds,
                }
            },
        )
        .collect();

    let EvaluateHOutput {
        host_poly: h_poly,
        device_values: h_device_values,
    } = {
        let _evaluate_scope = proof_scope.child("evaluation::evaluate_h");
        pk.ev.evaluate_h(
            pk,
            &advice
                .iter()
                .map(|a| a.advice_polys.as_slice())
                .collect::<Vec<_>>(),
            &instance
                .iter()
                .map(|i| i.instance_polys.as_slice())
                .collect::<Vec<_>>(),
            &challenges,
            *y,
            *beta,
            *gamma,
            *theta,
            &lookups,
            &shuffles,
            &permutations,
        )
    };

    let post_eval_start = Instant::now();
    log::info!("post-evaluate_h phase: begin");
    let vanishing = vanishing.construct(
        params,
        domain,
        h_poly,
        Some(h_device_values),
        &mut rng,
        transcript,
    )?;
    log::info!(
        "post-evaluate_h phase: vanishing::construct completed in {:.3?}",
        post_eval_start.elapsed()
    );

    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let xn = x.pow([params.n()]);

    if P::QUERY_INSTANCE {
        // Compute and hash instance evals for each circuit instance
        let instance_eval_start = Instant::now();
        for instance in instance.iter() {
            // Evaluate polynomials at omega^i x
            let instance_evals: Vec<_> = meta
                .instance_queries
                .iter()
                .map(|&(column, at)| {
                    eval_polynomial(
                        &instance.instance_polys[column.index()],
                        domain.rotate_omega(*x, at),
                    )
                })
                .collect();

            // Hash each instance column evaluation
            for eval in instance_evals.iter() {
                transcript.write_scalar(*eval)?;
            }
        }
        log::info!(
            "post-evaluate_h phase: instance evaluations completed in {:.3?}",
            instance_eval_start.elapsed()
        );
    }

    // Compute and hash advice evals for each circuit instance
    let advice_eval_start = Instant::now();
    for advice in advice.iter() {
        // Evaluate polynomials at omega^i x
        let advice_evals: Vec<_> = meta
            .advice_queries
            .par_iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &advice.advice_polys[column.index()],
                    domain.rotate_omega(*x, at),
                )
            })
            .collect();

        // Hash each advice column evaluation
        for eval in advice_evals.iter() {
            transcript.write_scalar(*eval)?;
        }
    }
    log::info!(
        "post-evaluate_h phase: advice evaluations completed in {:.3?}",
        advice_eval_start.elapsed()
    );

    // Compute and hash fixed evals (shared across all circuit instances)
    let fixed_eval_start = Instant::now();
    let fixed_evals: Vec<_> = meta
        .fixed_queries
        .par_iter()
        .map(|&(column, at)| {
            eval_polynomial(&pk.fixed_polys[column.index()], domain.rotate_omega(*x, at))
        })
        .collect();
    log::info!(
        "post-evaluate_h phase: fixed evaluations completed in {:.3?}",
        fixed_eval_start.elapsed()
    );
    // Hash each fixed column evaluation
    for eval in fixed_evals.iter() {
        transcript.write_scalar(*eval)?;
    }

    let vanishing_eval_start = Instant::now();
    let vanishing = vanishing.evaluate(x, xn, domain, transcript)?;
    log::info!(
        "post-evaluate_h phase: vanishing::evaluate completed in {:.3?}",
        vanishing_eval_start.elapsed()
    );

    // Evaluate common permutation data
    let permutation_eval_start = Instant::now();
    pk.permutation.evaluate(x, transcript)?;
    log::info!(
        "post-evaluate_h phase: permutation common evaluation completed in {:.3?}",
        permutation_eval_start.elapsed()
    );

    // Evaluate the permutations, if any, at omega^i x.
    let permutations_eval_start = Instant::now();
    let permutations: Vec<permutation::prover::Evaluated<Scheme::Curve>> = permutations
        .into_iter()
        .map(|permutation| -> Result<_, _> { permutation.construct().evaluate(pk, x, transcript) })
        .collect::<Result<Vec<_>, _>>()?;
    log::info!(
        "post-evaluate_h phase: permutation evaluations completed in {:.3?}",
        permutations_eval_start.elapsed()
    );

    // Evaluate the lookups, if any, at omega^i x.

    let lookups_eval_start = Instant::now();
    let lookups: Vec<Vec<lookup::prover::Evaluated<Scheme::Curve>>> = lookups
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|p| {
                    #[cfg(not(feature = "mv-lookup"))]
                    let res = { p.evaluate(pk, x, transcript) };
                    #[cfg(feature = "mv-lookup")]
                    let res = { p.evaluate(&pk.vk, x, transcript) };
                    res
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::info!(
        "post-evaluate_h phase: lookup evaluations completed in {:.3?}",
        lookups_eval_start.elapsed()
    );

    // Evaluate the shuffles, if any, at omega^i x.
    let shuffles_eval_start = Instant::now();
    let shuffles: Vec<Vec<shuffle::prover::Evaluated<Scheme::Curve>>> = shuffles
        .into_iter()
        .map(|shuffles| -> Result<Vec<_>, _> {
            shuffles
                .into_iter()
                .map(|p| p.evaluate(pk, x, transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    log::info!(
        "post-evaluate_h phase: shuffle evaluations completed in {:.3?}",
        shuffles_eval_start.elapsed()
    );

    let instances = instance
        .iter()
        .zip(advice.iter())
        .zip(permutations.iter())
        .zip(lookups.iter())
        .zip(shuffles.iter())
        .flat_map(|((((instance, advice), permutation), lookups), shuffles)| {
            iter::empty()
                .chain(
                    P::QUERY_INSTANCE
                        .then_some(pk.vk.cs.instance_queries.iter().map(move |&(column, at)| {
                            ProverQuery {
                                point: domain.rotate_omega(*x, at),
                                poly: &instance.instance_polys[column.index()],
                                blind: Blind::default(),
                            }
                        }))
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    pk.vk
                        .cs
                        .advice_queries
                        .iter()
                        .map(move |&(column, at)| ProverQuery {
                            point: domain.rotate_omega(*x, at),
                            poly: &advice.advice_polys[column.index()],
                            blind: advice.advice_blinds[column.index()],
                        }),
                )
                .chain(permutation.open(pk, x))
                .chain(lookups.iter().flat_map(move |p| p.open(pk, x)))
                .chain(shuffles.iter().flat_map(move |p| p.open(pk, x)))
        })
        .chain(
            pk.vk
                .cs
                .fixed_queries
                .iter()
                .map(|&(column, at)| ProverQuery {
                    point: domain.rotate_omega(*x, at),
                    poly: &pk.fixed_polys[column.index()],
                    blind: Blind::default(),
                }),
        )
        .chain(pk.permutation.open(x))
        // We query the h(X) polynomial at x
        .chain(vanishing.open(x));

    #[cfg(feature = "counter")]
    {
        use crate::{FFT_COUNTER, MSM_COUNTER};
        use std::collections::BTreeMap;
        log::debug!("MSM_COUNTER: {:?}", MSM_COUNTER.lock().unwrap());
        log::debug!("FFT_COUNTER: {:?}", *FFT_COUNTER.lock().unwrap());

        // reset counters at the end of the proving
        *MSM_COUNTER.lock().unwrap() = BTreeMap::new();
        *FFT_COUNTER.lock().unwrap() = BTreeMap::new();
    }

    let prover = P::new(params);
    let multi_open_start = Instant::now();
    prover
        .create_proof(rng, transcript, instances)
        .map_err(|_| Error::ConstraintSystemFailure)
        .map(|res| {
            log::info!(
                "post-evaluate_h phase: multi-open proof generation completed in {:.3?}",
                multi_open_start.elapsed()
            );
            log::info!(
                "post-evaluate_h phase: total tail completed in {:.3?}",
                post_eval_start.elapsed()
            );
            res
        })
}

#[test]
fn test_create_proof() {
    use crate::{
        circuit::SimpleFloorPlanner,
        plonk::{keygen_pk, keygen_vk},
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::ProverSHPLONK,
        },
        transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
    };
    use halo2curves::bn256::Bn256;
    use rand_core::OsRng;

    #[derive(Clone, Copy)]
    struct MyCircuit;

    impl<F: Field> Circuit<F> for MyCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            *self
        }

        fn configure(_meta: &mut ConstraintSystem<F>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl crate::circuit::Layouter<F>,
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    let params: ParamsKZG<Bn256> = ParamsKZG::setup(3, OsRng);
    let vk = keygen_vk(&params, &MyCircuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &MyCircuit).expect("keygen_pk should not fail");
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);

    // Create proof with wrong number of instances
    let proof = create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[],
        OsRng,
        &mut transcript,
    );
    assert!(matches!(proof.unwrap_err(), Error::InvalidInstances));

    // Create proof with correct number of instances
    create_proof::<KZGCommitmentScheme<_>, ProverSHPLONK<_>, _, _, _, _>(
        &params,
        &pk,
        &[MyCircuit, MyCircuit],
        &[&[], &[]],
        OsRng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
}
