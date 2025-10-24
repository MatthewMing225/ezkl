use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeTheta, ChallengeX, Error, ProvingKey, VerifyingKey,
};
use super::Argument;
use crate::helpers::SerdeCurveAffine;
use crate::icicle::device_vec_from_c_scalars;
use crate::plonk::evaluation::evaluate;
use crate::SerdeFormat;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    icicle::{
        c_scalars_from_device_vec, inplace_add, inplace_invert, inplace_mul, inplace_scalar_add,
        inplace_sub,
    },
    poly::{
        commitment::{Blind, Params},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::{ff::Field, Curve};
use icicle_core::traits::FieldImpl;
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};

use icicle_bn254::curve::ScalarField;
use rustc_hash::FxHashMap as HashMap;

use maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    borrow::Borrow,
    cell::RefCell,
    hash::{Hash, Hasher},
    iter,
    ops::{Mul, MulAssign},
    slice,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

const OOR_LOG_LIMIT: usize = 16;
static OOR_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
static OOR_TOTAL_COUNT: AtomicUsize = AtomicUsize::new(0);
static OOR_DETECTED: AtomicBool = AtomicBool::new(false);

struct ScalarKey<F>
where
    F: PrimeField,
{
    repr: F::Repr,
}

impl<F> Clone for ScalarKey<F>
where
    F: PrimeField,
    F::Repr: Clone,
{
    fn clone(&self) -> Self {
        ScalarKey {
            repr: self.repr.clone(),
        }
    }
}

impl<F> ScalarKey<F>
where
    F: PrimeField,
{
    fn from_scalar(value: &F) -> Self {
        ScalarKey {
            repr: value.to_repr(),
        }
    }

    fn as_bytes(&self) -> &[u8]
    where
        F::Repr: AsRef<[u8]>,
    {
        self.repr.as_ref()
    }
}

impl<F> PartialEq for ScalarKey<F>
where
    F: PrimeField,
    F::Repr: AsRef<[u8]>,
{
    fn eq(&self, other: &Self) -> bool {
        self.repr.as_ref() == other.repr.as_ref()
    }
}

impl<F> Eq for ScalarKey<F>
where
    F: PrimeField,
    F::Repr: AsRef<[u8]>,
{
}

impl<F> Hash for ScalarKey<F>
where
    F: PrimeField,
    F::Repr: AsRef<[u8]>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.repr.as_ref());
    }
}

impl<F> Borrow<[u8]> for ScalarKey<F>
where
    F: PrimeField,
    F::Repr: AsRef<[u8]>,
{
    fn borrow(&self) -> &[u8] {
        self.repr.as_ref()
    }
}

struct GrandSumWorkspace {
    accumulator: Option<DeviceVec<ScalarField>>,
    temp: Option<DeviceVec<ScalarField>>,
    m_values: Option<DeviceVec<ScalarField>>,
    table: Option<DeviceVec<ScalarField>>,
    zero_host: Vec<ScalarField>,
    capacity: usize,
}

impl GrandSumWorkspace {
    fn new() -> Self {
        GrandSumWorkspace {
            accumulator: None,
            temp: None,
            m_values: None,
            table: None,
            zero_host: Vec::new(),
            capacity: 0,
        }
    }

    fn ensure_capacity(&mut self, len: usize) {
        if self.capacity >= len {
            return;
        }

        self.accumulator = Some(DeviceVec::device_malloc(len).expect("allocate accumulator"));
        self.temp = Some(DeviceVec::device_malloc(len).expect("allocate temp buffer"));
        self.m_values = Some(DeviceVec::device_malloc(len).expect("allocate m buffer"));
        self.table = Some(DeviceVec::device_malloc(len).expect("allocate table buffer"));
        self.zero_host.resize(len, ScalarField::zero());
        self.capacity = len;
    }

    fn zero_accumulator(&mut self, stream: &IcicleStream, len: usize) {
        let host_slice =
            unsafe { HostSlice::from_slice(slice::from_raw_parts(self.zero_host.as_ptr(), len)) };
        self.accumulator
            .as_mut()
            .expect("grand-sum accumulator not initialized")
            .copy_from_host_async(host_slice, stream)
            .expect("zero accumulator");
    }

    fn copy_polynomial<C: CurveAffine>(
        stream: &IcicleStream,
        dest: &mut DeviceVec<ScalarField>,
        poly: &Polynomial<C::Scalar, LagrangeCoeff>,
    ) {
        let values: &[C::Scalar] = poly.as_ref();
        let icicle_values: &[ScalarField] =
            unsafe { slice::from_raw_parts(values.as_ptr() as *const ScalarField, values.len()) };
        let host_slice = HostSlice::from_slice(icicle_values);
        dest.copy_from_host_async(host_slice, stream)
            .expect("copy polynomial to device");
    }
}
thread_local! {
    static GRAND_SUM_WORKSPACE: RefCell<GrandSumWorkspace> = RefCell::new(GrandSumWorkspace::new());
    static GRAND_SUM_STREAM: RefCell<IcicleStream> = RefCell::new(IcicleStream::create().expect("failed to create mv-lookup stream"));
}

#[derive(Debug)]
pub(in crate::plonk) struct Prepared<C: CurveAffine> {
    compressed_inputs_expressions: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    m_values: Polynomial<C::Scalar, LagrangeCoeff>,
    pub(in crate::plonk) commitment: C,
}

pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) m_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) phi_poly: Polynomial<C::Scalar, Coeff>,
    pub(in crate::plonk) commitment: C,
}

impl<C: SerdeCurveAffine> Committed<C> {
    #[allow(dead_code)]
    pub fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        format: SerdeFormat,
    ) -> std::io::Result<()>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
    {
        self.m_poly.write(writer, format)?;
        self.phi_poly.write(writer, format)?;
        self.commitment.write(writer, format)
    }

    #[allow(dead_code)]
    pub fn read<R: std::io::Read>(reader: &mut R, format: SerdeFormat) -> std::io::Result<Self>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
    {
        let m_poly = Polynomial::read(reader, format)?;
        let phi_poly = Polynomial::read(reader, format)?;
        let commitment = C::read(reader, format)?;

        Ok(Committed {
            m_poly,
            phi_poly,
            commitment,
        })
    }
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    pub(in crate::plonk) fn prepare<'a, 'params: 'a, C, P: Params<'params, C>>(
        &self,
        vk: &VerifyingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
    ) -> Result<Prepared<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        F: PrimeField,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
        F::Repr: Clone + AsRef<[u8]> + Send + Sync,
    {
        let n = params.n() as usize;
        OOR_LOG_COUNT.store(0, Ordering::Relaxed);
        OOR_TOTAL_COUNT.store(0, Ordering::Relaxed);
        OOR_DETECTED.store(false, Ordering::Relaxed);
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        n,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        let start = instant::Instant::now();
        log::info!(
            "mv_lookup::prepare: evaluating {} compressed input expressions",
            self.inputs_expressions.len()
        );
        let compressed_inputs_expressions: Vec<_> = self
            .inputs_expressions
            .par_iter()
            .map(|input_expressions| compress_expressions(input_expressions))
            .collect();
        log::info!(
            "mv_lookup::prepare: compressed input expressions computed in {:.3?}",
            start.elapsed()
        );

        // Get values of table expressions involved in the lookup and compress them
        let start = instant::Instant::now();
        let compressed_table_expression = compress_expressions(&self.table_expressions);
        log::info!(
            "mv_lookup::prepare: compressed table expression computed in {:.3?}",
            start.elapsed()
        );

        let blinding_factors = vk.cs.blinding_factors();

        let chunk_size = n - blinding_factors - 1;

        // compute m(X)
        let start = instant::Instant::now();
        log::info!("mv_lookup::prepare: building table index mapping");
        let table_index_value_mapping: HashMap<ScalarKey<C::Scalar>, usize> =
            compressed_table_expression
                .par_iter()
                .take(chunk_size)
                .enumerate()
                .map(|(i, &x)| (ScalarKey::from_scalar(&x), i))
                .collect();
        log::info!(
            "mv_lookup::prepare: table index mapping built in {:.3?}",
            start.elapsed()
        );

        let start = instant::Instant::now();
        log::info!("mv_lookup::prepare: accumulating m(X)");
        let mut multiplicities: Vec<u64> = compressed_inputs_expressions
            .par_iter()
            .map(|compressed_input_expression| {
                let mut local_counts = vec![0u64; chunk_size];
                compressed_input_expression
                    .iter()
                    .take(chunk_size)
                    .for_each(|fi| {
                        let key = ScalarKey::from_scalar(fi);
                        if let Some(&index) = table_index_value_mapping.get(&key) {
                            local_counts[index] += 1;
                        } else {
                            let logged = OOR_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                            OOR_TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
                            OOR_DETECTED.store(true, Ordering::Relaxed);
                            if logged < OOR_LOG_LIMIT {
                                log::error!(
                                    "value is OOR of lookup (sample {}): {:?}",
                                    logged + 1,
                                    key.as_bytes()
                                );
                            }
                        }
                    });
                local_counts
            })
            .reduce(
                || vec![0u64; chunk_size],
                |mut acc, local_counts| {
                    acc.iter_mut()
                        .zip(local_counts.into_iter())
                        .for_each(|(acc, count)| *acc += count);
                    acc
                },
            );

        multiplicities.resize(params.n() as usize, 0);

        if OOR_DETECTED.load(Ordering::Relaxed) {
            let total_missed = OOR_TOTAL_COUNT.load(Ordering::Relaxed);
            log::error!(
                "{} lookup value(s) were missing from the table during mv-lookup preparation",
                total_missed
            );
            return Err(Error::ConstraintSystemFailure);
        }

        let m_values: Vec<F> = multiplicities.into_par_iter().map(F::from).collect();
        log::info!(
            "mv_lookup::prepare: m(X) accumulation completed in {:.3?}",
            start.elapsed()
        );
        let m_values = vk.domain.lagrange_from_vec(m_values);

        #[cfg(feature = "sanity-checks")]
        {
            // check that m is zero after blinders
            let invalid_ms = m_values
                .iter()
                .skip(params.n() as usize - blinding_factors)
                .collect::<Vec<_>>();
            assert_eq!(invalid_ms.len(), blinding_factors);
            for mi in invalid_ms {
                assert_eq!(*mi, C::Scalar::ZERO);
            }

            // check sums
            let alpha = C::Scalar::random(&mut rng);
            let cs_input_sum =
                |compressed_input_expression: &Polynomial<C::Scalar, LagrangeCoeff>| {
                    let mut lhs_sum = C::Scalar::ZERO;
                    for &fi in compressed_input_expression
                        .iter()
                        .take(params.n() as usize - blinding_factors - 1)
                    {
                        lhs_sum += (fi + alpha).invert().unwrap();
                    }

                    lhs_sum
                };

            let mut lhs_sum = C::Scalar::ZERO;

            for compressed_input_expression in compressed_inputs_expressions.iter() {
                lhs_sum += cs_input_sum(compressed_input_expression);
            }

            let mut rhs_sum = C::Scalar::ZERO;
            for (&ti, &mi) in compressed_table_expression.iter().zip(m_values.iter()) {
                rhs_sum += mi * (ti + alpha).invert().unwrap();
            }

            assert_eq!(lhs_sum, rhs_sum);
        }

        // commit to m(X)
        let start = instant::Instant::now();
        let m_commitment = params
            .commit_lagrange(&m_values, Blind::default())
            .to_affine();
        log::info!(
            "mv_lookup::prepare: m_commitment completed in {:.3?}",
            start.elapsed()
        );

        // write commitment of m(X) to transcript
        // transcript.write_point(m_commitment)?;

        Ok(Prepared {
            compressed_inputs_expressions,
            compressed_table_expression,
            m_values,
            commitment: m_commitment,
        })
    }
}

impl<C: CurveAffine> Prepared<C> {
    pub(in crate::plonk) fn commit_grand_sum<'params, P: Params<'params, C>>(
        self,
        vk: &VerifyingKey<C>,
        params: &P,
        beta: ChallengeBeta<C>,
        phi_blinds: &[C::Scalar],
    ) -> Result<Committed<C>, Error> {
        /*
            φ_i(X) = f_i(X) + α
            τ(X) = t(X) + α
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */

        let total_start = instant::Instant::now();
        let log_derivative_start = instant::Instant::now();
        log::info!("mv_lookup::commit_grand_sum: start");

        let Prepared {
            compressed_inputs_expressions,
            compressed_table_expression,
            m_values,
            commitment: _,
        } = self;

        let n = params.n() as usize;

        let committed = GRAND_SUM_STREAM.with(|stream_cell| {
            GRAND_SUM_WORKSPACE.with(|workspace_cell| {
                let stream = stream_cell.borrow_mut();
                let stream_ref = &*stream;
                let mut workspace = workspace_cell.borrow_mut();
                workspace.ensure_capacity(n);
                workspace.zero_accumulator(stream_ref, n);

                let beta_device = device_vec_from_c_scalars(&[*beta], stream_ref);

                let mut accumulator_buf = workspace
                    .accumulator
                    .take()
                    .expect("grand-sum accumulator not initialized");
                let mut temp_buf = workspace
                    .temp
                    .take()
                    .expect("grand-sum temp buffer not initialized");
                let mut m_values_buf = workspace
                    .m_values
                    .take()
                    .expect("grand-sum m buffer not initialized");
                let mut table_buf = workspace
                    .table
                    .take()
                    .expect("grand-sum table buffer not initialized");

                GrandSumWorkspace::copy_polynomial::<C>(stream_ref, &mut m_values_buf, &m_values);

                GrandSumWorkspace::copy_polynomial::<C>(
                    stream_ref,
                    &mut table_buf,
                    &compressed_table_expression,
                );

                for compressed_input_expression in &compressed_inputs_expressions {
                    GrandSumWorkspace::copy_polynomial::<C>(
                        stream_ref,
                        &mut temp_buf,
                        compressed_input_expression,
                    );
                    inplace_scalar_add(&mut temp_buf, &beta_device, stream_ref);
                    inplace_invert(&mut temp_buf, stream_ref);
                    inplace_add(&mut accumulator_buf, &temp_buf, stream_ref);
                }

                inplace_scalar_add(&mut table_buf, &beta_device, stream_ref);
                inplace_invert(&mut table_buf, stream_ref);

                inplace_mul(&mut m_values_buf, &mut table_buf, stream_ref);
                inplace_sub(&mut accumulator_buf, &mut m_values_buf, stream_ref);

                let log_derivatives_diff: Vec<C::Scalar> =
                    c_scalars_from_device_vec(&mut accumulator_buf, stream_ref);

                log::info!(
                    "mv_lookup::commit_grand_sum: log-derivatives computed in {:.3?}",
                    log_derivative_start.elapsed()
                );

                let phi_start = instant::Instant::now();
                let blinding_factors = vk.cs.blinding_factors();

                assert!(
                    phi_blinds.len() == blinding_factors,
                    "invalid number of blinding factors"
                );

                let phi_values = iter::once(C::Scalar::ZERO)
                    .chain(log_derivatives_diff)
                    .scan(C::Scalar::ZERO, |state, cur| {
                        *state += &cur;
                        Some(*state)
                    })
                    .take(n - blinding_factors)
                    .chain(phi_blinds.iter().copied())
                    .collect::<Vec<_>>();
                assert_eq!(phi_values.len(), n);

                #[cfg(feature = "sanity-checks")]
                let phi_for_checks = phi_values.clone();

                let phi_lagrange = vk.domain.lagrange_from_vec(phi_values);

                log::info!(
                    "mv_lookup::commit_grand_sum: phi constructed in {:.3?}",
                    phi_start.elapsed()
                );

                #[cfg(feature = "sanity-checks")]
                // This test works only with intermediate representations in this method.
                // It can be used for debugging purposes.
                {
                    // While in Lagrange basis, check that product is correctly constructed
                    let u = n - (blinding_factors + 1);

                    /*
                        φ_i(X) = f_i(X) + α
                        τ(X) = t(X) + α
                        LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                        RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                    */

                    // q(X) = LHS - RHS mod zH(X)
                    for i in 0..u {
                        // Π(φ_i(X))
                        let fi_prod = || {
                            let mut prod = C::Scalar::ONE;
                            for compressed_input_expression in compressed_inputs_expressions.iter()
                            {
                                prod *= *beta + compressed_input_expression[i];
                            }

                            prod
                        };

                        let fi_log_derivative = || {
                            let mut sum = C::Scalar::ZERO;
                            for compressed_input_expression in compressed_inputs_expressions.iter()
                            {
                                sum += (*beta + compressed_input_expression[i]).invert().unwrap();
                            }

                            sum
                        };

                        // LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                        let lhs = {
                            (*beta + compressed_table_expression[i])
                                * fi_prod()
                                * (phi_for_checks[i + 1] - phi_for_checks[i])
                        };

                        // RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                        let rhs = {
                            (*beta + compressed_table_expression[i])
                                * fi_prod()
                                * (fi_log_derivative()
                                    - m_values[i]
                                        * (*beta + compressed_table_expression[i])
                                            .invert()
                                            .unwrap())
                        };

                        assert_eq!(lhs - rhs, C::Scalar::ZERO);
                    }

                    assert_eq!(phi_for_checks[u], C::Scalar::ZERO);
                }

                let grand_sum_blind = Blind(C::Scalar::ZERO);
                let commitment_start = instant::Instant::now();
                let phi_commitment = params
                    .commit_lagrange_with_stream(&phi_lagrange, grand_sum_blind, stream_ref)
                    .to_affine();
                log::info!(
                    "mv_lookup::commit_grand_sum: phi commitment in {:.3?}",
                    commitment_start.elapsed()
                );

                let m_poly = vk.domain.lagrange_to_coeff_stream(m_values, stream_ref);
                let phi_poly = vk.domain.lagrange_to_coeff_stream(phi_lagrange, stream_ref);

                stream_ref.synchronize().unwrap();

                workspace.accumulator = Some(accumulator_buf);
                workspace.temp = Some(temp_buf);
                workspace.m_values = Some(m_values_buf);
                workspace.table = Some(table_buf);

                Committed {
                    m_poly,
                    phi_poly,
                    commitment: phi_commitment,
                }
            })
        });

        log::info!(
            "mv_lookup::commit_grand_sum: completed in {:.3?}",
            total_start.elapsed()
        );

        Ok(committed)
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        vk: &VerifyingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let phi_eval = eval_polynomial(&self.phi_poly, *x);
        let phi_next_eval = eval_polynomial(&self.phi_poly, x_next);
        let m_eval = eval_polynomial(&self.m_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(phi_eval))
            .chain(Some(phi_next_eval))
            .chain(Some(m_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.phi_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.m_poly,
                blind: Blind(C::Scalar::ZERO),
            }))
    }
}
