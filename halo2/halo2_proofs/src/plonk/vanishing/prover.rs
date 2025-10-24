use ff::Field;
use group::Curve;
use log::{debug, warn};
use maybe_rayon::iter::IndexedParallelIterator;
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rustc_hash::FxHashMap as HashMap;
use std::{env, iter};

use icicle_bn254::curve::ScalarField;
use icicle_core::{
    ntt::{ntt_inplace, NTTConfig, NTTDir},
    vec_ops::{mul_scalars, VecOpsConfig},
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice},
    stream::IcicleStream,
};

use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    icicle::{c_scalars_from_device_vec, device_vec_from_c_scalars},
    multicore::current_num_threads,
    plonk::{ChallengeX, Error},
    poly::{
        commitment::{Blind, ParamsProver},
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial, ProverQuery,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(in crate::plonk) struct Committed<C: CurveAffine> {
    random_poly: Polynomial<C::Scalar, Coeff>,
    random_blind: Blind<C::Scalar>,
}

#[allow(dead_code)]
pub(crate) struct VanishingDeviceData {
    pub quotient: DeviceVec<ScalarField>,
    pub piece_len: usize,
    pub num_pieces: usize,
}

pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    h_pieces: Vec<Polynomial<C::Scalar, Coeff>>,
    h_blinds: Vec<Blind<C::Scalar>>,
    device: Option<VanishingDeviceData>,
    committed: Committed<C>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    h_poly: Polynomial<C::Scalar, Coeff>,
    h_blind: Blind<C::Scalar>,
    #[allow(dead_code)]
    device: Option<VanishingDeviceData>,
    committed: Committed<C>,
}

impl<C: CurveAffine> Argument<C> {
    pub(in crate::plonk) fn commit<
        'params,
        P: ParamsProver<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        // Sample a random polynomial of degree n - 1
        let n = 1usize << domain.k() as usize;
        let mut rand_vec = vec![C::Scalar::ZERO; n];

        let num_threads = current_num_threads();
        let chunk_size = n / num_threads;
        let thread_seeds = (0..)
            .step_by(chunk_size + 1)
            .take(n % num_threads)
            .chain(
                (chunk_size != 0)
                    .then(|| ((n % num_threads) * (chunk_size + 1)..).step_by(chunk_size))
                    .into_iter()
                    .flatten(),
            )
            .take(num_threads)
            .zip(iter::repeat_with(|| {
                let mut seed = [0u8; 32];
                rng.fill_bytes(&mut seed);
                ChaCha20Rng::from_seed(seed)
            }))
            .collect::<HashMap<_, _>>();

        parallelize(&mut rand_vec, |chunk, offset| {
            let mut rng = thread_seeds[&offset].clone();
            chunk
                .iter_mut()
                .for_each(|v| *v = C::Scalar::random(&mut rng));
        });

        let random_poly: Polynomial<C::Scalar, Coeff> = domain.coeff_from_vec(rand_vec);

        // Sample a random blinding factor
        let random_blind = Blind(C::Scalar::random(rng));

        // Commit
        let c = params.commit(&random_poly, random_blind).to_affine();
        transcript.write_point(c)?;

        Ok(Committed {
            random_poly,
            random_blind,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    fn reset_active_device() {
        if let Ok(device_type) = env::var("ICICLE_DEVICE_TYPE") {
            let device = icicle_runtime::Device::new(device_type.as_str(), 0);
            if let Err(err) = icicle_runtime::runtime::set_device(&device) {
                warn!(
                    "vanishing::construct: failed to reset active device {}: {:?}",
                    device_type, err
                );
            }
        }
    }

    fn construct_host_pieces<'params, P: ParamsProver<'params, C> + Send + Sync>(
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        h_poly: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    ) -> Vec<Polynomial<C::Scalar, Coeff>> {
        let h_poly = domain.divide_by_vanishing_poly(h_poly);
        let h_poly = domain.extended_to_coeff(h_poly);

        h_poly
            .chunks_exact(params.n() as usize)
            .map(|v| domain.coeff_from_vec(v.to_vec()))
            .collect::<Vec<_>>()
    }

    fn try_construct_with_device<'params, P: ParamsProver<'params, C> + Send + Sync>(
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        mut device_values: DeviceVec<ScalarField>,
    ) -> Result<(Vec<Polynomial<C::Scalar, Coeff>>, VanishingDeviceData), ()> {
        let enable_gpu = env::var("EZKL_ENABLE_VANISHING_GPU")
            .map(|val| {
                let val_lower = val.to_ascii_lowercase();
                val_lower == "1" || val_lower == "true" || val_lower == "yes"
            })
            .unwrap_or(false);
        if !enable_gpu {
            debug!("vanishing::construct: device quotient path disabled (set EZKL_ENABLE_VANISHING_GPU=1 to enable)");
            drop(device_values);
            return Err(());
        }

        let stream = IcicleStream::default();
        match icicle_runtime::runtime::get_active_device() {
            Ok(device) => {
                debug!(
                    "vanishing::construct: active device before GPU path: {}:{}",
                    device.get_device_type(),
                    device.id
                );
            }
            Err(err) => {
                warn!(
                    "vanishing::construct: failed to fetch active device before GPU path: {:?}",
                    err
                );
            }
        }

        let result = (|| -> Result<_, ()> {
            let device_slice = &device_values[..];
            debug!(
                "vanishing::construct: h(X) device buffer len={}, on_device={}, on_active={}",
                device_slice.len(),
                device_slice.is_on_device(),
                device_slice.is_on_active_device()
            );

            let vanishing_factors = domain.repeated_vanishing_evals();
            if !vanishing_factors.is_empty() {
                let vanishing_device =
                    device_vec_from_c_scalars::<C::Scalar>(&vanishing_factors, &stream);
                let vanishing_slice = &vanishing_device[..];
                debug!(
                    "vanishing::construct: vanishing factors len={}, on_device={}, on_active={}",
                    vanishing_slice.len(),
                    vanishing_slice.is_on_device(),
                    vanishing_slice.is_on_active_device()
                );
                let mut cfg = VecOpsConfig::default();
                cfg.stream_handle = (&stream).into();
                cfg.is_async = false;

                let values_slice_mut = device_values.as_mut_slice();
                let len = values_slice_mut.len();
                let values_slice_const = unsafe {
                    DeviceSlice::from_slice(std::slice::from_raw_parts(
                        values_slice_mut.as_ptr(),
                        len,
                    ))
                };

                match mul_scalars(
                    values_slice_const,
                    &vanishing_device[..],
                    values_slice_mut,
                    &cfg,
                ) {
                    Ok(_) => {
                        debug!("vanishing::construct: vanishing multiply completed (sync)");
                    }
                    Err(err) => {
                        warn!("vanishing::construct: vanishing multiply failed: {:?}", err);
                        Self::reset_active_device();
                        return Err(());
                    }
                }
            }
            stream.synchronize().map_err(|err| {
                warn!(
                    "vanishing::construct: synchronizing after vanishing multiplication failed: {:?}",
                    err
                );
                Self::reset_active_device();
            })?;

            let mut cfg = NTTConfig::<ScalarField>::default();
            cfg.stream_handle = (&stream).into();
            cfg.is_async = true;
            ntt_inplace::<ScalarField, ScalarField>(&mut device_values[..], NTTDir::kInverse, &cfg)
                .map_err(|err| {
                    warn!(
                        "vanishing::construct: inverse NTT on device failed: {:?}",
                        err
                    );
                    Self::reset_active_device();
                })?;

            let coset_scaling = domain.coset_scaling_factors(false);
            if !coset_scaling.is_empty() {
                let coset_device = device_vec_from_c_scalars::<C::Scalar>(&coset_scaling, &stream);
                let coset_slice = &coset_device[..];
                debug!(
                    "vanishing::construct: coset factors len={}, on_device={}, on_active={}",
                    coset_slice.len(),
                    coset_slice.is_on_device(),
                    coset_slice.is_on_active_device()
                );

                let mut cfg = VecOpsConfig::default();
                cfg.stream_handle = (&stream).into();
                cfg.is_async = false;

                let values_slice_mut = device_values.as_mut_slice();
                let len = values_slice_mut.len();
                let values_slice_const = unsafe {
                    DeviceSlice::from_slice(std::slice::from_raw_parts(
                        values_slice_mut.as_ptr(),
                        len,
                    ))
                };

                match mul_scalars(values_slice_const, coset_slice, values_slice_mut, &cfg) {
                    Ok(_) => {
                        debug!("vanishing::construct: coset multiply completed (sync)");
                    }
                    Err(err) => {
                        warn!("vanishing::construct: coset multiply failed: {:?}", err);
                        Self::reset_active_device();
                        return Err(());
                    }
                }
            }

            stream.synchronize().map_err(|err| {
                warn!(
                    "vanishing::construct: synchronizing after coset scaling failed: {:?}",
                    err
                );
                Self::reset_active_device();
            })?;

            let quotient_len = domain.quotient_poly_len();
            let mut host_coeffs =
                c_scalars_from_device_vec::<C::Scalar>(&mut device_values, &stream);
            host_coeffs.truncate(quotient_len);

            let h_pieces = host_coeffs
                .chunks_exact(params.n() as usize)
                .map(|chunk| domain.coeff_from_vec(chunk.to_vec()))
                .collect::<Vec<_>>();

            let num_pieces = h_pieces.len();

            let quotient_device = device_vec_from_c_scalars::<C::Scalar>(&host_coeffs, &stream);
            stream.synchronize().map_err(|err| {
                warn!(
                    "vanishing::construct: synchronizing after quotient upload failed: {:?}",
                    err
                );
                Self::reset_active_device();
            })?;

            Ok((
                h_pieces,
                VanishingDeviceData {
                    quotient: quotient_device,
                    piece_len: params.n() as usize,
                    num_pieces,
                },
            ))
        })();

        result
    }

    pub(in crate::plonk) fn construct<
        'params,
        P: ParamsProver<'params, C> + Send + Sync,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        h_poly: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
        h_poly_device: Option<DeviceVec<ScalarField>>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Constructed<C>, Error> {
        let (h_pieces, device) = match h_poly_device {
            Some(device_values) => {
                match Self::try_construct_with_device(params, domain, device_values) {
                    Ok((pieces, device_data)) => (pieces, Some(device_data)),
                    Err(_) => {
                        log::warn!("vanishing::construct: falling back to host path");
                        let pieces = Self::construct_host_pieces(params, domain, h_poly);
                        (pieces, None)
                    }
                }
            }
            None => {
                let pieces = Self::construct_host_pieces(params, domain, h_poly);
                (pieces, None)
            }
        };

        let h_blinds: Vec<_> = h_pieces
            .iter()
            .map(|_| Blind(C::Scalar::random(&mut rng)))
            .collect();

        // Compute commitments to each h(X) piece
        let h_commitments_projective: Vec<_> = h_pieces
            .par_iter()
            .zip(h_blinds.par_iter())
            .map(|(h_piece, blind)| params.commit(h_piece, *blind))
            .collect();
        let mut h_commitments = vec![C::identity(); h_commitments_projective.len()];
        C::Curve::batch_normalize(&h_commitments_projective, &mut h_commitments);
        let h_commitments = h_commitments;

        // Hash each h(X) piece
        for c in h_commitments.iter() {
            transcript.write_point(*c)?;
        }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            device,
            committed: self,
        })
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        x: ChallengeX<C>,
        xn: C::Scalar,
        domain: &EvaluationDomain<C::Scalar>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let h_poly = self
            .h_pieces
            .iter()
            .rev()
            .fold(domain.empty_coeff(), |acc, eval| acc * xn + eval);

        let h_blind = self
            .h_blinds
            .iter()
            .rev()
            .fold(Blind(C::Scalar::ZERO), |acc, eval| acc * Blind(xn) + *eval);

        let random_eval = eval_polynomial(&self.committed.random_poly, *x);
        transcript.write_scalar(random_eval)?;

        Ok(Evaluated {
            h_poly,
            h_blind,
            device: self.device,
            committed: self.committed,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open(
        &self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'_, C>> + Clone {
        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.h_poly,
                blind: self.h_blind,
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.committed.random_poly,
                blind: self.committed.random_blind,
            }))
    }
}
