use super::{
    construct_intermediate_sets, ChallengeU, ChallengeV, ChallengeY, Commitment, RotationSet,
};
use crate::arithmetic::{
    eval_polynomial, evaluate_vanishing_polynomial, kate_division_deg2, kate_division_deg3,
    kate_division_par, lagrange_interpolate, powers_of_x, vector_mul_scalar_inplace, CurveAffine,
};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::{Blind, ParamsProver, Prover};
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::{PolynomialPointer, ProverQuery};
use crate::poly::{Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use ff::Field;
use group::Curve;
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};
use maybe_rayon::prelude::IntoParallelIterator;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;

fn div_by_vanishing_par<F: Field>(poly: Polynomial<F, Coeff>, roots: &[F]) -> Vec<F> {
    roots.iter().fold(poly.values, |coeffs, root| {
        kate_division_par(&coeffs, *root)
    })
}

struct CommitmentExtension<'a, C: CurveAffine> {
    commitment: Commitment<C::Scalar, PolynomialPointer<'a, C>>,
    low_degree_equivalent: Polynomial<C::Scalar, Coeff>,
}

impl<'a, C: CurveAffine> Commitment<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(&self, points: &[C::Scalar]) -> CommitmentExtension<'a, C> {
        let poly = lagrange_interpolate(points, &self.evals()[..]);

        let low_degree_equivalent = Polynomial {
            values: poly,
            _marker: PhantomData,
        };

        CommitmentExtension {
            commitment: self.clone(),
            low_degree_equivalent,
        }
    }
}

struct RotationSetExtension<'a, C: CurveAffine> {
    commitments: Vec<CommitmentExtension<'a, C>>,
    points: Vec<C::Scalar>,
}

impl<'a, C: CurveAffine> RotationSet<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(self, commitments: Vec<CommitmentExtension<'a, C>>) -> RotationSetExtension<'a, C> {
        RotationSetExtension {
            commitments,
            points: self.points,
        }
    }
}

/// Concrete KZG prover with SHPLONK variant
#[derive(Debug)]
pub struct ProverSHPLONK<'a, E: Engine> {
    params: &'a ParamsKZG<E>,
}

impl<'a, E: Engine> ProverSHPLONK<'a, E> {
    /// Given parameters creates new prover instance
    pub fn new(params: &'a ParamsKZG<E>) -> Self {
        Self { params }
    }
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>>
    for ProverSHPLONK<'params, E>
where
    E::Fr: Ord,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    const QUERY_INSTANCE: bool = false;

    fn new(params: &'params ParamsKZG<E>) -> Self {
        Self { params }
    }

    /// Create a multi-opening proof
    fn create_proof<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, Ch>,
        R,
        I,
    >(
        &self,
        _: R,
        transcript: &mut T,
        queries: I,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = ProverQuery<'com, E::G1Affine>> + Clone,
        R: RngCore,
    {
        // TODO: explore if it is safe to use same challenge
        // for different sets that are already combined with another challenge
        let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

        let params_n = self.params.n as usize;
        let quotient_contribution =
            |rotation_set: &RotationSetExtension<E::G1Affine>| -> (Polynomial<E::Fr, Coeff>, Polynomial<E::Fr, Coeff>) {
                let count = rotation_set.commitments.len();
                debug_assert!(count > 0);

                let y_pows = powers_of_x(*y, count);

                let mut combined = rotation_set.commitments[0].commitment.get().poly.clone();
                combined.mul_inplace_scalar(y_pows[0]);
                if count > 1 {
                    let refs: Vec<&Polynomial<E::Fr, Coeff>> = rotation_set.commitments[1..]
                        .iter()
                        .map(|commitment| commitment.commitment.get().poly)
                        .collect();
                    combined.add_mul_scalars(&refs, &y_pows[1..]);
                }

                let mut numerator = combined.clone();
                for (weight, commitment) in y_pows.iter().zip(rotation_set.commitments.iter()) {
                    numerator.sub_low_poly_mul_scalar(&commitment.low_degree_equivalent, *weight);
                }

                let points = rotation_set.points.as_slice();
                let mut quotient_values = match points.len() {
                    2 => kate_division_deg2(numerator.values.iter(), points),
                    3 => kate_division_deg3(numerator.values.iter(), points),
                    _ => div_by_vanishing_par(numerator, points),
                };
                quotient_values.resize(params_n, E::Fr::ZERO);

                (
                    combined,
                    Polynomial {
                        values: quotient_values,
                        _marker: PhantomData,
                    },
                )
            };

        let intermediate_sets = construct_intermediate_sets(queries);
        let (rotation_sets, super_point_set) = (
            intermediate_sets.rotation_sets,
            intermediate_sets.super_point_set,
        );

        let rotation_sets: Vec<RotationSetExtension<E::G1Affine>> = rotation_sets
            .into_par_iter()
            .map(|rotation_set| {
                let commitments: Vec<CommitmentExtension<E::G1Affine>> = rotation_set
                    .commitments
                    .as_slice()
                    .into_par_iter()
                    .map(|commitment_data| commitment_data.extend(&rotation_set.points))
                    .collect();
                rotation_set.extend(commitments)
            })
            .collect();

        let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();

        let (comb_polys, quotient_polynomials): (Vec<_>, Vec<_>) = rotation_sets
            .as_slice()
            .into_par_iter()
            .map(quotient_contribution)
            .unzip();

        assert!(
            !quotient_polynomials.is_empty(),
            "expected at least one rotation set"
        );

        let v_pows = powers_of_x(*v, quotient_polynomials.len());

        let mut h_acc = quotient_polynomials[0].clone();
        h_acc.mul_inplace_scalar(v_pows[0]);
        if quotient_polynomials.len() > 1 {
            let refs: Vec<&Polynomial<E::Fr, Coeff>> = quotient_polynomials[1..].iter().collect();
            h_acc.add_mul_scalars(&refs, &v_pows[1..]);
        }

        let h = self.params.commit(&h_acc, Blind::default()).to_affine();
        transcript.write_point(h)?;
        let u: ChallengeU<_> = transcript.squeeze_challenge_scalar();

        let comb_polys_ref = &comb_polys;
        let super_point_set_ref = &super_point_set;
        let linearisation_contribution = move |index: usize,
                                               rotation_set: RotationSetExtension<E::G1Affine>|
              -> (Polynomial<E::Fr, Coeff>, E::Fr) {
            let mut diffs = (*super_point_set_ref).clone();
            for point in rotation_set.points.iter() {
                diffs.remove(point);
            }
            let diffs_vec = diffs.into_iter().collect::<Vec<_>>();
            let z_i = evaluate_vanishing_polynomial(&diffs_vec[..], *u);

            let ys = powers_of_x(*y, rotation_set.commitments.len());
            let evals: Vec<E::Fr> = rotation_set
                .commitments
                .iter()
                .map(|commitment| eval_polynomial(&commitment.low_degree_equivalent.values[..], *u))
                .collect();

            let mut l_acc = comb_polys_ref[index].clone();
            if let Some(constant) = l_acc.values.get_mut(0) {
                for (weight, eval) in ys.iter().zip(evals.iter()) {
                    *constant -= *weight * *eval;
                }
            }

            (l_acc * z_i, z_i)
        };

        #[allow(clippy::type_complexity)]
        let (linearisation_contributions, z_diffs): (
            Vec<Polynomial<E::Fr, Coeff>>,
            Vec<E::Fr>,
        ) = rotation_sets
            .into_par_iter()
            .enumerate()
            .map(|(i, rotation_set)| linearisation_contribution(i, rotation_set))
            .unzip();

        assert!(
            !linearisation_contributions.is_empty(),
            "expected at least one linearisation contribution"
        );

        let mut l_acc = linearisation_contributions[0].clone();
        l_acc.mul_inplace_scalar(v_pows[0]);
        if linearisation_contributions.len() > 1 {
            let refs: Vec<&Polynomial<E::Fr, Coeff>> =
                linearisation_contributions[1..].iter().collect();
            l_acc.add_mul_scalars(&refs, &v_pows[1..]);
        }

        let super_points = super_point_set.iter().cloned().collect::<Vec<_>>();
        let zt_eval = evaluate_vanishing_polynomial(&super_points[..], *u);
        let l_acc = l_acc - &(h_acc * zt_eval);

        // sanity check
        #[cfg(debug_assertions)]
        {
            let must_be_zero = eval_polynomial(&l_acc.values[..], *u);
            assert_eq!(must_be_zero, E::Fr::ZERO);
        }

        let mut h_vec = div_by_vanishing_par(l_acc, &[*u]);

        // normalize coefficients by the coefficient of the first polynomial
        assert!(!z_diffs.is_empty(), "expected at least one z difference");
        let z_0_diff_inv = z_diffs[0].invert().unwrap();
        vector_mul_scalar_inplace(&mut h_vec, z_0_diff_inv);

        let h_poly = Polynomial {
            values: h_vec,
            _marker: PhantomData,
        };

        let h = self.params.commit(&h_poly, Blind::default()).to_affine();
        transcript.write_point(h)?;

        Ok(())
    }
}
