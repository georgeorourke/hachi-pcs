use ark_ff::{AdditiveGroup, Field};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::fs::FS;
use crate::arithmetic::sumcheck::{Univariate, eq, eq_bin};
use crate::arithmetic::{ExtField, FieldExtension, Logarithm, MultiLinearCoeff, PChal, PVec, Poly, RandFromRng, powers};

use crate::hachi::Hachi;
use crate::hachi::common::form_m_alpha;
use crate::hachi::evaluate::EvaluationProofRound;
use crate::hachi::setup::Parameters;
#[cfg(feature = "verbose")]
use crate::utils::verbose::tick_item;

/// Verification function.
pub trait Verify<F> {
    /// The verificiation function for the multilinear polynomial.
    fn verify(
        params: &Parameters,
        x: &[u64],
        y: F,
        com: &PVec,
        proof: &EvaluationProofRound
    );
}

impl Verify<u64> for Hachi{
    #[time_graph::instrument]
    fn verify(
        params: &Parameters,
        x: &[u64],
        y: u64,
        com : &PVec,
        proof: &EvaluationProofRound
        ) {
            #[cfg(feature = "verbose")]
            println!("\n==== Verify ====");

            // initialise a FS stream and push the commitment
            let mut fs = FS::init();
            fs.push(com);

            // verify the reduction to Rq
            let log_d = params.d.log();
            let x_v = &x[params.l - log_d..params.l];
            let mut v = vec![0u64; params.d];

            for i in 0..params.d {
                v[i] = u64::multi_lin_coeff(x_v, i, log_d, params.q);
            }

            let ring_y = proof.y.slice();
            assert_eq!(v.len(), ring_y.len());

            let mut y_actual = 0;

            for i in 0..params.d {
                y_actual = (y_actual + ring_y[i] * v[i]) % params.q;
            }

            assert_eq!(y, y_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify reduction to Rq");

            // sample challenges
            fs.push(&proof.y);
            fs.push(&proof.v);
            let seed = fs.get_seed();
            let challenges = PChal::rand_vec(1 << params.r, params.d, params.k, seed);

            // sample alpha
            fs.push(&proof.u_dash);
            let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
            let alpha = ExtField::rand(0, &mut rng);

            // Get the powers of alpha
            let alpha_pows = powers(alpha, params.d);

            // Get [M | -(X^d+1).In] evaluated at alpha
            let m_alpha = form_m_alpha(params, x, &challenges, &alpha_pows);

            // Get lengths for tau_0 and tau_1
            let mu = 
            (1 << params.r) * params.delta                          // w_hat
            + params.n * (1 << params.r) * params.delta             // t_hat
            + (1 << params.m) * params.delta * params.delta_z;      // z_hat    
            
            // length of r part of new witness (height of M)
            let n = params.n * 3 + 2;

            // new witness must have power of 2 length, so we pad (mu + n) to next power of 2
            let num_vars = ((mu + n) * params.d).log();

            // Sample the random field elements tau_0 and tau_1
            let mut tau_0 = vec![ExtField::ZERO; num_vars];
            
            for i in 0..num_vars {
                tau_0[i] = ExtField::rand(0, &mut rng);
            }

            let log_n = (n as u64).log();
            let mut tau_1 = vec![ExtField::ZERO; log_n];
            
            for i in 0..log_n {
                tau_1[i] = ExtField::rand(0, &mut rng);
            }

            // Get expected sum for F_alpha
            let sum_f_alpha = compute_expected_sum_f_alpha(params, com, proof, &alpha_pows, &tau_1);

            // Verify sum check
            let (f_alpha_expected, f_0_expected, chals_f_alpha, chals_f_0) = sumcheck_verify(
                &proof.univariates_f_alpha, 
                &proof.univariates_f_0, 
                sum_f_alpha, 
                &mut fs
            );

            // Check evaluation of f_0
            // evaluate indicator polynomial
            let mut i_r = ExtField::ZERO;
            let mut prefix = ExtField::ONE;

            let len_z = mu * params.d;

            for i in (0..chals_f_0.len()).rev() {
                // i-th bit of mu
                let mu_i = (len_z >> i) & 1;

                if mu_i == 1 {
                    i_r += prefix * (ExtField::ONE - chals_f_0[i]);
                    prefix *= chals_f_0[i];
                }
                else {
                    prefix *= ExtField::ONE - chals_f_0[i];
                }
            }

            // evaluate eq
            let mut eq_r = ExtField::ONE;
            assert_eq!(tau_0.len(), chals_f_0.len());

            for i in 0..tau_0.len() {
                eq_r *= eq(tau_0[i], chals_f_0[i]);
            }

            // perform the multiplication v_r=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
            let mut v_r = proof.y_dash * (proof.y_dash + ExtField::lift_int(params.b / 2));
            
            // multiply with difference of two squares to improve efficiency
            let w_r_squared = proof.y_dash * proof.y_dash;

            for r in 1..params.b as usize / 2 { 
                v_r *= w_r_squared - ExtField::lift_int((r * r) as u64);
            }

            let f_0_actual = i_r * eq_r * v_r;
            assert_eq!(f_0_expected, f_0_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify sum check on F_0");
            
            // Check evaluation of f_alpha
            let mut eq_r = ExtField::ONE;

            for i in 0..tau_1.len() {
                eq_r *= eq(tau_1[i], chals_f_alpha[i]);
            }

            assert_eq!(1 << log_d, alpha_pows.len());
            let alpha_mle = DenseMultilinearExtension::from_evaluations_vec(log_d, alpha_pows);
            let alpha_r = alpha_mle.evaluate(&chals_f_alpha[chals_f_alpha.len() - log_d..chals_f_alpha.len()].to_vec());

            let m_vars = (m_alpha.len() as u64).log();
            assert_eq!(chals_f_alpha.len() - log_d, m_vars);
            let m_mle = DenseMultilinearExtension::from_evaluations_vec(m_vars, m_alpha);
            let m_r = m_mle.evaluate(&chals_f_alpha[0..m_vars].to_vec());

            let f_alpha_actual = proof.y_dash * alpha_r * eq_r * m_r;
            assert_eq!(f_alpha_expected, f_alpha_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify sum check on F_alpha");

            #[cfg(feature = "verbose")]
            println!("==== Complete ====\n");
    }
}

/// Compute the expected result of sum check for F_alpha
fn compute_expected_sum_f_alpha(
    params: &Parameters, 
    com: &PVec, 
    proof: &EvaluationProofRound, 
    alpha_pows: &[ExtField], tau_1: &[ExtField]
) -> ExtField {
    // evaluate Y=[v u y 0 0] at alpha
    let mut y_alpha = Vec::<ExtField>::new();

    for i in 0..proof.v.length() {
        y_alpha.push(proof.v.element(i).eval(alpha_pows));
    }

    for i in 0..com.length() {
        y_alpha.push(com.element(i).eval(alpha_pows));
    }

    y_alpha.push(proof.y.element(0).eval(alpha_pows));

    y_alpha.push(ExtField::ZERO);

    for _ in 0..params.n {
        y_alpha.push(ExtField::ZERO);
    }

    // sense check
    assert_eq!(y_alpha.len().log(), tau_1.len());

    // compute a=sum_i eq(tau_1, i) . y_alpha[i]
    let mut a = ExtField::ZERO;

    for i in 0..y_alpha.len() {
        let mut eq = ExtField::ONE;

        for j in 0..tau_1.len() {
            eq *= eq_bin(tau_1[j], (i >> j) & 1);
        }

        a += eq * y_alpha[i];
    }

    a
}

/// Verify sum check, return expected evaluation of F_alpha and F_0
fn sumcheck_verify(
    univariates_f_alpha: &Vec<Univariate<ExtField>>, 
    univariates_f_0: &Vec<Univariate<ExtField>>,
    sum_f_alpha: ExtField,
    fs: &mut FS
) -> (ExtField, ExtField, Vec<ExtField>, Vec<ExtField>) {
    // get the number of rounds
    let rounds_f_0 = univariates_f_0.len();
    let rounds_f_alpha = univariates_f_alpha.len();
    let mut cur = rounds_f_alpha;

    let mut cur_check_f_alpha = sum_f_alpha;
    let mut cur_check_f_0 = ExtField::ZERO;

    let mut challenges_f_alpha = Vec::<ExtField>::new();
    let mut challenges_f_0 = Vec::<ExtField>::new();

    let mut valid = false;

    while cur > rounds_f_0 {
        // get the univariate polynomial g(x) for this round
        let univariate_f_alpha = univariates_f_alpha[rounds_f_alpha - cur].clone();

        // check g(0) + g(1)
        valid = cur_check_f_alpha == univariate_f_alpha.binary_sum();

        // sample the challenge
        fs.push(&univariate_f_alpha);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = ExtField::rand(0, &mut rng);
        challenges_f_alpha.push(r);

        // update the expected binary sum
        cur_check_f_alpha = univariate_f_alpha.eval(r);

        cur -= 1;
    }

    while cur > 0 {
        // get the univariate polynomial g(x) for this round
        let univariate_f_alpha = univariates_f_alpha[rounds_f_alpha - cur].clone();
        let univariate_f_0 = univariates_f_0[rounds_f_0 - cur].clone();

        // check g(0) + g(1)
        valid = cur_check_f_alpha == univariate_f_alpha.binary_sum();
        assert_eq!(cur_check_f_0, univariate_f_0.binary_sum());

        // sample the challenge
        fs.push(&univariate_f_alpha);
        fs.push(&univariate_f_0);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = ExtField::rand(0, &mut rng);
        challenges_f_alpha.push(r);
        challenges_f_0.push(r);

        // update the expected binary sum
        cur_check_f_alpha = univariate_f_alpha.eval(r);
        cur_check_f_0 = univariate_f_0.eval(r);

        cur -= 1;
    }

    assert!(valid);

    (cur_check_f_alpha, cur_check_f_0, challenges_f_alpha, challenges_f_0)
}