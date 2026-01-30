use ark_ff::AdditiveGroup;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::ExtField;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::ring::Ring;
use crate::arithmetic::fs::FS;
use crate::arithmetic::sumcheck::Univariate;
use crate::arithmetic::utils::{Logarithm, powers, rand_field};

use crate::hachi::prover_utils::sumcheck::{F0, FAlpha, sumcheck_proof};
use crate::stream::Stream;
use crate::stream::file_stream::U64FileStream;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{tick_item};

use crate::hachi::Hachi;
use crate::hachi::setup::{Setup, Parameters};
use crate::hachi::commit::{Commit, CommitmentWithState};
use crate::hachi::prover_utils::rq::{commit_w_and_lift, compute_z, lift_c_i_t_i, sample_matrices};
use crate::hachi::prover_utils::zq_zq::{compute_y_and_w, form_next_witness};
use crate::hachi::common::{form_m_alpha_different_matrices, form_m_alpha_same_matrix};

/// Structure to store a round of the evaluation proof.
pub struct ProofRound {
    // single ring element produced when reducing Zq to Rq
    pub y: PVec,

    // commitment to w
    pub v: PVec,

    // commitment to new witness
    pub u_dash: PVec,

    // sumcheck univariates for F_alpha
    pub univariates_f_alpha: Vec<Univariate<ExtField>>,

    // sumcheck univariates for 0
    pub univariates_f_0: Vec<Univariate<ExtField>>,

    // claimed new evaluation
    pub y_dash: ExtField
}

/// Evaluation proof function.
pub trait Prove<T, F> {
    /// Evaluation proof for the multilinear polynomial witness provided as T,
    /// with an evaluation point provided as a slice over F.
    fn prove(
        witness: T,                     // witness polynomial
        params: &Parameters,            // parameters
        x: &[F],                        // evaluation point
        com: &CommitmentWithState       // commitment with internal state
    ) -> ProofRound;
}

/// Implementation of the evaluation proof for an evaluation point over integers.
impl Prove<&mut U64FileStream, u64> for Hachi {
    #[time_graph::instrument]
    fn prove(
            witness: &mut U64FileStream,
            params: &Parameters,
            x: &[u64],
            CommitmentWithState { t, t_hat, u }: &CommitmentWithState
        ) -> ProofRound {
        #[cfg(feature = "verbose")]
        println!("\n==== Evaluate ====");
        
        // ensure stream has sufficient length
        assert!(witness.length() >= 1 << params.l);

        // initialise a FS stream and push the commitment
        let mut fs = FS::init();
        fs.push(u);

        // create the rings Zq[X] and Zq[X]/(X^d+1)
        let ring_full = Ring::init(params.q, params.d, false);
        let ring_cyclotomic = Ring::init(params.q, params.d, true);

        // sample commitment matrices
        let 
        (
            mat_a, mat_a_ntt,
            mat_b, mat_b_ntt, 
            mat_d, mat_d_ntt
        ) = sample_matrices(params, &ring_full);

        // ---- First prover message ----
        // reduce to an evaluation over Rq and calculate w
        let mut y = PVec::zero(1, params.d);
        let mut w = PVec::zero(1 << params.r, params.d);
        compute_y_and_w(witness, params, x, &ring_cyclotomic, &mut y, &mut w);

        // commit to w and lift to Zq[X]
        let mut w_hat = PVec::zero(w.length() * params.delta, params.d);
        let mut v_quo = PVec::zero(params.n, params.d);
        let mut v = PVec::zero(params.n, params.d);
        commit_w_and_lift(params, &ring_full, &mat_d_ntt, &w, &mut w_hat, &mut v, &mut v_quo);

        // ---- Sample challenges ----
        fs.push(&y);
        fs.push(&v);
        let seed = fs.get_seed();
        let challenges = PChal::rand_vec(1 << params.r, params.d, params.k, seed);

        #[cfg(feature = "verbose")]
        tick_item("Sample challenges");

        // ---- Prover response z ----
        let mut z = PVec::zero((1 << params.m) * params.delta, params.d);
        compute_z(witness, params, &ring_cyclotomic, &challenges, &mut z);

        // ---- Lift the verification equations to Zq[X]

        // - already calculated commitment to w over Zq[X]

        // - lift the outer commitment (re-calculate commitment to inner commitment t over Zq[X]).
        let mut u_full = PVec::zero(params.n, 2 * params.d);
        ring_full.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u_full, 0);

        let mut u_quo = PVec::zero(params.n, params.d);
        let mut u_rem = PVec::zero(params.n, params.d);
        u_full.cyclotomic_div(params.q, &mut u_quo, &mut u_rem);

        // sense check
        assert_eq!(u.slice(), u_rem.slice());

        // - the product [[b_1 ... b_2^r]^T [w_1 ... w^2^r]] does not need to be lifted
        // since b_i are integers so the quotient is zero.

        // - calculate the inner product [c_1 ... c_2^r]^T [w_1 ... w^2^r] over Zq[X]
        let mut c_i_w_i_full = PVec::zero(1, 2 * params.d);
        ring_full.chal_vec_mul_poly_vec(&challenges, &w, &mut c_i_w_i_full, 0);

        let mut c_i_w_i_quo = PVec::zero(1, params.d);
        let mut c_i_w_i_rem = PVec::zero(1, params.d);
        c_i_w_i_full.cyclotomic_div(params.q, &mut c_i_w_i_quo, &mut c_i_w_i_rem);

        // - the product [[a_1 ... a_2^m]^T [z'_1 ... z'^2^m]] does not need to be lifted
        // since a_i are integers so the quotient is zero.

        // - calculate the inner products ([c_1 ... c_2^r]^T o I_n] [t_1,1..t_1,n ... t_2^r,1..t_2^r,n] over Zq[X]       
        let mut c_i_t_i_quo = PVec::zero(params.n, params.d);
        let mut c_i_t_i_rem = PVec::zero(params.n, params.d);
        lift_c_i_t_i(params, &ring_full, &challenges, t, &mut c_i_t_i_quo, &mut c_i_t_i_rem);

        // - calculate A.z over Zq[X]
        let mut mat_a_z_full = PVec::zero(params.n, 2 * params.d);
        ring_full.mat_mul_vec(&mat_a_ntt, &z, &mut mat_a_z_full, 0);

        let mut mat_a_z_quo = PVec::zero(params.n, params.d);
        let mut mat_a_z_rem = PVec::zero(params.n, params.d);
        mat_a_z_full.cyclotomic_div(params.q, &mut mat_a_z_quo, &mut mat_a_z_rem);

        // sense check
        assert_eq!(c_i_t_i_rem.slice(), mat_a_z_rem.slice());

        #[cfg(feature = "verbose")]
        tick_item("Lift verification equations to Zq[X]");

        // Produce the new witness (z,r)
        let (num_vars, mu, n, z_r) = form_next_witness(
            params, &w_hat, &t_hat, &z, &v_quo, &u_quo, &c_i_w_i_quo, &c_i_t_i_quo, &mat_a_z_quo
        );

        // Commit to the next witness
        let next_params = Hachi::setup(num_vars, false);
        let com_dash = Hachi::commit(z_r.as_slice(), &next_params);

        // Sample a random field element
        fs.push(&com_dash.u);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let alpha = rand_field(params.q, &mut rng);

        // Get the powers of alpha
        let alpha_pows = powers(alpha, params.d);

        // Get [M | -(X^d+1).In] evaluated at alpha
        let m_alpha = if params.reuse_mats {
            form_m_alpha_same_matrix(params, x, &challenges, &alpha_pows, &mat_d)
        } else {
            form_m_alpha_different_matrices(params, x, &challenges, &alpha_pows, &mat_a, &mat_b, &mat_d)
        };

        // Sample the random field elements tau_0 and tau_1
        let mut tau_0 = vec![ExtField::ZERO; num_vars];
        
        for i in 0..num_vars {
            tau_0[i] = rand_field(params.q, &mut rng);
        }

        let log_n = n.log();
        let mut tau_1 = vec![ExtField::ZERO; log_n];
        
        for i in 0..log_n {
            tau_1[i] = rand_field(params.q, &mut rng);
        }

        // Form F_0,tau_0 and F_alpha_tau_1 for sum check
        let mut f_0 = F0::init(&z_r, params.b, tau_0, params.q, (mu + n * params.delta) * params.d);
        let mut f_alpha = FAlpha::init(&z_r, alpha_pows, tau_1, m_alpha, params.q);

        let (univariates_f_alpha, univariates_f_0, y_dash) = sumcheck_proof(&mut f_0, &mut f_alpha, &mut fs);

        #[cfg(feature = "verbose")]
        println!("==== Complete ====\n");

        ProofRound { y, v, u_dash: com_dash.u, univariates_f_alpha, univariates_f_0, y_dash }
    }
}
