use crate::arithmetic::{Decompose, Multiply, Ntt, PMat, PMatNtt, PVec, RandFromSeed, ZeroWithParams};
use crate::arithmetic::ring::Ring;

use crate::stream::Stream;

use crate::hachi::Hachi;
use crate::hachi::setup::Parameters;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar, tick_item};

/// Structure to store a commitment and associated state.
pub struct CommitmentWithState {
    pub t: PVec,         // inner commitment (state)
    pub t_hat: PVec,     // decomposed inner commitment (state)
    pub u: PVec          // outer commitment (public commitment)
}

/// Commitment function.
pub trait Commit {
    /// The commitment function for a multilinear polynomial
    /// provided as a stream of unsigned 64 bit coefficients.
    fn commit(witness: &mut impl Stream<u64>, params: &Parameters) -> CommitmentWithState;
}

impl Commit for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &mut impl Stream<u64>, params: &Parameters) -> CommitmentWithState
    {
        #[cfg(feature = "verbose")]
        println!("\n==== Commit ====");

        // ensure the stream has sufficient length
        assert!(witness.length() >= 1 << params.l);

        // create the ring Zq[X]/(X^d+1)
        let ring = Ring::init(params.q, params.d, true);

        // create vectors for the commitment vectors
        let mut t = PVec::zero((params.n * (1 << params.r), params.d));
        let mut t_hat = PVec::zero((t.length() * params.delta, params.d));
        let mut u = PVec::zero((params.n, params.d));

        // --- inner commitment ---
        // sample outer commitment matrix and perform forward NTT.
        let mat_a = PMat::rand((params.n, params.width_a, params.d, params.q), params.a_seed);
        let mut mat_a_ntt = PMatNtt::zero((params.n, params.width_a, params.d));
        ring.fwd(&mat_a, &mut mat_a_ntt);

        // create a vector for f_i and s_i
        let mut f_i = PVec::zero((1 << params.m, params.d));
        let mut s_i = PVec::zero((f_i.length() * params.delta, params.d));
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            #[cfg(feature = "verbose")]
            progress_bar("Inner Commitment", i, 1 << params.r);

            // read the next chunk f_i
            witness.read(f_i.mut_slice());

            // if decomposing
            if params.decomp_witness {
                f_i.b_decomp(params.b, params.delta, &mut s_i);
                ring.mul(&mat_a_ntt, &s_i, &mut t, i * params.n);
            }
            // if not decomposing
            else {
                ring.mul(&mat_a_ntt, &f_i, &mut t, i * params.n);
            };
        }

        // --- outer commitment ---
        // get outer commitment matrix
        let mat_b_ntt = {
            if params.reuse_mats { mat_a_ntt }
            else {
                let mat_b = PMat::rand((params.n, params.width_b, params.d, params.q), params.b_seed);
                let mut mat_b_ntt = PMatNtt::zero((params.n, params.width_b, params.d));
                ring.fwd(&mat_b, &mut mat_b_ntt);
                mat_b_ntt
            }
        };

        // decompose the inner commitment
        t.b_decomp(params.b, params.delta, &mut t_hat);

        // commit over Zq[X]
        ring.mul(&mat_b_ntt, &t_hat, &mut u, 0);

        #[cfg(feature = "verbose")]
        tick_item("Outer Commitment");

        #[cfg(feature = "verbose")]
        println!("==== Complete ====\n");

        CommitmentWithState { t, t_hat, u }
    }
}