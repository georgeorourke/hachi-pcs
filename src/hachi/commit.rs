use crate::arithmetic::utils::Logarithm;
use crate::{arithmetic::poly_vec::PVec, stream::file_stream::U64FileStream};
use crate::arithmetic::poly_mat::PMat;
use crate::arithmetic::poly_mat_ntt::PMatNtt;
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
pub trait Commit<T> {
    /// Commit to the witness provided as type T.
    fn commit(witness: T, params: &Parameters) -> CommitmentWithState;
}

/// Commitment function - witness provided as file stream of u64 integers.
/// The witness is a multilinear polynomial in l variables over Zq,
/// or equivalently a multilinear polynomial in l - log d variables over Rq,
/// where each block of d coefficients define an Rq element.
impl Commit<&mut U64FileStream> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &mut U64FileStream, params: &Parameters) -> CommitmentWithState
    {
        commit_stream(witness, params)
    }
}

/// Commitment function - witness provided as slice of u64 integers.
/// The witness is a multilinear polynomial in l variables over Zq,
/// or equivalently a multilinear polynomial in l - log d variables over Rq,
/// where each block of d coefficients define an Rq element.
impl Commit<&[u64]> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &[u64], params: &Parameters) -> CommitmentWithState
    {
        commit_slice(witness, params)
    }
}

/// Commitment function - witness provided as a vector over Rq.
impl Commit<&PVec> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &PVec, params: &Parameters) -> CommitmentWithState {
        commit_slice(witness.slice(), params)
    }
}

/// Generic u64 stream commitment.
fn commit_stream(witness: &mut impl Stream<u64>, params: &Parameters) -> CommitmentWithState {
    #[cfg(feature = "verbose")]
    println!("\n==== Commit ====");

    // ensure the stream has sufficient length
    assert!(witness.length() >= 1 << params.l);

    // create the ring Zq[X]/(X^d+1)
    let ring = Ring::init(params.q, params.d, true);

    // create vectors for the commitment vectors
    let mut t = PVec::zero(params.n * (1 << params.r), params.d);
    let mut t_hat = PVec::zero(t.length() * params.delta, params.d);
    let mut u = PVec::zero(params.n, params.d);

    // --- inner commitment ---
    // sample outer commitment matrix and perform forward NTT.
    let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
    let mut mat_a_ntt = PMatNtt::zero(params.n, params.width_a, params.d);
    ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

    // create a vector for f_i and s_i
    let mut f_i = PVec::zero(1 << params.m, params.d);
    let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
    
    // iterate over 0..2^r
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Inner Commitment", i, 1 << params.r);

        // read the next chunk f_i
        witness.read(f_i.mut_slice());

        // if decomposing
        if params.decomp_witness {
            f_i.b_decomp(params.b, params.delta, &mut s_i);
            ring.mat_mul_vec(&mat_a_ntt, &s_i, &mut t, i * params.n);
        }
        // if not decomposing
        else {
            ring.mat_mul_vec(&mat_a_ntt, &f_i, &mut t, i * params.n);
        };
    }

    // --- outer commitment ---
    // get outer commitment matrix
    let mat_b_ntt = {
        if params.reuse_mats { mat_a_ntt }
        else {
            let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
            let mut mat_b_ntt = PMatNtt::zero(params.n, params.width_b, params.d);
            ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);
            mat_b_ntt
        }
    };

    // decompose the inner commitment
    t.b_decomp(params.b, params.delta, &mut t_hat);

    // commit over Zq[X]
    ring.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u, 0);

    #[cfg(feature = "verbose")]
    tick_item("Outer Commitment");

    #[cfg(feature = "verbose")]
    println!("==== Complete ====\n");

    CommitmentWithState { t, t_hat, u }
}

/// u64 slice commitment.
fn commit_slice(witness: &[u64], params: &Parameters) -> CommitmentWithState {
    #[cfg(feature = "verbose")]
    println!("\n==== Commit ====");

    // ensure the stream has sufficient length
    assert!(witness.len() >= 1 << params.l);

    // create the ring Zq[X]/(X^d+1)
    let ring = Ring::init(params.q, params.d, true);

    // log ring dimension
    let alpha = params.d.log();

    // create vectors for the commitment vectors
    let mut t = PVec::zero(params.n * (1 << params.r), params.d);
    let mut t_hat = PVec::zero(t.length() * params.delta, params.d);
    let mut u = PVec::zero(params.n, params.d);

    // --- inner commitment ---
    // sample outer commitment matrix and perform forward NTT.
    let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
    let mut mat_a_ntt = PMatNtt::zero(params.n, params.width_a, params.d);
    ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

    // create a vector for f_i and s_i
    let mut f_i = PVec::zero(1 << params.m, params.d);
    let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
    
    // iterate over 0..2^r
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Inner Commitment", i, 1 << params.r);

        // copy the next chunk f_i
        f_i.mut_slice().copy_from_slice(&witness[i << (params.m + alpha)..(i + 1 << (params.m + alpha))]);

        // if decomposing
        if params.decomp_witness {
            f_i.b_decomp(params.b, params.delta, &mut s_i);
            ring.mat_mul_vec(&mat_a_ntt, &s_i, &mut t, i * params.n);
        }
        // if not decomposing
        else {
            ring.mat_mul_vec(&mat_a_ntt, &f_i, &mut t, i * params.n);
        };
    }

    // --- outer commitment ---
    // get outer commitment matrix
    let mat_b_ntt = {
        if params.reuse_mats { mat_a_ntt }
        else {
            let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
            let mut mat_b_ntt = PMatNtt::zero(params.n, params.width_b, params.d);
            ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);
            mat_b_ntt
        }
    };

    // decompose the inner commitment
    t.b_decomp(params.b, params.delta, &mut t_hat);

    // commit over Zq[X]
    ring.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u, 0);

    #[cfg(feature = "verbose")]
    tick_item("Outer Commitment");

    #[cfg(feature = "verbose")]
    println!("==== Complete ====\n");

    CommitmentWithState { t, t_hat, u }
}