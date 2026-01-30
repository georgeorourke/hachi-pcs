#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar, tick_item};

use crate::stream::Stream;

use crate::arithmetic::poly_mat_ntt::PMatNtt;
use crate::arithmetic::poly_mat::PMat;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::ring::Ring;

use crate::hachi::setup::Parameters;

/// Prover utility functions generic to prover operations of Rq.

/// Sample the commitment matrices.
pub fn sample_matrices(params: &Parameters, ring: &Ring) -> (PMat, PMatNtt, PMat, PMatNtt, PMat, PMatNtt) {
    // Sample D and NTT
    let mat_d = PMat::rand(params.n, params.width_d, params.d, params.q, params.d_seed);
    let mut mat_d_ntt = PMatNtt::zero(mat_d.height(), mat_d.width(), 2 * params.d);
    ring.mat_fwd_ntt(&mat_d, &mut mat_d_ntt);

    // If reusuing matrices, copy into B and A
    if params.reuse_mats {
        let mat_b = mat_d.clone();
        let mat_b_ntt = mat_d_ntt.clone();
        let mat_a = mat_d.clone();
        let mat_a_ntt = mat_d_ntt.clone();
        
        (mat_a, mat_a_ntt, mat_b, mat_b_ntt, mat_d, mat_d_ntt)
    }
    // If not reusing matrices, sample again
    else {
        // Sample B and NTT
        let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
        let mut mat_b_ntt = PMatNtt::zero(mat_b.height(), mat_b.width(), 2 * params.d);
        ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);

        // Sample A and NTT
        let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
        let mut mat_a_ntt = PMatNtt::zero(mat_a.height(), mat_a.width(), 2 * params.d);
        ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

        (mat_a, mat_a_ntt, mat_b, mat_b_ntt, mat_d, mat_d_ntt)
    }   
}

/// Commit to w over Zq[X] and divide by (X^d+1) to obtain commitmemnt
/// over Rq and the quotient to lift it to Zq[X].
/// v = D.w_hat + (X^d+1).v_quo
pub fn commit_w_and_lift(
    params: &Parameters, ring: &Ring, mat_d_ntt: &PMatNtt, 
    w: &PVec, w_hat: &mut PVec, v: &mut PVec, v_quo: &mut PVec
) {
    // decompose w
    w.b_decomp(params.b, params.delta, w_hat);

    // multiply, v = D.w_hat over Zq[X]
    let mut v_full = PVec::zero(params.n, 2 * params.d);
    ring.mat_mul_vec(&mat_d_ntt, &w_hat, &mut v_full, 0);

    // cyclotomic reduction    
    v_full.cyclotomic_div(params.q, v_quo, v);

    #[cfg(feature = "verbose")]
    tick_item("Commit to w and lift");
}

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Compute the prover response z.
pub fn compute_z(
        witness: &mut impl Stream<u64>,
        params: &Parameters,
        ring: &Ring,
        challenges: &Vec<PChal>,
        z: &mut PVec
    ) {
        assert_eq!((1 << params.m) * params.delta, z.length());
        assert_eq!(1 << params.r, challenges.len());
        witness.reset();

        // create a vector for f_i and s_i
        let mut f_i = PVec::zero(1 << params.m, params.d);
        let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            #[cfg(feature = "verbose")]
            progress_bar("Computing response z", i, 1 << params.r);

            // read the next chunk f_i
            witness.read(f_i.mut_slice());

            // if decomposing
            if params.decomp_witness {
                f_i.b_decomp(params.b, params.delta, &mut s_i);
                ring.chal_mul_poly_vec(&challenges[i], &s_i, z, 0);
            }
            // if not decomposing
            else {
                ring.chal_mul_poly_vec(&challenges[i], &f_i, z, 0);
            };
        }

        // Check the norm of z
        let mut min = 0;
        let mut max = 0;
        
        for x in z.slice() {
            if (*x as i64) > max { max = *x as i64 };
            if (*x as i64) < min { min = *x as i64 }
        }

        assert!(min >= - (params.z_bound as i64));
        assert!(max <= params.z_bound as i64);

        #[cfg(feature = "verbose")]
        tick_item("z is within heuristic bound");
}

/// Lift the verification equation ([c_1 ... c_2^r]^T o I_n] [t_1,1..t_1,n ... t_2^r,1..t_2^r,n] to Zq[X].
pub fn lift_c_i_t_i(
        params: &Parameters,
        ring: &Ring,
        challenges: &Vec<PChal>,
        t: &PVec,
        quo: &mut PVec,
        rem: &mut PVec
    ) {
        assert_eq!(1 << params.r, challenges.len());
        assert_eq!(params.n * (1 << params.r), t.length());
        assert_eq!(params.n, quo.length());
        assert_eq!(params.n, rem.length());

        let mut prod_full = PVec::zero(params.n, 2 * params.d);
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            // iterate over height of matrix
            for j in 0..params.n {
                // multiply c_i by t_i,j and store in out[j]
                ring.chal_mul_poly(&challenges[i], t.element(i * params.n + j), prod_full.mut_element(j));
            }
        }

        // cyclotomic reduction
        prod_full.cyclotomic_div(params.q, quo, rem);
}

