#[cfg(feature = "verbose")]
use crate::utils::verbose::progress_bar;

use crate::stream::Stream;

use crate::arithmetic::ring::Ring;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::utils::{Logarithm, multi_lin_coeff_int};

use crate::hachi::setup::Parameters;

/// Prover utility functions when the original witness is over Zq
/// and the evaluation point is over Zq.

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Reduce evaluation over Zq to evaluation over Rq and calculate the first prover message.
/// Both done in a single function to avoid reading the witness twice.
///  - Compute the new evaluation point Y \in Rq.
///  - Compute the first prover message [w_i]_i := [a^T s_i]_i.
pub fn compute_y_and_w(
        witness: &mut impl Stream<u64>,
        params: &Parameters,
        x: &[u64],
        ring: &Ring,
        y: &mut PVec,
        w: &mut PVec
) {
    assert_eq!(1, y.length());
    assert_eq!(1 << params.r, w.length());

    let y = y.mut_element(0);
    witness.reset();

    // create a vector for reading in the witness
    let mut f_i = PVec::zero(1 << params.m, params.d);

    // pre-process the 2^m coefficients a^T
    let mut a = vec![0u64; 1 << params.m];

    for i in 0..1 << params.m {
        a[i] = multi_lin_coeff_int(&x[params.r..params.r + params.m], i, params.m, params.q)
    }

    // read in the whole witness in 2^r chunks of length 2^{m + alpha}
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Compute y and compute w", i, 1 << params.r);

        // since the polynomial is over integers, do not need to do anything more to map each f_i to Rq elements.
        witness.read(f_i.mut_slice());

        // compute w_i
        ring.int_vec_mul_poly_vec(&a, &f_i, w, i);
        
        // iterate over each f_ij
        for j in 0..1 << params.m {
            // the index in {0,1}^{r+m}
            let index = i << params.r | j;

            // get the coefficient for this index
            let coeff = multi_lin_coeff_int(&x[0..params.r + params.m], index, params.r + params.m, params.q);

            // get the current ring element
            let f_ij = f_i.element(j);

            // add coeff * f_ij to y
            ring.int_mul_poly(coeff, f_ij, y);
        }
    }
}

/// Prouduce the new witness (z,r).
pub fn form_next_witness(
        params: &Parameters,
        w_hat: &PVec,
        t_hat: &PVec,
        z: &PVec,
        v_quo: &PVec,
        u_quo: &PVec,
        c_i_w_i_quo: &PVec,
        c_i_t_i_quo: &PVec,
        mat_a_z_quo: &PVec

    ) -> (usize, usize, usize, Vec<u64>) {
        assert_eq!((1 << params.r) * params.delta, w_hat.length());
        assert_eq!(params.n * (1 << params.r) * params.delta, t_hat.length());
        assert_eq!((1 << params.m) * params.delta, z.length());
        assert_eq!(params.n, v_quo.length());
        assert_eq!(params.n, u_quo.length());
        assert_eq!(1, c_i_w_i_quo.length());
        assert_eq!(params.n, c_i_t_i_quo.length());
        assert_eq!(params.n, mat_a_z_quo.length());

        // length of the z part of new witness (width of M)
        let mu = w_hat.length() + t_hat.length() + z.length() * params.delta_z;
        
        // length of r part of new witness (height of M * decomposition expansion)
        let n = (v_quo.length() + u_quo.length() + 1 + c_i_w_i_quo.length() + c_i_t_i_quo.length()) * params.delta;

        // new witness must have power of 2 length, so we pad (mu + n) to next power of 2
        let num_vars = ((mu + n) * params.d).log();
        
        // create a vector to store the next witness
        let mut witness = vec![0u64; 1 << num_vars];

        // copy in w_hat
        let mut cur = 0;
        witness[cur..cur + w_hat.length() * params.d].copy_from_slice(w_hat.slice());
        cur += w_hat.length() * params.d;

        // copy in t_hat
        witness[cur..cur + t_hat.length() * params.d].copy_from_slice(t_hat.slice());
        cur += t_hat.length() * params.d;

        // decompose z into z_hat and copy in
        let mut z_hat = PVec::zero(z.length() * params.delta_z, params.d);
        z.b_decomp(params.b, params.delta_z, &mut z_hat);
        witness[cur..cur + z_hat.length() * params.d].copy_from_slice(z_hat.slice());
        cur += z_hat.length() * params.d;

        // sense check
        assert_eq!(cur, mu * params.d);

        // decompose and copy in quotient for commitment v=D.w_hat
        let mut v_quo_hat = PVec::zero(v_quo.length() * params.delta, params.d);
        v_quo.b_decomp(params.b, params.delta, &mut v_quo_hat);
        witness[cur..cur + v_quo_hat.length() * params.d].copy_from_slice(v_quo_hat.slice());
        cur += v_quo_hat.length() * params.d;

        // decompose and copy in quotient for commitment u=B.t_hat
        let mut u_quo_hat = PVec::zero(u_quo.length() * params.delta, params.d);
        u_quo.b_decomp(params.b, params.delta, &mut u_quo_hat);
        witness[cur..cur + u_quo_hat.length() * params.d].copy_from_slice(u_quo_hat.slice());
        cur += u_quo_hat.length() * params.d;

        // third row has 0 quotient 
        cur += params.d * params.delta;

        // decompose and copy in quotient for sum_i c_i_w_i
        let mut c_i_w_i_quo_hat = PVec::zero(c_i_w_i_quo.length() * params.delta, params.d);
        c_i_w_i_quo.b_decomp(params.b, params.delta, &mut c_i_w_i_quo_hat);
        witness[cur..cur + c_i_w_i_quo_hat.length() * params.d].copy_from_slice(c_i_w_i_quo_hat.slice());
        cur += c_i_w_i_quo_hat.length() * params.d;

        // decompose and copy in sum c_i_i_t_i - A.z
        assert_eq!(c_i_t_i_quo.length(), mat_a_z_quo.length());
        let mut quo = PVec::zero(c_i_t_i_quo.length(), params.d);
        let coeffs = quo.mut_slice();

        for i in 0..coeffs.len() {
            let a = c_i_t_i_quo.slice()[i];
            let b = mat_a_z_quo.slice()[i];
            let mut quo_coeff = a as i64 - b as i64;

            if quo_coeff < 0 {
                quo_coeff = params.q as i64 + quo_coeff;
            }

            coeffs[i] = quo_coeff as u64;
        }

        let mut quo_hat = PVec::zero(quo.length() * params.delta, params.d);
        quo.b_decomp(params.b, params.delta, &mut quo_hat);
        witness[cur..cur + quo_hat.length() * params.d].copy_from_slice(quo_hat.slice());
        cur += quo_hat.length() * params.d;

        // sense check
        assert_eq!((mu + n) * params.d, cur);

        #[cfg(feature = "verbose")]
        println!("Formed next witness ({} variables)", num_vars);

        (num_vars, mu, n/params.delta, witness)
}