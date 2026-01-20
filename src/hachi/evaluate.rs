use ark_ff::{AdditiveGroup, Field};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::sumcheck::{SumCheckPoly, Univariate, eq, eq_bin, fix_first_variable};
use crate::arithmetic::{Decompose, ExtField, FieldExtension, Logarithm, MultiLinearCoeff, Multiply, Ntt, PChal, PMat, PMatNtt, PVec, RandFromRng, RandFromSeed, ZeroWithParams, powers};
use crate::arithmetic::ring::Ring;
use crate::arithmetic::fs::FS;

use crate::stream::Stream;
use crate::stream::vec_stream::VectorStream;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar, tick_item};

use crate::hachi::Hachi;
use crate::hachi::setup::{Setup, Parameters};
use crate::hachi::commit::{Commit, CommitmentWithState};
use crate::hachi::common::{form_m_alpha_different_matrices, form_m_alpha_same_matrix};

/// Structure to store a round of the evaluation proof.
pub struct EvaluationProofRound {
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

/// Evaluation function.
pub trait Evaluate<F> {
    /// The evaluation function for the multilinear polynomial 
    /// provided as a stream of unsigned 64 bit coefficients 
    /// and the evaluation point provided as a slice of type F.
    fn evaluate(
        witness: &mut impl Stream<u64>, // witness polynomial
        params: &Parameters,            // parameters
        x: &[F],                        // evaluation point
        com: &CommitmentWithState       // commitment with internal state
    ) -> EvaluationProofRound;
}

/// Implementation of the evaluation proof for an evaluation point over integers.
impl Evaluate<u64> for Hachi {
    #[time_graph::instrument]
    fn evaluate(
            witness: &mut impl Stream<u64>,
            params: &Parameters,
            x: &[u64],
            CommitmentWithState { t, t_hat, u }: &CommitmentWithState
        ) -> EvaluationProofRound {
        #[cfg(feature = "verbose")]
        println!("\n==== Evaluate ====");

        // initialise a FS stream and push the commitment
        let mut fs = FS::init();
        fs.push(u);

        // ensure stream has sufficient length
        assert!(witness.length() >= 1 << params.l);

        // create the rings Zq[X] and Zq[X]/(X^d+1)
        let ring_full = Ring::init(params.q, params.d, false);
        let ring_cyclotomic = Ring::init(params.q, params.d, true);

        // ---- First prover message ----
        // reduce to an evaluation over Rq and calculate w
        let mut y = PVec::zero((1, params.d));
        let mut w = PVec::zero((1 << params.r, params.d));
        reduce_to_rq_and_compute_w(witness, params, x, &ring_cyclotomic, &mut y, &mut w);

        // commit to w over Zq[X]
        let mat_d = PMat::rand((params.n, params.width_d, params.d, params.q), params.d_seed);
        let mut mat_d_ntt = PMatNtt::zero((mat_d.height(), mat_d.width(), 2 * params.d));
        ring_full.fwd(&mat_d, &mut mat_d_ntt);
        
        // decompose
        let mut w_hat = PVec::zero((w.length() * params.delta, params.d));
        w.b_decomp(params.b, params.delta, &mut w_hat);

        // multiply
        let mut v_full = PVec::zero((params.n, 2 * params.d));
        ring_full.mul(&mat_d_ntt, &w_hat, &mut v_full, 0);

        // cyclotomic reduction
        let mut v_quo = PVec::zero((params.n, params.d));
        let mut v = PVec::zero((params.n, params.d));
        v_full.cyclotomic_div(params.q, &mut v_quo, &mut v);

        #[cfg(feature = "verbose")]
        tick_item("Commit to w");

        // ---- Sample challenges ----
        fs.push(&y);
        fs.push(&v);
        let seed = fs.get_seed();
        let challenges = PChal::rand_vec(1 << params.r, params.d, params.k, seed);

        #[cfg(feature = "verbose")]
        tick_item("Sample challenges");

        // ---- Prover response z ----
        let mut z = PVec::zero(((1 << params.m) * params.delta, params.d));
        compute_z(witness, params, &ring_cyclotomic, &challenges, &mut z);

        // ---- Lift the verification equations to Zq[X]

        // - already calculated commitment to w over Zq[X]

        // lift the outer commitment (re-calculate commitment to inner commitment t
        // over Zq[X]).
        let mat_b:  Option<PMat>;
        let mat_b_ntt = {
            if params.reuse_mats {
                mat_b = None;
                mat_d_ntt 
            }
            else {
                let mat = PMat::rand((params.n, params.width_b, params.d, params.q), params.b_seed);
                let mut mat_b_ntt = PMatNtt::zero((mat.height(), mat.width(), 2 * params.d));
                ring_full.fwd(&mat, &mut mat_b_ntt);
                mat_b = Some(mat);
                mat_b_ntt
            }
        };

        // multiply
        let mut u_full = PVec::zero((params.n, 2 * params.d));
        ring_full.mul(&mat_b_ntt, &t_hat, &mut u_full, 0);

        // cyclotomic reduction
        let mut u_quo = PVec::zero((params.n, params.d));
        let mut u_rem = PVec::zero((params.n, params.d));
        u_full.cyclotomic_div(params.q, &mut u_quo, &mut u_rem);

        // sense check
        assert_eq!(u.slice(), u_rem.slice());

        // - the product [[b_1 ... b_2^r]^T [w_1 ... w^2^r]] does not need to be lifted
        // since b_i are integers so the quotient is zero.

        // - calculate the inner product [c_1 ... c_2^r]^T [w_1 ... w^2^r] over Zq[X]
        let mut c_i_w_i_full = PVec::zero((1, 2 * params.d));
        ring_full.mul(&challenges, &w, &mut c_i_w_i_full, 0);

        let mut c_i_w_i_quo = PVec::zero((1, params.d));
        let mut c_i_w_i_rem = PVec::zero((1, params.d));
        c_i_w_i_full.cyclotomic_div(params.q, &mut c_i_w_i_quo, &mut c_i_w_i_rem);

        // - the product [[a_1 ... a_2^m]^T [z'_1 ... z'^2^m]] does not need to be lifted
        // since a_i are integers so the quotient is zero.

        // - calculate the inner products ([c_1 ... c_2^r]^T o I_n] [t_1,1..t_1,n ... t_2^r,1..t_2^r,n] over Zq[X]       
        let mut c_i_t_i_quo = PVec::zero((params.n, params.d));
        let mut c_i_t_i_rem = PVec::zero((params.n, params.d));
        lift_c_i_t_i(params, &ring_full, &challenges, t, &mut c_i_t_i_quo, &mut c_i_t_i_rem);

        // - calculate A.z over Zq[X]
        let mat_a:  Option<PMat>;
        let mat_a_ntt = {
            if params.reuse_mats {
                mat_a = None;
                mat_b_ntt 
            }
            else {
                let mat = PMat::rand((params.n, params.width_a, params.d, params.q), params.a_seed);
                let mut mat_a_ntt = PMatNtt::zero((mat.height(), mat.width(), 2 * params.d));
                ring_full.fwd(&mat, &mut mat_a_ntt);
                mat_a = Some(mat);
                mat_a_ntt
            }
        };

        // multiply
        let mut mat_a_z_full = PVec::zero((params.n, 2 * params.d));
        ring_full.mul(&mat_a_ntt, &z, &mut mat_a_z_full, 0);

        // cyclotomic reduction
        let mut mat_a_z_quo = PVec::zero((params.n, params.d));
        let mut mat_a_z_rem = PVec::zero((params.n, params.d));
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
        let mut witness_stream = VectorStream::init(z_r.clone());
        let com_dash = Hachi::commit(&mut witness_stream, &next_params);

        // Sample a random field element
        fs.push(&com_dash.u);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let alpha = ExtField::rand(params.q, &mut rng);

        // Get the powers of alpha
        let alpha_pows = powers(alpha, params.d);

        // Get [M | -(X^d+1).In] evaluated at alpha
        let m_alpha = if params.reuse_mats {
            form_m_alpha_same_matrix(params, x, &challenges, &alpha_pows, &mat_d)
        } else {
            form_m_alpha_different_matrices(params, x, &challenges, &alpha_pows, &mat_a.unwrap(), &mat_b.unwrap(), &mat_d)
        };

        // Sample the random field elements tau_0 and tau_1
        let mut tau_0 = vec![ExtField::ZERO; num_vars];
        
        for i in 0..num_vars {
            tau_0[i] = ExtField::rand(params.q, &mut rng);
        }

        let log_n = n.log();
        let mut tau_1 = vec![ExtField::ZERO; log_n];
        
        for i in 0..log_n {
            tau_1[i] = ExtField::rand(params.q, &mut rng);
        }

        // Form F_0,tau_0 and F_alpha_tau_1 for sum check
        let mut f_0 = F0::init(&z_r, mu * params.d, params.q, params.b, tau_0);
        let mut f_alpha = FAlpha::init(&z_r, alpha_pows, tau_1, m_alpha, params.q);

        let (univariates_f_alpha, univariates_f_0, y_dash) = sumcheck_proof(&mut f_0, &mut f_alpha, &mut fs);

        #[cfg(feature = "verbose")]
        println!("==== Complete ====\n");

        EvaluationProofRound { y, v, u_dash: com_dash.u, univariates_f_alpha, univariates_f_0, y_dash }
    }
}

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// First part of the evaluation proof requires reducing proving an evaluation
/// of a polynomial over T and evaluation point over F to an evaluation over Rq.
/// Store the new evaluation in y.
/// Also compute the first prover message w so we only read the witness once.
fn reduce_to_rq_and_compute_w(
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
    let mut f_i = PVec::zero((1 << params.m, params.d));

    // pre-process the 2^m coefficients a^T
    let mut a = vec![0u64; 1 << params.m];

    for i in 0..1 << params.m {
        a[i] = u64::multi_lin_coeff(&x[params.r..params.r + params.m], i, params.m, params.q)
    }

    // read in the whole witness in 2^r chunks of length 2^{m + alpha}
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Reduce to Rq", i, 1 << params.r);

        // since the polynomial is over integers, do not need to do anything more to map each f_i to Rq elements.
        witness.read(f_i.mut_slice());

        // compute w_i
        ring.mul(&a, &f_i, w, i);
        
        // iterate over each f_ij
        for j in 0..1 << params.m {
            // the index in {0,1}^{r+m}
            let index = i << params.r | j;

            // get the coefficient for this index
            let coeff = u64::multi_lin_coeff(&x[0..params.r + params.m], index, params.r + params.m, params.q);

            // get the current ring element
            let f_ij = f_i.element(j);

            // add coeff * f_ij to y
            ring.int_mul_poly(coeff, f_ij, y);
        }
    }
}

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Compute the prover response z.
fn compute_z(
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
        let mut f_i = PVec::zero((1 << params.m, params.d));
        let mut s_i = PVec::zero((f_i.length() * params.delta, params.d));
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            #[cfg(feature = "verbose")]
            progress_bar("Computing response z", i, 1 << params.r);

            // read the next chunk f_i
            witness.read(f_i.mut_slice());

            // if decomposing
            if params.decomp_witness {
                f_i.b_decomp(params.b, params.delta, &mut s_i);
                ring.mul(&challenges[i], &s_i, z, 0);
            }
            // if not decomposing
            else {
                ring.mul(&challenges[i], &f_i, z, 0);
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

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Lift the verification equation ([c_1 ... c_2^r]^T o I_n] [t_1,1..t_1,n ... t_2^r,1..t_2^r,n] to Zq[X].
fn lift_c_i_t_i(
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

        let mut prod_full = PVec::zero((params.n, 2 * params.d));
        
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

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Prouduce the new witness (z,r).
fn form_next_witness(
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
        
        // length of r part of new witness (height of M)
        // n is typically much smaller than mu
        let n = v_quo.length() + u_quo.length() + 1 + c_i_w_i_quo.length() + c_i_t_i_quo.length();

        // new witness must have power of 2 length, so we pad (mu + n) to next power of 2
        let num_vars = (((mu + n) * params.d) as u64).log();
        
        // create a vector to store the next witness
        let mut witness = vec![0u64; 1 << num_vars];
        let mut cur = 0;

        // copy in w_hat
        witness[cur..cur + w_hat.length() * params.d].clone_from_slice(w_hat.slice());
        cur += w_hat.length() * params.d;

        // copy in t_hat
        witness[cur..cur + t_hat.length() * params.d].clone_from_slice(t_hat.slice());
        cur += t_hat.length() * params.d;

        // (balanced) decompose z into z_hat and copy in
        let mut z_hat = PVec::zero((z.length() * params.delta_z, params.d));
        z.b_decomp(params.b, params.delta_z, &mut z_hat);
        witness[cur..cur + z_hat.length() * params.d].clone_from_slice(z_hat.slice());
        cur += z_hat.length() * params.d;

        // sense check
        assert_eq!(cur, mu * params.d);

        // copy in quotient for commitment v=D.w_hat
        witness[cur..cur + v_quo.length() * params.d].clone_from_slice(v_quo.slice());
        cur += v_quo.length() * params.d;

        // copy in quotient for commitment u=B.t_hat
        witness[cur..cur + u_quo.length() * params.d].clone_from_slice(u_quo.slice());
        cur += u_quo.length() * params.d;

        // third row has 0 quotient
        cur += params.d;

        // copy in quotient for sum_i c_i_w_i
        witness[cur..cur + c_i_w_i_quo.length() * params.d].clone_from_slice(c_i_w_i_quo.slice());
        cur += c_i_w_i_quo.length() * params.d;

        for (a, b) in c_i_t_i_quo.slice().iter().zip(mat_a_z_quo.slice()) {
            let (a, b) = (*a as i64, *b as i64);
            let mut quo = a - b;

            if quo < 0 {
                quo = params.q as i64 + quo;
            }

            witness[cur] = quo as u64;
            cur += 1;
        }

        // sense check
        assert_eq!((mu + n) * params.d, cur);

        #[cfg(feature = "verbose")]
        println!("Formed next witness ({} variables)", num_vars);

        (num_vars, mu, n, witness)
}

/// Representation of the polynomial f_0 = eq(tau_0, x) w(x)(w(x)-1)(w(x)+1)...
struct F0 {
    z_r: Vec<u64>,                  // table of evaluations for w in integers
    w_eval_table: Vec<ExtField>,         // table of evaluations for w
    indicator_eval_table: Vec<ExtField>, // table of evaluations for indicator polynomial that restricts norm check to z part
    q: u64,                         // modulus q
    base: u64,                      // decomposition base
    tau_0: Vec<ExtField>,                // random vector tau_0
    eq_scalar: ExtField,                 // scalar carried through for eq multiplication
}

impl F0 {
    /// Construct F0.
    fn init(z_r: &Vec<u64>, len_z: usize, q: u64, base: u64, tau_0: Vec<ExtField>) -> Self {
        // lift evaluations to field
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| ExtField::lift_int(*x)).collect();

        // create function table for the indicator MLE to restricts norm check to z part
        let mut indicator_eval_table = vec![ExtField::ONE; len_z];
        let pad = vec![ExtField::ZERO; w_eval_table.len() - len_z];
        indicator_eval_table.extend_from_slice(&pad);

        Self { z_r: z_r.clone(), w_eval_table, indicator_eval_table, q, base, tau_0, eq_scalar: ExtField::ONE }
    }
}

impl SumCheckPoly<ExtField> for F0 {
    fn degree(&self) -> usize {
        (self.base + 2) as usize
    }

    fn num_vars(&self) -> usize {
        (self.w_eval_table.len() as u64).log()
    }

    fn get_univariate(&self) -> Univariate<ExtField> {
        let num_vars = self.num_vars();
        assert_eq!(num_vars, self.tau_0.len());

        // pre-compute eq(tau, suffix) for each possible suffix
        let mut eq_suffix = vec![ExtField::ONE; 1 << (num_vars - 1)];

        for suffix in 0..1 << (num_vars - 1) {
            for i in 0..num_vars - 1 {
                eq_suffix[suffix] *= eq_bin(self.tau_0[i + 1], (suffix >> i) & 1);
            }
        }

        // evaluate the univariate polynomial at degree + 1 different points
        let deg = self.degree();
        let mut ys = Vec::<ExtField>::with_capacity(deg + 1);

        for x_i in 0..=deg {
            let x_i = x_i as u64;

            // build sum_{0,1}^n-1 F0(x_i, b_2, ..., b_n)
            let mut y_i = ExtField::ZERO;

            for suffix in 0..1 << (num_vars - 1) {
                // get I(0, suffix) and I(1, suffix)
                let i_0 = self.indicator_eval_table[suffix << 1];
                let i_1 = self.indicator_eval_table[(suffix << 1) | 1];

                // evaluation of indicator at x is linear interpolation
                let i_x = i_0 + (i_1 - i_0).mul_int(x_i);

                // don't perform all calculations needlessly if masked off
                if i_x == ExtField::ZERO { continue; }

                let v_x =
                // round 1 - perform operations on w over integers for improved efficiency
                if self.w_eval_table.len() == self.z_r.len() {
                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.z_r[suffix << 1];
                    let w_1 = self.z_r[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = (w_0 as i64 + (w_1 as i64 - w_0 as i64) * x_i as i64) % self.q as i64;

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = (w_x * (w_x + self.base as i64 / 2)) % self.q as i64;
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = (w_x * w_x) % self.q as i64;

                    for r in 1..self.base as usize / 2 { 
                        v_x = (v_x * (w_x_squared - r as i64 * r as i64)) % self.q as i64;
                    }

                    ExtField::lift_int(v_x as u64)
                }
                // subsequent rounds
                else {
                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.w_eval_table[suffix << 1];
                    let w_1 = self.w_eval_table[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = w_0 + (w_1 - w_0).mul_int(x_i);

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = w_x * (w_x + ExtField::lift_int(self.base / 2));
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = w_x * w_x;

                    for r in 1..self.base as usize / 2 { 
                        v_x *= w_x_squared - ExtField::lift_int((r * r) as u64);
                    }

                    v_x
                };

                // calculate equality with tau_0
                let eq_x = self.eq_scalar * eq(self.tau_0[0], ExtField::lift_int(x_i)) * eq_suffix[suffix];

                // multiply by indicator and equality
                let y = i_x * eq_x * v_x;

                y_i += y;
            }

            ys.push(y_i);
        }

        Univariate::init(ys)
    }

    fn fix_first_variable(&mut self, r: ExtField) {
        let half = self.num_vars() - 1;

        // update the tables of evaluations on boolean inputs
        self.w_eval_table = fix_first_variable(&self.w_eval_table, r);
        self.indicator_eval_table = fix_first_variable(&self.indicator_eval_table, r);

        // track eq(tau, r)
        self.eq_scalar *= eq(self.tau_0[0], r);

        // update tau_0
        let mut tmp = vec![ExtField::ZERO; half];
        tmp.copy_from_slice(&self.tau_0[1..=half]);
        self.tau_0 = tmp;
    }
}

/// Representation of the polynomial f_alpha(i,x,y) = w(x,y).alpha(y).eq(i).M(i,x)
struct FAlpha {
    z_r: Vec<u64>,                      // table of evaluations for w (integers)
    w_eval_table: Vec<ExtField>,             // table of evaluations for w
    alpha_pows_eval_table: Vec<ExtField>,    // table of evaluations for powers of alpha
    eq_eval_table: Vec<ExtField>,            // table of evaluations for eq_tau_1
    m_alpha_eval_table: Vec<ExtField>,       // table of evaluations for M_alpha
    z_r_alpha: Vec<ExtField>,                // pre-compute values of z_r*alpha
    q: u64                              // modulus
}

impl FAlpha {
    /// Construct F0.
    fn init(
        z_r: &Vec<u64>, 
        alpha_pows_eval_table: Vec<ExtField>,
        tau_1: Vec<ExtField>,
        m_alpha_eval_table: Vec<ExtField>,
        q: u64
    ) -> Self {
        // lift evaluations to field
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| ExtField::lift_int(*x)).collect();

        // compute evaluation table for eq(tau, i)
        let log_n = tau_1.len();
        let mut eq_eval_table = vec![ExtField::ONE; 1 << log_n];

        for bin in 0..1 << log_n {
            for i in 0..log_n {
                eq_eval_table[bin] *= eq_bin(tau_1[i], (bin >> i) & 1);
            }
        }

        // pre-compute z_r[x_y_suffix]*alpha[y_suffix]
        let len_i = eq_eval_table.len();
        let len_x = m_alpha_eval_table.len() / len_i;
        let len_y = w_eval_table.len() / len_x;
        let log_y = (len_y as u64).log();

        let mut z_r_alpha = vec![ExtField::ZERO; len_x * len_y];

        for x_suffix in 0..len_x {
            for y_suffix in 0..len_y {
                let x_y_suffix = (x_suffix << log_y) | y_suffix;

                // still have complete (non-folded) w so can use integers
                let w = z_r[x_y_suffix];

                // get alpha(y_suffix)
                let alpha = alpha_pows_eval_table[y_suffix];

                z_r_alpha[x_y_suffix] = alpha.mul_int(w);
            }
        }

        Self { z_r: z_r.clone(), w_eval_table, alpha_pows_eval_table, eq_eval_table, m_alpha_eval_table, z_r_alpha, q }
    }
}

impl SumCheckPoly<ExtField> for FAlpha {
    fn degree(&self) -> usize {
        2
    }

    fn num_vars(&self) -> usize {
        (self.w_eval_table.len() as u64).log() + (self.eq_eval_table.len() as u64).log()
    }

    fn get_univariate(&self) -> Univariate<ExtField> {
        // evaluate the univariate polynomial at degree + 1 different points
        let deg = self.degree();
        let xs: Vec<ExtField> = (0..=deg).into_iter().map(|i| ExtField::lift_int(i as u64)).collect();
        let mut ys = Vec::<ExtField>::with_capacity(deg + 1);

        // get lengths of remaining variables
        let len_i = self.eq_eval_table.len();
        let len_x = self.m_alpha_eval_table.len() / len_i;
        let len_y = self.w_eval_table.len() / len_x;

        let log_x = (len_x as u64).log();
        let log_y = (len_y as u64).log();

        // sense check
        assert_eq!(len_y, self.alpha_pows_eval_table.len());

        for (i, x_i) in xs.iter().enumerate() {
            // build sum_{0,1}^n-1 F_alpha(x_i, b_2, ..., b_n)
            let mut y_i = ExtField::ZERO;

            // TODO: calculation of univariate when i or x not fully folded does not match expected sum
            // still folding in i
            if len_i > 1 {
                for i_suffix in 0..len_i / 2 {
                    // get eq(0, suffix) and eq(1, suffix)
                    let eq_0 = self.eq_eval_table[i_suffix << 1];
                    let eq_1 = self.eq_eval_table[(i_suffix << 1) | 1];

                    // evaluation at x is linear interpolation
                    let eq_x = eq_0 + (eq_1 - eq_0) * x_i;

                    for x_suffix in 0..len_x {
                        let i_x_suffix = (i_suffix << log_x) | x_suffix;

                        // get M_alpha(0, i_x_suffix) and M_alpha(1, i_x_suffix)
                        let m_0 = self.m_alpha_eval_table[i_x_suffix << 1];
                        let m_1 = self.m_alpha_eval_table[(i_x_suffix << 1) | 1];

                        // evaluation at x is linear interpolation
                        let m_x = m_0 + (m_1 - m_0) * x_i;

                        let mut y = eq_x * m_x;

                        for y_suffix in 0..len_y {
                            let x_y_suffix = (x_suffix << log_y) | y_suffix;
                            y *=  self.z_r_alpha[x_y_suffix];
                            y_i += y;
                        }
                    }
                }
            }

            // still folding in x
            else if len_x > 1 {
                let eq = self.eq_eval_table[0];

                for x_suffix in 0..len_x / 2 {
                    // get M_alpha(0, x_suffix) and eq(1, x_suffix)
                    let m_0 = self.m_alpha_eval_table[x_suffix << 1];
                    let m_1 = self.m_alpha_eval_table[(x_suffix << 1) | 1];

                    // evaluation at x is linear interpolation
                    let m_x = m_0 + (m_1 - m_0) * x_i;

                    let mut y = eq * m_x;

                    for y_suffix in 0..len_y {
                        let x_y_suffix = (x_suffix << log_y) | y_suffix;

                        // if this is first round of x being folded then w can use integers for w
                        if self.w_eval_table.len() == self.z_r.len() {
                            // get w(0, x_y_suffix) and w(1, x_y_suffix)
                            let w_0 = self.z_r[x_y_suffix << 1];
                            let w_1 = self.z_r[(x_y_suffix << 1) | 1];

                            // evaluation at x is linear interpolation
                            let w_x = (w_0 as i64 + (w_1 as i64 - w_0 as i64) * i as i64) % self.q as i64;

                            // get alpha(y_suffix)
                            let alpha = self.alpha_pows_eval_table[y_suffix];

                            y *= alpha.mul_int(w_x as u64);
                            y_i += y;

                        }
                        // otherwise use field elements
                        else 
                        {
                            // get w(0, x_y_suffix) and w(1, x_y_suffix)
                            let w_0 = self.w_eval_table[x_y_suffix << 1];
                            let w_1 = self.w_eval_table[(x_y_suffix << 1) | 1];

                            // evaluation at x is linear interpolation
                            let w_x = w_0 + (w_1 - w_0) * x_i;

                            // get alpha(y_suffix)
                            let alpha = self.alpha_pows_eval_table[y_suffix];

                            y *= w_x * alpha;
                            y_i += y;
                        }
                    }
                }
            }

            // still folding in y
            else {
                let eq = self.eq_eval_table[0];
                let m_alpha = self.m_alpha_eval_table[0];

                for y_suffix in 0..len_y / 2 {
                    // get w(0, y_suffix) and w(1, y_suffix)
                    let w_0 = self.w_eval_table[y_suffix << 1];
                    let w_1 = self.w_eval_table[(y_suffix << 1) | 1];

                    // evaluation of eq at x is linear interpolation
                    let w_x = w_0 + (w_1 - w_0) * x_i;

                    // get alpha(0, y_suffix) and alpha(1, y_suffix)
                    let alpha_0 = self.alpha_pows_eval_table[y_suffix << 1];
                    let alpha_1 = self.alpha_pows_eval_table[(y_suffix << 1) | 1];

                    // evaluation of eq at x is linear interpolation
                    let alpha_x = alpha_0 + (alpha_1 - alpha_0) * x_i;

                    let y = w_x * alpha_x * eq * m_alpha;
                    y_i += y;
                }
            }

            ys.push(y_i);
        }

        Univariate::init(ys)
    }

    fn fix_first_variable(&mut self, r: ExtField) {
        // folding in the variable i
        if self.eq_eval_table.len() > 1 {
            // fold eq(i)
            self.eq_eval_table = fix_first_variable(&self.eq_eval_table, r);

            // fold M_alpha(i, x)
            self.m_alpha_eval_table = fix_first_variable(&self.m_alpha_eval_table, r);
        }

        // folding in the variable x
        else if self.m_alpha_eval_table.len() > 1 {
            // fold in w(x, y)
            self.w_eval_table = fix_first_variable(&self.w_eval_table, r);

            // fold M_alpha(x)
            self.m_alpha_eval_table = fix_first_variable(&self.m_alpha_eval_table, r);
        }

        // folding in the variable y
        else {
            // fold in w(y)
            self.w_eval_table = fix_first_variable(&self.w_eval_table, r);

            // fold in alpha(y)
            self.alpha_pows_eval_table = fix_first_variable(&self.alpha_pows_eval_table, r);
        }
    }
}

#[time_graph::instrument]
/// Sum check proof.
pub fn sumcheck_proof(f_0: &mut F0, f_alpha: &mut FAlpha, fs: &mut FS) -> (Vec<Univariate<ExtField>>, Vec<Univariate<ExtField>>, ExtField) {
    // get the number of variables in the two polynomials (x,y) for f_0 and (i,x,y) for f_alpha
    let rounds_f_0 = f_0.num_vars();
    let rounds_f_alpha = f_alpha.num_vars();
    let mut cur = rounds_f_alpha;

    // store univariate polynomials of F_0 and F_alpha
    let mut univariates_f_0 = Vec::<Univariate<ExtField>>::with_capacity(rounds_f_0);
    let mut univariates_f_alpha = Vec::<Univariate<ExtField>>::with_capacity(rounds_f_alpha);

    while cur > rounds_f_0 {
        #[cfg(feature = "verbose")]
        progress_bar("Sum Check", rounds_f_alpha - cur, rounds_f_alpha);

        let univariate_f_alpha = f_alpha.get_univariate();
        fs.push(&univariate_f_alpha);
        univariates_f_alpha.push(univariate_f_alpha);

        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = ExtField::rand(0, &mut rng);

        f_alpha.fix_first_variable(r);

        cur -= 1;
    }

    while cur > 0 {
        #[cfg(feature = "verbose")]
        progress_bar("Sum Check", rounds_f_alpha - cur, rounds_f_alpha);

        let univariate_f_alpha = f_alpha.get_univariate();
        fs.push(&univariate_f_alpha);
        univariates_f_alpha.push(univariate_f_alpha);

        let univariate_f_0 = f_0.get_univariate();
        fs.push(&univariate_f_0);
        univariates_f_0.push(univariate_f_0);

        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = ExtField::rand(0, &mut rng);

        f_alpha.fix_first_variable(r);
        f_0.fix_first_variable(r);

        cur -= 1;
    }

    // get the evaluation of the new witness
    let y_dash = f_0.w_eval_table[0];

    // sense check
    assert_eq!(y_dash, f_alpha.w_eval_table[0]);

    (univariates_f_alpha, univariates_f_0, y_dash)
}