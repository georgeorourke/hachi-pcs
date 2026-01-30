use ark_ff::{AdditiveGroup, Field};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar};

use crate::arithmetic::utils::{Logarithm, eq, eq_bin, lift_int, mul_int_field, rand_field};
use crate::arithmetic::sumcheck::fix_first_variable;
use crate::arithmetic::sumcheck::Univariate;
use crate::arithmetic::sumcheck::SumCheckPoly;
use crate::arithmetic::fs::FS;
use crate::arithmetic::ExtField;

use crate::hachi::setup::Q;

/// Representation of the polynomial f_0 = eq(tau_0, x) w(x)(w(x)-1)(w(x)+1)...
pub struct F0 {
    w: Vec<u64>,                            // witness w
    w_eval_table: Vec<ExtField>,            // table of evaluations for w
    base: u64,                              // decomposition base
    tau_0: Vec<ExtField>,                   // random vector tau_0
    eq_scalar: ExtField,                    // scalar carried through for eq multiplication
    q: u64,                                 // modulus,
    len: usize                              // length of non-padded part of w
}

impl F0 {
    /// Construct F0.
    pub fn init(z_r: &Vec<u64>, base: u64, tau_0: Vec<ExtField>, q: u64, len: usize) -> Self {
        // lift evaluations to field
        // let w: Vec<BaseField> = z_r.iter().map(|x| BaseField::from(*x)).collect();
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();

        Self { w: z_r.clone(), w_eval_table, base, tau_0, eq_scalar: ExtField::ONE, q, len }
    }
}

impl SumCheckPoly<ExtField> for F0 {
    fn degree(&self) -> usize {
        (self.base + 1) as usize
    }

    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log()
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
                let v_x =
                // round 1 - perform operations on w over integers for improved efficiency
                if self.w_eval_table.len() == self.w.len() {
                    // short circuit if in padded section
                    if (suffix << 1) >= self.len {
                        break;
                    }

                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.w[suffix << 1];
                    let w_1 = self.w[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = (w_0 as i64 + (w_1 as i64 - w_0 as i64) * x_i as i64) % self.q as i64;

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = (w_x * (w_x + self.base as i64 / 2)) % self.q as i64;
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = (w_x * w_x) % self.q as i64;

                    for r in 1..self.base as usize / 2 { 
                        v_x = (v_x * (w_x_squared - r as i64 * r as i64)) % self.q as i64;
                    }

                    lift_int(v_x as u64)
                }
                // subsequent rounds
                else {
                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.w_eval_table[suffix << 1];
                    let w_1 = self.w_eval_table[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = w_0 + mul_int_field(x_i, w_1 - w_0);

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = w_x * (w_x + lift_int(self.base / 2));
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = w_x * w_x;

                    for r in 1..self.base as usize / 2 { 
                        v_x *= w_x_squared - lift_int((r * r) as u64);
                    }

                    v_x
                };

                // calculate equality with tau_0
                let eq_x = self.eq_scalar * eq(self.tau_0[0], lift_int(x_i)) * eq_suffix[suffix];

                // multiply by equality
                let y = eq_x * v_x;

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

        // track eq(tau, r)
        self.eq_scalar *= eq(self.tau_0[0], r);

        // update tau_0
        let mut tmp = vec![ExtField::ZERO; half];
        tmp.copy_from_slice(&self.tau_0[1..=half]);
        self.tau_0 = tmp;
    }
}

/// Representation of the polynomial f_alpha(i,x,y) = w(x,y).alpha(y).eq(i).M(i,x)
pub struct FAlpha {
    w: Vec<u64>,                             // table of evaluations for w (integers)
    w_eval_table: Vec<ExtField>,             // table of evaluations for w
    alpha_pows_eval_table: Vec<ExtField>,    // table of evaluations for powers of alpha
    eq_eval_table: Vec<ExtField>,            // table of evaluations for eq_tau_1
    m_alpha_eval_table: Vec<ExtField>,       // table of evaluations for M_alpha
    w_alpha: Vec<ExtField>,                  // pre-compute values of w*alpha,
    q: u64                                   // modulus
}

impl FAlpha {
    /// Construct F0.
    pub fn init(
        z_r: &Vec<u64>, 
        alpha_pows_eval_table: Vec<ExtField>,
        tau_1: Vec<ExtField>,
        m_alpha_eval_table: Vec<ExtField>,
        q: u64
    ) -> Self {
        // lift evaluations to field
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();

        // compute evaluation table for eq(tau, i)
        let log_n = tau_1.len();
        let mut eq_eval_table = vec![ExtField::ONE; 1 << log_n];

        for bin in 0..1 << log_n {
            for i in 0..log_n {
                eq_eval_table[bin] *= eq_bin(tau_1[i], (bin >> i) & 1);
            }
        }

        // pre-compute w[x_y_suffix]*alpha[y_suffix]
        let len_i = eq_eval_table.len();
        let len_x = m_alpha_eval_table.len() / len_i;
        let len_y = w_eval_table.len() / len_x;
        let log_y = len_y.log();

        let mut w_alpha = vec![ExtField::ZERO; len_x * len_y];

        for x_suffix in 0..len_x {
            for y_suffix in 0..len_y {
                let x_y_suffix = (x_suffix << log_y) | y_suffix;

                // still have complete (non-folded) w so can use integers
                let w = z_r[x_y_suffix];

                // get alpha(y_suffix)
                let alpha = alpha_pows_eval_table[y_suffix];

                w_alpha[x_y_suffix] = mul_int_field(w, alpha);
            }
        }

        Self { w: z_r.clone(), w_eval_table, alpha_pows_eval_table, eq_eval_table, m_alpha_eval_table, w_alpha, q }
    }
}

impl SumCheckPoly<ExtField> for FAlpha {
    fn degree(&self) -> usize {
        2
    }

    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log() + self.eq_eval_table.len().log()
    }

    fn get_univariate(&self) -> Univariate<ExtField> {
        // evaluate the univariate polynomial at degree + 1 different points
        let deg = self.degree();
        let mut ys = Vec::<ExtField>::with_capacity(deg + 1);

        // get lengths of remaining variables
        let len_i = self.eq_eval_table.len();
        let len_x = self.m_alpha_eval_table.len() / len_i;
        let len_y = self.w_eval_table.len() / len_x;

        let log_x = len_x.log();
        let log_y = len_y.log();

        // sense check
        assert_eq!(len_y, self.alpha_pows_eval_table.len());

        for x_i in 0..=deg {
            let x_i = x_i as u64;

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
                    let eq_x = eq_0 + mul_int_field(x_i, eq_1 - eq_0);

                    for x_suffix in 0..len_x {
                        let i_x_suffix = (i_suffix << log_x) | x_suffix;

                        // get M_alpha(0, i_x_suffix) and M_alpha(1, i_x_suffix)
                        let m_0 = self.m_alpha_eval_table[i_x_suffix << 1];
                        let m_1 = self.m_alpha_eval_table[(i_x_suffix << 1) | 1];

                        // evaluation at x is linear interpolation
                        let m_x = m_0 + mul_int_field(x_i, m_1 - m_0);

                        let mut y = eq_x * m_x;

                        for y_suffix in 0..len_y {
                            let x_y_suffix = (x_suffix << log_y) | y_suffix;
                            y *=  self.w_alpha[x_y_suffix];
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
                    let m_x = m_0 + mul_int_field(x_i, m_1 - m_0);

                    let mut y = eq * m_x;

                    for y_suffix in 0..len_y {
                        let x_y_suffix = (x_suffix << log_y) | y_suffix;

                        // if this is first round of x being folded then w can use integers for w
                        if self.w_eval_table.len() == self.w.len() {
                            // get w(0, x_y_suffix) and w(1, x_y_suffix)
                            let w_0 = self.w[x_y_suffix << 1];
                            let w_1 = self.w[(x_y_suffix << 1) | 1];

                            // evaluation at x is linear interpolation
                            let w_x = (w_0 as i64 + (w_1 as i64 - w_0 as i64) * x_i as i64) % self.q as i64;

                            // get alpha(y_suffix)
                            let alpha = self.alpha_pows_eval_table[y_suffix];

                            y *= mul_int_field(w_x as u64, alpha);
                            y_i += y;

                        }
                        // otherwise use field elements
                        else 
                        {
                            // get w(0, x_y_suffix) and w(1, x_y_suffix)
                            let w_0 = self.w_eval_table[x_y_suffix << 1];
                            let w_1 = self.w_eval_table[(x_y_suffix << 1) | 1];

                            // evaluation at x is linear interpolation
                            let w_x = w_0 + mul_int_field(x_i,w_1 - w_0);

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
                    let w_x = w_0 + mul_int_field(x_i,w_1 - w_0);

                    // get alpha(0, y_suffix) and alpha(1, y_suffix)
                    let alpha_0 = self.alpha_pows_eval_table[y_suffix << 1];
                    let alpha_1 = self.alpha_pows_eval_table[(y_suffix << 1) | 1];

                    // evaluation of eq at x is linear interpolation
                    let alpha_x = alpha_0 + mul_int_field(x_i,alpha_1 - alpha_0);

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
        progress_bar("Sum Check" , rounds_f_alpha - cur, rounds_f_alpha);

        let univariate_f_alpha = f_alpha.get_univariate();
        fs.push(&univariate_f_alpha);
        univariates_f_alpha.push(univariate_f_alpha);

        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = rand_field(Q, &mut rng);

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
        let r = rand_field(Q, &mut rng);

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