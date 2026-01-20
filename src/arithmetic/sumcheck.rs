use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};

use crate::arithmetic::{ExtField, FieldExtension, Serialise};

/// Representation of a polynomial that can be used for sum check.
/// The polynomial must have at most 64 variables.
pub trait SumCheckPoly<F: Field> {
    /// Number of variables in the polynomial.
    fn num_vars(&self) -> usize;

    /// Degree (per variable) of the polynomial.
    fn degree(&self) -> usize;

    /// Get the next univariate polynomial p(X_1) = sum_{b \in {0,1}^n-1} p(X_1, b)
    /// Represent this polynomial as k = degree + 1 evaluations on {0, 1, ..., k} where
    /// k is the per-variable degree of p.
    fn get_univariate(&self) -> Univariate<F>;

    /// Fold in the first variable by evaluating it at f.
    fn fix_first_variable(&mut self, r: F);
}

#[derive(Debug, Clone)]
/// Representation of a univariate polynomial of given degree d
/// as evaluations on x = 0, 1, ..., d
pub struct Univariate<F : Field> {
    evals: Vec<F>
}

impl Univariate<ExtField> {
    pub fn init(evals: Vec<ExtField>) -> Self {
        Self { evals }
    }

    /// f(0) + f(1)
    pub fn binary_sum(&self) -> ExtField {
        self.evals[0] + self.evals[1]
    }

    /// f(x)
    pub fn eval(&self, x: ExtField) -> ExtField {
        let n = self.evals.len();
        let mut y = ExtField::ZERO;

        for i in 0..n {
            let mut l_i = ExtField::ONE;

            for j in 0..n {
                if i != j {
                    let x_i = ExtField::lift_int(i as u64);
                    let x_j = ExtField::lift_int(j as u64);
                    l_i *= (x - x_j) * (x_i - x_j).inverse().unwrap();
                }
            }

            y += self.evals[i] * l_i;
        }

        y
    }
}

impl Serialise for Univariate<ExtField> {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for y in &self.evals {
            bytes.extend_from_slice(&y.c0.c0.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c0.c1.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c1.c0.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c1.c1.into_bigint().to_bytes_le());
        }

        bytes
    }
}

/// Equality function on two field elements.
pub fn eq(a: ExtField, b: ExtField) -> ExtField {
    a * b + (ExtField::ONE - a) * (ExtField::ONE - b)
}

/// Equality function on field element and binary element.
pub fn eq_bin(a: ExtField, bin: usize) -> ExtField {
    if bin == 1 { a }
    else { ExtField::ONE - a }
}

/// Given a function table on boolean hypercube, fix the first variable
/// to a particular field element.
pub fn fix_first_variable(eval_table: &Vec<ExtField>, r: ExtField) -> Vec<ExtField> {
    let half = eval_table.len() / 2;
    let mut new_table = vec![ExtField::ZERO; half];

    for suffix in 0..half {
        // p(0, suffix)
        let p_0 = eval_table[suffix << 1];

        // p(1, suffix)
        let p_1 = eval_table[(suffix << 1) | 1];

        // interpolate
        new_table[suffix] = p_0 + (p_1 - p_0) * r;
    }

    new_table
}