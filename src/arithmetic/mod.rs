use ark_ff::Field;
use rand::RngCore;
use std::ops::{Add, Mul};

use crate::arithmetic::field::Fq4;
use crate::arithmetic::poly_vec::PolyVec;
use crate::arithmetic::poly_mat::PolyMat;
use crate::arithmetic::sparse_poly::SparsePoly;

pub mod int;
pub mod field;
pub mod poly;
pub mod poly_vec;
pub mod poly_mat;
pub mod sparse_poly;
pub mod ring;
pub mod sumcheck;
pub mod fs;


/// Type used to store coefficients of a polynomial.
pub type CoeffType = u64;

/// Type used to store NTT coefficients.
#[cfg(not(feature = "nightly"))]
pub type NttType = u32;
#[cfg(feature = "nightly")]
pub type NttType = u64;

/// Type used to store elements of extension field.
pub type ExtField = Fq4;

/// Number of NTT friendly rings used in ring multiplication.
#[cfg(not(feature = "nightly"))]
pub const NTT_N: usize = 5;
#[cfg(feature = "nightly")]
pub const NTT_N: usize = 3;

/// Type used to store a vector of polynomials in coefficient form.
pub type PVec = PolyVec<1, CoeffType>;

/// Type used to store a vector of polynomials in NTT form.
pub type PVecNtt = PolyVec<NTT_N, NttType>;

/// Type used to store a matrix of polynomials in coefficient form.
pub type PMat = PolyMat<1, CoeffType>;

/// Type used to store a matrix of polynomials in NTT form.
pub type PMatNtt = PolyMat<NTT_N, NttType>;

// Type used to represent a challenge polynomial.
pub type PChal = SparsePoly;


/// Object can represent a coefficient of a polynomial (normal or NTT).
pub trait Coefficient where Self : Add + Mul + Clone + Copy + Sized + Zero + RandFromRng<(CoeffType, usize)> + Logarithm { }

/// Object can represent an element of the field extension.
pub trait FieldExtension {
    /// Lift an integer in the base field to the field extension
    fn lift_int(x: CoeffType) -> Self;

    /// Multiply by an integer.
    fn mul_int(&self, x: CoeffType) -> Self;
}

/// Object represents a polynomial stored in coefficient form.
pub trait Poly {
    /// Evaluate the polynomial at a given extension field element alpha in
    /// given the powers [1, alpha, alpha^2 ... alpha^(d-1)].
    fn eval(&self, alpha_pows: &[ExtField]) -> ExtField;

    /// Divide by X^d+1 mod q and store quotient and remainder.
    /// Assume the polynomial is of length 2d.
    fn cyclotomic_div(&self, q: CoeffType, quotient: &mut [CoeffType], remainder: &mut [CoeffType]);
}


/// Object can be initialised as zero without parameters.
pub trait Zero {
    fn zero() -> Self;
}

/// Object can be initialised as zero with parameters.
pub trait ZeroWithParams<T> {
    fn zero(params: T) -> Self;
}

/// Object can be randomly sampled from parameters and random number generator.
pub trait RandFromRng<T> {
    fn rand(params: T, rng: &mut impl RngCore) ->  Self;
}

/// Object can be randomly sampled from parameters and seed.
pub trait RandFromSeed<T> {
    fn rand(params: T, seed: [u8; 32]) ->  Self;
}

/// Object can take the logarithm of itself.
pub trait Logarithm {
    fn log(&self) -> usize;
}

/// Object can compute the multilinear coefficient for index i (regarded as a binary index of length len).
/// i = b_1, ..., b_len, return \prod x_j^b_j
pub trait MultiLinearCoeff {
    fn multi_lin_coeff(x: &[Self], i: usize, len: usize, q: CoeffType) -> Self where Self : Sized;
}

/// Object can perform multiplication on provided types.
pub trait Multiply<A, B, C> {
    /// Multiply lhs by rhs and store the result in out (starting at index).
    fn mul(&self, lhs: &A, rhs: &B, out: &mut C, index: usize);
}

///  Object can perform Number Theoretic Transform on provided types.
pub trait Ntt<A, B> {
    /// Forward NTT.
    fn fwd(&self, coeffs: &A, ntt: &mut B);
}

/// Object can be decomposed.
pub trait Decompose<T> {
    /// Unbalanced decomposition and store in the provided buffer.
    /// Coefficients result in range 0..(base-1)
    #[allow(dead_code)]
    fn u_decomp(&self, base: CoeffType, delta: usize, out: T);

    /// Perform balanced decomposition and store in the provided buffer.
    /// Coefficients result in range floor(-base/2)..floor(base/2)
    fn b_decomp(&self, base: CoeffType, delta: usize, out: T);
}

/// Object can be serialised.
pub trait Serialise {
    /// Represent the object as a byte array.
    fn serialise(&self) -> Vec<u8>;
}

/// Compute the powers [1, x, x^2, ..., x^{l-1}]
pub fn powers(x: Fq4, l: usize) -> Vec<Fq4> {
    let mut powers = vec![Fq4::ONE; l];

    for i in 1..l {
        powers[i] = x * powers[i-1];
    }

    powers
}

/// Compute the gadget vector [1, b, b^2, ..., b^{l-1}]
pub fn gadget(b: u64, l: usize) -> Vec<u64> {
    let mut gadget = vec![1u64; l];

    for i in 1..l {
        gadget[i] = b * gadget[i-1];
    }

    gadget
}