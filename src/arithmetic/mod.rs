pub mod field;
pub mod poly;
pub mod poly_vec;
pub mod poly_vec_ntt;
pub mod poly_mat;
pub mod poly_mat_ntt;
pub mod poly_chal;
pub mod ring;
pub mod sumcheck;
pub mod fs;
pub mod utils;

/// Type used to store coefficients of a polynomial.
pub type CoeffType = u64;

/// Type used to store NTT coefficients.
#[cfg(not(feature = "nightly"))]
pub type NttType = u32;
#[cfg(feature = "nightly")]
pub type NttType = u64;

/// Type used to store elements of extension field.
use crate::arithmetic::field::Fq4;
pub type ExtField = Fq4;
