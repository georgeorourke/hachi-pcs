use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{CoeffType, fs::Serialise, poly::Poly, utils::{Logarithm, b_decomp, rand_int}};

/// Representation of a vectors of polynomials over Zq with maximum degree d-1.
/// Stored in default (coefficient) format one after another in a single vector
/// (Array of Structure).
pub struct PVec {
    len: usize,
    d: usize,
    logd: usize,
    vec: Vec<CoeffType>
}

impl PVec {
    /// Initialize a zero vector.
    pub fn zero(len: usize, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        Self { len, d, logd, vec: vec![0; len * d] }
    }

    /// Sample a vector with elements that have uniform random Zq coefficients.
    pub fn rand(len: usize, d: usize, q: CoeffType, seed: [u8; 32]) ->  Self {
        assert!(d.is_power_of_two());
        let logd = d.log();
        let logq = q.log();
        let mut rng = ChaCha12Rng::from_seed(seed);

        let vec: Vec<CoeffType> = (0..len * d).map(|_| rand_int(q, logq, &mut rng)).collect();

        Self { len, d, logd, vec }
    }

    /// Length of vector.
    pub fn length(&self) -> usize {
        self.len
    }

    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, i: usize) -> &[CoeffType] {
        &self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, i: usize) -> &mut [CoeffType] {
        &mut self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// Get a reference to the whole internal vector.
    pub fn slice(&self) -> &[CoeffType] {
        &self.vec
    }

    /// Get a mutable reference to the whole internal vector.
    pub fn mut_slice(&mut self) -> &mut [CoeffType] {
        &mut self.vec
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Balanced decomposition of the vector of polynomials.
    pub fn b_decomp(&self, base: u64, delta: usize, out: &mut Self) {
        assert_eq!(self.length() * delta, out.length());

        let logb = base.log();
        let mut decomp_element = vec![0u64; delta];
        let out = out.mut_slice();

        // iterate over each poly
        for i in 0..self.len {
            // iterate over coefficients of this poly
            for j in 0..self.d {
                // decompose and place in correct coefficient
                b_decomp(self.vec[i * self.d + j], logb, delta, &mut decomp_element);

                for k in 0..delta {
                    out[i * self.d * delta + j + k * self.d] = decomp_element[k];
                }
            }
        }
    }

    /// Perform cyclotomic reduction of each element of the vector.
    pub fn cyclotomic_div(&self, q: CoeffType, quotient: &mut Self, remainder: &mut Self) {
        for i in 0..self.length() {
            self.element(i).cyclotomic_div(q, quotient.mut_element(i), remainder.mut_element(i));
        }
    }
}

/// Serialisation of polynomial in coefficient form.
impl Serialise for PVec {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for x in &self.vec {
            bytes.append(&mut x.to_be_bytes().to_vec());
        }

        bytes
    }
}

/// Clone the vector.
impl Clone for PVec {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, vec: self.vec.clone() }
    }
}

#[cfg(test)]
/// Tests for PolVec.
mod test_poly_vec {
    use super::*;

    #[test]
    fn test_init() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(vec![0u64; 256*32], p1.slice());
    }

    #[test]
    fn test_rand() {
        // just test length of vector and not all zero - rand already tested in Int.
        let q = (1u64 << 32) - 324321;
        let seed = [1u8; 32];

        let p1 = PVec::rand(256, 32, q, seed);
        assert_eq!(256 * 32, p1.slice().len());
        assert_ne!(vec![0u64; 256*32], p1.slice());
    }

    #[test]
    fn test_len() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(256, p1.length());
    }

    #[test]
    fn test_element() {
        let mut p1 = PVec::zero(256, 32);
        let s = p1.mut_slice();

        for i in 0..256 {
            for j in 0..32 {
                s[i * 32 + j] = i as u64;
            }
        }

        for i in 0..256 {
            assert_eq!([i as u64; 32].as_slice(), p1.element(i));
        }
    }

    #[test]
    fn test_mut_element() {
        let mut p1 = PVec::zero(256, 32);
        let s = p1.mut_slice();

        for i in 0..256 {
            for j in 0..32 {
                s[i * 32 + j] = i as u64;
            }
        }

        for i in 0..256 {
            assert_eq!([i as u64; 32].as_mut_slice(), p1.mut_element(i));
        }
    }

    #[test]
    fn test_slice() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(&p1.vec, p1.slice());
    }

    #[test]
    fn test_poly_vec_b_decomp() {
        let base = 16;
        let delta = 8;
        let n = 1024;
        let d = 64;

        // create a random decomposed vector
        let mut decomp_expected = PVec::rand(n * delta, d, base, [1; 32]);
        
        for i in 0..decomp_expected.slice().len() {
            decomp_expected.mut_slice()[i] = decomp_expected.slice()[i].wrapping_sub(base / 2);
        }

        // create the composed vector
        let mut p_vec = PVec::zero(n, d);

        for i in 0..n {
            let v = p_vec.mut_element(i);

            for j in 0..d {
                for k in 0..delta {
                    v[j] = v[j].wrapping_add((decomp_expected.element(i * delta + k)[j] as i64 * base.pow(k as u32) as i64) as u64); 
                }
            }
        }

        // decompose
        let mut decomp_actual = PVec::zero(n * delta, d);
        p_vec.b_decomp(base, delta, &mut decomp_actual);
        assert_eq!(decomp_expected.slice(), decomp_actual.slice());
    }

    #[test]
    fn test_poly_vec_cyclotomic_div() {
        let q = 54351;
        let p = PVec::rand(16, 64, q, [1u8; 32]);
        let mut quo = PVec::zero(16, 32);
        let mut rem = PVec::zero(16, 32);
        p.cyclotomic_div(q, &mut quo, &mut rem);

        let mut expected_quo = vec![0u64; 32];
        let mut expected_rem = vec![0u64; 32];

        for i in 0..p.length() {
            p.element(i).cyclotomic_div(q, &mut expected_quo, &mut expected_rem);
            assert_eq!(expected_quo, quo.element(i));
            assert_eq!(expected_rem, rem.element(i))
        }
    }
}