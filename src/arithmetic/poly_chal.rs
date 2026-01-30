use ark_ff::AdditiveGroup;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{ExtField, utils::{Logarithm, rand_int}};

/// Representation of a sparse polynomial challenge which contains exactly k coefficients
/// with value -1 or 1, and all other coefficients zero.
/// Store the k non-zero coefficients in a vector of integers where each element 
/// stores the sign in the LSB and the index in the rest of the bits.
pub struct PChal {
    coeffs: Vec<u32>,
}

impl PChal {
    /// Sample a random challenge.
    pub fn rand(d: usize, k: usize, rng: &mut impl RngCore) -> Self {
        // create array [0, 1, 2, ..., D]
        let mut arr = vec![0u32; d];

        for i in 1..d {
            arr[i] = i as u32;
        }

        // iterate from 0 to k-1
        for i in 0..k {
            // generate a random number between 0 and 2(d-i)-1
            let n = (d as u64 - i as u64) * 2;
            let val = rand_int(n, n.log(), rng) as u32;

            let index = (val >> 1) as usize;
            let sign = val & 1;

            // set the next output as index-th unstruck item
            let tmp = arr[i];
            arr[i] = (arr[i + index] << 1) | sign;

            // move the overwritten element into the unstruck set
            if i < k - 1 && index > 0 {
                arr[i + index] = tmp
            }
        }

        // sort the vector to help memory access when multiplying
        let mut v = arr[0..k].to_vec();
        v.sort();

        Self { coeffs : v }
    }

    /// Sample a vector of random challenge polynomials from the given seed.
    pub fn rand_vec(num: usize, d: usize, k: usize, seed: [u8; 32]) -> Vec<Self> {
        let mut vec: Vec<Self> = Vec::with_capacity(num);
        let mut rng = ChaCha12Rng::from_seed(seed);

        for _ in 0..num {
            vec.push(Self::rand(d, k, &mut rng));
        }

        vec
    }

    /// Return the number of non-zero coefficients.
    pub fn k(&self) -> usize {
        self.coeffs.len()
    }

    /// Return the index and sign of the j-th non-zero coefficient of the i-th element.
    /// Sign will be binary. False => -1, True => 1
    pub fn get(&self, i: usize) -> (usize, bool) {
        ((self.coeffs[i] >> 1) as usize, (self.coeffs[i] & 1) == 1)
    }

    /// Evaluate the polynomial at a given extension field element alpha in
    /// given the powers [1, alpha, alpha^2 ... alpha^(d-1)].
    pub fn eval(&self, alpha_pows: &[ExtField]) -> ExtField { 
        let mut out = ExtField::ZERO;

        for i in 0..self.k() {
            let (exp, sign) = self.get(i);

            if sign {
                out += alpha_pows[exp];
            }
            else {
                out -= alpha_pows[exp];
            }
        }

        out
    }
}


#[cfg(test)]
/// Tests for Challenge Polynomial.
mod test_pchal {
    use rand::rng;

    use crate::arithmetic::{poly::Poly, utils::{powers, rand_field}};

    use super::*;

    #[test]
    fn test_rand() {
        // generate a random sparse poly
        let mut rng = rng();
        let p = PChal::rand(1024, 17, &mut rng);

        // check that p has distinct exponents and print the signs for manual inspection
        fn count(p: &PChal, exp: usize) -> usize {
            let mut n = 0;
            
            for i in 0..p.k() {
                let (exp_2,  _) = p.get(i);
                if exp == exp_2 { n += 1 ;}
            }

            n
        }

        let mut total_neg = 0;
        let mut total_pos = 0;

        for i in 0..p.k() {
            let (exp, sign) = p.get(i);
            assert_eq!(1, count(&p, exp));

            if sign {
                total_pos += 1;
            }
            else {
                total_neg += 1;
            }
        }

        println!("Total positive: {}. Total negative: {}", total_pos, total_neg);  
    }

    #[test]
    fn test_rand_vec() {
        // just test it generates the correct number of elements
        let num = 100;
        assert_eq!(num, PChal::rand_vec(num, 64, 40, [1u8; 32]).len());
    }

    #[test]
    fn test_k() {
        let poly = PChal::rand(1024, 17, &mut rng());
        assert_eq!(17, poly.k());

        let poly = PChal::rand(512, 32, &mut rng());
        assert_eq!(32, poly.k());
    }

    #[test]
    fn test_get() {
        let poly = PChal::rand(1024, 17, &mut rng());
        
        for i in 0..poly.k() {
            let (exp, _) = poly.get(i);
            assert!(exp < 1024);
        }
    }

    #[test]
    fn test_eval() {
        let d = 1024;
        let sparse_poly = PChal::rand(d, 17, &mut rng());
        let mut poly = vec![0u64; d];

        // store sparse poly as normal poly
        for i in 0..sparse_poly.k() {
            let (exp, sign) = sparse_poly.get(i);
            
            if sign {
                poly[exp] = 1;
            }
            else {
                poly[exp] = 4294967196; // -1 mod q;
            }
        }

        // get a random evaluation point
        let alpha = rand_field(4294967197, & mut rng());

        // generate the powers [1, alpha, ..., alpha^d-1] and calculate the correct evaluation
        let pows = powers(alpha, d);

        assert_eq!(poly.as_slice().eval(&pows), sparse_poly.eval(&pows));
    }
}