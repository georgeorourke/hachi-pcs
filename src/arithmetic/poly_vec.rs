use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{CoeffType, Coefficient, Decompose, Logarithm, Poly, RandFromSeed, Serialise, ZeroWithParams};

/// Representation of a vector of polynomials over ZZq with maximum
/// degree d-1.
/// 
/// n is the number of coefficients that must be stored per standard
/// coefficient of the polynomial. I.e. n=1 when storing the polynomial
/// in coefficient domain, and n=3/5 when storing the polynomial in
/// multi-modular NTT domain.
pub struct PolyVec<const N: usize, T : Coefficient> {
    length: usize,
    d: usize,
    logd: usize,
    vec: Vec<T>
}

/// Zero initialisation.
impl<const N: usize, T: Coefficient> ZeroWithParams<(usize, usize)> for PolyVec<N, T> {
    fn zero((length, d): (usize, usize)) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        Self { length, d, logd, vec: vec![T::zero(); length * d * N] }
    }
}

/// Random initialisation from seed.
impl<const N: usize, T: Coefficient> RandFromSeed<(usize, usize, CoeffType)> for PolyVec<N, T> {
    fn rand((length, d, q): (usize, usize, CoeffType), seed: [u8; 32]) ->  Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        let len = length * d * N;
        let mut vec = vec![T::zero(); len];
        let mut rng = ChaCha12Rng::from_seed(seed);

        let logq = q.log();  

        for i in 0..len {
            vec[i] = T::rand((q, logq), &mut rng);
        }

        Self { length, d, logd, vec }
    }
}

/// Generic methods.
impl<const N: usize, T : Coefficient> PolyVec<N, T> {
    /// Length of vector.
    pub fn length(&self) -> usize {
        self.length
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, i: usize) -> &[T] {
        &self.vec[(i << self.logd) * N..((i + 1) << self.logd) * N]
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, i: usize) -> &mut [T] {
        &mut self.vec[(i << self.logd) * N..((i + 1) << self.logd) * N]
    }

    /// Get a reference to the whole internal vector.
    pub fn slice(&self) -> &[T] {
        &self.vec
    }

    /// Get a mutable reference to the whole internal vector.
    pub fn mut_slice(&mut self) -> &mut [T] {
        &mut self.vec
    }
}

/// Methods specific to polynomial in coefficient form.
impl PolyVec<1, CoeffType> {
    /// Perform cyclotomic reduction of each element of the vector.
    pub fn cyclotomic_div(&self, q: u64, quotient: &mut Self, remainder: &mut Self) {
        for i in 0..self.length() {
            self.element(i).cyclotomic_div(q, quotient.mut_element(i), remainder.mut_element(i));
        }
    }
}

/// Decomposition of polynomial in coefficient form.
impl Decompose<&mut Self> for PolyVec<1, CoeffType> {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn u_decomp(&self, base: u64, delta: usize, out: &mut Self) {
        assert_eq!(self.length() * delta, out.length());

        let logb = base.log() as u64;
        let mut decomp_element = vec![0u64; delta];
        let out = out.mut_slice();

        // iterate over each poly
        for i in 0..self.length {
            // iterate over coefficients of this poly
            for j in 0..self.d {
                // decompose and place in correct coefficient
                self.vec[i * self.d + j].u_decomp(logb, delta, &mut decomp_element);

                for k in 0..delta {
                    out[i * self.d * delta + j + k * self.d] = decomp_element[k];
                }
            }
        }
    }
    
    // Below AVX-512 implementation is slower than no SIMD.
    // #[cfg(feature = "nightly")]
    // #[cfg_attr(feature = "stats", time_graph::instrument)]
    // /// Optimise unbalanced decomposition with AVX-512.
    // /// Assume the ring dimension is always a power of two >= 16.
    // /// Actually (marginally) slower than not using vector instructions,
    // /// probably due to having to load vector registers.
    // fn u_decomp(&self, base: u64, delta: usize, out: &mut Self) {
    //     use core::arch::x86_64::*;

    //     // assume that the decomposition base is hard coded as 16=2^4
    //     // needed since _mm512_srli_epi64 requires constant shift
    //     // TODO: extend for other bases
    //     assert_eq!(16, base);

    //     // compute and store the next element of the decomposition for each of a slice of integers
    //     // i.e. the remainders mod b
    //     // return the next elements to decompose, i.e. division by q
    //     #[cfg_attr(feature = "stats", time_graph::instrument)]
    //     unsafe fn extract_one(v_cur: __m512i, v_mask: __m512i, decomp: &mut [u64]) -> __m512i {
    //         // take remainder mod base
    //         let v_rem = unsafe { _mm512_and_si512(v_cur, v_mask) };

    //         // store in decomp
    //         let decomp_ptr = decomp.as_mut_ptr() as *mut __m512i;
    //         unsafe { _mm512_storeu_si512(decomp_ptr, v_rem) };

    //         // compute next values by dividing by base (shifting right 4 bits = divide by 16)
    //         unsafe { _mm512_srli_epi64(v_cur, 4) }
    //     }

    //     // create the vector mask
    //     let v_mask = unsafe { _mm512_set1_epi64((base - 1) as i64) };

    //     // we can decompose 8 64 bit integers as a single vector
    //     let lanes = 8;

    //     // number of integer vector we need to decompose per ring element
    //     // assume 8 divides d
    //     let reps_per_element = self.d / lanes;

    //     // pointer the the vector
    //     let mut x_ptr = self.vec.as_ptr();

    //     for i in 0..self.length() {
    //         for j in 0..reps_per_element {
    //             // load in the next <lanes> integers
    //             let mut v_cur: __m512i = unsafe { _mm512_loadu_si512(x_ptr as *const __m512i) };

    //             // decompose
    //             for k in 0..delta {
    //                 v_cur = unsafe { extract_one(v_cur, v_mask, &mut out.mut_element(i * delta + k)[j * lanes..(j + 1) * lanes]) };
    //             }

    //             x_ptr = unsafe { x_ptr.add(lanes) };
    //         }
    //     }
    // }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn b_decomp(&self, base: u64, delta: usize, out: &mut Self) {
        assert_eq!(self.length() * delta, out.length());

        let logb = base.log() as u64;
        let mut decomp_element = vec![0u64; delta];
        let out = out.mut_slice();

        // iterate over each poly
        for i in 0..self.length {
            // iterate over coefficients of this poly
            for j in 0..self.d {
                // decompose and place in correct coefficient
                self.vec[i * self.d + j].b_decomp(logb, delta, &mut decomp_element);

                for k in 0..delta {
                    out[i * self.d * delta + j + k * self.d] = decomp_element[k];
                }
            }
        }
    }
}

/// Serialisation  of polynomial in coefficient form.
impl Serialise for PolyVec<1, CoeffType> {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for x in &self.vec {
            bytes.append(&mut x.serialise());
        }

        bytes
    }
}

#[cfg(test)]
/// Tests for PolVec.
mod test_poly_vec {
    use super::*;

    #[test]
    fn test_init() {
        let p1 = PolyVec::<1, u64>::zero((256, 32));
        assert_eq!(vec![0u64; 256*32], p1.slice());

        let p2 = PolyVec::<3, u64>::zero((512, 1024));
        assert_eq!(vec![0u64; 3*512*1024], p2.slice());

        let p3 = PolyVec::<5, u32>::zero((512, 64));
        assert_eq!(vec![0u32; 5*512*64], p3.slice());
    }

    #[test]
    fn test_rand() {
        // just test length of vector and not all zero - rand already tested in Int.
        let q = (1u64 << 32) - 324321;
        let seed = [1u8; 32];

        let p1 = PolyVec::<1, u64>::rand((256, 32, q), seed);
        assert_eq!(256 * 32, p1.slice().len());
        assert_ne!(vec![0u64; 256*32], p1.slice());

        let p2 = PolyVec::<3, u64>::rand((512, 1024, q), seed);
        assert_eq!(3 * 512 * 1024, p2.slice().len());
        assert_ne!(vec![0u64; 3*512*1024], p2.slice());

        let p3 = PolyVec::<5, u32>::rand((512, 64, q), seed);
        assert_eq!(5 * 512 * 64, p3.slice().len());
        assert_ne!(vec![0u32; 5*512*64], p3.slice());
    }

    #[test]
    fn test_len() {
        let p1 = PolyVec::<1, u64>::zero((256, 32));
        assert_eq!(256, p1.length());

        let p2 = PolyVec::<3, u64>::zero((512, 1024));
        assert_eq!(512, p2.length());

        let p3 = PolyVec::<5, u32>::zero((512, 64));
        assert_eq!(512, p3.length());
    }

    #[test]
    fn test_element() {
        let mut p1 = PolyVec::<1, u64>::zero((256, 32));
        let s = p1.mut_slice();

        for i in 0..256 {
            for j in 0..32 {
                s[i * 32 + j] = i as u64;
            }
        }

        for i in 0..256 {
            assert_eq!([i as u64; 32].as_slice(), p1.element(i));
        }

        let p2 = PolyVec::<3, u64>::rand((16, 64, 100), [1u8; 32]);

        for i in 0..16 {
            assert_eq!(&p2.vec[3 * i * 64..3 * (i + 1) * 64], p2.element(i));
        }
    }

    #[test]
    fn test_mut_element() {
        let mut p1 = PolyVec::<1, u64>::zero((256, 32));
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
        let p1 = PolyVec::<1, u64>::zero((256, 32));
        assert_eq!(&p1.vec, p1.slice());

        let p2 = PolyVec::<3, u64>::rand((16, 64, 100), [1u8; 32]);
        assert_eq!(&p2.vec, p2.slice());
    }
}

#[cfg(test)]
/// Tests for PolyVec cyclotomic reduction.
mod test_poly_vec_cyclotomic_div {
    use super::*;

    #[test]
    fn test_poly_vec_cyclotomic_div() {
        let q = 54351;
        let p = PolyVec::<1, u64>::rand((16, 64, q), [1u8; 32]);
        let mut quo = PolyVec::<1, u64>::zero((16, 32));
        let mut rem = PolyVec::<1, u64>::zero((16, 32));
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

#[cfg(test)]
/// Tests for PolyVec decomposition.
mod test_poly_vec_decomp {
    use super::*;

    #[test]
    fn test_poly_vec_u_decomp() {
        let base = 16;
        let delta = 8;
        let n = 256;
        let d = 1024;

        // create a random decomposed vector
        let decomp_expected = PolyVec::<1, u64>::rand((n * delta, d, base), [1; 32]);

        // create the composed vector
        let mut p_vec = PolyVec::<1, u64>::zero((n, d));

        for i in 0..n {
            let v = p_vec.mut_element(i);

            for j in 0..d {
                for k in 0..delta {
                    v[j] += decomp_expected.element(i * delta + k)[j] * base.pow(k as u32); 
                }
            }
        }

        // decompose
        let mut decomp_actual = PolyVec::<1, u64>::zero((n * delta, d));
        p_vec.u_decomp(base, delta, &mut decomp_actual);
        assert_eq!(decomp_expected.slice(), decomp_actual.slice());
    }

    #[test]
    fn test_poly_vec_b_decomp() {
        let base = 16;
        let delta = 8;
        let n = 1024;
        let d = 64;

        // create a random decomposed vector
        let mut decomp_expected = PolyVec::<1, u64>::rand((n * delta, d, base), [1; 32]);
        
        for i in 0..decomp_expected.slice().len() {
            decomp_expected.mut_slice()[i] = decomp_expected.slice()[i].wrapping_sub(base / 2);
        }

        // create the composed vector
        let mut p_vec = PolyVec::<1, u64>::zero((n, d));

        for i in 0..n {
            let v = p_vec.mut_element(i);

            for j in 0..d {
                for k in 0..delta {
                    v[j] = v[j].wrapping_add((decomp_expected.element(i * delta + k)[j] as i64 * base.pow(k as u32) as i64) as u64); 
                }
            }
        }

        // decompose
        let mut decomp_actual = PolyVec::<1, u64>::zero((n * delta, d));
        p_vec.b_decomp(base, delta, &mut decomp_actual);
        assert_eq!(decomp_expected.slice(), decomp_actual.slice());
    }
}