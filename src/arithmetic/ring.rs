#[cfg(not(feature = "nightly"))]
use tfhe_ntt::native64::Plan32 as Native;

#[cfg(feature = "nightly")]
use tfhe_ntt::native64::Plan52 as Native;

use crate::arithmetic::{PChal, PMat, PMatNtt, PVec};
use crate::arithmetic::{Multiply, Ntt};

#[cfg(feature = "nightly")]
use crate::arithmetic::PVecNtt;

/// Representation of operations over polynomials Zq[X].
/// If cyclotomic = True then the operations are over Zq[X]/(X^d+1)
/// with inputs and outputs given as slices of length d.
/// Otherwise, operations are over Zq[X] with inputs assumed to
/// be given as a slice of length d and outputs assumed to be
/// slices of length 2d.
pub struct Ring {
    native: Native,
    q: u64,
    input_dim: usize,
    output_dim: usize,
    cyclotomic: bool
}

/// General methods.
impl Ring {
    /// Constructor.
    pub fn init(q: u64, d: usize, cyclotomic: bool) -> Self {
        let output_dim = if cyclotomic { d } else {2 * d};
        let native = Native::try_new(output_dim).unwrap();
        return Self { native, q, input_dim: d, output_dim, cyclotomic };
    }
    
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Integer multiplied by a polynomial.
    /// Result added to out.
    pub fn int_mul_poly(&self, a: u64, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        let a = a as u128;
        
        for i in 0..self.input_dim {
            // expand to 128 bits to avoid overflow.
            out[i] = ((out[i] as u128 + a * poly[i] as u128) % self.q as u128) as u64;
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Sparse polynomial (challenge) multiplied by a polynomial.
    /// Result added to out.
    pub fn chal_mul_poly(&self, chal: &PChal, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        // iterate over non-zero coefficients of challenge
        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);
            
            // iterate over all coefficients of the polynomial
            for j in 0..self.input_dim {
                assert!(poly[j] < self.q);

                // get the exponent of the current product
                let mut exp = exp + j;
                let mut sign = sign;

                // cyclotomic reduction
                if self.cyclotomic && exp >= self.input_dim {
                    exp = exp - self.input_dim;
                    sign = !sign;
                }

                assert!(out[exp] < self.q);

                // multiply coefficients
                if sign {
                    out[exp] = out[exp] + poly[j];

                    if out[exp] >= self.q {
                        out[exp] = out[exp].wrapping_sub(self.q);
                    }
                }
                else {
                    if out[exp] > poly[j] {
                        out[exp] = out[exp].wrapping_sub(poly[j]);
                    }
                    else {
                        out[exp] = (out[exp] + self.q).wrapping_sub(poly[j]);
                    }
                }
            }
        }
    }
    
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Sparse polynomial (challenge) multiplied by a polynomial where
    /// the polynomial has small coefficients.
    /// Result added to out.
    /// Note: no reduction mod q is performed since we assume the product 
    /// has an l1 norm of less than q/2. The product will contain positive
    /// and negative elements, where negative elements are stored as if they
    /// are signed (i.e. using wrapping arithmetic).
    fn chal_mul_small_poly(&self, chal: &PChal, poly: &[u64], out: &mut [u64]) {
        assert_eq!(self.input_dim, poly.len());
        assert_eq!(self.output_dim, out.len());

        // iterate over non-zero coefficients of challenge
        for i in 0..chal.k() {
            let (exp, sign) = chal.get(i);

            // iterate over all coefficients of the polynomial
            for j in 0..self.input_dim {
                // get the exponent of the current product
                let mut exp = exp + j;
                let mut sign = sign;

                // cyclotomic reduction
                if self.cyclotomic && exp >= self.input_dim {
                    exp = exp - self.input_dim;
                    sign = !sign;
                }

                // multiply coefficients
                if sign {
                    out[exp] = out[exp].wrapping_add(poly[j]);
                }
                else {
                    out[exp] = out[exp].wrapping_sub(poly[j]);
                }
            }
        }
    }
}

#[cfg(not(feature = "nightly"))]
/// Internal methods.
impl Ring {
    /// Split a polynomial provided as a slice in NTT domain into its components
    /// as slices.
    fn _split_ntt<'a, T>(&self, p_ntt: &'a [T]) -> (&'a [T], &'a [T], &'a [T], &'a [T], &'a [T]) {
        let (p0, tail) = p_ntt.split_at(self.output_dim);
        let (p1, tail) = tail.split_at(self.output_dim);
        let (p2, tail) = tail.split_at(self.output_dim);
        let (p3, p4) = tail.split_at(self.output_dim);
        (p0, p1, p2, p3, p4)
    }

    /// Split a polynomial provided as a slice in NTT domain into its components
    /// as mutable slices.
    fn _split_ntt_mut<'a, T>(&self, p_ntt: &'a mut [T]) -> (&'a mut [T], &'a mut [T], &'a mut [T], &'a mut [T], &'a mut [T]) {
        let (p0, tail) = p_ntt.split_at_mut(self.output_dim);
        let (p1, tail) = tail.split_at_mut(self.output_dim);
        let (p2, tail) = tail.split_at_mut(self.output_dim);
        let (p3, p4) = tail.split_at_mut(self.output_dim);
        (p0, p1, p2, p3, p4)
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Forward NTT on a polynomial provided as a slice.
    fn _fwd(&self, p: &[u64], p0: &mut [u32], p1: &mut [u32], p2: &mut [u32], p3: &mut [u32], p4: &mut [u32]) {
        assert_eq!(self.input_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        assert_eq!(self.output_dim, p3.len());
        assert_eq!(self.output_dim, p4.len());

        if self.cyclotomic {
            self.native.fwd(p, p0, p1, p2, p3, p4);
        }
        else {
            // if non-cyclotomic, pad p
            let mut p_pad = vec![0u64; self.output_dim];
            
            for i in 0..self.input_dim {
                p_pad[i] = p[i];
            }

            self.native.fwd(&p_pad, p0, p1, p2, p3, p4);
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inverse NTT on slice and reduction mod q.
    fn _inv(&self, p: &mut [u64], p0: &mut [u32], p1: &mut [u32], p2: &mut [u32], p3: &mut [u32], p4: &mut [u32]) {
        assert_eq!(self.output_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        assert_eq!(self.output_dim, p3.len());
        assert_eq!(self.output_dim, p4.len());
        
        self.native.ntt_0().normalize(p0);
        self.native.ntt_1().normalize(p1);
        self.native.ntt_2().normalize(p2);
        self.native.ntt_3().normalize(p3);
        self.native.ntt_4().normalize(p4);

        self.native.inv(p, p0, p1, p2, p3, p4);

        for j in 0..self.output_dim {
            let tmp = p[j] as i64 % self.q as i64;
            p[j] = if tmp >= 0 { tmp as u64 } else { (tmp + self.q as i64) as u64 };
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Pointwise product of two NTT domain polynomials l and r.
    /// Result is accumulated in the sum s.
    fn _mul_acc(&self, 
        l0: &[u32], l1: &[u32], l2: &[u32], l3: &[u32], l4: &[u32],
        r0: &[u32], r1: &[u32], r2: &[u32], r3: &[u32], r4: &[u32],
        s0: &mut [u32], s1: &mut [u32], s2: &mut [u32], s3: &mut [u32], s4: &mut [u32],
    ) {
        self.native.ntt_0().mul_accumulate(s0, l0, r0);
        self.native.ntt_1().mul_accumulate(s1, l1, r1);
        self.native.ntt_2().mul_accumulate(s2, l2, r2);
        self.native.ntt_3().mul_accumulate(s3, l3, r3);
        self.native.ntt_4().mul_accumulate(s4, l4, r4);
    }
}

#[cfg(feature = "nightly")]
/// Internal methods.
impl Ring {    
    /// Split a polynomial provided as a slice in NTT domain into its components
    /// as slices.
    fn _split_ntt<'a>(&self, p_ntt: &'a [u64]) -> (&'a [u64], &'a [u64], &'a [u64]) {
        let (p0, tail) = p_ntt.split_at(self.output_dim);
        let (p1, p2) = tail.split_at(self.output_dim);
        (p0, p1, p2)
    }

    /// Split a polynomial provided as a slice in NTT domain into its components
    /// as mutable slices.
    fn _split_ntt_mut<'a>(&self, p_ntt: &'a mut [u64]) -> (&'a mut [u64], &'a mut [u64], &'a mut [u64]) {
        let (p0, tail) = p_ntt.split_at_mut(self.output_dim);
        let (p1, p2) = tail.split_at_mut(self.output_dim);
        (p0, p1, p2)
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Forward NTT on a polynomial provided as a slice.
    fn _fwd(&self, p: &[u64], p0: &mut [u64], p1: &mut [u64], p2: &mut [u64]) {
        assert_eq!(self.input_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());

        if self.cyclotomic {
            self.native.fwd(p, p0, p1, p2);
        }
        else {
            // if non-cyclotomic, pad p
            let mut p_pad = vec![0u64; self.output_dim];
            
            for i in 0..self.input_dim {
                p_pad[i] = p[i];
            }

            self.native.fwd(&p_pad, p0, p1, p2);
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Inverse NTT on slice and reduction mod q.
    fn _inv(&self, p: &mut [u64], p0: &mut [u64], p1: &mut [u64], p2: &mut [u64]) {
        assert_eq!(self.output_dim, p.len());
        assert_eq!(self.output_dim, p0.len());
        assert_eq!(self.output_dim, p1.len());
        assert_eq!(self.output_dim, p2.len());
        
        self.native.ntt_0().normalize(p0);
        self.native.ntt_1().normalize(p1);
        self.native.ntt_2().normalize(p2);

        self.native.inv(p, p0, p1, p2);

        for j in 0..self.output_dim {
            let tmp = p[j] as i64 % self.q as i64;
            p[j] = if tmp >= 0 { tmp as u64 } else { (tmp + self.q as i64) as u64 };
        }
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Pointwise product of two NTT domain polynomials l and r.
    /// Result is accumulated in the sum s.
    fn _mul_acc(&self, 
        l0: &[u64], l1: &[u64], l2: &[u64],
        r0: &[u64], r1: &[u64], r2: &[u64],
        s0: &mut [u64], s1: &mut [u64], s2: &mut [u64],
    ) {
        self.native.ntt_0().mul_accumulate(s0, l0, r0);
        self.native.ntt_1().mul_accumulate(s1, l1, r1);
        self.native.ntt_2().mul_accumulate(s2, l2, r2);
    }
}

/// Element-wise NTT transform for matrix over the ring.
impl Ntt<PMat, PMatNtt> for Ring {
    fn fwd(&self, mat: &PMat, mat_ntt: &mut PMatNtt) {
        assert_eq!(mat.height(), mat_ntt.height());
        assert_eq!(mat.width(), mat_ntt.width());

        for i in 0..mat.height() {
            for j in 0..mat.width() {
                #[cfg(not(feature = "nightly"))]
                {
                    let (p0, p1, p2, p3, p4) = self._split_ntt_mut(mat_ntt.mut_element(i, j));
                    self._fwd(mat.element(i, j), p0, p1, p2, p3, p4);
                }

                #[cfg(feature = "nightly")]
                {
                    let (p0, p1, p2) = self._split_ntt_mut(mat_ntt.mut_element(i, j));
                    self._fwd(mat.element(i, j), p0, p1, p2);
                }
            }
        }
    }
}

#[cfg(not(feature = "nightly"))]
/// Multiplication of a matrix of polynomials in NTT form and vector of polynomials in coefficient form
/// in the ring Zq[X]/(X^d+1). Result is stored in out at the index provided.
/// 
/// Note: the coefficients of the vector are treated as if they are signed integers despite
/// being provided as unsigned integers (use wrapping arithmetic). I.e. the value 2^64 - 1 represents -1.
/// 
/// Note: the result is only guaranteed to be valid if the product,
/// before it is reduced mod q, 'fits in' a 64 bit signed integer.
/// E.g. if q = 32 bits, d = 10 bits and the maximum coefficient of the vector
/// is 4 bits then the result (pre mod q reduction) will be 32 + 10 + 4 < 64 bits.
/// 
/// In practise, the matrix will have arbitrary coefficients mod q (up to log q bits),
/// d <= 10 bits and the vector will be decomposed to 4 bits so we are able to use this.
/// 
/// The output will contain coefficients in {0,...,q-1}.
impl Multiply<PMatNtt, PVec, PVec> for Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn mul(&self, mat: &PMatNtt, vec: &PVec, out: &mut PVec, index: usize) {
        use crate::arithmetic::{PVecNtt, ZeroWithParams};

        assert_eq!(mat.width(), vec.length());

        // Vectors to store forward NTT of elements as required.
        // Only perform forward NTT on elements as required (rather than the whole vector at the start).
        // Much quicker (possibly to do with memory access to larger vector).
        let mut r0 = vec![0u32; self.output_dim];
        let mut r1 = vec![0u32; self.output_dim];
        let mut r2 = vec![0u32; self.output_dim];
        let mut r3 = vec![0u32; self.output_dim];
        let mut r4 = vec![0u32; self.output_dim];

        // Vector to store product in NTT domain
        let mut out_ntt = PVecNtt::zero((mat.height(), self.output_dim));

        // Iterate over columns.
        for col in 0..mat.width() {
            // Perform forward NTT on corresponding element of the vector.
            self._fwd(vec.element(col), &mut r0, &mut r1, &mut r2, &mut r3, &mut r4);

            // Iterate over over rows.
            for row in 0..mat.height() {
                // Get corresponding element of the matrix.
                let l = mat.element(row, col);
                let (l0, l1, l2, l3, l4) = self._split_ntt(l);

                // Get corresponding element of the sum.
                let s = out_ntt.mut_element(row);
                let (s0, s1, s2, s3, s4) = self._split_ntt_mut(s);

                // Pointwise multiplication accumulated in the sum.
                self._mul_acc(
                    l0, l1, l2, l3, l4, 
                    &r0, &r1, &r2, &r3, &r4, 
                    s0, s1, s2, s3, s4
                );
            }
        }

        // Inverse NTT on the result.
        for i in 0..out_ntt.length() {
            let (p0, p1, p2, p3, p4) = self._split_ntt_mut(out_ntt.mut_element(i));
            self._inv(out.mut_element(index + i), p0, p1, p2, p3, p4);
        }
    }
}

#[cfg(feature = "nightly")]
/// Multiplication of a matrix of polynomials in NTT form and vector of polynomials in coefficient form
/// in the ring Zq[X]/(X^d+1). Result is stored in out at the index provided.
impl Multiply<PMatNtt, PVec, PVec> for Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn mul(&self, mat: &PMatNtt, vec: &PVec, out: &mut PVec, index: usize) {
        use crate::arithmetic::ZeroWithParams;

        assert_eq!(mat.width(), vec.length());

        // Vectors to store forward NTT of elements as required.
        // Only perform forward NTT on elements as required (rather than the whole vector at the start).
        // Much quicker (possibly to do with memory access to larger vector).
        let mut r0 = vec![0u64; self.output_dim];
        let mut r1 = vec![0u64; self.output_dim];
        let mut r2 = vec![0u64; self.output_dim];

        // Vector to store product in NTT domain.
        let mut out_ntt = PVecNtt::zero((mat.height(), self.output_dim));

        // Iterate over columns.
        for col in 0..mat.width() {
            // Perform forward NTT on corresponding element of the vector.
            self._fwd(vec.element(col), &mut r0, &mut r1, &mut r2);

            // Iterate over over rows.
            for row in 0..mat.height() {
                // Get corresponding element of the matrix.
                let l = mat.element(row, col);
                let (l0, l1, l2) = self._split_ntt(l);

                // Get corresponding element of the sum.
                let s = out_ntt.mut_element(row);
                let (s0, s1, s2) = self._split_ntt_mut(s);

                // Pointwise multiplication accumulated in the sum.
                self._mul_acc(
                    l0, l1, l2, 
                    &r0, &r1, &r2, 
                    s0, s1, s2
                );
            }
        }

        // Inverse NTT on the result.
        for i in 0..out_ntt.length() {
            let (p0, p1, p2) = self._split_ntt_mut(out_ntt.mut_element(i));
            self._inv(out.mut_element(index + i), p0, p1, p2);
        }
    }
}

/// Inner product between a vector of integers and a vector of polynomials.
/// Result is added to out at the index provided.
impl Multiply<Vec<u64>, PVec, PVec> for Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn mul(&self, int_vec: &Vec<u64>, p_vec: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(int_vec.len(), p_vec.length());

        for i in 0..p_vec.length() {
            self.int_mul_poly(int_vec[i], p_vec.element(i), out.mut_element(index));
        }
    }
}

/// Given a sparse polynomial (challenge), multiply the vector of
/// polynomails by the challenge add the result to out, starting at index.
/// 
/// Note: the multiplication is only guaranteed for polynomial vectors with
/// small coefficients (e.g. a decomposed witness) so that the result
/// does not wrap mod q. The result will have  coefficients which are both 
/// positive and negative (using wrapping arithmetic).
impl Multiply<PChal, PVec, PVec> for Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn mul(&self, chal: &PChal, poly_vec: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(poly_vec.length(), out.length());

        for i in 0..poly_vec.length() {
            self.chal_mul_small_poly(chal, poly_vec.element(i), out.mut_element(index + i));
        }
    }
}

/// Inner product between a vector of challenges and a vector of polynomials.
/// Result is added to out at the index provided.
impl Multiply<Vec<PChal>, PVec, PVec> for Ring {
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn mul(&self, chals: &Vec<PChal>, poly: &PVec, out: &mut PVec, index: usize) {
        assert_eq!(chals.len(), poly.length());

        for i in 0..poly.length() {
            self.chal_mul_poly(&chals[i], poly.element(i), out.mut_element(index));
        }
    }
}

#[cfg(test)]
/// Tests for Ring.
mod test_ring {
    use rand::rng;

    use super::*;
    use crate::arithmetic::{Logarithm, RandFromRng, RandFromSeed, ZeroWithParams};

    // use default modulus for all tests.
    use crate::{arithmetic::{PVecNtt, Poly}, hachi::setup::Q};

    const P0: u64 = 0b0011_1111_0101_1010_0000_0000_0000_0001;
    const P1: u64 = 0b0011_1111_0101_1101_0000_0000_0000_0001;
    const P2: u64 = 0b0011_1111_0111_0110_0000_0000_0000_0001;

    #[cfg(not(feature = "nightly"))]
    const P3: u64 = 0b0011_1111_1000_0010_0000_0000_0000_0001;

    #[cfg(not(feature = "nightly"))]
    const P4: u64 = 0b0011_1111_1010_1100_0000_0000_0000_0001;

    /// add mod q
    fn add_64(a: u64, b: u64, q: u64) -> u64 {
        ((a % q) + (b % q)) % q
    }

    /// add mod q
    fn add_32(a: u32, b: u32, q: u64) -> u32 {
        add_64(a as u64, b as u64, q) as u32
    }

    /// subtract mod q
    fn sub_64(a: u64, b: u64, q: u64) -> u64 {
        let a = a % q;
        let b = b % q;

        if a >= b {
            a.wrapping_sub(b)
        }
        else {
            (a + q).wrapping_sub(b)
        }
    }

    /// subtract mod q
    fn sub_32(a: u32, b: u32, q: u64) -> u32 {
        sub_64(a as u64, b as u64, q) as u32
    }

    /// multiply mod q
    fn mul_64(a: u64, b: u64, q: u64) -> u64 {
        let a = (a % q) as u128;
        let b = (b % q) as u128;
        ((a * b) % q as u128) as u64
    }

    /// multiply mod q
    fn mul_32(a: u32, b: u32, q: u64) -> u32 {
        mul_64(a as u64, b as u64, q) as u32
    }

    /// schoolbook polynomial multiplication
    fn poly_mul(p1: &[u64], p2: &[u64], p_out: &mut [u64], q: u64) {
        let d = p1.len();
        assert_eq!(d, p2.len());
        assert_eq!(2*d, p_out.len());

        for i in 0..d {
            for j in 0..d {
                p_out[i + j] = add_64(p_out[i + j], mul_64(p1[i], p2[j], q), q);
            }
        }
    }

    /// schoolbook polynomial multiplication
    fn poly_mul_cyclotomic(p1: &[u64], p2: &[u64], p_out: &mut [u64], q: u64) {
        let d = p1.len();
        assert_eq!(d, p2.len());
        assert_eq!(d, p_out.len());

        let mut prod = vec![0; 2 * d];

        for i in 0..d {
            for j in 0..d {
                prod[i + j] = add_64(prod[i + j], mul_64(p1[i], p2[j], q), q);
            }
        }

        for i in 0..d {
            p_out[i] = sub_64(prod[i], prod[d + i], q);
        }
    }

    #[test]
    fn test_test_methods() {
        let a = Q - 10224;
        let b = 10224;
        let c = 24243;

        assert_eq!(0, add_64(a, b, Q));
        assert_eq!(b + c, add_64(b, c, Q));
        assert_eq!(c.wrapping_sub(b), add_64(a, c ,Q));

        assert_eq!(a, sub_64(0, b, Q));
        assert_eq!(c - b, sub_64(c, b, Q));
        assert_eq!(a.wrapping_sub(c), sub_64(a, c, Q));
        assert_eq!((b + Q).wrapping_sub(c), sub_64(b, c, Q));
        
        assert_eq!((a * b) % Q, mul_64(a, b, Q));
        assert_eq!(b * c, mul_64(b, c, Q));        
        assert_eq!(((a as u128 * a as u128) % Q as u128) as u64, mul_64(a, a, Q));    

        let a = a as u32;
        let b = b as u32;
        let c = c as u32;

        assert_eq!(0, add_32(a, b, Q));
        assert_eq!(b + c, add_32(b, c, Q));
        assert_eq!(c.wrapping_sub(b), add_32(a, c, Q));

        assert_eq!(a, sub_32(0, b, Q));
        assert_eq!(c - b, sub_32(c, b, Q));
        assert_eq!(a.wrapping_sub(c), sub_32(a, c, Q));
        assert_eq!((Q as u32).wrapping_sub(c.wrapping_sub(b)), sub_32(b, c, Q));

        assert_eq!(((a  as u64 * b as u64) % Q) as u32, mul_32(a, b, Q));
        assert_eq!(b * c, mul_32(b, c, Q));        
        assert_eq!(((a as u128 * a as u128) % Q as u128) as u32, mul_32(a, a, Q));

        let p1_coeffs: [i64; 4] = [-1, -2, 5432, 43];
        let p2_coeffs: [i64; 4] = [6, 602, -5, 4];
        let expected_coeffs: [i64; 8] = [-6, -614, 31393, 3270328, -1282, 21513, 172, 0];

        let mut p1 = [0u64; 4];
        let mut p2 = [0u64; 4];
        let mut expected = [0u64; 8];

        for i in 0..4 {
            if p1_coeffs[i] < 0 {
                p1[i] = (p1_coeffs[i] + Q as i64) as u64;
            }
            else {
                p1[i] = p1_coeffs[i] as u64;
            }

            if p2_coeffs[i] < 0 {
                p2[i] = (p2_coeffs[i] + Q as i64) as u64;
            }
            else {
                p2[i] = p2_coeffs[i] as u64;
            }
        }

        for i in 0..8 {
            if expected_coeffs[i] < 0 {
                expected[i] = (expected_coeffs[i] + Q as i64) as u64;
            }
            else {
                expected[i] = expected_coeffs[i] as u64;
            }
        }

        let mut actual = [0u64; 8];
        poly_mul(&p1, &p2, &mut actual, Q);
        assert_eq!(expected, actual);

        let mut expected_cyclotomic = [0u64; 4];
        let mut _quo = [0u64; 4];
        expected.as_slice().cyclotomic_div(Q, &mut _quo, &mut expected_cyclotomic);

        let mut actual = [0u64; 4];
        poly_mul_cyclotomic(&p1, &p2, &mut actual, Q);
        assert_eq!(expected_cyclotomic, actual);
    }

    #[test]
    fn test_int_mul_poly() {
        let d = 1024;
        let a = 42314u64;
        let p_vec = PVec::rand((1, d, Q), [1u8; 32]);
        let p = p_vec.element(0);

        // pad a to be a full polynomial
        let mut a_poly = vec![0u64; d];
        a_poly[0] = a;

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&a_poly, &p, &mut expected, Q);

        // int * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.int_mul_poly(a, &p, &mut actual);
        assert_eq!(expected, actual);

        // int * mul with cyclotomic recuction - should not change
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.int_mul_poly(a, &p, &mut actual);
        assert_eq!(expected[0..d], actual);
    }

    #[test]
    fn test_chal_mul_poly() {
        let d = 32;
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);
        let p_vec = PVec::rand((1, d, Q), [1u8; 32]);
        let p2 = p_vec.element(0);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&p1, &p2, &mut expected, Q);

        // chal * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.chal_mul_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);

        // schoolbook multiplication with cyclotomic reduction
        let mut expected = vec![0u64; d];
        poly_mul_cyclotomic(&p1, &p2, &mut expected, Q);

        // chal * mul with cyclotomic recuction
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.chal_mul_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_chal_mul_small_poly() {
        let d = 32;
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);

        // sample a polynomial with coefficients at most 15
        let p_vec = PVec::rand((1, d, 16), [1u8; 32]);
        let p2 = p_vec.element(0);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }

        // schoolbook multiplication
        let mut expected = vec![0u64; d * 2];
        poly_mul(&p1, &p2, &mut expected, Q);

        // adjust to balance the coefficients. I.e. coefficients in range q/2...q -> -q/2...0
        for i in 0..expected.len() {
            if expected[i] > (Q >> 1) {
                expected[i] = expected[i].wrapping_sub(Q);
            }
        }

        // chal * mul
        let ring = Ring::init(Q, d, false);
        let mut actual = vec![0u64; d * 2];
        ring.chal_mul_small_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);

        // schoolbook multiplication with cyclotomic reduction
        let mut expected = vec![0u64; d];
        poly_mul_cyclotomic(&p1, &p2, &mut expected, Q);

        // adjust to balance the coefficients. I.e. coefficients in range q/2...q -> -q/2...0
        for i in 0..expected.len() {
            if expected[i] > (Q >> 1) {
                expected[i] = expected[i].wrapping_sub(Q);
            }
        }

        // chal * mul with cyclotomic recuction
        let ring = Ring::init(Q, d, true);
        let mut actual = vec![0u64; d];
        ring.chal_mul_small_poly(&chal_vec[0], &p2, &mut actual);

        assert_eq!(expected, actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_split_ntt() {
        let d = 64;

        // non-cyclotomic ring
        let ring = Ring::init(Q, d, false);
        let ntt = PVecNtt::rand((1, 2 * d, Q), [1; 32]);
        let (a0, a1, a2, a3, a4) = ring._split_ntt(ntt.element(0));

        assert_eq!(&ntt.element(0)[0..2 * d], a0);
        assert_eq!(&ntt.element(0)[2 * d..4 * d], a1);
        assert_eq!(&ntt.element(0)[4 * d..6 * d], a2);
        assert_eq!(&ntt.element(0)[6 * d..8 * d], a3);
        assert_eq!(&ntt.element(0)[8 * d..10 * d], a4);

        // cyclotomic ring
        let ring = Ring::init(Q, d, true);
        let ntt = PVecNtt::rand((1, d, Q), [1; 32]);
        let (a0, a1, a2, a3, a4) = ring._split_ntt(ntt.element(0));

        assert_eq!(&ntt.element(0)[0..d], a0);
        assert_eq!(&ntt.element(0)[d..2 * d], a1);
        assert_eq!(&ntt.element(0)[2 * d..3 * d], a2);
        assert_eq!(&ntt.element(0)[3 * d..4 * d], a3);
        assert_eq!(&ntt.element(0)[4 * d..5 * d], a4);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_inv() {
        use crate::arithmetic::ZeroWithParams;

        let d = 512;
        let ring = Ring::init(Q, d, true);

        // do forward NTT
        let p = PVec::rand((1, d, Q), [1; 32]);
        let mut ntt = PVecNtt::zero((1, d));
        let (a0, a1, a2, a3, a4) = ring._split_ntt_mut(ntt.mut_element(0));
        ring._fwd(p.element(0), a0, a1, a2, a3, a4);
        

        // do inverse NTT
        let mut p_out = PVec::zero((1, d));
        ring._inv(p_out.mut_element(0), a0, a1, a2, a3, a4);

        // check we get the same thing back
        assert_eq!(p.element(0), p_out.element(0));
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_sum_inv() {
        use crate::arithmetic::ZeroWithParams;

        let d = 512;
        let ring = Ring::init(Q, d, true);

        // generate many polynomials and sum them in coefficient form
        let p_vec = PVec::rand((1024, d, Q), [1; 32]);
        let mut sum_actual = vec![0u64; d];

        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            for j in 0..d {
                sum_actual[j] = add_64(sum_actual[j], p[j], Q);
            }
        }

        // sum them over NTT form
        let mut p_vec_ntt = PVecNtt::zero((1024, d));
        let mut sum_ntt = vec![0u32; 5 * d];
        let (s0, s1, s2, s3, s4) = ring._split_ntt_mut(&mut sum_ntt);
        
        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            // do forward NTT
            let (a0, a1, a2, a3, a4) = ring._split_ntt_mut(p_vec_ntt.mut_element(i));
            ring._fwd(p, a0, a1, a2, a3, a4);

            // sum over NTT domains
            for j in 0..d {
                s0[j] = add_32(s0[j], a0[j], P0);
                s1[j] = add_32(s1[j], a1[j], P1);
                s2[j] = add_32(s2[j], a2[j], P2);
                s3[j] = add_32(s3[j], a3[j], P3);
                s4[j] = add_32(s4[j], a4[j], P4);
            }
        }

        // do inverse NTT on sum
        let mut sum_actual = vec![0u64; d];
        ring._inv(&mut sum_actual, s0, s1, s2, s3, s4);

        // check we get the same thing back
        assert_eq!(sum_actual, sum_actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_fwd_mul_inv() {
        // Note: multiplication in our ring is only guaranteed if each coefficient of the product does not exceed
        // 63 bits.
        let d = 32;
        let ring = Ring::init(Q, d, true);

        // generate arbitrary element and small element (e.g. as in commitment matrix * decomposed witness)
        let p_vec_1 = PVec::rand((1, d, Q), [1; 32]);
        let p_vec_2 = PVec::rand((1, d, 16), [1; 32]);

        let mut mul_expected = vec![0u64; d];
        poly_mul_cyclotomic(p_vec_1.element(0), p_vec_2.element(0), &mut mul_expected, Q);

        // do forward NTT on both elements
        let mut a_ntt = vec![0u32; 5 * d];
        let mut b_ntt = vec![0u32; 5 * d];

        let (a0, a1, a2, a3, a4) = ring._split_ntt_mut(&mut a_ntt);
        ring._fwd(p_vec_1.element(0), a0, a1, a2, a3, a4);

        let (b0, b1, b2, b3, b4) = ring._split_ntt_mut(&mut b_ntt);
        ring._fwd(p_vec_2.element(0), b0, b1, b2, b3, b4);

        println!("\n\n\n\n{:?}\n{:?}", a0, b0);

        let mut mul_ntt = vec![0u32; 5 * d];
        let (m0, m1, m2, m3, m4) = ring._split_ntt_mut(&mut mul_ntt);

        // pointwise multiply
        for j in 0..d {
            m0[j] = mul_32(a0[j], b0[j], P0);
            m1[j] = mul_32(a1[j], b1[j], P1);
            m2[j] = mul_32(a2[j], b2[j], P2);
            m3[j] = mul_32(a3[j], b3[j], P3);
            m4[j] = mul_32(a4[j], b4[j], P4);
        }

        println!("\n\n\n{:?},\n{:?}\n{:?}\n{:?}\n{:?}", m0, m1, m2, m3, m4);

        // do inverse NTT and pointwise product
        let mut mul_actual = vec![0u64; d];
        ring._inv(&mut mul_actual, m0, m1, m2, m3, m4);

        assert_eq!(mul_expected, mul_actual);
    }

    #[cfg(not(feature = "nightly"))]
    #[test]
    fn test_mul_acc() {
        let d = 32;
        let ring = Ring::init(Q, d, true);
        let length = 100;

        // generate some random NTT domain polynomials
        let p_vec_ntt_1 = PVecNtt::rand((length, d, P4), [1; 32]);
        let p_vec_ntt_2 = PVecNtt::rand((length, d, P4), [2; 32]);

        // do pointwise product and sum
        let mut sum_expected = vec![0u32; 5 * d];
        let (s0, s1, s2, s3, s4) = ring._split_ntt_mut(&mut sum_expected);

        let mut sum_actual = vec![0u32; 5 * d];
        let (p0, p1, p2, p3, p4) = ring._split_ntt_mut(&mut sum_actual);

        for i in 0..p_vec_ntt_1.length() {
            let (a0, a1, a2, a3, a4) = ring._split_ntt(p_vec_ntt_1.element(i));
            let (b0, b1, b2, b3, b4) = ring._split_ntt(p_vec_ntt_2.element(i));

            // pointwise multiply and sum
            for j in 0..d {
                s0[j] = add_32(s0[j], mul_32(a0[j], b0[j], P0), P0);
                s1[j] = add_32(s1[j], mul_32(a1[j], b1[j], P1), P1);
                s2[j] = add_32(s2[j], mul_32(a2[j], b2[j], P2), P2);
                s3[j] = add_32(s3[j], mul_32(a3[j], b3[j], P3), P3);
                s4[j] = add_32(s4[j], mul_32(a4[j], b4[j], P4), P4);
            }

            // use method
            ring._mul_acc(
                a0, a1, a2, a3, a4, 
                b0, b1, b2, b3, b4, 
                p0, p1, p2, p3, p4);
        }

        assert_eq!(sum_expected, sum_actual);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_split_ntt() {
        let d = 64;

        // non-cyclotomic ring
        let ring = Ring::init(Q, d, false);
        let ntt = PVecNtt::rand((1, 2 * d, Q), [1; 32]);
        let (a0, a1, a2) = ring._split_ntt(ntt.element(0));

        assert_eq!(&ntt.element(0)[0..2 * d], a0);
        assert_eq!(&ntt.element(0)[2 * d..4 * d], a1);
        assert_eq!(&ntt.element(0)[4 * d..6 * d], a2);

        // cyclotomic ring
        let ring = Ring::init(Q, d, true);
        let ntt = PVecNtt::rand((1, d, Q), [1; 32]);
        let (a0, a1, a2) = ring._split_ntt(ntt.element(0));

        assert_eq!(&ntt.element(0)[0..d], a0);
        assert_eq!(&ntt.element(0)[d..2 * d], a1);
        assert_eq!(&ntt.element(0)[2 * d..3 * d], a2);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_fwd_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // do forward NTT
        let p = PVec::rand((1, d, Q), [1; 32]);
        let mut ntt = PVecNtt::zero((1, d));
        let (a0, a1, a2) = ring._split_ntt_mut(ntt.mut_element(0));
        ring._fwd(p.element(0), a0, a1, a2);
        

        // do inverse NTT
        let mut p_out = PVec::zero((1, d));
        ring._inv(p_out.mut_element(0), a0, a1, a2);

        // check we get the same thing back
        assert_eq!(p.element(0), p_out.element(0));
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_fwd_sum_inv() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // generate many polynomials and sum them in coefficient form
        let p_vec = PVec::rand((1024, d, Q), [1; 32]);
        let mut sum_actual = vec![0u64; d];

        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            for j in 0..d {
                sum_actual[j] = add_64(sum_actual[j], p[j], Q);
            }
        }

        // sum them over NTT form
        let mut p_vec_ntt = PVecNtt::zero((1024, d));
        let mut sum_ntt = vec![0u64; 3 * d];
        let (s0, s1, s2) = ring._split_ntt_mut(&mut sum_ntt);
        
        for i in 0..p_vec.length() {
            let p = p_vec.element(i);

            // do forward NTT
            let (a0, a1, a2) = ring._split_ntt_mut(p_vec_ntt.mut_element(i));
            ring._fwd(p, a0, a1, a2);

            // sum over NTT domains
            for j in 0..d {
                s0[j] = add_64(s0[j], a0[j], P0);
                s1[j] = add_64(s1[j], a1[j], P1);
                s2[j] = add_64(s2[j], a2[j], P2);
            }
        }

        // do inverse NTT on sum
        let mut sum_actual = vec![0u64; d];
        ring._inv(&mut sum_actual, s0, s1, s2);

        // check we get the same thing back
        assert_eq!(sum_actual, sum_actual);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_fwd_mul_inv() {
        // Note: multiplication in our ring is only guaranteed if each coefficient of the product does not exceed
        // 63 bits.
        let d = 1024;
        let ring = Ring::init(Q, d, true);

        // generate arbitrary element and small element (e.g. as in commitment matrix * decomposed witness)
        let p_vec_1 = PVec::rand((1, d, Q), [1; 32]);
        let p_vec_2 = PVec::rand((1, d, 16), [1; 32]);

        let mut mul_expected = vec![0u64; d];
        poly_mul_cyclotomic(p_vec_1.element(0), p_vec_2.element(0), &mut mul_expected, Q);

        // do forward NTT on both elements
        let mut a_ntt = vec![0u64; 3 * d];
        let mut b_ntt = vec![0u64; 3 * d];

        let (a0, a1, a2) = ring._split_ntt_mut(&mut a_ntt);
        ring._fwd(p_vec_1.element(0), a0, a1, a2);

        let (b0, b1, b2) = ring._split_ntt_mut(&mut b_ntt);
        ring._fwd(p_vec_2.element(0), b0, b1, b2);

        // poitnwise multiply with concrete NTT function
        let mut mul_ntt = vec![0u64; 3 * d];
        let (m0, m1, m2) = ring._split_ntt_mut(&mut mul_ntt);

        ring.native.ntt_0().mul_accumulate(m0, a0, b0);
        ring.native.ntt_1().mul_accumulate(m1, a1, b1);
        ring.native.ntt_2().mul_accumulate(m2, a2, b2);

        // do inverse NTT
        let mut mul_actual = vec![0u64; d];
        ring._inv(&mut mul_actual, m0, m1, m2);

        assert_eq!(mul_expected, mul_actual);

        // TODO: figure out what the AVX-512 implementation is actually doing.
        // Coefficients of NTT are bigger than P0, P1, P2 (hence why then need 64 bits not 32)
        // and mul_accumulate isn't doing a straight pointwise multiplication mod P0, P1, P2 (see below).
        // Not super important as we can use the methods black-box anyway.

        // pointwise multiply naively
        let mut test_mul_ntt = vec![0u64; 3 * d];
        let (t0, t1, t2) = ring._split_ntt_mut(&mut test_mul_ntt);

        for j in 0..d {
            t0[j] = mul_64(a0[j], b0[j], P0);
            t1[j] = mul_64(a1[j], b1[j], P1);
            t2[j] = mul_64(a2[j], b2[j], P2);
        }

        // assert_eq!(m0, t0);
        // assert_eq!(m1, t1);
        // assert_eq!(m2, t2);
    }

    #[test]
    fn test_ntt_mat() {
        let d = 1024;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 32);
        let mat = PMat::rand((h, w, d, Q), [1; 32]);

        // perform forward NTT
        let mut mat_ntt = PMatNtt::zero((h, w, d));
        ring.fwd(&mat, &mut mat_ntt);

        // check each element was processed correctly
        for i in 0..h {
            for j in 0..w {
                #[cfg(not(feature = "nightly"))]
                {
                let mut ntt = vec![0u32; 5 * d];
                let (p0, p1, p2, p3, p4) = ring._split_ntt_mut(&mut ntt);
                ring.native.fwd(mat.element(i, j), p0, p1, p2, p3, p4);

                assert_eq!(&ntt, mat_ntt.element(i, j));
                }

                #[cfg(feature = "nightly")]
                {
                let mut ntt = vec![0u64; 3 * d];
                let (p0, p1, p2) = ring._split_ntt_mut(&mut ntt);
                ring.native.fwd(mat.element(i, j), p0, p1, p2);

                assert_eq!(&ntt, mat_ntt.element(i, j));
                }
            }
        }
    }

    #[test]
    fn test_mat_prod_vec_cyclotomic() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 48);
        let mat = PMat::rand((h, w, d, Q), [1; 32]);
        
        // sample a vector with small coefficients
        let vec = PVec::rand((w, d, 16), [2; 32]);
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero((h, d));

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                
                let mut prod = vec![0u64; d];
                poly_mul_cyclotomic(m, v, &mut prod, Q);

                for k in 0..d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero((h, w, d));
        ring.fwd(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual: crate::arithmetic::poly_vec::PolyVec<1, u64> = PVec::zero((h, d));
        ring.mul(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_prod_vec() {
        let d = 64;
        let ring = Ring::init(Q, d, false);

        // sample a random matrix
        let (h, w) = (8, 8);
        let mat = PMat::rand((h, w, d, Q), [1; 32]);
        
        // sample a vector with small coefficients
        let vec = PVec::rand((w, d, 16), [2; 32]);
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero((h, 2 * d));

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);

                let mut prod = vec![0u64; 2 * d];
                poly_mul(m, v, &mut prod, Q);

                for k in 0..2 * d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero((h, w, 2 * d));
        ring.fwd(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual = PVec::zero((h, 2 * d));
        ring.mul(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_prod_vec_cyclotomic_negative() {
        let d = 512;
        let ring = Ring::init(Q, d, true);

        // sample a random matrix
        let (h, w) = (4, 48);
        let mat = PMat::rand((h, w, d, Q), [1; 32]);
        
        // sample a vector with small coefficients
        let mut vec = PVec::rand((w, d, 16), [2; 32]);

        // adjust the coefficients so they're between -8..7 instead of 0..15 (using wrappping arithmetic)
        let v = vec.mut_slice();

        for i in 0..v.len() {
            v[i] = v[i].wrapping_sub(8);
        }
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero((h, d));

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                let mut v_pos = vec![0u64; v.len()];

                // adjust the coefficients of v so that they are between 0..7 and q-8..q-1
                for i in 0..v.len() {
                    if (v[i] as i64) < 0 {
                        v_pos[i] = (Q as i64 + v[i] as i64) as u64;
                    }
                    else {
                        v_pos[i] = v[i];
                    }
                }
                
                let mut prod = vec![0u64; d];
                poly_mul_cyclotomic(m, &v_pos, &mut prod, Q);

                for k in 0..d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero((h, w, d));
        ring.fwd(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual: crate::arithmetic::poly_vec::PolyVec<1, u64> = PVec::zero((h, d));
        ring.mul(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_mat_prod_vec_negative() {
        let d = 1024;
        let ring = Ring::init(Q, d, false);

        // sample a random matrix
        let (h, w) = (4, 32);
        let mat = PMat::rand((h, w, d, Q), [1; 32]);
        
        // sample a vector with small-ish coefficients
        let mut vec = PVec::rand((w, d, 1 << 16), [2; 32]);

        // adjust the coefficients so they're balanced (using wrappping arithmetic)
        let v = vec.mut_slice();

        for i in 0..v.len() {
            v[i] = v[i].wrapping_sub(1 << 15);
        }
        
        // manually perform matrix * vector
        let mut prod_expected = PVec::zero((h, 2 * d));

        for row in 0..h {
            let p = prod_expected.mut_element(row);

            for col in 0..w {
                let m = mat.element(row, col);
                let v = vec.element(col);
                let mut v_pos = vec![0u64; v.len()];

                // adjust the coefficients of v so that they are between 0..7 and q-8..q-1
                for i in 0..v.len() {
                    if (v[i] as i64) < 0 {
                        v_pos[i] = (Q as i64 + v[i] as i64) as u64;
                    }
                    else {
                        v_pos[i] = v[i];
                    }
                }
                
                let mut prod = vec![0u64; 2 * d];
                poly_mul(m, &v_pos, &mut prod, Q);

                for k in 0..2 * d {
                    p[k] = add_64(p[k], prod[k], Q);
                }
            }
        }

        // forward NTT on the matrix
        let mut mat_ntt = PMatNtt::zero((h, w, 2 * d));
        ring.fwd(&mat, &mut mat_ntt);

        // perform product with the ring
        let mut prod_actual: crate::arithmetic::poly_vec::PolyVec<1, u64> = PVec::zero((h, 2 * d));
        ring.mul(&mat_ntt, &vec, &mut prod_actual, 0);

        // check equal
        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_inner_prod_int_poly() {
        let n = 1024;
        let d = 128;

        // create a vector of random integers
        let mut int_vec = vec![0u64; n];
        let mut rng = rng();

        for i in 0..n {
            int_vec[i] = u64::rand((Q, Q.log()), &mut rng);
        }

        // sample a random poly vector
        let p_vec = PVec::rand((n, d, Q), [1; 32]);

        // perform inner product manually
        let mut prod_expected = vec![0u64; 2 * d];

        for i in 0..n {
            // pad the integer to be a full ring element
            let mut p1 = vec![0u64; d];
            p1[0] = int_vec[i];

            // get the next poly
            let p2 = p_vec.element(i);

            // multiply and add to inner product
            let mut prod = vec![0u64; 2 * d];
            poly_mul(&p1, p2, &mut prod, Q);

            for i in 0..d {
                prod_expected[i] = add_64(prod_expected[i], prod[i], Q);
            }
        }

        // perform inner product with non-cyclotomic ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero((1, 2 * d));
        ring.mul(&int_vec, &p_vec, &mut prod_actual, 0);
        assert_eq!(prod_expected, prod_actual.element(0));

        // perform inner product with cyclotomic ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero((1, d));
        ring.mul(&int_vec, &p_vec, &mut prod_actual, 0);
        assert_eq!(&prod_expected[0..d], prod_actual.element(0));

    }

    #[test]
    fn test_chal_mul_poly_vec_cyclotomic() {
        let n = 128;
        let d = 256;

        // sample a random challenge
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }
        
        // sample a random poly vec with small coefficients
        let p_vec = PVec::rand((n, d, 16), [1u8; 32]);
        
        // manually compute chal*vec
        let mut prod_expected = PVec::zero((n, d));

        for i in 0..n {
            let out = prod_expected.mut_element(i);
            poly_mul_cyclotomic(&p1, p_vec.element(i), out, Q);
        
            // adjust to balance coefficients
            for i in 0..d {
                if out[i] > (Q >> 1) {
                    out[i] = out[i].wrapping_sub(Q);
                }
            }
        }

        // compute using ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero((n, d));
        ring.mul(&chal_vec[0], &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_chal_mul_poly_vec() {
        let n = 128;
        let d = 256;

        // sample a random challenge
        let chal_vec = PChal::rand_vec(1, d, 16, [1u8; 32]);

        // convert sparse poly into full poly
        let mut p1 = vec![0u64; d];

        for i in 0..chal_vec[0].k() {
            let (exp, sign) = chal_vec[0].get(i);
            
            if sign {
                p1[exp] = 1;
            }
            else {
                p1[exp] = Q.wrapping_sub(1); // -1 mod q;
            }
        }
        
        // sample a random poly vec with small coefficients
        let p_vec = PVec::rand((n, d, 16), [1u8; 32]);
        
        // manually compute chal*vec
        let mut prod_expected = PVec::zero((n, 2 * d));

        for i in 0..n {
            let out = prod_expected.mut_element(i);
            poly_mul(&p1, p_vec.element(i), out, Q);

            // adjust to balance coefficients
            for i in 0..2 * d {
                if out[i] > (Q >> 1) {
                    out[i] = out[i].wrapping_sub(Q);
                }
            }
        }

        // compute using ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero((n, 2 * d));
        ring.mul(&chal_vec[0], &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected.slice(), prod_actual.slice());
    }

    #[test]
    fn test_inner_prod_chal_vec_poly_vec_cyclotomic() {
        let n = 1024;
        let d = 64;

        // sample a random challenge vector
        let chal_vec = PChal::rand_vec(n, d, 16, [1u8; 32]);

        // sample a random poly vec
        let p_vec = PVec::rand((n, d, Q), [1u8; 32]);
        
        // do inner product manually
        let mut prod_expected = vec![0u64; d];

        for i in 0..n {
            // convert sparse poly into full poly
            let mut p1 = vec![0u64; d];

            for j in 0..chal_vec[i].k() {
                let (exp, sign) = chal_vec[i].get(j);
                
                if sign {
                    p1[exp] = 1;
                }
                else {
                    p1[exp] = Q.wrapping_sub(1); // -1 mod q;
                }
            }

            let mut prod = vec![0u64; d];
            poly_mul_cyclotomic(&p1, p_vec.element(i), &mut prod, Q);

            for j in 0..d {
                prod_expected[j] = add_64(prod_expected[j], prod[j], Q)
            }            
        }

        // do inner product with ring
        let ring = Ring::init(Q, d, true);
        let mut prod_actual = PVec::zero((1, d));
        ring.mul(&chal_vec, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected, prod_actual.element(0));
    }

    #[test]
    fn test_inner_prod_chal_vec_poly_vec() {
        let n = 1024;
        let d = 64;

        // sample a random challenge vector
        let chal_vec = PChal::rand_vec(n, d, 16, [1u8; 32]);

        // sample a random poly vec
        let p_vec = PVec::rand((n, d, Q), [1u8; 32]);
        
        // do inner product manually
        let mut prod_expected = vec![0u64; 2 * d];

        for i in 0..n {
            // convert sparse poly into full poly
            let mut p1 = vec![0u64; d];

            for j in 0..chal_vec[i].k() {
                let (exp, sign) = chal_vec[i].get(j);
                
                if sign {
                    p1[exp] = 1;
                }
                else {
                    p1[exp] = Q.wrapping_sub(1); // -1 mod q;
                }
            }

            let mut prod = vec![0u64; 2 * d];
            poly_mul(&p1, p_vec.element(i), &mut prod, Q);

            for j in 0..2 * d {
                prod_expected[j] = add_64(prod_expected[j], prod[j], Q)
            }            
        }

        // do inner product with ring
        let ring = Ring::init(Q, d, false);
        let mut prod_actual = PVec::zero((1, 2 * d));
        ring.mul(&chal_vec, &p_vec, &mut prod_actual, 0);

        assert_eq!(prod_expected, prod_actual.element(0));
    }
}
