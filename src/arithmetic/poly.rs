use ark_ff::AdditiveGroup;

use crate::arithmetic::{CoeffType, ExtField};
use crate::arithmetic::utils::mul_int_field;

/// Object represents a polynomial stored in coefficient form.
pub trait Poly {
    /// Evaluate the polynomial at a given extension field element alpha in
    /// given the powers [1, alpha, alpha^2 ... alpha^(d-1)].
    fn eval(&self, alpha_pows: &[ExtField]) -> ExtField;

    /// Divide by X^d+1 mod q and store quotient and remainder.
    /// Assume the polynomial is of length 2d.
    fn cyclotomic_div(&self, q: CoeffType, quotient: &mut [CoeffType], remainder: &mut [CoeffType]);
}

/// Implementation of Poly for a slice of unsinged 64 bit integers.
impl Poly for &[u64] {
    fn eval(&self, alpha_pows: &[ExtField]) -> ExtField {
        let mut out = ExtField::ZERO;

        for i in 0..self.len() {
            out += mul_int_field(self[i], alpha_pows[i]);
        }

        out
    }

    fn cyclotomic_div(&self, q: CoeffType, quotient: &mut [CoeffType], remainder: &mut [CoeffType]) {
        let d = self.len() / 2;
        assert_eq!(d, quotient.len());
        assert_eq!(d, remainder.len());

        for i in 0..d {
            remainder[i] = self[i].wrapping_sub(self[i+d]);
            quotient[i] = self[i+d];

            if (remainder[i] as i64) < 0 {
                remainder[i] = remainder[i].wrapping_add(q);
            }
        }
    }
}

#[cfg(test)]
/// Tests for Poly.
mod test_poly {
    use ark_ff::{Field, UniformRand};

    use crate::arithmetic::field::Fq4;

    use super::*;

    #[test]
    fn test_eval() {
        // 3 + x + 5x^2 + 3432x^3 + 4313x^4 + 8761x^5 + 2x^6 + 535430x^7
        let poly = [3u64, 1, 5, 3432, 4313, 8761, 2, 535430].as_slice();

        // get a random evaluation point
        let mut rng = ark_std::test_rng();
        let alpha = Fq4::rand(&mut rng);

        // generate the powers [1, alpha, ..., alpha^7] and calculate the correct evaluation
        let mut pows: Vec<Fq4> = Vec::new();
        let mut expected = Fq4::ZERO;

        for i in 0..8 {
            pows.push(alpha.pow([i]));
            expected += mul_int_field(poly[i as usize], alpha.pow([i]));
        }

        // evaluate the polynomial
        let actual = poly.eval(&pows);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_cyclotomic_div() {
        // 3 + x + 5x^2 + 3432x^3 + 4313x^4 + 8761x^5 + 2x^6 + 535430x^7
        let poly = [3u64, 1, 5, 3432, 4313, 8761, 2, 535430].as_slice();

        let q = (1i64 << 32) - 43221;

        // expected division
        let expected_quo = [4313, 8761, 2, 535430].as_slice();
        let expected_rem = [4294919765, 4294915315, 3, 4294392077].as_slice();

        // actual division
        let mut actual_quo = [0u64; 4];
        let mut actual_rem = [0u64; 4];

        poly.cyclotomic_div(q as u64, &mut actual_quo, &mut actual_rem);

        assert_eq!(expected_quo, actual_quo);
        assert_eq!(expected_rem, actual_rem);
    }
}