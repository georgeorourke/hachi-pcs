use ark_ff::{AdditiveGroup, Field};
use rand::RngCore;

use crate::arithmetic::{CoeffType, ExtField, field::{Fq, Fq2, Fq4}};

/// Compute the multilinear coefficient for index i (regarded as a binary index of length len).
/// i = b_1, ..., b_len, return \prod x_j^b_j
pub fn multi_lin_coeff_int(x: &[u64], i: usize, len: usize, q: u64) -> u64 {
    let mut c = 1;

    for j in 0..len {
        if ((i >> j) & 1) == 1 {
            c = (c * x[j]) % q;
        }
    }

    c
}

/// Compute the powers [1, x, x^2, ..., x^{l-1}]
pub fn powers<F: Field>(x: F, l: usize) -> Vec<F> {
    let mut powers = vec![F::ONE; l];

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

/// Take the logarithm of the given type.
pub trait Logarithm {
    fn log(&self) -> usize;
}

/// Logarithm of usize.
impl Logarithm for usize {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// Logarithm of u64.
impl Logarithm for u64 {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// Logarithm of u32.
impl Logarithm for u32 {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 32 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// Sample a single coefficient.
pub fn rand_int(q: CoeffType, logq: usize, rng: &mut impl RngCore) ->  CoeffType {
    // assume q is at most 63 bits.
    let mask = (1 << logq) - 1;
    let mut rnd = q;

    // perform rejection sampling to produce output in range
    while rnd >= q {
        // generate random bits and mask to higher bits to get pow random bits
        // all 0 <= i < q produced with equal likliehood
        rnd = rng.next_u64() & mask;
    }

    rnd
}

/// Sample a field extension element with coefficients up to q
pub fn rand_field(q: CoeffType, rng: &mut impl RngCore) -> ExtField {
    let logq = q.log();

    let a = rand_int(q, logq, rng);
    let b = rand_int(q, logq, rng);
    let c = rand_int(q, logq, rng);
    let d = rand_int(q, logq, rng);

    Fq4::new(Fq2::new(Fq::from(a), Fq::from(b)), Fq2::new(Fq::from(c), Fq::from(d)))
}

/// Balanced power of two decomposition of a coefficient into a slice.
/// Assume the base is given as log base.
/// Assume the integer is essentially signed (wrapping arithmetic).
/// Out will contain coefficients in range floor(-base/2)..floor(base/2) with
/// wrapping arithmetic.
pub fn b_decomp(x: CoeffType, logb: usize, delta: usize, out: &mut [CoeffType]) {
    let mut cur = x;
    let base = 1 << logb;
    let mask = (1 << logb) - 1;
    let half_base = (1 << logb) >> 1;

    for i in 0..delta {
        let r = cur & mask;
        let a = if r >= half_base { r.wrapping_sub(base) } else { r };
        cur = (cur.wrapping_sub(a)) >> logb;
        out[i] = a;
    }
}

/// Lift the type storing an integer coefficient into a field extension element.
pub fn lift_int(x: CoeffType) -> ExtField {
    Fq4::new(Fq2::new(Fq::from(x as i64), Fq::from(0)), Fq2::ZERO)
}

/// Multiply the type storing an integer coefficient by a field extension element.
pub fn mul_int_field(x: CoeffType, f: ExtField) -> ExtField {
    f.mul_by_base_prime_field(&Fq::from(x))
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