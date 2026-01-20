/// Module containing implementations for primitive integers (u64, u32, usize).

use rand::RngCore;

use crate::arithmetic::{
    CoeffType, 
    Decompose, 
    Coefficient,
    Logarithm, 
    MultiLinearCoeff,
    RandFromRng, 
    Serialise, 
    Zero
};

// ---- 64-bit integer ---- //

/// u64 used to represent integers in arithmetic library where 64 bits needed,
/// such as representing coefficients of a polynomial to be passed to concrete-ntt library.
/// u64 may represent an unsigned number or signed number (using wrapping arithmetic,
/// as in concrete-ntt) depending on the context. Functions should specify which is the case.
impl Coefficient for u64 { }

/// Zero initialisation of u64
impl Zero for u64 {
    fn zero() -> Self {
        0
    }
}

/// Random sample of u64 from RNG
impl RandFromRng<(CoeffType, usize)> for u64 {
    fn rand((q, logq): (CoeffType, usize), rng: &mut impl RngCore) ->  Self {
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
}

/// Logarithm of u64.
impl Logarithm for u64 {
    // log method assumes the given integer is unsigned.
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

/// Multi-linear coefficient.
impl MultiLinearCoeff for u64 {
    fn multi_lin_coeff(x: &[Self], i: usize, len: usize, q: CoeffType) -> Self {
        let mut c = 1;

        for j in 0..len {
            if ((i >> j) & 1) == 1 {
                c = (c * x[j]) % q;
            }
        }

        c
    }
}

/// Serialise 64 bit integer
impl Serialise for u64 {
    fn serialise(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

/// Implement decomposition for a 64 bit integer.
impl Decompose<&mut [u64]> for u64 {
    /// Unbalanced power of two decomposition.
    /// Assume the base is given as log base.
    /// Assume the integer is always positive and less than 63 bits (so we do not have to consider the sign).
    /// Out will contain 
    fn u_decomp(&self, logb: CoeffType, delta: usize, out: &mut [u64]) {
        let logb = logb as usize;
        let mut cur = *self;

        let mask = (1 << logb) - 1;

        for i in 0..delta {
            let r = cur & mask;
            cur = cur >> logb;
            out[i] = r;
        }
    }

    /// Balanced power of two decomposition.
    /// Assume the base is given as log base.
    /// Assume the integer is essentially signed (wrapping arithmetic).
    /// Out will contain coefficients in range floor(-base/2)..floor(base/2) with
    /// wrapping arithmetic.
    fn b_decomp(&self, logb: CoeffType, delta: usize, out: &mut [u64]) {
        let logb = logb as usize;
        let mut cur = *self;

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
}

// ---- 32-bit integer ---- //

/// u32 used to represent integers in arithmetic library where 32 bits needed,
/// such as representing NTT coefficients to be passed to concrete-ntt library.
/// u32 is always unsigned.
impl Coefficient for u32 { }

/// Zero initialisation of u32.
impl Zero for u32 {
    fn zero() -> Self {
        0
    }
}

/// Random sample of u32 from RNG
impl RandFromRng<(u64, usize)> for u32 {
    fn rand((q, logq): (CoeffType, usize), rng: &mut impl RngCore) ->  Self {
        let q = q as u32;
        let mask = if logq < 32 { (1 << logq) - 1 } else { 0u32.wrapping_sub(1) };
        let mut rnd = q;

        // perform rejection sampling to produce output in range
        while rnd >= q {
            // generate random bits and mask to higher bits to get pow random bits
            // all 0 <= i < q produced with equal likliehood
            rnd = rng.next_u32() & mask;
        }

        rnd
    }
}

/// Logarithm of u32.
impl Logarithm for u32 {
    // log method assumes the given integer is unsigned.
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 32 && *self > (1 << logx) { logx += 1; }
        logx
    }
}

// ---- usize ----

/// Useful to take logarithm of usize to avoid conversion.
impl Logarithm for usize {
    fn log(&self) -> usize {
        let mut logx = 0;
        while logx < 64 && *self > (1 << logx) { logx += 1; }
        logx
    }
}


#[cfg(test)]
/// Tests for Int.
mod test_int {
    use rand::rng;

    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(0u64, u64::zero());
        assert_eq!(0u32, u32::zero());
    }

    #[test]
    fn test_log() {
        assert_eq!(0, 1u64.log());
        assert_eq!(0, 1u32.log());

        assert_eq!(1, 2u64.log());
        assert_eq!(1, 2u32.log());

        assert_eq!(17, (1u64 << 17).log());
        assert_eq!(17, (1u32 << 17).log());

        assert_eq!(63, (1u64 << 63).log());
        assert_eq!(31, (1u32 << 31).log());

        assert_eq!(64, ((1u64 << 63) + 42123).log());
        assert_eq!(32, ((1u32 << 31) + 4263).log());

        assert_eq!(63, ((1u64 << 63) - 42123).log());
        assert_eq!(31, ((1u32 << 31) - 4263).log());
    }

    #[test]
    fn test_rand_u64() {
        let mut rng = rng();

        // 32 bit modulus
        let q = (1u64 << 32) - 5432;
        let logq = q.log();
        assert_eq!(32, logq);

        // generate loads of random integers
        let n = 10000;
        let mut ints = vec![0u64; n];
        let mut total = 0u64;

        for i in 0..n {
            ints[i] = u64::rand((q, logq), &mut rng);
            assert!(ints[i] < q);

            total += ints[i]
        }

        // find the average and check it's near q/2
        let avg = total / n as u64;
        println!("Average is {}. Should be about {}.", avg, q/2);
    }

    #[test]
    fn test_rand_u32() {
        let mut rng = rng();

        // 24 bit integer
        let q = (1u64 << 24) - 5432;
        let logq = q.log();
        assert_eq!(24, logq);

        // generate loads of random integers
        let n = 10000;
        let mut ints = vec![0u32; n];
        let mut total = 0u64;

        for i in 0..n {
            ints[i] = u32::rand((q, logq), &mut rng);
            assert!(ints[i] < q as u32);

            total += ints[i] as u64
        }

        // find the average and check it's near q/2
        let avg = total / n as u64;
        println!("Average is {}. Should be about {}.", avg, q/2);
    }
}

#[cfg(test)]
/// Tests for serialisation of integers.
mod test_serialise {
    use super::*;

    #[test]
    fn test_serialise_u64() {
        let zero = 0u64;
        assert_eq!(vec![0; 8], zero.serialise());

        let a = 19u64;
        assert_eq!(vec![19, 0, 0, 0, 0, 0, 0, 0], a.serialise());

        let b_expected: Vec<u8> = vec![19, 120, 43, 21, 0, 0, 0, 0];
        let b: u64 = 19 | 120 << 8 | 43 << 16 | 21 << 24;
        assert_eq!(b_expected, b.serialise());

        let c: u64 = (1 << 45) + 542;
        assert_eq!(c.to_le_bytes().to_vec(), c.serialise());
    }
}

#[cfg(test)]
/// Tests for decomposition of integers.
mod test_decompose {
    use rand::rng;

    use super::*;

    #[test]
    fn test_u_decomp_zero() {
        let zero = 0u64;

        let logb = 1;
        let delta = 1;
        let mut actual = [5];
        zero.u_decomp(logb, delta, &mut actual);
        assert_eq!([0], actual);

        let logb = 4;
        let delta = 8;
        let mut actual = [5; 8];
        zero.u_decomp(logb, delta, &mut actual);
        assert_eq!([0; 8], actual);

    }

    #[test]
    fn test_u_decomp_small() {
        let a = 7u64;
        let logb = 2;
        let delta = 2;
        let mut actual = [0u64; 2];
        a.u_decomp(logb, delta, &mut actual);

        // 7 = [1 4]^T [3 1]
        let expected = [3, 1];
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_u_decomp_rand() {
        let base: u64 = 16;
        let delta: usize = 8;
        let mut rng = rng();
        let mut expected = Vec::<u64>::new();

        for _ in 0..delta {
            let r = u64::rand((base, base.log()), &mut rng);
            expected.push(r);
        }

        // create the element to be decomposed
        let mut a = 0u64;
        for (i, x) in expected.iter().enumerate() {
            a += x * base.pow(i as u32);
        }

        let mut decomp = vec![0u64; delta];
        a.u_decomp(base.log() as u64, delta, &mut decomp);
        assert_eq!(decomp.to_vec(), decomp);
    }

    #[test]
    fn test_b_decomp_zero() {
        let zero = 0u64;

        let logb = 1;
        let delta = 1;
        let mut actual = [5];
        zero.b_decomp(logb, delta, &mut actual);
        assert_eq!([0], actual);

        let logb = 4;
        let delta = 8;
        let mut actual = [5; 8];
        zero.b_decomp(logb, delta, &mut actual);
        assert_eq!([0; 8], actual);

    }

    #[test]
    fn test_b_decomp_small() {
        let a = 7u64;
        let logb = 2;
        let delta = 3;
        let mut actual = [0u64; 3];
        a.b_decomp(logb, delta, &mut actual);

        // 7 = [1 4 16]^T [-1 -2 1]
        let expected = [-1i64 as u64, -2i64 as u64, 1];
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_b_decomp_neg() {
        let a = -7i64 as u64;
        let logb = 2;
        let delta = 2;
        let mut actual = [0u64; 2];
        a.b_decomp(logb, delta, &mut actual);

        // -7 = [1 4]^T [1 -2]
        let expected = [1, -2i64 as u64];
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_b_decomp_rand() {
        let base: u64 = 16;
        let delta: usize = 8;
        let mut rng = rng();
        let mut expected = Vec::<u64>::new();

        for _ in 0..delta {
            let r = u64::rand((base, base.log()), &mut rng);
            expected.push(r.wrapping_sub(8));
        }

        // create the element to be decomposed
        let mut a = 0i64;
        for (i, x) in expected.iter().enumerate() {
            a += (*x as i64) * base.pow(i as u32) as i64;
        }

        let mut decomp = vec![0u64; delta];
        (a as u64).b_decomp(base.log() as u64, delta, &mut decomp);
        assert_eq!(decomp.to_vec(), decomp);
    }
}

