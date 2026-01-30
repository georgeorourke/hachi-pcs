/// Definition of fields provided by arkworks.

use ark_ff::fields::{Fp, Fp2, Fp2Config, Fp4, Fp4Config, MontBackend, MontConfig};
use ark_ff::BigInt;

// ---- The base prime field Zq ---- //
#[derive(MontConfig)]
#[modulus = "4294967197"]
#[generator = "3"]
pub struct FqConfig;

pub type Fq = Fp<MontBackend<FqConfig, 1>, 1>;

// ---- Degree 2 field extension. ---- //
pub struct Fq2Config;

impl Fp2Config for Fq2Config {
    type Fp = Fq;

    const NONRESIDUE: Self::Fp = Fq::new_unchecked(BigInt!("5"));
    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        Fq::new_unchecked(BigInt!("1")),
        Fq::new_unchecked(BigInt!("-1"))
    ];
}

pub type Fq2 = Fp2<Fq2Config>;

// ---- Degree 4 field extension. ---- //
pub struct Fq4Config;

impl Fp4Config for Fq4Config {
    type Fp2Config = Fq2Config;

    const NONRESIDUE: Fp2<Self::Fp2Config> = Fq2::new(
        Fp::new_unchecked(BigInt!("0")), 
        Fp::new_unchecked(BigInt!("1"))
    );

    const FROBENIUS_COEFF_FP4_C1: &'static [<Self::Fp2Config as Fp2Config>::Fp] = &[
        Fq::new_unchecked(BigInt!("1")),
        Fq::new_unchecked(BigInt!("-1"))
    ];
}

pub type Fq4 = Fp4<Fq4Config>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fq() {
        let a = Fq::from(10);
        let b = Fq::from(30);

        assert_eq!(Fq::from(40), a+b);
        assert_eq!(Fq::from(300), a*b);
    }

    #[test]
    fn test_fq2() {
        let a = Fq2::new(Fq::from(10), Fq::from(50));
        let b = Fq2::new(Fq::from(30), Fq::from(-8));

        assert_eq!(Fq2::new(Fq::from(40), Fq::from(42)), a+b);
        assert_eq!(Fq2::new(Fq::from(3212570907i64), Fq::from(1420)), a*b);
    }

    #[test]
    fn test_fq4() {
        let a = Fq2::new(Fq::from(10), Fq::from(50));
        let b = Fq2::new(Fq::from(30), Fq::from(-8));

        let c = Fq4::new(a, b);
        let d = Fq4::new(b, a);

        let sum = Fq4::new(
            Fq2::new(Fq::from(40), Fq::from(42)), Fq2::new(Fq::from(40), Fq::from(42)));

        let prod = Fq4::new(
            Fq2::new(Fq::from(612628006), Fq::from(3212572327i64)), Fq2::new(Fq::from(3931686104i64), Fq::from(520)));
        
        assert_eq!(sum, c+d);
        assert_eq!(prod, c*d);
    }
}