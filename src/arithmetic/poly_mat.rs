use crate::arithmetic::{CoeffType, Coefficient, RandFromSeed, ZeroWithParams};
use crate::arithmetic::poly_vec::PolyVec;

/// Representation of a matrix of polynomials over ZZq with maximum
/// degree d-1.
/// 
/// n is the number of coefficients that must be stored per standard
/// coefficient of the polynomial. I.e. n=1 when storing the polynomial
/// in coefficient domain, and n=3/5 when storing the polynomial in
/// multi-modular NTT domain.
pub struct PolyMat<const N: usize, T: Coefficient> {
    height: usize,
    width: usize,
    vec: PolyVec<N, T>
}

/// Zero initialisation.
impl<const N: usize, T : Coefficient> ZeroWithParams<(usize, usize, usize)> for PolyMat<N, T> {
    fn zero((height, width, d): (usize, usize, usize)) -> Self {
        let vec = PolyVec::<N, T>::zero((height * width, d));
        Self { height, width, vec }
    }
}

/// Random initialisation from seed.
impl<const N: usize, T : Coefficient> RandFromSeed<(usize, usize, usize, CoeffType)> for PolyMat<N, T> {
    fn rand((height, width, d, q): (usize, usize, usize, CoeffType), seed: [u8; 32]) ->  Self {
        let vec = PolyVec::<N, T>::rand((height * width, d, q), seed);
        Self { height, width, vec }
    }
}

/// General methods.
impl<const N: usize, T : Coefficient> PolyMat<N, T> {
    /// Height of the matrix.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Width of the matrix.
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Specified element of the matrix.
    pub fn element(&self, row: usize, col: usize) -> &[T] {
        self.vec.element(row * self.width + col)
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, row: usize, col: usize) -> &mut [T] {
        self.vec.mut_element(row * self.width + col)
    }
}

#[cfg(test)]
/// Tests for PolVec.
mod test_poly_mat {
    use super::*;

    #[test]
    fn test_init() {
        let m1 = PolyMat::<1, u64>::zero((1, 256, 32));
        assert_eq!(vec![0u64; 256*32], m1.vec.slice());

        let m2 = PolyMat::<3, u64>::zero((2, 512, 1024));
        assert_eq!(vec![0u64; 3*2*512*1024], m2.vec.slice());

        let m3 = PolyMat::<5, u32>::zero((3, 512, 64));
        assert_eq!(vec![0u32; 5*3*512*64], m3.vec.slice());
    }

    #[test]
    fn test_rand() {
        // just test length of vector and not all zero - rand already tested in Int.
        let q = (1u64 << 32) - 324321;
        let seed = [1u8; 32];

        let m1 = PolyMat::<1, u64>::rand((1, 256, 32, q), seed);
        assert_eq!(256 * 32, m1.vec.slice().len());
        assert_ne!(vec![0u64; 256*32], m1.vec.slice());

        let m2 = PolyMat::<3, u64>::rand((2, 512, 1024, q), seed);
        assert_eq!(3 * 2 * 512 * 1024, m2.vec.slice().len());
        assert_ne!(vec![0u64; 3*512*1024], m2.vec.slice());

        let m3 = PolyMat::<5, u32>::rand((3, 512, 64, q), seed);
        assert_eq!(5 * 3 *512 * 64, m3.vec.slice().len());
        assert_ne!(vec![0u32; 5*512*64], m3.vec.slice());
    }

    #[test]
    fn test_height() {
        let m1 = PolyMat::<1, u64>::zero((1, 256, 32));
        assert_eq!(1, m1.height());

        let m2 = PolyMat::<3, u64>::zero((2, 512, 1024));
        assert_eq!(2, m2.height());

        let m3 = PolyMat::<5, u32>::zero((3, 512, 64));
        assert_eq!(3, m3.height());
    }

    #[test]
    fn test_width() {
        let m1 = PolyMat::<1, u64>::zero((1, 256, 32));
        assert_eq!(256, m1.width());

        let m2 = PolyMat::<3, u64>::zero((1, 512, 1024));
        assert_eq!(512, m2.width());

        let p3 = PolyMat::<5, u32>::zero((3, 512, 64));
        assert_eq!(512, p3.width());
    }

    #[test]
    fn test_element() {
        let m = PolyMat::<3, u64>::rand((8, 16, 64, 100), [1u8; 32]);

        for i in 0..8 {
            for j in 0..16 {
                let start = (16 * i + j) * 3 * 64;
                let end = (16 * i + j + 1) * 3 * 64;
                assert_eq!(&m.vec.slice()[start..end], m.element(i, j));
            }
        }
    }
}