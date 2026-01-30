use crate::arithmetic::{CoeffType, poly_vec::PVec};

/// Representation of a matrix of polynomials over ZZq with maximum
/// degree d-1.
pub struct PMat {
    height: usize,
    width: usize,
    vec: PVec
}

impl PMat {
    /// Sample a vector with elements that have uniform random Zq coefficients.
    pub fn rand(height: usize, width: usize, d: usize, q: CoeffType, seed: [u8; 32]) ->  Self {
        let vec = PVec::rand(height * width, d, q, seed);
        Self { height, width, vec }
    }

    /// Height of the matrix.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Width of the matrix.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Specified element of the matrix.
    pub fn element(&self, row: usize, col: usize) -> &[CoeffType] {
        self.vec.element(row * self.width + col)
    }
}

impl Clone for PMat {
    fn clone(&self) -> Self {
        Self { height: self.height, width: self.width, vec: self.vec.clone() }
    }
}