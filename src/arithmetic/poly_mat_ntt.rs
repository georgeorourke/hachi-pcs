use crate::arithmetic::{NttType, poly_vec_ntt::PVecNtt};

/// Representation of a matrix of polynomials over ZZq with maximum
/// degree d-1 stored in multi-modular NTT form.
pub struct PMatNtt {
    height: usize,
    width: usize,
    vec: PVecNtt
}

impl PMatNtt {
    /// Initialize a zero vector.
    pub fn zero(height: usize, width: usize, d: usize) -> Self {
        let vec = PVecNtt::zero(height * width, d);
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

    #[cfg(not(feature = "nightly"))]
    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, row: usize, col: usize) -> (&[NttType], &[NttType], &[NttType], &[NttType], &[NttType]) {
        self.vec.element(row * self.width + col)
    }

    #[cfg(not(feature = "nightly"))]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, row: usize, col: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType]) {
        self.vec.mut_element(row * self.width + col)
    }

    #[cfg(feature = "nightly")]
    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, row: usize, col: usize) -> (&[NttType], &[NttType], &[NttType]) {
        self.vec.element(row * self.width + col)
    }

    #[cfg(feature = "nightly")]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, row: usize, col: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType]) {
        self.vec.mut_element(row * self.width + col)
    }
}

impl Clone for PMatNtt {
    fn clone(&self) -> Self {
        Self { height: self.height, width: self.width, vec: self.vec.clone() }
    }
}