use crate::arithmetic::{NttType, utils::Logarithm};

/// Representation of a vectors of polynomials over Zq with maximum degree n-1.
/// Stored in multi-modular NTT coefficients, which are store in vectors as
/// Arrays of Structures.
pub struct PVecNtt {
    len: usize,
    d: usize,
    logd: usize,
    vec0: Vec<NttType>,
    vec1: Vec<NttType>,
    vec2: Vec<NttType>,
    #[cfg(not(feature = "nightly"))]
    vec3: Vec<NttType>,
    #[cfg(not(feature = "nightly"))]
    vec4: Vec<NttType>,
}

impl PVecNtt {
    /// Initialize a zero vector.
    pub fn zero(len: usize, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        let vec0 = vec![0; len * d];
        let vec1 = vec![0; len * d];
        let vec2 = vec![0; len * d];
        #[cfg(not(feature = "nightly"))]
        let vec3 = vec![0; len * d];
        #[cfg(not(feature = "nightly"))]
        let vec4 = vec![0; len * d];

        #[cfg(not(feature = "nightly"))]
        return Self { len, d, logd, vec0, vec1, vec2, vec3, vec4 };

        #[cfg(feature = "nightly")]
        return Self { len, d, logd, vec0, vec1, vec2 }
    }

    /// Length of vector.
    pub fn length(&self) -> usize {
        self.len
    }

    #[cfg(not(feature = "nightly"))]
    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, i: usize) -> (&[NttType], &[NttType], &[NttType], &[NttType], &[NttType]) {
        let s0 = &self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &self.vec2[(i << self.logd)..((i + 1) << self.logd)];
        let s3 = &self.vec3[(i << self.logd)..((i + 1) << self.logd)];
        let s4 = &self.vec4[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2, s3, s4)
    }

    #[cfg(not(feature = "nightly"))]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, i: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType]) {
        let s0 = &mut self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &mut self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &mut self.vec2[(i << self.logd)..((i + 1) << self.logd)];
        let s3 = &mut self.vec3[(i << self.logd)..((i + 1) << self.logd)];
        let s4 = &mut self.vec4[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2, s3, s4)
    }

    #[cfg(feature = "nightly")]
    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, i: usize) -> (&[NttType], &[NttType], &[NttType]) {
        let s0 = &self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &self.vec2[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2)
    }

    #[cfg(feature = "nightly")]
    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, i: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType]) {
        let s0 = &mut self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &mut self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &mut self.vec2[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2)
    }
}


#[cfg(not(feature = "nightly"))]
impl Clone for PVecNtt {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, 
            vec0: self.vec0.clone(), vec1: self.vec1.clone(), vec2: self.vec2.clone(), vec3: self.vec3.clone(), vec4: self.vec4.clone() }
    }
}

#[cfg(feature = "nightly")]
impl Clone for PVecNtt {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, 
            vec0: self.vec0.clone(), vec1: self.vec1.clone(), vec2: self.vec2.clone() }
    }
}