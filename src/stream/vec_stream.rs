use crate::stream::Stream;

/// Stream of T from a vector.
pub struct VectorStream<T : Clone> {
    vec: Vec<T>,
    cur: usize
}

impl<T : Clone> VectorStream<T> {
    /// Construct the stream from the vector.
    pub fn init(vec: Vec<T>) -> Self {
        Self { vec, cur: 0 }
    }
}

/// Implement the Stream trait.
impl<T : Clone> Stream<T> for VectorStream<T> {
    fn length(&self) -> usize {
        self.vec.len()
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn read(&mut self, arr: &mut [T]) {
        assert!(self.cur + arr.len() <= self.length());
        arr.clone_from_slice(&self.vec[self.cur..self.cur + arr.len()]);
        self.cur += arr.len()
    }

    fn reset(&mut self) {
        self.cur = 0;
    }
}