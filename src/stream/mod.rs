pub mod file_stream;
pub mod vec_stream;

/// Representation of a stream of data from some source.
pub trait Stream<T> {
    /// Length of the stream.
    fn length(&self) -> usize;

    /// Fill the provided slice with the next elements from the stream.
    fn read(&mut self, arr: &mut [T]);

    /// Reset the stream.
    fn reset(&mut self);
}