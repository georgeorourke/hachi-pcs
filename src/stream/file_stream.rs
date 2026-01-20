use memmap2::Mmap;
use std::fs::File;

use crate::stream::Stream;

/// Stream of u64 integers read from a file.
/// The file is assumed to contain bytes representing 32 bit little-endian integers.
pub struct U64FileStream {
    mmap: Mmap,
    cur_byte: usize,
    offset: usize
}

impl U64FileStream {
    /// Construct the file stream.
    pub fn init(path: &str, offset: usize) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        Self {
            mmap,
            cur_byte: offset,
            offset
        }
    }
}

/// Implement the stream.
impl Stream<u64> for U64FileStream {
    fn length(&self) -> usize {
        self.mmap.len() / 4
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn read(&mut self, arr: &mut [u64]) {
        assert!(self.cur_byte + arr.len() * 4 <= self.mmap.len());

        for i in 0..arr.len() {
            let b0 = self.mmap[self.cur_byte];
            let b1 = self.mmap[self.cur_byte + 1];
            let b2 = self.mmap[self.cur_byte + 2];
            let b3 = self.mmap[self.cur_byte + 3];

            arr[i] = u32::from_le_bytes([b0, b1, b2, b3]) as u64;
            self.cur_byte += 4;
        }
    }

    fn reset(&mut self) {
        self.cur_byte = self.offset;
    }
}