use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

#[cfg(feature = "verbose")]
use crate::utils::verbose::progress_bar;

use crate::arithmetic::{RandFromRng, Logarithm};

// bytes per integer
const INT_WIDTH: usize = 4;

/// Produce a file containing 2^l random integers mod q.
pub fn write_random_data(
    filename: &str,
    l: usize,
    q: u64
) {
    assert!(l >= 20);
    
    if Path::new(filename).exists() {
        println!("Not generating file: already exists");
        return;
    }

    // Create the file in append mode
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename)
        .expect("Failed to open the file");

    let reps = 1 << (l - 20);
    let buf_size = INT_WIDTH * (1 << 20);
    let mut arr = vec![0u64; 1 << 20];
    let mut buf = vec![0u8; buf_size];

    let mut rng = ChaCha12Rng::from_os_rng();

    // Write all of the coefficients
    for rep in 0..reps {
        // Fill the array
        for i in 0..arr.len() {
            arr[i] = u64::rand((q, q.log()), &mut rng);
        }

        // Iterate over every element and convert it to 4 bytes
        for i in 0..(1 << 20) {
            let bytes: [u8; INT_WIDTH] = (arr[i] as u32).to_le_bytes();

            for j in 0..INT_WIDTH {
                buf[INT_WIDTH * i + j] = bytes[j];
            }
        }

        file.write_all(&buf)
                .expect("Failed to write to the file");

        #[cfg(feature = "verbose")]
        progress_bar("Generating file", rep, reps);
    }
}