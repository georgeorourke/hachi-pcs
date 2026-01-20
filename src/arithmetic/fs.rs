use sha256::digest;
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};

use crate::arithmetic::Serialise;

/// Structure used to perform FS transform.
pub struct FS {
    bytes: Vec<u8>
}

impl FS {
    pub fn init() -> Self {
        Self { bytes: Vec::new() }
    }

    pub fn push(&mut self, a: &impl Serialise) {
        self.bytes.append(&mut a.serialise());
    }

    pub fn get_seed(&self) -> [u8; 32] {
        // Base 64 encode then hash
        let b64 = STANDARD_NO_PAD.encode(&self.bytes);
        let digest = digest(b64);
        let mut seed = [0u8; 32];

        for i in 0..32 {
            seed[i] = u8::from_str_radix(&digest[2*i..2*(i+1)], 16).unwrap();
        }

        seed
    }
}

