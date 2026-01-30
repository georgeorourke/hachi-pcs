use crate::hachi::Hachi;

// Defualt parameters
pub const Q: u64 = 4294967197;
const MAX_ALPHA: usize = 10; // Maximum ring dimension is 2^10

const DECOMP_BASE: u64 = 16;
const DECOMP_DELTA: usize = 8;

const Z_DECOMP_DELTA: usize = 4;
// require z in range [-8 . (16^3 + 16^2 + 16 + 1) ... 7 . (16^3 + 16^2 + 16 + 1)] = [-34952 ... 30583]
const Z_BOUND: u64 = 30583; 

// TODO: K should be calculated based on alpha (this is correct for alpha=10).
const K: usize = 16;

/// The public parameters available to all parties.
pub struct Parameters {
    pub l: usize,                   // number of variables of the multilinear polynomial
    pub q: u64,                     // prime modulus = 5 (mod 8)
    pub d: usize,                   // ring dimension of R_{q,d}
    pub m: usize,                   // folding parameter
    pub r: usize,                   // folding parameter

    pub k: usize,                   // number of non-zero elements in challenges

    // Decomposition
    pub decomp_witness: bool,       // whether to decompose the witness
    pub z_bound: u64,               // bound l1 norm z
    pub b: u64,                     // decomposition base
    pub delta: usize,               // expansion factor from decomposing arbitrary element
    pub delta_z: usize,             // expansion factor from decomposing z
    
    // Dimensions of commitment matrices.
    pub n: usize,                   // height
    pub width_a: usize,             // width of A
    pub width_b: usize,             // width of B
    pub width_d: usize,             // width of D
    pub reuse_mats: bool,           // use the same matrix for all commitments if all dimensions match

    // Commitment matrix seeds.
    pub a_seed: [u8; 32],
    pub b_seed: [u8; 32],
    pub d_seed: [u8; 32],
}

/// Set up function.
pub trait Setup  {
    /// The setup function for a multilinear polynomial in l variables.
    fn setup(l: usize, decompose: bool) -> Parameters;
}

impl Setup for Hachi {
    fn setup(l: usize, decompose: bool) -> Parameters {
        // Modulus
        let q = Q;

        // Ring dimension. TODO: should be chosen adaptively.
        let alpha = MAX_ALPHA;
        let d: usize = 1 << alpha; 

        // folding parameters - set as equal
        let m = (l - alpha) / 2;
        let r = l - m - alpha;

        // number of non-zero coefficients in challenges
        let k = K;

        // decomposition
        let decomp_witness = decompose;
        let z_bound = Z_BOUND;
        let b = DECOMP_BASE;
        let delta = DECOMP_DELTA;
        let delta_z = Z_DECOMP_DELTA;

        // height of matrices - we require height * ring dimension at least 2^10
        let n = 1 << (10-alpha);

        // width of matrices
        let width_a = if decomp_witness {(1 << m) * delta } else { 1 << m };
        let width_b = n * (1 << r) * delta;
        let width_d = (1 << r) * delta;
        let reuse_mats = width_a == width_b && width_b == width_d;

        // seeds. TODO: make non-deterministic.
        let a_seed = [1u8; 32];
        let b_seed = if reuse_mats { a_seed } else { [2u8; 32] };
        let d_seed = if reuse_mats { a_seed } else { [3u8; 32] };

        Parameters { l, q, d, m, r, k, decomp_witness, z_bound, b, delta, delta_z, n, width_a, width_b, width_d, reuse_mats, a_seed, b_seed, d_seed }
    }
}