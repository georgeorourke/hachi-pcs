# Hachi PCS prototpye 

## Build and Run
Build: `cargo build --release`

Usage: `./target/release/hachi <witness-file> <l>`
    - Runs the PCS scheme for witness of size 2^l (streamed from the specified file)
    - Outputs benchmarks for Commit, Prove, Verify

### Generate witness file and run
- `cargo build --release --features=gen_file`

### Run with AVX-512 (requires compatible CPU)
- `cargo +nightly build --release --features=nightly`