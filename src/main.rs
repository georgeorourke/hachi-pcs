mod arithmetic;
mod stream;
mod utils;
mod hachi;

#[cfg(feature = "gen_file")]
use crate::utils::gen_file::write_random_data;

use crate::stream::Stream;
use crate::stream::file_stream::U64FileStream;

use crate::hachi::Hachi;
use crate::hachi::setup::Setup;
use crate::hachi::commit::Commit;
use crate::hachi::evaluate::Evaluate;
use crate::hachi::verify::Verify;

use crate::arithmetic::MultiLinearCoeff;

fn main() {
    time_graph::enable_data_collection(true);

    // get witness file and length
    let args: Vec<String> = std::env::args().collect();
    let witness_file = &args[1];
    let l = *&args[2].parse::<usize>().unwrap();

    // public parameters
    let params = Hachi::setup(l, true);

    #[cfg(feature = "gen_file")]
    // Produce a dummy witness file containing random coefficients.
    write_random_data(witness_file, params.l, params.q);

    // commit to the witness
    let mut witness = U64FileStream::init(witness_file, 0);
    let com = Hachi::commit(&mut witness, &params);

    // create an evaluation point and run the evaluation proof
    let x = vec![1234; params.l];
    let proof = Hachi::evaluate(&mut witness, &params, &x, &com);

    // compute claimed evaluation
    println!("\nCalculating evaluation...");
    let mut y = 0;

    witness.reset();

    let r = 20;
    let m = params.l - r;
    let mut buf = vec![0u64; 1 << r];

    for i in 0..1 << m {
        witness.read(&mut buf);

        for j in 0..1 << r {
            let a = u64::multi_lin_coeff(&x, i << r | j, params.l, params.q);
            y = (y + a * buf[j]) % params.q;
        }
    }

    // verification
    Hachi::verify(&params, &x, y, &com.u, &proof);

    let graph = time_graph::get_full_graph();
    println!("{}", graph.as_dot());    
}
