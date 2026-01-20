use ark_ff::AdditiveGroup;

use crate::arithmetic::{ExtField, FieldExtension, Logarithm, MultiLinearCoeff, PChal, PMat, Poly, RandFromSeed, gadget};

use crate::hachi::setup::Parameters;

/// Form M_alpha (as evaluation table on boolean hypercube).
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField]   
) -> Vec<ExtField>
{
    if params.reuse_mats {
        let mat = PMat::rand((params.n, params.width_d, params.d, params.q), params.d_seed);
        form_m_alpha_same_matrix(params, x, challenges, alpha_pows, &mat)
    }
    else {
        let mat_a = PMat::rand((params.n, params.width_a, params.d, params.q), params.a_seed);
        let mat_b = PMat::rand((params.n, params.width_b, params.d, params.q), params.b_seed);
        let mat_d = PMat::rand((params.n, params.width_d, params.d, params.q), params.d_seed);
        form_m_alpha_different_matrices(params, x, challenges, alpha_pows, &mat_a, &mat_b, &mat_d)
    }
}

/// Form M_alpha (as evaluation table on boolean hypercube) given the three commitment matrices A, B, D
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha_different_matrices(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField],
    mat_a: &PMat,
    mat_b: &PMat,
    mat_d: &PMat
) -> Vec<ExtField>
{
    // We are constructing M (evaluated at alpha) such that M.[z|r] = y.
    // M'.z = y + (X^d-1).r where M' is the verification matrix for equations from previous part of the protocol.
    // So M = [M' | (-X^d-1).In], and padded with zeros so its height and width are powers of two.

    // length of z (width of M')
    let mu = params.width_d + params.width_b + params.width_a * params.delta_z;

    // length of r (height of M')
    let n = params.n + params.n + 1 + 1 + params.n;

    // pad n and (mu + n) to the next power of two
    let height = 1 << (n.log());
    let width = 1 << ((mu + n).log());

    // create the vector of evaluations of M_alpha_mle
    let mut evals = vec![ExtField::ZERO; height * width];

    // set a particular element of the evaluations
    fn set(evals: &mut Vec<ExtField>, f: ExtField, width: usize, row: usize, col: usize) {
        evals[row * width + col] = f;
    }

    // copy in D
    for row in 0..params.n {
        for col in 0..params.width_d {
            let cur = mat_d.element(row, col);
            let f = cur.eval(alpha_pows);
            set(&mut evals, f, width, row, col);
        }
    }

    // copy in B
    let row_offset = params.n;
    let col_offset = params.width_d;

    for row in 0..params.n {
        for col in 0..params.width_b {
            let cur = mat_b.element(row, col);
            let f = cur.eval(alpha_pows);
            set(&mut evals, f, width, row + row_offset, col + col_offset);
        }
    }

    // copy in b^T . G_{2^r}
    let row = params.n + params.n;
    let gadget_vec = gadget(params.b, params.delta);

    for i in 0..1 << params.r {
        // get the current coefficient
        let b_i = u64::multi_lin_coeff(&x[0..params.r], i, params.r, params.q);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = ExtField::lift_int(b_i * gadget_vec[j]);
            set(&mut evals, f, width, row, i * params.delta + j);
        }
    }

    // copy in c^T o G_1 = c^T . G_{2^r} (fourth row section) and c^T o G_n (fifth row section)
    let row = params.n + params.n + 1;              // for c^T o G_1 (fourth row section)
    let row_offset = params.n + params.n + 1 + 1;   // for c^T o G_n (fifth row section)
    let col_offset = params.width_d;                // for c^T o G_n (fifth row section)

    for i in 0..1 << params.r {
        // get the current challenge and evaluate at alpha
        let c_i = challenges[i].eval(alpha_pows);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = c_i.mul_int(gadget_vec[j]);

            // set in fourth row section
            set(&mut evals, f, width, row, i * params.delta + j);

            // set in fifth row section
            for k in 0..params.n {
                set(&mut evals, f, width, row_offset + k, col_offset + i * params.delta * params.n + k * params.delta + j);
            }
        }
    }

    // copy in -[a^T . G_{2^m} . G_z]
    let row = params.n + params.n + 1;
    let col_offset = params.width_d + params.width_b;
    let gadget_vec_z = gadget(params.b, params.delta_z);

    for i in 0..1 << params.m {
        // get the current coefficient
        let a_i = u64::multi_lin_coeff(&x[params.r..params.r + params.m], i, params.m, params.q);

        // expand with normal gadget vector
        for j in 0..params.delta {
            // expand with gadget matrix for composition of z_hat into z
            for k in 0..params.delta_z {
                let f = - ExtField::lift_int(a_i * gadget_vec[j] * gadget_vec_z[k]);
                set(&mut evals, f, width, row, col_offset + i * params.delta * params.delta_z + j * params.delta_z + k);
            }
        }
    }

    // copy in -A.G_z
    let row_offset = params.n + params.n + 1 + 1;
    let col_offset = params.width_d + params.width_b;

    for row in 0..params.n {
        for col in 0..params.width_a {
            // evaluate the current element of A
            let cur = mat_a.element(row, col).eval(alpha_pows);

            // expand with gadget vector
            for j in 0..params.delta_z {
                let f = - cur.mul_int(gadget_vec_z[j]);
                set(&mut evals, f, width, row_offset + row, col_offset + col * params.delta_z + j);
            }
        }
    }

    // Set -alpha^d + 1 in the n places
    let minus_alpha_d_plus_one = -(alpha_pows[params.d-1] * alpha_pows[1] + alpha_pows[0]);
    
    for i in 0..n {
        set(&mut evals, minus_alpha_d_plus_one, width, i, mu + i);
    }

    evals
}

/// Form M_alpha (as evaluation table on boolean hypercube) given the a single commitment matrix (repeated for A, B and D)
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha_same_matrix(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField],
    mat: &PMat
) -> Vec<ExtField>
{
    // We are constructing M (evaluated at alpha) such that M.[z|r] = y.
    // M'.z = y + (X^d-1).r where M' is the verification matrix for equations from previous part of the protocol.
    // So M = [M' | (-X^d-1).In], and padded with zeros so its height and width are powers of two.
    // Optimise for the case when the commitment matrices are the same.

    assert!(params.reuse_mats);

    // length of z (width of M')
    let mu = params.width_d + params.width_b + params.width_a * params.delta_z;

    // length of r (height of M')
    let n = params.n + params.n + 1 + 1 + params.n;

    // pad n and (mu + n) to the next power of two
    let height = 1 << (n as u64).log();
    let width = 1 << ((mu + n) as u64).log();

    // create the vector of evaluations of M_alpha_mle
    let mut evals = vec![ExtField::ZERO; height * width];

    // set a particular element of the evaluations
    fn set(evals: &mut Vec<ExtField>, f: ExtField, width: usize, row: usize, col: usize) {
        evals[row * width + col] = f;
    }

    // gadget vectors
    let gadget_vec = gadget(params.b, params.delta);
    let gadget_vec_z = gadget(params.b, params.delta_z);

    // copy in D, B and -A.G_z
    for row in 0..params.n {
        for col in 0..params.width_d {
            let f = mat.element(row, col).eval(alpha_pows);

            // set D
            set(&mut evals, f, width, row, col);

            // set B
            set(&mut evals, f, width, params.n + row, params.width_d + col);

            // gadget expand and set A.G_z
            let row_offset = params.n + params.n + 1 + 1;
            let col_offset = params.width_d + params.width_b;

            for j in 0..params.delta_z {
                let f = - f.mul_int(gadget_vec_z[j]);
                set(&mut evals, f, width, row_offset + row, col_offset + col * params.delta_z + j);
            }
        }
    }

    // copy in b^T . G_{2^r}
    let row = params.n + params.n;

    for i in 0..1 << params.r {
        // get the current coefficient
        let b_i = u64::multi_lin_coeff(&x[0..params.r], i, params.r, params.q);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = ExtField::lift_int(b_i * gadget_vec[j]);
            set(&mut evals, f, width, row, i * params.delta + j);
        }
    }

    // copy in c^T o G_1 = c^T . G_{2^r} (fourth row section) and c^T o G_n (fifth row section)
    let row = params.n + params.n + 1;              // for c^T o G_1 (fourth row section)
    let row_offset = params.n + params.n + 1 + 1;   // for c^T o G_n (fifth row section)
    let col_offset = params.width_d;                // for c^T o G_n (fifth row section)

    for i in 0..1 << params.r {
        // get the current challenge and evaluate at alpha
        let c_i = challenges[i].eval(alpha_pows);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = c_i.mul_int(gadget_vec[j]);

            // set in fourth row section
            set(&mut evals, f, width, row, i * params.delta + j);

            // set in fifth row section
            for k in 0..params.n {
                set(&mut evals, f, width, row_offset + k, col_offset + i * params.delta * params.n + k * params.delta + j);
            }
        }
    }

    // copy in -[a^T . G_{2^m} . G_z]
    let row = params.n + params.n + 1;
    let col_offset = params.width_d + params.width_b;

    for i in 0..1 << params.m {
        // get the current coefficient
        let a_i = u64::multi_lin_coeff(&x[params.r..params.r + params.m], i, params.m, params.q);

        // expand with normal gadget vector
        for j in 0..params.delta {
            // expand with gadget matrix for composition of z_hat into z
            for k in 0..params.delta_z {
                let f = - ExtField::lift_int(a_i * gadget_vec[j] * gadget_vec_z[k]);
                set(&mut evals, f, width, row, col_offset + i * params.delta * params.delta_z + j * params.delta_z + k);
            }
        }
    }

    // Set -alpha^d + 1 in the n places
    let minus_alpha_d_plus_one = -(alpha_pows[params.d-1] * alpha_pows[1] + alpha_pows[0]);
    
    for i in 0..n {
        set(&mut evals, minus_alpha_d_plus_one, width, i, mu + i);
    }

    evals
}