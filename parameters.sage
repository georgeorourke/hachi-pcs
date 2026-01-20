import sys
import random


# Importing lattice estimator for SIS hardness.
# IMPORTANT: Remember to download and extract the lattice estimator folder
sys.path.insert(1, '../lattice-estimator-main')
from estimator import *


# Global Parameters
secpar = 120            # Security parameter
logq = 32               # Log of the prime modulus  
kappa = 2               # Log of the degree of the field extension
poly_length = 30        # Length of the initial polynomial
repetitions = 1         # Ignore it if we don't do fully-splitting rings
JL = 0                  # Boolean to check if we use JL or not
TESTS = 50              # Number of tests we are willing to spend on

# Powers of two
q = 2^logq
k = 2^kappa

# Array containing information about the challenge space. We consider power-of-two d between 2^5 and 2^9. 
# That is, ChSpaceL1norm[i]/ChSpaceSize[i] provides the bound on the L1 norm/size of challenges in the ring Rq with d = 2^{5+i}
ChSpaceL1norm = [256, 128, 128, 25, 22]
ChSpaceSize = [16^32, 5^64, 3^128, binomial(256, ChSpaceL1norm[3]) * 2^(ChSpaceL1norm[3]), binomial(512, ChSpaceL1norm[4]) * 2^(ChSpaceL1norm[4])]

# Algorithm to check whether SIS is hard against known lattice attacks
def is_SIS_secure(n, q, beta):

    # Checking the trivial condition
    if beta >= (q-1)/2:
        return 0

    # Running the lattice estimator
    params = SIS.Parameters(n, q, beta, norm=oo)
    params = SIS.lattice(params)

    # Considering the best lattice attack
    if min(log(params["rop"],2), log(params["red"],2) , log(params["sieve"], 2)) < secpar:
        return 0
    else:
        return 1


# Binary search algorithm to find a SIS rank (e.g. nA, nB, nD)
def find_SIS_rank(lower_bound, upper_bound, d, q, beta):
    if lower_bound == upper_bound:
        return lower_bound
    else:
        n = int((lower_bound + upper_bound)/2)
        if is_SIS_secure(n*d, q , beta):
            return find_SIS_rank(lower_bound, n, d,  q, beta)
        else:
            return find_SIS_rank(n + 1, upper_bound, d, q, beta)

######################## Parameter script #########################################

round = 0                               # Variable to keep track of the number of rounds
single_round_proof_size = 1             # The best possible communication in a single round
proof_size = 0                          # Total proof size. Currently not ouput


while single_round_proof_size != 0:

    print("ROUND ", round+1, ".  Current witness log length: ", poly_length)

    min_folded_witness_length = poly_length
    single_round_proof_size = 0

    # The following Boolean checks whether we commit to the polynomial (in the first round), or its MLE (second round onwards)
    if_witness_already_decomposed = min(round, 1)

    # Initializing variables for the best possible parameters for r and d
    final_logr = 0
    finald_d = 0

    # Probabilistic sampling
    for _ in range(TESTS):

        # Sampling random parameters
        alpha = random.randint(7, 9)
        delta = random.randint(2, 16)
        logr = random.randint(1, ceil((poly_length - alpha) / 2))

        # Setting main parameters
        r = 2^logr
        m = 2^(poly_length - alpha) / r
        d = 2^alpha

        # L1 norm of the challenge
        omega = ChSpaceL1norm[alpha-5]

        # Bounding infinity norm of z
        expected_inf_bound = (q^(1/delta) - 1)/2 * omega * sqrt(r)

        # Parameters to check relaxed binding of the inner-outer commitment scheme
        com_beta = expected_inf_bound
        com_omega = 2 * omega
        binding_length_bound = 2 * com_omega * com_beta

        # The following is the length of hat(z), i.e. decomposed z.
        # We note that from second round, when we commit to MLE, the ``delta'' factor doesn't appear for z.
        length_of_z = repetitions * m * d * log(2 * expected_inf_bound, q^(1/delta)) * (if_witness_already_decomposed  + (1 - if_witness_already_decomposed) * delta)

        # The following "if" checks:
        # (i) if binding is trivially broken by MSIS, 
        # (ii) the CWSS knowledge error is large, and 
        # (iii) if the length of z is already larger than the initial vector

        if binding_length_bound < (q-1)/2 and r/ChSpaceSize[alpha-5] < 2^(-secpar) and log(length_of_z,2) < poly_length:

            # Finding SIS dimensions for matrix A
            nA = find_SIS_rank(1, 2^11/d, d, q, binding_length_bound)

            # Finding SIS dimension for matrix B. Actually, we should run the script, but it gives an error because SIS is too hard.
            # So, let's just set nB = 1.
            # nB = find_SIS_rank(1, 2^11/d, d, q, q^(1/delta))
            nB = 1

            # We can fairly confidently assume the dimension for D is the same as for B
            nD = nB

            # Computing the rest of folded witnesses, e.g. hat(t) and hat(w)
            length_of_t_hat = r * nA * d * log(q, q^(1/delta))

            # For hat(w), remember that it is over field extension (for second round onwards, not first). Hence, the k factor.
            length_of_w_hat = r * d * log(q, q^(1/delta)) * (if_witness_already_decomposed * k + (1 - if_witness_already_decomposed))

            # Don't forget the carry vector
            length_of_carry_vector = (repetitions * nA + nB + nD + 2) * d

            # Total witness length
            log_witness_length = ceil(log(length_of_z + length_of_t_hat + length_of_w_hat + length_of_carry_vector, 2))
            witness_length = 2^log_witness_length

            # Computing the partial proof sizes. There are two choices, via sumcheck or Johnson-Lindnestrauss
            rangeproof = 0
            if JL == 1:
                rangeproof = 256 * logq
            else:
                rangeproof = ceil(q^(1/delta)) * logq * k * log_witness_length

            # Sumcheck for linear equations
            linear_sumcheck = 2 * logq * k * log_witness_length

            total_sumcheck = rangeproof + linear_sumcheck
            outer_commitment = nB * d * logq
            Zq_to_Rq_transformation_cost = d * logq
            
            # "If" to check whether we have founded even smaller folded witness
            if log_witness_length <= min_folded_witness_length:
                # We update the witness length
                min_folded_witness_length = log_witness_length

                # Updating the best possible single-round proof size
                if single_round_proof_size == 0 or single_round_proof_size > total_sumcheck + outer_commitment + Zq_to_Rq_transformation_cost:
                    single_round_proof_size =  total_sumcheck + outer_commitment + Zq_to_Rq_transformation_cost
                    final_logr = logr
                    final_d = d

    # Checking if we managed to actually fold anything. If not, break the loop.
    if min_folded_witness_length >= poly_length:
        single_round_proof_size = 0

    # Increase the round number. Update the witness size for the next round.
    round += 1
    poly_length = min_folded_witness_length

    # Update the proof size
    proof_size += single_round_proof_size

    print("Round completed. Folding parameters: ", d, delta, logr, poly_length - alpha - logr, (single_round_proof_size/2^13).n(), " KB")


print("Final witness length in bits: ", poly_length)
