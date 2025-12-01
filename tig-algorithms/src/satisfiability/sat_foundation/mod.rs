use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> anyhow::Result<()> {
    let _ = save_solution(&Solution { variables: vec![false; challenge.num_variables] });
    // Initialize deterministic RNG using StdRng with the provided seed
    let seed_u64 = u64::from_le_bytes(challenge.seed[..8].try_into().unwrap());
    let mut rng = StdRng::seed_from_u64(seed_u64);

    let num_variables = challenge.num_variables;
    let num_clauses_initial = challenge.clauses.len();

    // Ensure the number of variables does not exceed 64 for bitmasking
    // For larger numbers, extend the bitmask with additional u64s or use a bitset library
    assert!(
        num_variables <= 64,
        "Number of variables exceeds 64, which is not supported by the current bitmask implementation."
    );

    // Use bitmask representations for p_single and n_single
    let mut p_single: u64 = 0;
    let mut n_single: u64 = 0;

    // Define a fixed-size array for clauses since it's 3-SAT
    type Clause = [i32; 3];
    let mut clauses_: Vec<Clause> = Vec::with_capacity(num_clauses_initial);

    // Preprocess clauses: ensure each clause has exactly three literals
    for c in &challenge.clauses {
        if c.len() == 3 {
            clauses_.push([c[0], c[1], c[2]]);
        }
    }

    let mut clauses = Vec::with_capacity(clauses_.len());

    let mut rounds = 0;
    let mut dead = false;

    // Preallocate a buffer for filtered clauses to reuse memory
    let mut c_buffer: Vec<i32> = Vec::with_capacity(3);

    // Preprocessing loop to handle unit clauses and remove redundant literals
    while !dead {
        let mut done = true;
        clauses.clear(); // Reuse the allocated space

        for &c in &clauses_ {
            c_buffer.clear();
            let mut skip = false;

            for (i, &l) in c.iter().enumerate() {
                let var_index = (l.abs() - 1) as usize;

                // Manual check for complementary literals instead of using .contains()
                let mut found_complement = false;
                for &other in &c[(i + 1)..] {
                    if other == -l {
                        found_complement = true;
                        break;
                    }
                }

                if (p_single & (1 << var_index) != 0 && l > 0)
                    || (n_single & (1 << var_index) != 0 && l < 0)
                    || found_complement
                {
                    skip = true;
                    break;
                } else {
                    // Manual check for duplicate literals
                    let mut found_duplicate = false;
                    for &other in &c[(i + 1)..] {
                        if other == l {
                            found_duplicate = true;
                            break;
                        }
                    }

                    if (p_single & (1 << var_index) != 0)
                        || (n_single & (1 << var_index) != 0)
                        || found_duplicate
                    {
                        done = false;
                        continue;
                    } else {
                        c_buffer.push(l);
                    }
                }
            }

            if skip {
                done = false;
                continue;
            }

            match c_buffer.len() {
                0 => {
                    // Clause has no literals, unsatisfiable
                    dead = true;
                    break;
                }
                1 => {
                    // Unit clause detected
                    done = false;
                    let l = c_buffer[0];
                    let var_index = (l.abs() - 1) as usize;
                    if l > 0 {
                        if n_single & (1 << var_index) != 0 {
                            // Conflict detected
                            dead = true;
                            break;
                        } else {
                            // Assign variable to true
                            p_single |= 1 << var_index;
                        }
                    } else {
                        if p_single & (1 << var_index) != 0 {
                            // Conflict detected
                            dead = true;
                            break;
                        } else {
                            // Assign variable to false
                            n_single |= 1 << var_index;
                        }
                    }
                }
                _ => {
                    // Multi-literal clause, add to clauses
                    let mut new_clause = [0i32; 3];
                    for (j, &lit) in c_buffer.iter().enumerate() {
                        new_clause[j] = lit;
                    }
                    clauses.push(new_clause);
                }
            }
        }

        if done {
            break;
        } else {
            clauses_ = clauses.clone(); // Clone clauses for the next iteration
            clauses.clear(); // Clear clauses for the next iteration
        }
    }

    if dead {
        // Unsatisfiable due to conflicts
        return Ok(());
    }

    // Recalculate the number of clauses after preprocessing
    let num_clauses = clauses.len();

    // Initialize clause mappings with fixed-size arrays for better cache locality
    let avg_clauses_per_var = (num_clauses + num_variables - 1) / num_variables; // Ceiling division
    let capacity_per_var = avg_clauses_per_var + 1; // Slightly more to prevent reallocations

    // Initialize p_clauses and n_clauses with preallocated capacities
    let mut p_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(capacity_per_var); num_variables];
    let mut n_clauses: Vec<Vec<usize>> = vec![Vec::with_capacity(capacity_per_var); num_variables];

    // Populate p_clauses and n_clauses
    for (i, &c) in clauses.iter().enumerate() {
        for &l in &c {
            let var = (l.abs() - 1) as usize;
            if l > 0 {
                p_clauses[var].push(i);
            } else {
                n_clauses[var].push(i);
            }
        }
    }

    // Use bitmask for variable assignments
    let mut variables: u64 = 0;

    // Initialize variable assignments using integer-based heuristics
    for v in 0..num_variables {
        let num_p = p_clauses[v].len();
        let num_n = n_clauses[v].len();

        // Replace floating-point calculations with scaled integers
        // Scale ratio by 100 to maintain two decimal places (1.28 becomes 128)
        let nad_scaled = 128; // Equivalent to 1.28 scaled by 100
        let vad_scaled = if num_n > 0 {
            (num_p * 100) / num_n
        } else {
            100 + 1 // Equivalent to nad + 1.0 scaled
        };

        if vad_scaled <= nad_scaled {
            // Assign false (bit remains 0)
            // No action needed as variables are initialized to 0
        } else {
            // Calculate probability as integer scaled by 1000
            let prob_scaled = (num_p * 1000) / (num_p + num_n).max(1);
            // Convert scaled probability to f64 between 0 and 1
            if rng.gen_bool((prob_scaled as f64) / 1000.0) {
                variables |= 1 << v;
            }
        }
    }

    // Track satisfied clauses using a vector of counts
    let mut num_good_so_far: Vec<u8> = vec![0; num_clauses];
    for (i, &c) in clauses.iter().enumerate() {
        for &l in &c {
            let var = (l.abs() - 1) as usize;
            if (l > 0 && (variables & (1 << var)) != 0)
                || (l < 0 && (variables & (1 << var)) == 0)
            {
                num_good_so_far[i] += 1;
            }
        }
    }

    // Identify residual clauses with zero satisfied literals
    let mut residual_: Vec<usize> = Vec::with_capacity(num_clauses / 2); // Estimate half as residual
    let mut residual_indices: Vec<Option<usize>> = vec![None; num_clauses];

    for (i, &num_good) in num_good_so_far.iter().enumerate() {
        if num_good == 0 {
            residual_indices[i] = Some(residual_.len());
            residual_.push(i);
        }
    }

    // Calculate fuel and rounds based on provided parameters
    let clauses_ratio = (challenge.clauses.len() * 100 / challenge.num_variables) as f64;
    let num_vars_f64 = num_variables as f64;
    let max_fuel = 2_000_000_000.0;
    let base_fuel = (2000.0 + 40.0 * clauses_ratio) * num_vars_f64;
    let flip_fuel = 350.0 + 0.9 * clauses_ratio;
    let max_num_rounds = ((max_fuel - base_fuel) / flip_fuel) as usize;

    // Main loop for flipping variables to satisfy residual clauses
    loop {
        if !residual_.is_empty() {
            // Select a residual clause randomly
            let rand_val = rng.gen::<usize>();
            let clause_idx = residual_[rand_val % residual_.len()];
            let c = clauses[clause_idx];

            // Shuffle the clause literals to add randomness to the selection
            let mut c_shuffled = c.clone();
            if c_shuffled.len() > 1 {
                let random_index = rand_val % c_shuffled.len();
                c_shuffled.swap(0, random_index);
            }

            let mut min_sad = usize::MAX;
            let mut v_min_sad = usize::MAX;

            // Determine the variable to flip using MCV
            for &l in &c_shuffled {
                let abs_l = (l.abs() - 1) as usize;
                let clauses_to_check = if (variables & (1 << abs_l)) != 0 {
                    &p_clauses[abs_l]
                } else {
                    &n_clauses[abs_l]
                };

                let mut sad = 0;
                for &c_idx in clauses_to_check {
                    if num_good_so_far[c_idx] == 1 {
                        sad += 1;
                    }
                }

                if sad < min_sad {
                    min_sad = sad;
                    v_min_sad = abs_l;
                }
            }

            // Choose variable to flip based on SAD
            let v = if min_sad == 0 {
                v_min_sad
            } else {
                // Introduce randomness in selection to escape local minima
                if rng.gen_bool(0.5) {
                    (c_shuffled[0].abs() - 1) as usize
                } else {
                    v_min_sad
                }
            };

            // Flip the variable and update clause satisfaction accordingly
            if (variables & (1 << v)) != 0 {
                // Currently True, flipping to False
                variables &= !(1 << v); // Flip to False

                // Update clauses affected by this flip
                for &c_idx in &n_clauses[v] {
                    num_good_so_far[c_idx] += 1;
                    if num_good_so_far[c_idx] == 1 {
                        residual_indices[c_idx] = Some(residual_.len());
                        residual_.push(c_idx);
                    }
                }
                for &c_idx in &p_clauses[v] {
                    if num_good_so_far[c_idx] == 1 {
                        // Remove from residual
                        if let Some(idx) = residual_indices[c_idx] {
                            let last = residual_.pop().unwrap();
                            if idx < residual_.len() {
                                residual_[idx] = last;
                                residual_indices[last] = Some(idx);
                            }
                        }
                        residual_indices[c_idx] = None;
                    }
                    num_good_so_far[c_idx] = num_good_so_far[c_idx].saturating_sub(1);
                }
            } else {
                // Currently False, flipping to True
                variables |= 1 << v; // Flip to True

                // Update clauses affected by this flip
                for &c_idx in &p_clauses[v] {
                    num_good_so_far[c_idx] += 1;
                    if num_good_so_far[c_idx] == 1 {
                        residual_indices[c_idx] = Some(residual_.len());
                        residual_.push(c_idx);
                    }
                }
                for &c_idx in &n_clauses[v] {
                    if num_good_so_far[c_idx] == 1 {
                        // Remove from residual
                        if let Some(idx) = residual_indices[c_idx] {
                            let last = residual_.pop().unwrap();
                            if idx < residual_.len() {
                                residual_[idx] = last;
                                residual_indices[last] = Some(idx);
                            }
                        }
                        residual_indices[c_idx] = None;
                    }
                    num_good_so_far[c_idx] = num_good_so_far[c_idx].saturating_sub(1);
                }
            }

            rounds += 1;
            if rounds >= max_num_rounds {
                // Exceeded maximum allowed rounds without finding a solution
                return Ok(());
            }
        } else {
            // All clauses are satisfied
            break;
        }
    }

    // Convert the bitmask back to Vec<bool> for the solution
    let mut solution_variables = Vec::with_capacity(num_variables);
    for v in 0..num_variables {
        solution_variables.push((variables & (1 << v)) != 0);
    }

    let _ = save_solution(&Solution {
        variables: solution_variables,
    });
    return Ok(());
}

pub fn help() {
    println!("No help information available.");
}
