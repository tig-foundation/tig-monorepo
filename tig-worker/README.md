# tig-worker

A Rust crate for verifying and computing solutions.

Solutions are computed by executing an algorithm in a WASM virtual machine ([TIG's fork of wasmi](https://github.com/tig-foundation/wasmi)).

# Compiling

`tig-worker` is setup such that you can compile an algorithm to the native environment by setting the feature. This is entirely optional, and is useful for performance testing (see useful scripts).

Command for normal compilation:
```
cargo build -p tig-worker --release
./target/release/tig-worker --help
```

Command for compiling specific algorithms to native environment:
```
ALGOS_TO_COMPILE="" # see notes
cargo build -p tig-worker --release --features "${ALGOS_TO_COMPILE}"
./target/release/tig-worker --help
```

**Notes:**

* Setting `ALGOS_TO_COMPILE` will run the selected algorithms directly in your execution environment to filter for nonces that results in solutions. Nonces that are filtered will then be re-run in the WASM virtual machine to compute the necessary solution data for submission.

* **WARNING** before setting `ALGOS_TO_COMPILE`, be sure to thoroughly review the algorithm code for malicious routines as it will be ran directly in your execution environment (not within a sandboxed WASM virtual machine)!

* `ALGOS_TO_COMPILE` is a space separated string of algorithms with format `<challenge_name>_<algorithm_name>`. Example: 

    ```
    ALGOS_TO_COMPILE="satisfiability_schnoing vehicle_routing_clarke_wright knapsack_dynamic"
    ```

# Usage

`tig-worker` has sub-commands `compute_solution` and `verify_solution`. These are used in 2 scripts:

* [Test algorithm performance](../scripts/test_algorithm_performance.sh)
* [Verify benchmark solutions](../scripts/verify_benchmark_solutions.sh)

## Compute Solution

Given settings, nonce and the WASM for an algorithm, `tig-worker` computes the solution data (runtime_signature, fuel_consumed, solution). This sub-command does not verify whether the solution is valid or not.

* If the algorithm results in an error, `tig-worker` will terminate with exit code 1 and print error to stderr.

* If the algorithm returns a solution, `tig-worker` will terminate with exit code 0 and print the solution data to stdout.

```
Usage: tig-worker compute_solution [OPTIONS] <SETTINGS> <NONCE> <WASM>

Arguments:
  <SETTINGS>  Settings json string or path to json file
  <NONCE>     Nonce value
  <WASM>      Path to a wasm file

Options:
      --fuel [<FUEL>]  Optional maximum fuel parameter for WASM VM [default: 1000000000]
      --mem [<MEM>]    Optional maximum memory parameter for WASM VM [default: 1000000000]
  -h, --help           Print help
```

## Verify Solution

Given settings, nonce and a solution, `tig-worker` verifies the solution is a valid solution for the challenge instance.

* If the solution is valid, `tig-worker` will terminate with exit code 0

* If the solution is invalid, `tig-worker` will terminate with exit code 1

```
Usage: tig-worker verify_solution <SETTINGS> <NONCE> <SOLUTION>

Arguments:
  <SETTINGS>  Settings json string or path to json file
  <NONCE>     Nonce value
  <SOLUTION>  Solution json string or path to json file

Options:
  -h, --help  Print help
```

# License

[End User License Agreement](../docs/agreements/end_user_license_agreement.pdf)