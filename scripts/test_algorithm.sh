#!/bin/bash
REPO_DIR=$(dirname $(dirname "$(realpath "$0")"))
TIG_WORKER_PATH="$REPO_DIR/target/release/tig-worker"

if [ ! -f $TIG_WORKER_PATH ]; then
    echo "Error: tig-worker binary not found at ./target/release/tig-worker"
    echo "Run: cd $REPO_DIR && cargo build -p tig-worker --release"
    exit 1
fi


options=()
index=0
echo "Available WASMs:"
for w in $(find $REPO_DIR/tig-algorithms/wasm -name '*.wasm'); do
    a_name=$(basename $w .wasm)
    c_name=$(basename $(dirname $w))
    echo "$index) $c_name/$a_name"
    options+=("$c_name/$a_name")
    index=$((index + 1))
done
echo "Don't see the algorithm you're looking for? Can run: git pull origin <challenge_name>/<algorithm_name> --no-edit"
read -p "Enter the index of the algorithm to test: " selected_index
if [[ $selected_index =~ ^[0-9]+$ ]] && [ "$selected_index" -ge 0 ] && [ "$selected_index" -lt "$index" ]; then
    selected_option=${options[$selected_index]}
    echo "Testing: $selected_option"
else
    echo "Invalid index"
    exit 1
fi

CHALLENGE=$(dirname $selected_option)
ALGORITHM=$(basename $selected_option)

case $CHALLENGE in
    satisfiability)
        CHALLENGE_ID="c001"
        ;;
    vehicle_routing)
        CHALLENGE_ID="c002"
        ;;
    knapsack)
        CHALLENGE_ID="c003"
        ;;
    vector_search)
        CHALLENGE_ID="c004"
        ;;
    *)
        echo "Error: Challenge '$CHALLENGE' is not recognized."
        exit 1
        ;;
esac

read -p "Enter difficulty for $CHALLENGE in format [x,y]: " difficulty
regex='^\[[0-9]+,[0-9]+\]$'
if ! [[ $difficulty =~ $regex ]]; then
    echo "Error: Difficulty must be in the format [x,y] where x and y are positive integers."
    exit 1
fi
is_positive_integer() {
    [[ $1 =~ ^[0-9]+$ ]] && [ "$1" -ge 0 ]
}
read -p "Enter starting nonce: " start_nonce
if ! is_positive_integer "$start_nonce"; then
    echo "Error: Starting nonce must be a positive integer."
    exit 1
fi
read -p "Enter number of nonces: " num_nonces
if ! is_positive_integer "$num_nonces"; then
    echo "Error: Number of nonces must be a positive integer."
    exit 1
fi
read -p "Enable debug mode? (leave blank to disable) " enable_debug
if [[ -n $enable_debug ]]; then
    debug_mode=true
else
    debug_mode=false
fi

SETTINGS="{\"challenge_id\":\"$CHALLENGE_ID\",\"difficulty\":$difficulty,\"algorithm_id\":\"\",\"player_id\":\"\",\"block_id\":\"\"}"
num_solutions=0
num_invalid=0
num_errors=0
total_ms=0

echo "----------------------------------------------------------------------"
echo "Testing performance of $CHALLENGE/$ALGORITHM"
echo "Settings: $SETTINGS"
echo "Starting nonce: $start_nonce"
echo "Number of nonces: $num_nonces"
echo -ne ""
for ((nonce=start_nonce; nonce<start_nonce+num_nonces; nonce++)); do
    start_time=$(date +%s%3N)
    stdout=$(mktemp)
    stderr=$(mktemp)
    ./target/release/tig-worker compute_solution "$SETTINGS" $nonce $REPO_DIR/tig-algorithms/wasm/$CHALLENGE/$ALGORITHM.wasm >"$stdout" 2>"$stderr"
    exit_code=$?
    output_stdout=$(cat "$stdout")
    output_stderr=$(cat "$stderr")
    end_time=$(date +%s%3N)
    duration=$((end_time - start_time))
    total_ms=$((total_ms + duration))
    if [ $exit_code -eq 0 ]; then
        num_solutions=$((num_solutions + 1))
    else
      if echo "$output_stderr" | grep -q "Invalid solution\|No solution found"; then
          num_invalid=$((num_invalid + 1))
      else
          num_errors=$((num_errors + 1))
      fi
    fi
    if [ $((num_solutions)) -eq 0 ]; then
        avg_ms_per_solution=0
    else
        avg_ms_per_solution=$((total_ms / num_solutions))
    fi
    if [[ $debug_mode == true ]]; then
        echo "    Nonce: $nonce"
        echo "    Exit code: $exit_code"
        echo "    Stdout: $output_stdout"
        echo "    Stderr: $output_stderr"
        echo "    Duration: $duration ms"
        echo "#instances: $((num_solutions + num_invalid + num_errors)), #solutions: $num_solutions, #invalid: $num_invalid, #errors: $num_errors, average ms/solution: $avg_ms_per_solution"
    else
        echo -ne "#instances: $((num_solutions + num_invalid + num_errors)), #solutions: $num_solutions, #invalid: $num_invalid, #errors: $num_errors, average ms/solution: $avg_ms_per_solution\033[K\r"
    fi
done
echo
echo "----------------------------------------------------------------------"
echo "To re-run this test, run the following commands:"
echo "    git clone https://github.com/tig-foundation/tig-monorepo.git"
echo "    cd tig-monorepo"
echo "    git pull origin/$CHALLENGE/$ALGORITHM --no-edit"
echo "    bash scripts/test_algorithm.sh"
echo "----------------------------------------------------------------------"
echo "Share your results on https://www.reddit.com/r/TheInnovationGame"
echo "----------------------------------------------------------------------"
rm "$stdout" "$stderr"