#!/bin/bash

show_help() {
    echo "Usage: $(basename $0) [OPTIONS] \"challange1/algorithm1 challange2/algorithm2 challange3/algorithm3 ...\""
    echo
    echo "Merge selected algorithms to local branch and generate cargo build command based on \
the specified algorithms with JSON algorithm selection."
    echo "This script must be run from the repository root directory."
    echo "It is required to use inside 'tig-monorepo' or custom repository with added it as a remote like:"
    echo "git remote add public https://github.com/tig-foundation/tig-monorepo.git"
    echo
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo "  -c, --command-only    Generate only the cargo build command"
    echo "  -j, --json-only       Generate only the JSON algorithm selection content"
    echo "  -o, --output <path>   Specify a file path to save the JSON algorithm selection content"
    echo
    echo "Arguments:"
    echo "  algorithms            Space-separated list of algorithms to be processed"
    echo
    echo "Example:"
    echo "  $(basename $0) \"satisfiability/sprint_sat vehicle_routing/clarke_wright_opt\""
    echo "  $(basename $0) -c \"satisfiability/sprint_sat vehicle_routing/clarke_wright_opt\""
    echo "  $(basename $0) -o algo_selection.json \"satisfiability/sprint_sat vehicle_routing/clarke_wright_opt\""
    exit 0
}

merge_branches=true
generate_command=false
generate_json=false
output_path=""

# Parse options
while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do case $1 in
    -h | --help )
        show_help
        ;;
    -c | --command-only )
        merge_branches=false
        generate_command=true
        ;;
    -j | --json-only )
        merge_branches=false
        generate_json=true
        ;;
    -o | --output )
        shift; output_path=$1
        ;;
esac; shift; done

if ! $generate_command && ! $generate_json; then
    generate_command=true
    generate_json=true
fi

if [ -z "$1" ]; then
    echo "No branches provided. Please provide branches to merge as a space-separated string."
    exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
branches_to_merge=$1

if $merge_branches; then
    git fetch public || exit 1

    for branch in $branches_to_merge; do
        echo "Checking out and updating branch '$branch'..."

        git checkout "$branch"
        if [ $? -ne 0 ]; then
            echo "Failed to checkout branch '$branch'. Please make sure you are in 'tig-monorepo' \
or you have added it as a remote repository."
            exit 1
        fi

        git pull
        if [ $? -ne 0 ]; then
            echo "Failed to pull latest changes for branch '$branch'. Please resolve any issues and try again."
            exit 1
        fi

        git checkout "$current_branch" || exit 1

        echo "Merging branch '$branch' into $current_branch..."
        git merge "$branch" -m "Merging branch '$branch' into $current_branch"
        if [ $? -ne 0 ]; then
            echo "Failed to merge branch '$branch'. Please resolve conflicts and try again."
            exit 1
        fi
    done

    echo "All branches merged successfully into $current_branch!"
    echo "----------------------------------------------------------------------"
fi

declare -A algorithms
features="standalone"

for branch in $branches_to_merge; do
  formatted_branch=$(echo "$branch" | sed 's/\// /')
  
  challenge=$(echo "$formatted_branch" | awk '{print $1}')
  algorithm=$(echo "$formatted_branch" | awk '{print $2}')
  
  algorithms["$challenge"]="$algorithm"
  features+=" ${challenge}_${algorithm}"
done

build_command="cargo build -p tig-benchmarker --release --no-default-features --features \"$features\""

json_content="{"
for key in "${!algorithms[@]}"; do
  json_content+="\"$key\":\"${algorithms[$key]}\","
done
json_content="${json_content%,}}"

if $generate_command; then
    echo "Build command:"
    echo "$build_command"
fi
if $generate_json; then
    echo "----------------------------------------------------------------------"
    echo "Algorithm selection file content:"
    echo $json_content
    if [ -n "$output_path" ]; then
        echo "$json_content" > "$output_path"
        echo "JSON content written to $output_path"
    fi
fi
