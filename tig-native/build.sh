#!/bin/bash

if command -v apt-get >/dev/null 2>&1
then
    if ! command -v curl >/dev/null 2>&1
    then
        echo "Installing curl..."
        if command -v sudo >/dev/null 2>&1
        then
            sudo apt-get update && sudo apt-get install -y curl
        else
            apt-get update && apt-get install -y curl
        fi

        if [ $? -ne 0 ]
        then
            echo "Error: Failed to install curl"
            exit 1
        fi
    fi
else
    if ! command -v curl >/dev/null 2>&1
    then
        echo "Error: curl is not installed and apt-get is not available"
        exit 1
    fi
fi

if ! command -v rustup >/dev/null 2>&1
then
    echo "Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    if [ $? -ne 0 ]
    then
        echo "Error: Failed to install rustup"
        exit 1
    fi

    source "$HOME/.cargo/env"
fi

if command -v apt-get >/dev/null 2>&1
then
    if ! command -v cc >/dev/null 2>&1
    then
        echo "Installing build-essential..."
        if command -v sudo >/dev/null 2>&1
        then
            sudo apt-get update && sudo apt-get install -y build-essential
        else
            apt-get update && apt-get install -y build-essential
        fi

        if [ $? -ne 0 ]
        then
            echo "Error: Failed to install build-essential"
            exit 1
        fi
    fi
else
    if ! command -v cc >/dev/null 2>&1
    then
        echo "Error: C compiler is not installed and apt-get is not available"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLVM_RELEASES=(
    "https://github.com/tig-foundation/llvm/releases/download/build-0971e080be6c87bf551a9f41897bda4aebad32b1/llvm.tar.zst", #TESTING ONLY
    "https://github.com/tig-foundation/llvm/releases/download/testing-0.0.1/llvm.tar.zst"
)

LLVM_CHECKSUMS=(
    "db302cce4320024258f3e9bd7839b418e5e683214eafbfc3833dfd0e2a63b244"
    "381b3492edfc9ebd305f79200f53c8b1de7eae8f6d5bc7e31be21fa619778522"
)

TOOLCHAIN="${RUST_TOOLCHAIN:-nightly-2025-01-16}" # default to latest nightly using llvm 19.1.6
LLVM_RELEASE_IDX=${LLVM_RELEASE:-${#LLVM_RELEASES[@]}-1}
CURRENT_RELEASE="${LLVM_RELEASES[$LLVM_RELEASE_IDX]}"
ARTIFACT_ID=$(echo "$CURRENT_RELEASE" | sed -E 's|.*/([^/]+)/llvm.tar.zst|\1|')
LLVM_ARCHIVE="llvm-${ARTIFACT_ID}.tar.zst"
LLVM_DIR="llvm-${ARTIFACT_ID}"
LLVM_CHECKSUM="${LLVM_CHECKSUMS[$LLVM_RELEASE_IDX]}"
if [ ! -d "$SCRIPT_DIR/$LLVM_DIR" ]; then
    rustup install $TOOLCHAIN &&
    rustup +$TOOLCHAIN component add rust-src &&
    rustup +$TOOLCHAIN target add aarch64-unknown-linux-gnu

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install rust toolchain"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/$LLVM_ARCHIVE" ]; then
        echo "Using LLVM release: $CURRENT_RELEASE"
        echo "Downloading $LLVM_ARCHIVE..."
        curl -L "$CURRENT_RELEASE" -o "$SCRIPT_DIR/$LLVM_ARCHIVE"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download LLVM release"
            exit 1
        fi

        echo "Verifying SHA256 checksum..."
        COMPUTED_CHECKSUM=$(sha256sum "$SCRIPT_DIR/$LLVM_ARCHIVE" | cut -d' ' -f1)
        
        if [ "$COMPUTED_CHECKSUM" != "$LLVM_CHECKSUM" ]
        then
            rm -f "$SCRIPT_DIR/$LLVM_ARCHIVE"
            echo "Error: SHA256 checksum verification failed"
            echo "Expected: $LLVM_CHECKSUM"
            echo "Got: $COMPUTED_CHECKSUM"
            exit 1
        fi
    fi

    mkdir -p "$SCRIPT_DIR/$LLVM_DIR"

    if command -v apt-get >/dev/null 2>&1
    then
        if ! command -v zstd >/dev/null 2>&1
        then
            echo "Installing zstd..."
            if command -v sudo >/dev/null 2>&1
            then
                sudo apt-get update && sudo apt-get install -y zstd
            else
                apt-get update && apt-get install -y zstd
            fi
            if [ $? -ne 0 ]
            then
                echo "Error: Failed to install zstd"
                exit 1
            fi
        fi
    else
        if ! command -v zstd >/dev/null 2>&1
        then
            echo "Error: zstd is not installed and apt-get is not available"
            exit 1
        fi
    fi

    tar -xf "$SCRIPT_DIR/$LLVM_ARCHIVE" -C "$SCRIPT_DIR/$LLVM_DIR"

    if [ $? -ne 0 ]; then
        rm -rf "$SCRIPT_DIR/$LLVM_DIR"
        echo "Error: Failed to extract llvm.tar.zst"
        exit 1
    fi

    if [ -d "$SCRIPT_DIR/$LLVM_DIR/llvm/build" ]; then
        mv "$SCRIPT_DIR/$LLVM_DIR/llvm/build"/* "$SCRIPT_DIR/$LLVM_DIR/"
        rm -rf "$SCRIPT_DIR/$LLVM_DIR/llvm"
    fi
fi

PATH_TO_LLVM="$SCRIPT_DIR/$LLVM_DIR"

if [ $# -eq 0 ]
then
    echo "Error: Project folder argument required"
    exit 1
fi

PROJECT_FOLDER=""
FEATURES="entry_point"
while [[ $# -gt 0 ]]
do
    case "$1" in
        --knapsack)
            FEATURES="$FEATURES knapsack"
            shift
        ;;
        --vector-search)
            FEATURES="$FEATURES vector_search"
            shift
        ;;
        --satisfiability)
            FEATURES="$FEATURES satisfiability"
            shift
        ;;
        --vehicle-routing)
            FEATURES="$FEATURES vehicle_routing"
            shift
        ;;
        *)
            PROJECT_FOLDER="$1"
            shift
        ;;
    esac
done

PROJECT_FOLDER=$(realpath "$PROJECT_FOLDER")
PROJECT_BASE_DIR=$(dirname "$PROJECT_FOLDER")

rm -rf "$PROJECT_FOLDER/target"

if [ -z "$ALGORITHM_NAME" ]
then
    echo "Error: ALGORITHM_NAME environment variable is required"
    exit 1
fi

CATEGORY_NAME=""
for feature in $FEATURES
do
    case "$feature" in
        "knapsack") CATEGORY_NAME="knapsack" ;;
        "vector_search") CATEGORY_NAME="vector_search" ;;
        "satisfiability") CATEGORY_NAME="satisfiability" ;;
        "vehicle_routing") CATEGORY_NAME="vehicle_routing" ;;
    esac
done

if [ -z "$CATEGORY_NAME" ]
then
    echo "Error: No valid category feature enabled"
    exit 1
fi

cp "$PROJECT_FOLDER/solve.rs" "$PROJECT_FOLDER/src/solve.rs"
sed -i.bak "s/{challenge_type}/$CATEGORY_NAME/g" "$PROJECT_FOLDER/src/solve.rs"
sed -i.bak "s/{algorithm_name}/$ALGORITHM_NAME/g" "$PROJECT_FOLDER/src/solve.rs"
rm -f "$PROJECT_FOLDER/src/solve.rs.bak"

pushd "$SCRIPT_DIR/$LLVM_DIR"
export TOOLCHAIN
"$SCRIPT_DIR/build-rust-project.sh" "$PROJECT_FOLDER" -r --shared -o "$PROJECT_FOLDER/rtsig_blob.dylib" -f 2500 --features "$FEATURES"
popd

echo "$PROJECT_FOLDER/rtsig_blob.dylib"

if [ $? -ne 0 ]
then
    exit 1
fi
