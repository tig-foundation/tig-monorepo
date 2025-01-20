#!/bin/bash

if ! command -v curl >/dev/null 2>&1
then
    echo "Installing curl..."
    sudo apt-get update && sudo apt-get install -y curl

    if [ $? -ne 0 ]
    then
        echo "Error: Failed to install curl"
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

if ! command -v cc >/dev/null 2>&1
then
    echo "Installing build-essential..."
    sudo apt-get update && sudo apt-get install -y build-essential

    if [ $? -ne 0 ]
    then
        echo "Error: Failed to install build-essential"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLVM_RELEASES=(
    "https://nightly.link/tig-foundation/llvm/actions/runs/12875091437/llvm.zip" # TESTING ONLY
)

LLVM_RELEASE_IDX=${LLVM_RELEASE:-${#LLVM_RELEASES[@]}-1}
CURRENT_RELEASE="${LLVM_RELEASES[$LLVM_RELEASE_IDX]}"
ARTIFACT_ID=$(echo "$CURRENT_RELEASE" | grep -o 'runs/[0-9]*' | cut -d'/' -f2)
LLVM_ARCHIVE="llvm-${ARTIFACT_ID}.tar.zst"
LLVM_DIR="llvm-${ARTIFACT_ID}"
if [ ! -d "$SCRIPT_DIR/$LLVM_DIR" ]; then
    rustup install nightly-2024-12-17 &&
    rustup +nightly-2024-12-17 component add rust-src &&
    rustup +nightly-2024-12-17 target add aarch64-unknown-linux-gnu

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install rust toolchain"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/$LLVM_ARCHIVE.zip" ]; then
        echo "Using LLVM release: $CURRENT_RELEASE"
        echo "Downloading $LLVM_ARCHIVE..."
        curl -L "$CURRENT_RELEASE" -o "$SCRIPT_DIR/$LLVM_ARCHIVE.zip"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download LLVM release"
            exit 1
        fi
    fi

    mkdir -p "$SCRIPT_DIR/$LLVM_DIR"

    if ! command -v unzip >/dev/null 2>&1; then
        echo "Installing unzip..."
        sudo apt-get update && sudo apt-get install -y unzip
        if [ $? -ne 0 ]
        then
            echo "Error: Failed to install unzip"
            exit 1
        fi
    fi

    if ! command -v zstd >/dev/null 2>&1
    then
        echo "Installing zstd..."
        sudo apt-get update && sudo apt-get install -y zstd
        if [ $? -ne 0 ]
        then
            echo "Error: Failed to install zstd"
            exit 1
        fi
    fi

    unzip "$SCRIPT_DIR/$LLVM_ARCHIVE.zip" -d "$SCRIPT_DIR/$LLVM_DIR"
    if [ $? -ne 0 ]
    then
        echo "Error: Failed to unzip LLVM release"
        exit 1
    fi
    tar -xf "$SCRIPT_DIR/$LLVM_DIR/llvm.tar.zst" -C "$SCRIPT_DIR/$LLVM_DIR"

    if [ $? -ne 0 ]; then
        rm -rf "$SCRIPT_DIR/$LLVM_DIR"
        echo "Error: Failed to extract llvm.tar.zst"
        exit 1
    fi

    if [ -d "$SCRIPT_DIR/$LLVM_DIR/llvm/build" ]; then
        mv "$SCRIPT_DIR/$LLVM_DIR/llvm/build"/* "$SCRIPT_DIR/$LLVM_DIR/"
        rm -rf "$SCRIPT_DIR/$LLVM_DIR/llvm"
    fi

    rm -rf "$SCRIPT_DIR/$LLVM_DIR/llvm.tar.zst"
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

pushd "$SCRIPT_DIR/$LLVM_DIR"
"$SCRIPT_DIR/build-rust-project.sh" "$PROJECT_FOLDER" -r --shared -o "$PROJECT_FOLDER/rtsig_blob.dylib" -f 2500 --features "$FEATURES"
popd

echo "$PROJECT_FOLDER/rtsig_blob.dylib"

if [ $? -ne 0 ]
then
    exit 1
fi

