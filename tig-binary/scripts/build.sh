#!/bin/bash

set -e

CUDA=false

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 CHALLENGE ALGORITHM [--cuda]"
    echo "Example: $0 knapsack dynamic"
    echo "Example with CUDA: $0 knapsack dynamic --cuda"
    exit 1
fi

CHALLENGE="$1"
ALGORITHM="$2"

shift 2
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda)
            CUDA=true
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 CHALLENGE ALGORITHM [--cuda]"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$CHALLENGE" ]; then
    echo "Error: CHALLENGE argument is empty"
    exit 1
fi

if [ -z "$ALGORITHM" ]; then
    echo "Error: ALGORITHM argument is empty"
    exit 1
fi

echo "Compiling algorithm $ALGORITHM for challenge $CHALLENGE"
cp tig-binary/src/entry_point_template.rs tig-binary/src/entry_point.rs
sed -i "s/{CHALLENGE}/$CHALLENGE/g" tig-binary/src/entry_point.rs
sed -i "s/{ALGORITHM}/$ALGORITHM/g" tig-binary/src/entry_point.rs

if [ "$CUDA" = true ]; then
    FEATURES="entry_point cuda"
else
    FEATURES="entry_point"
fi

RUSTFLAGS="--emit=llvm-ir -C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 -C lto=no -C debuginfo=2 -C relocation-model=pic" \
cargo +nightly-2025-02-10 build \
    -p tig-binary \
    --target=$RUST_TARGET \
    --release \
    -Z build-std=core,alloc,std \
    -Z build-std-features=panic-unwind \
    --features=$FEATURES

ll_files=()
while IFS= read -r line; do
    if [[ $line != *"panic_abort"* ]]; then
        ll_files+=("$line")
    fi
done < <(find "target/$RUST_TARGET/release/deps" -name "*.ll")


object_files=()
for ll_file in "${ll_files[@]}"
do
    echo "Running LLVM pass on $ll_file"
    temp_obj=$(mktemp).o

    if [ ${#object_files[@]} -eq 0 ]
    then
        IS_FIRST_SRC=1
    else
        IS_FIRST_SRC=0
    fi

    cat "$ll_file" | \
    IS_FIRST_SRC=$IS_FIRST_SRC INSTRUMENT_FUEL=1 INSTRUMENT_RTSIG=1 INSTRUMENT_MEMORY=1 opt \
        -load-pass-plugin LLVMFuelRTSig.so \
        -passes="fuel-rt-sig" -S -o - | \
    llc -relocation-model=pic -o - | \
    clang -fPIC -c -x assembler - -o "$temp_obj"

    if [ $? -ne 0 ]
    then
        echo "Failed to process $ll_file"
        exit 1
    fi

    object_files+=("$temp_obj")
done

RUST_TARGET_LIBDIR=$(rustc --print target-libdir --target=$RUST_TARGET)
LIBSTD_HASH=$(find "$RUST_TARGET_LIBDIR" -name "libstd-*.rlib" -exec basename {} \; | head -n1 | sed -E 's/libstd-(.*)\..*$/\1/')

if [ ! -L "$RUST_TARGET_LIBDIR/libstd.so" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH.so" "$RUST_TARGET_LIBDIR/libstd.so"
fi

output=tig-algorithms/$ARCH/$CHALLENGE/$ALGORITHM.so
mkdir -p $(dirname $output)

echo "Linking into shared library '$output'"
cat > export_symbols.map << 'EOF'
{
  global:
    entry_point;
    __fuel_remaining;
    __runtime_signature;
  local: *;
};
EOF
clang "${object_files[@]}" \
    -shared \
    -fPIC \
    -O3 \
    -o $output \
    -L "$RUST_TARGET_LIBDIR" \
    -lstd \
    -Wl,--gc-sections \
    -ffunction-sections \
    -fdata-sections \
    -Wl,-z,noexecstack \
    -Wl,--build-id=none \
    -Wl,--eh-frame-hdr \
    -Wl,--version-script=export_symbols.map \
    -Wl,-Map=output.map

strip --strip-debug $output

if [ "$CUDA" = true ]; then
    echo "Compiling CUDA code"
    PTX_FILE="tig-algorithms/cuda/$CHALLENGE/$ALGORITHM.ptx"

    mkdir -p "$(dirname "$PTX_FILE")"

    echo "Combining .cu source files for algorithm $ALGORITHM and challenge $CHALLENGE"
    cat tig-binary/src/framework.cu tig-challenges/src/$CHALLENGE.cu tig-algorithms/src/$CHALLENGE/$ALGORITHM.cu > /tmp/temp.cu

    echo "Compiling PTX @ $PTX_FILE"
    nvcc -ptx /tmp/temp.cu -o "$PTX_FILE" \
        -arch compute_70 \
        -code sm_70 \
        --use_fast_math \
        -dopt=on

    rm -f /tmp/temp.cu

    echo "Adding runtime signature to PTX file"
    add_runtime_signature.py $PTX_FILE
fi

echo "Done"