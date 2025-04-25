#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 CHALLENGE ALGORITHM"
    echo "Example: $0 knapsack dynamic"
    exit 1
fi

CHALLENGE="$1"
ALGORITHM="$2"

if [ -z "$CHALLENGE" ]; then
    echo "Error: CHALLENGE argument is empty"
    exit 1
fi

if [ -z "$ALGORITHM" ]; then
    echo "Error: ALGORITHM argument is empty"
    exit 1
fi

echo "Compiling algorithm $ALGORITHM for challenge $CHALLENGE"
cp tig-aarch64/src/entry_point_template.rs tig-aarch64/src/entry_point.rs
sed -i "s/{CHALLENGE}/$CHALLENGE/g" tig-aarch64/src/entry_point.rs
sed -i "s/{ALGORITHM}/$ALGORITHM/g" tig-aarch64/src/entry_point.rs

RUSTFLAGS="--emit=llvm-ir -C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 -C lto=no -C debuginfo=2 -C relocation-model=pic" \
cargo build \
    -p tig-aarch64 \
    --target=aarch64-unknown-linux-gnu \
    --release \
    -Z build-std=core,alloc,std \
    -Z build-std-features=panic-unwind \
    --features="entry_point"

ll_files=()
while IFS= read -r line; do
    if [[ $line != *"panic_abort"* ]]; then
        ll_files+=("$line")
    fi
done < <(find "target/aarch64-unknown-linux-gnu/release/deps" -name "*.ll")


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
    IS_FIRST_SRC=$IS_FIRST_SRC opt \
        -load-pass-plugin /opt/llvm-aarch64/lib/LLVMFuel.so \
        -load-pass-plugin /opt/llvm-aarch64/lib/LLVMRuntimeSig.so \
        -passes="runtime-signature,fuel" -S -o - | \
    llc -relocation-model=pic -o - | \
    clang -fPIC -c -x assembler - -o "$temp_obj"

    if [ $? -ne 0 ]
    then
        echo "Failed to process $ll_file"
        exit 1
    fi

    object_files+=("$temp_obj")
done

RUST_TARGET_LIBDIR=$(rustc --print target-libdir --target=aarch64-unknown-linux-gnu)
LIBSTD_HASH=$(find "$RUST_TARGET_LIBDIR" -name "libstd-*.dylib" -o -name "libstd-*.rlib" -exec basename {} \; | head -n1 | sed -E 's/libstd-(.*)\..*$/\1/')

if [ ! -L "$RUST_TARGET_LIBDIR/libstd.so" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH.so" "$RUST_TARGET_LIBDIR/libstd.so"
fi

if [ ! -L "$RUST_TARGET_LIBDIR/libstd.rlib" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH.rlib" "$RUST_TARGET_LIBDIR/libstd.rlib"
fi

output=tig-algorithms/aarch64/$CHALLENGE/$ALGORITHM.so
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

        ./bin/clang "${object_files[@]}" \
            -shared \
            -fPIC \
            -o "$output_name" \
            -L "$RUST_TARGET_LIBDIR" \
            -lm \
            -Wl,--gc-sections \
            -ffunction-sections \
            -fdata-sections \
            -Wl,-z,noexecstack \
            -Wl,--build-id=none \
            -Wl,--eh-frame-hdr \
            -Wl,-Map=output.map \
            -l$LIBSTD
            
strip --strip-debug $output

cat tig-ptx/framework.cu tig-challenges/src/${CHALLENGE}.cu tig-algorithms/src/${CHALLENGE}/${ALGORITHM}.cu > temp.cu

nvcc -ptx "$CUDA_TEMP_FILE" -o "$PTX_FILE" \
    -arch compute_70 \
    -code sm_70 \
    --use_fast_math \
    -dopt=on

# tar stuff

echo "Done"