#!/bin/bash

if [ $# -eq 0 ]
then
    echo "Usage: $0 project_dir [-o output_name] [-r] [-f fuel] [--shared]"
    exit 1
fi

output_name="program.out"
project_dir=""
config=""
shared=0
features=""

while [[ $# -gt 0 ]]
do
    case $1 in
        -o)
            output_name="$2"
            shift 2
        ;;
        -r|--release)
            config="--release"
            shift
        ;;
        -f|--fuel)
            FUEL="$2"
            shift 2
        ;;
        --shared)
            shared=1
            shift
        ;;
        --features)
            features="$2"
            shift 2
        ;;
        *)
            project_dir="$1"
            shift
        ;;
    esac
done

pushd "$project_dir"

if [[ "$OSTYPE" == "darwin"* ]]
then
    PLUGIN_EXT=".dylib"
    LINKER_FLAGS="-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem"
    ENTRY_POINT="_main"
else
    PLUGIN_EXT=".so"
    ENTRY_POINT="main"
fi

TARGET=""
if [[ "$OSTYPE" == "darwin"* ]]
then
    TARGET="aarch64-apple-darwin"
elif [[ "$(uname -m)" == "arm64" ]] || [[ "$(uname -m)" == "aarch64" ]]
then
    TARGET="aarch64-unknown-linux-gnu"
else
    TARGET="x86_64-unknown-linux-gnu"
fi

if [ -z "$TARGET" ]
then
    echo "Failed to determine target architecture"
    exit 1
fi

if [ $shared -eq 1 ]
then
    export RUSTFLAGS="--emit=llvm-ir -C embed-bitcode=yes -C codegen-units=1 -C lto=no -C debuginfo=2 -C relocation-model=pic"
else
    export RUSTFLAGS="--emit=llvm-ir -C embed-bitcode=yes -C codegen-units=1 -C lto=no -C debuginfo=2 -C relocation-model=static"
fi

cargo +$TOOLCHAIN build \
    --target=$TARGET \
    $config \
    -Z build-std=core,alloc,std \
    -Z build-std-features=panic-unwind \
    --features="$features"

if [ $? -ne 0 ]
then
    echo "Failed to build Rust project"
    exit 1
fi

popd

if [ -z "$FUEL" ]
then
    FUEL=10000
fi

export FUEL

if [ -z "$config" ]
then
    config="debug"
else
    config="release"
fi

ll_files=()
while IFS= read -r line; do
    if [[ $line != *"panic_abort"* ]]; then
        ll_files+=("$line")
    fi
done < <(find "$project_dir/target/$TARGET/$config/deps" -name "*.ll" 2>/dev/null || find "$project_dir/../target/$TARGET/$config/deps" -name "*.ll")

echo "LL files: ${ll_files[@]}"

object_files=()

for ll_file in "${ll_files[@]}"
do
    temp_obj=$(mktemp).o

    if [ ${#object_files[@]} -eq 0 ]
    then
        IS_FIRST_SRC=1
    else
        IS_FIRST_SRC=0
    fi

    cat "$ll_file" | \
    IS_FIRST_SRC=$IS_FIRST_SRC ./bin/opt \
        -load-pass-plugin ./lib/LLVMFuel$PLUGIN_EXT \
        -load-pass-plugin ./lib/LLVMRuntimeSig$PLUGIN_EXT \
        -passes="runtime-signature,fuel" -S -o - | \
    ./bin/llc $([ $shared -eq 1 ] && echo "-relocation-model=pic" || echo "-relocation-model=static") -o - | \
    ./bin/clang $([ $shared -eq 1 ] && echo "-fPIC" || echo "-fno-PIC") -c -x assembler - -o "$temp_obj"

    if [ $? -ne 0 ]
    then
        echo "Failed to process $ll_file"
        exit 1
    fi

    object_files+=("$temp_obj")
done

RUST_TARGET_LIBDIR=$(rustc +$TOOLCHAIN --print target-libdir --target=$TARGET)
LIBSTD_HASH=$(find "$RUST_TARGET_LIBDIR" -name "libstd-*.dylib" -o -name "libstd-*.rlib" -exec basename {} \; | head -n1 | sed -E 's/libstd-(.*)\..*$/\1/')

if [ ! -L "$RUST_TARGET_LIBDIR/libstd$PLUGIN_EXT" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH$PLUGIN_EXT" "$RUST_TARGET_LIBDIR/libstd$PLUGIN_EXT"
fi

if [ ! -L "$RUST_TARGET_LIBDIR/libstd.rlib" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH.rlib" "$RUST_TARGET_LIBDIR/libstd.rlib"
fi

if [ "$(uname)" = "Darwin" ]; then
    if [ $shared -eq 1 ]
    then
        ./bin/clang "${object_files[@]}" \
            -shared \
            -fPIC \
            -o "$output_name" \
            -L "$RUST_TARGET_LIBDIR" \
            -lstd \
            $LINKER_FLAGS
    else
        ./bin/clang "${object_files[@]}" \
            -o "$output_name" \
            -L "$RUST_TARGET_LIBDIR" \
            -fno-PIE \
            -no-pie \
            -lstd \
            $LINKER_FLAGS
    fi
else
    if [ $shared -eq 1 ]
    then
        ./bin/clang "${object_files[@]}" \
            -shared \
            -fPIC \
            -o "$output_name" \
            -L "$RUST_TARGET_LIBDIR" \
            -lstd \
            -Wl,--gc-sections \
            -ffunction-sections \
            -fdata-sections \
            -Wl,-z,noexecstack \
            -Wl,--build-id=none \
            -Wl,--eh-frame-hdr \
            -Wl,-Map=output.map
    else
        ./bin/clang "${object_files[@]}" \
            -o "$output_name" \
            -L "$RUST_TARGET_LIBDIR" \
            -fno-PIE \
            -no-pie \
            -lstd \
            -Wl,--gc-sections \
            -ffunction-sections \
            -fdata-sections \
            -Wl,-z,noexecstack \
            -Wl,--build-id=none \
            -Wl,--eh-frame-hdr \
            -Wl,-Map=output.map
    fi
fi

#rm -f "${ll_files[@]}"

if [ $? -eq 0 ]
then
    echo "Successfully compiled to $output_name"
    rm -f "${object_files[@]}"

    echo "LD_LIBRARY_PATH=$RUST_TARGET_LIBDIR:\$LD_LIBRARY_PATH ./$output_name"
else
    echo "Linking failed"
    exit 1
fi
