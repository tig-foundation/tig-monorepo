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
    --features="$FEATURES"

ll_files=()
while IFS= read -r line; do
    if [[ $line != *"panic_abort"* ]]; then
        ll_files+=("$line")
    fi
done < <(find "target/$RUST_TARGET/release/deps" -name "*.ll")

# Use number of processors available or default to 4
MAX_PROCS=$(nproc 2>/dev/null || echo 4)

object_files=()
temp_objs=()
pids=()
file_pids=()
pid_files=()
pid_objs=()
max_jobs=$(nproc)

# Process a single file
process_file() {
    local ll_file="$1"
    local is_first=$2
    local temp_obj="$3"
    
    echo "Processing $ll_file"
    
    cat "$ll_file" | \
    IS_FIRST_SRC=$is_first INSTRUMENT_FUEL=1 INSTRUMENT_RTSIG=1 INSTRUMENT_MEMORY=1 opt \
        -load-pass-plugin /opt/llvm/lib/LLVMFuelRTSig.so \
        -passes="fuel-rt-sig" -S -o - | \
    llc -relocation-model=pic -o - | \
    clang -fPIC -c -x assembler - -o "$temp_obj"
    
    return $?
}

# Launch processes up to max_jobs
for ll_file in "${ll_files[@]}"
do
    temp_obj=$(mktemp).o
    temp_objs+=("$temp_obj")
    
    # Set IS_FIRST_SRC flag - first file gets 1, all others get 0
    if [ "$ll_file" = "${ll_files[0]}" ]
    then
        is_first=1
    else
        is_first=0
    fi
    
    # Launch in background
    process_file "$ll_file" $is_first "$temp_obj" &
    pid=$!
    
    # Store associations
    pids+=($pid)
    pid_files[$pid]="$ll_file"
    pid_objs[$pid]="$temp_obj"
    
    # If we've reached max_jobs, wait for one to finish
    if [ ${#pids[@]} -ge $max_jobs ]
    then
        wait -n
        exit_status=$?
        
        if [ $exit_status -ne 0 ]
        then
            # Find which process failed
            for p in "${pids[@]}"
            do
                if ! kill -0 $p 2>/dev/null
                then
                    echo "Failed to process ${pid_files[$p]}"
                    # Kill all remaining processes
                    for active_pid in "${pids[@]}"
                    do
                        if kill -0 $active_pid 2>/dev/null
                        then
                            kill $active_pid 2>/dev/null || true
                        fi
                    done
                    exit 1
                fi
            done
            
            # If we couldn't identify which process failed, just exit
            echo "An error occurred during processing"
            exit 1
        fi
        
        # Remove completed process(es) from tracking
        for idx in "${!pids[@]}"
        do
            if ! kill -0 ${pids[$idx]} 2>/dev/null
            then
                # Success - add object file
                object_files+=("${pid_objs[${pids[$idx]}]}")
                unset pids[$idx]
            fi
        done
        
        # Reindex arrays
        pids=("${pids[@]}")
    fi
done

# Wait for all remaining processes
while [ ${#pids[@]} -gt 0 ]
do
    wait -n
    exit_status=$?
    
    if [ $exit_status -ne 0 ]
    then
        # Find which process failed
        for p in "${pids[@]}"
        do
            if ! kill -0 $p 2>/dev/null
            then
                echo "Failed to process ${pid_files[$p]}"
                # Kill all remaining processes
                for active_pid in "${pids[@]}"
                do
                    if kill -0 $active_pid 2>/dev/null
                    then
                        kill $active_pid 2>/dev/null || true
                    fi
                done
                exit 1
            fi
        done
        
        echo "An error occurred during processing"
        exit 1
    fi
    
    # Remove completed process(es) from tracking
    for idx in "${!pids[@]}"
    do
        if ! kill -0 ${pids[$idx]} 2>/dev/null
        then
            # Success - add object file
            object_files+=("${pid_objs[${pids[$idx]}]}")
            unset pids[$idx]
        fi
    done
    
    # Reindex arrays
    pids=("${pids[@]}")
done

echo "Successfully processed all files"

RUST_TARGET_LIBDIR=$(rustc --print target-libdir --target=$RUST_TARGET)
LIBSTD_HASH=$(find "$RUST_TARGET_LIBDIR" -name "libstd-*.rlib" -exec basename {} \; | head -n1 | sed -E 's/libstd-(.*)\..*$/\1/')

if [ ! -L "$RUST_TARGET_LIBDIR/libstd.so" ]
then
    ln -sf "$RUST_TARGET_LIBDIR/libstd-$LIBSTD_HASH.so" "$RUST_TARGET_LIBDIR/libstd.so"
fi

ARCH=$(if [ "$(uname -i)" = "aarch64" ] || [ "$(uname -i)" = "arm64" ]; then
    echo "aarch64"
else
    echo "amd64"
fi)
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
    # use benchmarker_outbound?
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

# Clean up temp files
rm -f /tmp/*.o.path

echo "Done"