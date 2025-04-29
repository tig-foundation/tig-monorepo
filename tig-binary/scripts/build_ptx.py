#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Import the dictionary from ptx_instructions.py
instruction_prime_map = {
    'add.u32': (0xEDC1A7F47ACD2EE3, 2),
    'add.u64': (0xBE8098FE5CFCC7B5, 3),
    'add.f32': (0x90B2AD995AF897CD, 4),
    'add.f64': (0xC3EA137E165AECED, 5),
    'add.s32': (0xE620952D37DDFF91, 2),
    'add.s64': (0x9BD972EB4677C3F1, 3),
    'sub.u32': (0x87353D33C2976737, 2),
    'sub.u64': (0xE155AD1E824B25E3, 3),
    'sub.f32': (0xC256B33A0EF0745F, 4),
    'sub.f64': (0x8C4AAED901B0472B, 5),
    'mul.u32': (0x91C6A1D9B4EDC1AB, 4),
    'mul.u64': (0x9A16DFEE16D4BBF5, 5),
    'mul.f32': (0x9AF1495384190163, 5),
    'mul.f64': (0x883C081D0CDE3065, 6),
    'div.u32': (0xFD9B02FDC35C72B3, 10),
    'div.u64': (0xF5F2BCF367F68ED1, 12),
    'div.f32': (0xB161AB7FB04F004D, 15),
    'div.f64': (0x96E8220B9C5E8C97, 20),
    'mul.wide.u32': (0x9B8C768C1401BA7B, 6),
    'mul.wide.u64': (0xBC49DE9B3A818899, 8),
    'mad.wide.u32': (0xA17576DF232E7669, 8),
    'mad.wide.u64': (0xD298F1C0BF940F95, 10),
    'mov.u32': (0xED571105D6049077, 1),
    'mov.u64': (0xEF0F967771856DC3, 1),
    'mov.f32': (0xC73484F9874FAEDF, 1),
    'mov.f64': (0xC41D2D54E517F109, 1),
    'and.b32': (0x80B429DFBFC318EB, 1),
    'and.b64': (0xBDA9ECF782F0D3FB, 1),
    'or.b32': (0xC26355832083F5A1, 1),
    'or.b64': (0xEB578CCCB8F3BD37, 1),
    'xor.b32': (0xE58E31633D73A4A5, 1),
    'xor.b64': (0xB62B0F67BFA44C95, 1),
    'shl.b32': (0x88330E6E1BFA5411, 2),
    'shl.b64': (0x8374F43D807A6F91, 3),
    'shr.b32': (0xF1B8109D2F948463, 2),
    'shr.b64': (0xA3230556089777C7, 3),
    'cvt.u32.u64': (0xCEDC3D307D8D8683, 2),
    'cvt.f32.f64': (0x9DA4540AE2D7A161, 3),
    'cvt.u64.u32': (0x9A9E961B54B29955, 2),
    'cvt.f64.f32': (0x8B1B3D4D77BC7BB3, 3),
    'setp.eq.u32': (0xC161CEDAF256D5E5, 2),
    'setp.eq.u64': (0xD1E5DA2FDCD5E157, 3),
    'setp.lt.u32': (0x986CCCCFA9F5B10B, 2),
    'setp.lt.u64': (0x8CC7C690FF547D63, 3),
    'setp.gt.u32': (0x80FF2A3B14A4D19D, 2),
    'setp.gt.u64': (0xE0E9526F53C79197, 3),
    'setp.ne.u32': (0xB19319E767B773DF, 2),
    'setp.ne.u64': (0x8AC38410037C32D5, 3),
    'selp.u32': (0xAB0A8BAC52D5D76B, 3),
    'selp.u64': (0x9CDBFC00628D628B, 4),
    'abs.s32': (0x84378B096D13B6A3, 2),
    'abs.s64': (0xC04FBAAA56FA0DAB, 3),
    'abs.f32': (0xCE9AA2EB4B22456B, 3),
    'abs.f64': (0xF165D7826D16DF47, 4),
    'min.u32': (0xE95BA56F275D3EC5, 2),
    'min.u64': (0x8C3F2F6F2EB0C34F, 3),
    'min.f32': (0xBF2A12007525FEED, 3),
    'min.f64': (0xF3A8D718FEC1B393, 4),
    'max.u32': (0xF4692B3CA0566779, 2),
    'max.u64': (0xADB80126D86C7295, 3),
    'max.f32': (0xE6FD4C5BCBC70E3D, 3),
    'max.f64': (0xD89E33A78DBF9527, 4),
    'sqrt.rn.f32': (0xECE9FBD3A6D77023, 15),
    'sqrt.rn.f64': (0xC7D7E4D1245D5CCD, 20),
    'rsqrt.rn.f32': (0xBFA9439C30B70919, 15),
    'rsqrt.rn.f64': (0xDF9DC483E11B08A5, 20),
    'sqrt.approx.ftz.f32': (0x8062539FCA30F685, 8),
    'sqrt.approx.ftz.f64': (0xBEAA214049557A1B, 10),
    'sin.approx.f32': (0x8062539FCA30F685, 8),
    'sin.approx.f64': (0xBEAA214049557A1B, 10),
    'cos.approx.f32': (0x854FD92C1227AF13, 8),
    'cos.approx.f64': (0xE4ED779797574CBD, 10),
    'tanh.approx.f32': (0x8062539FCA30F685, 8),
    'tanh.approx.f64': (0xBEAA214049557A1B, 10),
    'add.f16': (0xA5234567890ABCDE, 1),
    'add.f16x2': (0xB5234567890ABCDE, 1),
    'add.bf16': (0xC5234567890ABCDE, 1), 
    'add.bf16x2': (0xD5234567890ABCDE, 1),
    'fma.rn.bf16': (0xE5234567890ABCDE, 1),
    'fma.rn.bf16x2': (0xF5234567890ABCDE, 1),
    'cvt.rn.bf16.f32': (0xA6234567890ABCDE, 1),
    'cvt.rn.f32.bf16': (0xB6234567890ABCDE, 1),
    'cvt.rn.tf32.f32': (0xC6234567890ABCDE, 1),
    'cvt.rn.f32.tf32': (0xD6234567890ABCDE, 1),
    'atom.add.u32': (0xF2D231CD2E1FBA23, 8),
    'atom.add.u64': (0xE511FE4AEA87A429, 10),
    'atom.min.u32': (0xC4FA795EB531A38B, 8),
    'atom.min.u64': (0xF0FB61E6281360FD, 10),
    'atom.max.u32': (0x8F28A03020BF8813, 8),
    'atom.max.u64': (0xD80622EB6110253F, 10),
    'tex.1d.v4.f32': (0xC56DD2999CD1234F, 15),
    'tex.2d.v4.f32': (0x918C3A31D782B0E5, 20),
    'tex.3d.v4.f32': (0xC002087960B604D5, 25),
    'ld.param.u32': (0xE97ADA4F02ABD567, 3),
    'ld.param.u64': (0x8521CA309251BB1D, 4),
    'st.param.u32': (0xF9DD000BE29F68F1, 3),
    'st.param.u64': (0xEC242CB6C99E502D, 4),
    'ld.const.u32': (0xCCBED9D942A60229, 3),
    'ld.const.u64': (0x8E8D513AAC06F061, 4),
    'popc.b32': (0xC6051FFCF3752D2B, 3),
    'popc.b64': (0xE94FDBF317D00AB7, 4),
    'clz.b32': (0xA2E613C950EA7F17, 3),
    'clz.b64': (0x8DB562EEC3BBD64F, 4),
    'brev.b32': (0xAE6A290706DD70E7, 3),
    'brev.b64': (0xD4FDC03C4401C533, 4),
    'unused': (0xA3E6942DA60A926F, 1),
    'unused': (0xBCB5CD76C9DC3253, 1),
    'unused': (0x966C40D3884717FF, 1),
    'unused': (0xFED4DB903BB79241, 1),
    'unused': (0xDA31E485E5E0C445, 1),
    'unused': (0xB8290AAC45720989, 1),
    'unused': (0xD0C39B467979E695, 1),
    'unused': (0xE12B4AF73AAABEEF, 1),
    'unused': (0xC1B8CDB5496BAFD7, 1),
}

def read_ptx_file(file_path):
    with open(file_path, 'r') as file:
        ptx_code = file.readlines()
    return ptx_code

def write_ptx_file(modified_code, output_path):
    with open(output_path, 'w') as file:
        file.writelines(modified_code)

def rotate_left(x, n):
    n = n & 0x3F
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF

def rotate_right(x, n):
    n = n & 0x3F
    return ((x >> n) | (x << (64 - n))) & 0xFFFFFFFFFFFFFFFF

def get_position_modifier(bb_idx, inst_idx, func_hash):
    left_rot = ((bb_idx * 7 + inst_idx * 13 + func_hash * 17) & 0x3F)
    right_rot = ((bb_idx * 11 + inst_idx * 17 + func_hash * 23) & 0x3F)
    return left_rot, right_rot

# Function to modify the PTX code by inserting XOR commands after certain instructions
def add_xor_commands(ptx_code, instruction_prime_map, ignore_patterns):
    modified_code = []
    inside_kernel_function = False
    skip_kernel = False
    r_signature_inserted = False
    current_block_signature = 0
    current_block_fuel = 0
    bb_idx = 0
    inst_idx = 0
    func_hash = 0
    current_func = ""

    # Track branch targets and their frequency
    branch_targets = {}
    # First pass - identify potential loop headers
    for line in ptx_code:
        stripped_line = line.strip()
        if stripped_line.startswith("$L") or stripped_line.startswith("BB"):
            label = stripped_line.split(":")[0]
            branch_targets[label] = 0
        elif "bra" in stripped_line:
            target = stripped_line.split()[-1].rstrip(";")
            if target in branch_targets:
                branch_targets[target] += 1

    # Labels with multiple branches to them are likely loop headers
    loop_headers = {label for label, count in branch_targets.items() if count > 0}
    print(f"Identified potential loop headers: {loop_headers}")

    # Main processing pass
    for line in enumerate(ptx_code):
        stripped_line = line[1].strip()

        if stripped_line.startswith(".visible .entry") or stripped_line.startswith(".func"):
            current_func = stripped_line.split()[-1]
            func_hash = hash(current_func) & 0xFFFFFFFFFFFFFFFF
            print(f"\nProcessing function: {current_func} (hash: {func_hash:016x})")
            
            # Check if current function should be skipped
            skip_kernel = current_func in ignore_patterns
            if skip_kernel:
                print(f"Skipping kernel: {current_func[:-1]}")
                    
            inside_kernel_function = True
            r_signature_inserted = False
            bb_idx = 0
            inst_idx = 0

        if inside_kernel_function and skip_kernel:
            modified_code.append(line[1])
            if stripped_line == "}":
                inside_kernel_function = False
            continue

        if inside_kernel_function and stripped_line == "{" and not r_signature_inserted:
            modified_code.append(line[1])
            modified_code.append("\t.reg .u64 \tr_signature;\n")
            modified_code.append("\t.reg .u64 \tr_fuelusage;\n")
            modified_code.append("\t.reg .u64 \tr_fuel_backup;\n")
            modified_code.append("\t.reg .u64 \tr_fuel_addr;\n")
            modified_code.append("\t.reg .u64 \tr_temp_fuel;\n")
            modified_code.append("\t.reg .u64 \tr_sig_addr;\n")
            modified_code.append("\t.reg .pred \tp_fuel;\n")
            modified_code.append("\tmov.u64 \tr_signature, 0x1111111111111111;\n")
            modified_code.append("\tmov.u64 \tr_fuelusage, 0;\n")
            modified_code.append("\tmov.u64 \tr_temp_fuel, 0;\n")
            modified_code.append("\tmov.u64 \tr_sig_addr, gbl_SIGNATURE;\n")
            modified_code.append("\tmov.u64 \tr_fuel_addr, gbl_FUELUSAGE;\n")
            r_signature_inserted = True
            continue

        # Handle branch instructions
        if inside_kernel_function and not skip_kernel:
            is_branch = ("bra" in stripped_line and not stripped_line.startswith("//")) or \
                       (stripped_line.startswith("@") and "bra" in stripped_line)
            is_return = stripped_line == "ret;"
            
            if (is_branch or is_return):
                if current_block_signature != 0 or current_block_fuel != 0:
                    left_rot, right_rot = get_position_modifier(bb_idx, inst_idx, func_hash)
                    rotated_sig = rotate_left(rotate_right(current_block_signature, right_rot), left_rot)
                    modified_code.append(f"\txor.b64 \tr_signature, r_signature, 0x{rotated_sig:016x};\n")
                    modified_code.append(f"\tadd.u64 \tr_fuelusage, r_fuelusage, {current_block_fuel};\n")
                
                if is_return:
                    modified_code.append("\tmov.u64 \tr_fuel_backup, r_fuelusage;\n")
                    modified_code.append("\tatom.global.add.u64 \tr_temp_fuel, [r_fuel_addr], r_fuelusage;\n")
                    modified_code.append("\tadd.u64 \tr_temp_fuel, r_temp_fuel, r_fuel_backup;\n")
                    modified_code.append("\tmov.u64 \tr_fuelusage, 0;\n")
                    modified_code.append("\tsetp.gt.u64 p_fuel, r_temp_fuel, 0xdeadbeefdeadbeef;\n")
                    modified_code.append("\t@p_fuel bra $FUEL_EXCEEDED;\n")
                    modified_code.append("\tbra $NORMAL_EXIT;\n")

                    modified_code.append("$FUEL_EXCEEDED:\n")
                    modified_code.append("\tmov.u64 \tr_temp_fuel, 1;\n")
                    modified_code.append("\tst.global.u64 \t[gbl_ERRORSTAT], r_temp_fuel;\n")

                    modified_code.append("$NORMAL_EXIT:\n")
                    modified_code.append("\tatom.global.xor.b64 \tr_sig_addr, [r_sig_addr], r_signature;\n")
                    modified_code.append("\tatom.global.add.u64 \tr_fuel_addr, [r_fuel_addr], r_fuelusage;\n")
                    modified_code.append("\tret;\n")
                    continue
                
                modified_code.append(line[1])
                current_block_signature = 0
                current_block_fuel = 0
                bb_idx += 1
                inst_idx = 0
                continue

        if stripped_line.startswith("$L") or stripped_line.startswith("BB"):
            if inside_kernel_function and not skip_kernel:
                if current_block_signature != 0 or current_block_fuel != 0:
                    left_rot, right_rot = get_position_modifier(bb_idx, inst_idx, func_hash)
                    rotated_sig = rotate_left(rotate_right(current_block_signature, right_rot), left_rot)
                    print(f"BB{bb_idx}: Signature: {current_block_signature:016x} -> {rotated_sig:016x} (rotl: {left_rot}, rotr: {right_rot})")
                    print(f"BB{bb_idx}: Fuel: {current_block_fuel}")
                    modified_code.append(f"\txor.b64 \tr_signature, r_signature, 0x{rotated_sig:016x};\n")
                    modified_code.append(f"\tadd.u64 \tr_fuelusage, r_fuelusage, {current_block_fuel};\n")
                current_block_signature = 0
                current_block_fuel = 0
                bb_idx += 1
                inst_idx = 0

        if inside_kernel_function and not skip_kernel:
            if "trap;" in stripped_line:
                modified_code.append("\tmov.u64 \tr_fuel_backup, r_fuelusage;\n")
                modified_code.append("\tatom.global.add.u64 \tr_temp_fuel, [r_fuel_addr], r_fuelusage;\n")
                modified_code.append("\tadd.u64 \tr_temp_fuel, r_temp_fuel, r_fuel_backup;\n")
                modified_code.append("\tmov.u64 \tr_fuelusage, 0;\n")
                modified_code.append("\tsetp.gt.u64 p_fuel, r_temp_fuel, 0xdeadbeefdeadbeef;\n")
                modified_code.append("\t@p_fuel bra $FUEL_EXCEEDED;\n")
                continue

            for instr, (prime, fuel_cost) in instruction_prime_map.items():
                if stripped_line.lstrip().startswith(f"{instr} "):
                    # XOR with rotation to prevent nullification
                    left_rot, right_rot = get_position_modifier(bb_idx, inst_idx, func_hash)
                    rotated_prime = rotate_left(rotate_right(prime, right_rot), left_rot)
                    current_block_signature ^= rotated_prime
                    current_block_fuel += fuel_cost
                    inst_idx += 1
                    print(f"Found instruction {instr} in BB{bb_idx} (idx: {inst_idx}, cost: {fuel_cost}, prime: {prime:016x} -> {rotated_prime:016x}, rotl: {left_rot}, rotr: {right_rot})")
                    break

        if inside_kernel_function and stripped_line == "}":
            if not skip_kernel:
                if current_block_signature != 0 or current_block_fuel != 0:
                    left_rot, right_rot = get_position_modifier(bb_idx, inst_idx, func_hash)
                    rotated_sig = rotate_left(rotate_right(current_block_signature, right_rot), left_rot)
                    modified_code.append(f"\txor.b64 \tr_signature, r_signature, 0x{rotated_sig:016x};\n")
                    modified_code.append(f"\tadd.u64 \tr_fuelusage, r_fuelusage, {current_block_fuel};\n")
                    
            modified_code.append(line[1])
            inside_kernel_function = False
            continue

        modified_code.append(line[1])

    return modified_code

def main():
    parser = argparse.ArgumentParser(description='Compile PTX with injected runtime signature')
    parser.add_argument('challenge', help='Challenge name')
    parser.add_argument('algorithm', help='Algorithm name')
    
    args = parser.parse_args()

    print(f"Compiling .ptx for {args.challenge}/{args.algorithm}")


    framework_cu = "tig-binary/src/framework.cu"
    if not os.path.exists(framework_cu):
        raise FileNotFoundError(
            f"Framework code does not exist @ '{framework_cu}'. This script must be run from the root of tig-monorepo"
        )

    challenge_cu = f"tig-challenges/src/{args.challenge}.cu"
    if not os.path.exists(challenge_cu):
        raise (
            f"Challenge code does not exist @ '{challenge_cu}'. Is the challenge name correct?"
        )

    algorithm_cu = f"tig-algorithms/src/{args.challenge}/{args.algorithm}.cu"
    algorithm_cu2 = f"tig-algorithms/src/{args.challenge}/{args.algorithm}/benchmarker_outbound.cu"
    if not os.path.exists(algorithm_cu) and not os.path.exists(algorithm_cu2):
        raise FileNotFoundError(
            f"Algorithm code does not exist @ '{algorithm_cu}' or '{algorithm_cu2}'. Is the algorithm name correct?"
        )
    if not os.path.exists(algorithm_cu):
        algorithm_cu = algorithm_cu2

    # Combine .cu source files into a temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_cu = os.path.join(temp_dir, "temp.cu")
        temp_ptx = os.path.join(temp_dir, "temp.ptx")
        
        with open(framework_cu, 'r') as f:
            code = f.read() + "\n"
        with open(challenge_cu, 'r') as f:
            code += f.read() + "\n"
        func_regex = r'(?:extern\s+"C"\s+__global__|__device__)\s+\w+\s+(?P<func>\w+)\s*\('
        funcs_to_ignore = [match.group('func') for match in re.finditer(func_regex, code)]
        with open(algorithm_cu, 'r') as f:
            code += f.read()
        with open(temp_cu, 'w') as f:
            f.write(code)

        # Compile the temporary .cu file into a .ptx file using nvcc
        nvcc_command = [
            "nvcc", "-ptx", temp_cu, "-o", temp_ptx,
            "-arch", "compute_70",
            "-code", "sm_70",
            "--use_fast_math",
            "-dopt=on"
        ]

        print(f"Running nvcc command: {' '.join(nvcc_command)}")
        subprocess.run(nvcc_command, check=True)
        print(f"Successfully compiled")

        print("Adding runtime signature opcodes")
        with open(temp_ptx, 'r') as f:
            ptx_code = f.readlines()
        modified_ptx_code = add_xor_commands(
            ptx_code, 
            instruction_prime_map, 
            set(f"{x}(" for x in funcs_to_ignore)
        )
        
        output_path = f"tig-algorithms/ptx/{args.challenge}/{args.algorithm}.ptx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.writelines(modified_ptx_code)
        print(f"Wrote ptx to {output_path}")
        print(f"Done")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)