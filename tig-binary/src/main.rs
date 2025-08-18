#![feature(naked_functions)]

extern crate tig_algorithms;
extern crate tig_challenges;

#[cfg(not(feature = "entry_point"))]
fn main() {
    eprintln!("This binary has not been compiled with tig-binary/scripts/build_binary. Please use the build_binary script to compile it.");
    std::process::exit(1);
}

#[cfg(feature = "entry_point")]
use std::sync::atomic::{AtomicI64, AtomicU64};

#[cfg(feature = "entry_point")]
use tig_structs::core::{BenchmarkSettings, CPUArchitecture, OutputData, Solution};
#[cfg(feature = "entry_point")]
use getargs::{Options, Opt};

#[cfg(feature = "entry_point")]
mod entry_point;
#[cfg(feature = "entry_point")]
use entry_point::{Challenge};

#[cfg(feature = "entry_point")]
#[inline(never)]
#[no_mangle]
unsafe fn __switch_stack_and_call(
    stack_top_ptr: *mut u8,
    func_to_call: *const core::ffi::c_void,
    arg: *const core::ffi::c_void,
) {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::asm!(
            "mov r12, rsp", // backup old stack pointer

            "mov rsp, {stack_top}", // switch to new stack
            "and rsp, -16", // align stack to 16 bytes
            
            "mov rdi, {arg}",
            "call {func}",
            
            "mov rsp, r12", // restore original stack
            
            stack_top = in(reg) stack_top_ptr,
            func = in(reg) func_to_call,
            arg = in(reg) arg,
            clobber_abi("C"),
        );
    }

    // clear all registers to ensure deterministic execution
    #[cfg(target_arch = "aarch64")]
    {
        crate::clear_registers!();

        std::arch::asm!(
            "mov x19, sp", // backup original stack pointer
            
            "mov x10, x2", // x2 already contains stack_top_ptr
            
            "bic x10, x10, #15", // clear lowest 4 bits to ensure alignment
            "mov sp, x10", // move aligned value to sp
            
            // x0 already contains arg, x1 already contains func_to_call
            "blr x1",
            
            "mov sp, x19", // restore original stack
            
            in("x0") arg,
            in("x1") func_to_call,
            in("x2") stack_top_ptr,
            out("x10") _,
            clobber_abi("C"),
        );
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        compile_error!("Unsupported architecture for stack switching");
    }
}

#[cfg(feature = "entry_point")]
fn alloc_stack(
    base_address: *mut u8,
    size: usize,
) -> *mut u8 {
    unsafe {
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE | libc::MAP_GROWSDOWN;
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        
        let result = libc::mmap(
            base_address as *mut libc::c_void,
            size,
            prot,
            flags,
            -1,
            0,
        );
        
        if result == libc::MAP_FAILED {
            panic!("Failed to allocate stack memory at address {:#x}: {}", base_address as usize, std::io::Error::last_os_error());
        }
        
        (result as *mut u8).wrapping_add(size)
    }
}

#[cfg(feature = "entry_point")]
fn load_settings(settings: &str) -> BenchmarkSettings {
    let settings = if settings.ends_with(".json") {
        std::fs::read_to_string(settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        })
    } else {
        settings.to_string()
    };

    tig_utils::dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    })
}

#[cfg(feature = "entry_point")]
#[no_mangle]
pub static __max_fuel: AtomicI64 = AtomicI64::new(i64::MAX);

#[cfg(feature = "entry_point")]
#[no_mangle]
pub static __nonce: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "entry_point")]
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut opts = Options::new(args.iter().map(String::as_str));

    let mut settings = None;
    let mut rand_hash = None;
    let mut nonce = None;
    let mut ptx = None;
    let mut fuel = None;
    let mut output = None;
    let mut compress = None;
    let mut gpu = None;

    while let Some(opt) = opts.next_opt().expect("Argument parsing error") {
        match opt {
            Opt::Short('p') | Opt::Long("ptx") => {
                ptx = Some(opts.value());
            }

            Opt::Short('f') | Opt::Long("fuel") => {
                fuel = Some(opts.value());
            }

            Opt::Short('o') | Opt::Long("output") => {
                output = Some(opts.value());
            }

            Opt::Short('c') | Opt::Long("compress") => {
                compress = Some(opts.value());
            }

            Opt::Short('g') | Opt::Long("gpu") => {
                gpu = Some(opts.value());
            }

            _ => {
                eprintln!("Unexpected option: {:?}", opt);
                std::process::exit(1);
            }
        }       
    }

    for (i, arg) in opts.positionals().enumerate() {
        match i {
            0 => settings = Some(arg.to_string()),
            1 => rand_hash = Some(arg.to_string()),
            2 => nonce = Some(arg.parse().expect("Nonce parsing error")),
            _ => {
                eprintln!("Unexpected positional argument: {}", arg);
                std::process::exit(1);
            }
        }
    }

    let settings = load_settings(&settings.expect("Settings not provided"));
    let nonce = nonce.expect("Nonce not provided");
    let seed = settings.calc_seed(&rand_hash.expect("Rand hash not provided"), nonce);

    let solve_addr = solve as *const core::ffi::c_void;
    let stack_top = alloc_stack(0x30000000000 as *mut u8, 0x100000); // 1mb stack

    let challenge = Box::new(Challenge::generate_instance(&seed, &settings.difficulty.into()).expect("Failed to generate challenge"));
    let ptr_to_challenge = Box::into_raw(challenge) as *const _ as *const core::ffi::c_void;
    
    __nonce.store(nonce, std::sync::atomic::Ordering::Relaxed);
    __runtime_signature.store(u64::from_le_bytes(seed[0..8].try_into().unwrap()), std::sync::atomic::Ordering::Relaxed);
    let max_fuel = match fuel {
        Some(fuel) => match fuel.expect("Fuel not provided").parse::<i64>() {
            Ok(fuel) => fuel,
            Err(_) => {
                eprintln!("Invalid fuel value");
                std::process::exit(1);
            }
        },
        None => i64::MAX,
    };
    
    _flush_tls(); 
    {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        __max_fuel.store(max_fuel, std::sync::atomic::Ordering::Relaxed);
        __fuel_remaining.store(max_fuel, std::sync::atomic::Ordering::Relaxed);
        //__max_allowed_memory_usage.store(settings.max_memory_usage as u64, std::sync::atomic::Ordering::Relaxed);
    }

    unsafe {
        __switch_stack_and_call(
            stack_top,
            solve_addr,
            ptr_to_challenge,
        );
    }
}

#[cfg(feature = "entry_point")]
#[inline(never)]
extern "C" fn _flush_tls() {
    std::hint::black_box(());
}

#[cfg(feature = "entry_point")]
#[no_mangle]
#[inline(never)]
fn __copy_to_restore_region(restore_chunk: &[u8]) -> *mut u8 {
    unsafe {
        let restore_arena = __restore_region.load(std::sync::atomic::Ordering::Relaxed);
        let restore_arena_ptr = restore_arena as *mut u8;
        std::ptr::copy_nonoverlapping(restore_chunk.as_ptr(), restore_arena_ptr, restore_chunk.len());
        restore_arena_ptr
    }
}

#[cfg(feature = "entry_point")]
extern "C" fn solve(ptr_to_challenge: *const core::ffi::c_void) {
    let (mut sp1, mut sp2, mut sp3): (u64, u64, u64);
    unsafe {
        std::arch::asm!("mov x0, sp", out("x0") sp1);
    }

    let snapshot = snapshot::Snapshot::new();

    unsafe {
        std::arch::asm!("mov x0, sp", out("x0") sp2);
    }

    //let snapshot2 = snapshot::Snapshot::new();

    unsafe {
        std::arch::asm!("mov x0, sp", out("x0") sp3);
    }
    println!("SP1: {:?}, SP2: {:?}, SP3: {:?}", sp1, sp2, sp3);

    //let delta = snapshot::DeltaSnapshot::delta_from(&snapshot, &snapshot2);
    //println!("Delta: {:?}", delta);

    //let restore_chunk = delta.generate_restore_chunk();
    //let restore_region = __copy_to_restore_region(&restore_chunk);
    //println!("Restore region: {:?}, written: {}", restore_region, restore_chunk.len());

    //let stack_ptr: usize;
    let challenge_box = unsafe { Box::from_raw(ptr_to_challenge as *mut Challenge) };
    let challenge = &*challenge_box;
    
    let(solution, fuel_consumed, runtime_signature) = { 
        let result = crate::entry_point::entry_point(&challenge);
        if let Err(e) = result {
            eprintln!("Runtime error: {}", e);
            std::process::exit(84);
        }

        let runtime_signature = __runtime_signature.load(std::sync::atomic::Ordering::Relaxed);
        let fuel_left = __fuel_remaining.load(std::sync::atomic::Ordering::Relaxed);
        __fuel_remaining.store(i64::MAX, std::sync::atomic::Ordering::Relaxed);

        let fuel_consumed = __max_fuel.load(std::sync::atomic::Ordering::Relaxed) - if fuel_left < 0 {
            0
        } else {
            fuel_left
        };

        (result, fuel_consumed, runtime_signature)
    };

    if fuel_consumed >= __max_fuel.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("Out of fuel");
        std::process::exit(87);
    }

    let (solution, invalid_reason) = match solution.unwrap() {
        Some(s) => match challenge.verify_solution(&s) {
            Ok(_) => (
                serde_json::to_value(&s)
                    .unwrap()
                    .as_object()
                    .unwrap()
                    .to_owned(),
                None,
            ),
            Err(e) => (Solution::new(), Some(e.to_string())),
        },
        None => (Solution::new(), None),
    };

    match invalid_reason {
        Some(e) => {
            eprintln!("Invalid solution: {}", e);
            std::process::exit(86);
        }
        None => {}
    }

    let output_data = OutputData {
        nonce: __nonce.load(std::sync::atomic::Ordering::Relaxed),
        runtime_signature,
        fuel_consumed: fuel_consumed as u64,
        solution,
        #[cfg(target_arch = "x86_64")]
        cpu_arch: CPUArchitecture::AMD64,
        #[cfg(target_arch = "aarch64")]
        cpu_arch: CPUArchitecture::ARM64,
    };

    println!("{}", tig_utils::jsonify(&output_data));

    if output_data.solution.len() == 0 {
        eprintln!("No solution found");
        std::process::exit(85);
    }

    std::process::exit(0);
}

#[cfg(feature = "entry_point")]
#[no_mangle]
static __runtime_signature: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __fuel_remaining: AtomicI64 = AtomicI64::new(i64::MAX);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __max_allowed_memory_usage: AtomicU64 = AtomicU64::new(u64::MAX);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __max_memory_usage: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __total_memory_usage: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __curr_memory_usage: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "entry_point")]
#[no_mangle]
static __restore_region: AtomicU64 = AtomicU64::new(0x60000000000);

#[cfg(feature = "entry_point")]
mod snapshot;