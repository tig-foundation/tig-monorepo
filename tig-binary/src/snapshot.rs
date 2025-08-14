use {
    std::sync::atomic::{AtomicU64, Ordering},
    std::sync::Arc,
    crate::{__total_memory_usage, __max_memory_usage, __max_allowed_memory_usage, __curr_memory_usage},
};

#[cfg(feature = "entry_point")]
#[repr(C)]
pub struct Snapshot {
    pub total_memory_usage: u64,
    pub max_memory_usage: u64,
    pub max_allowed_memory_usage: u64,
    pub curr_memory_usage: u64,
    pub registers: RegisterSnapshot,
}

#[cfg(target_arch = "aarch64")]
#[repr(C)]
#[derive(Debug)]
pub struct RegisterSnapshot {
    pub gprs: [u64; 31], // x0-x30

    pub sp: u64,
    pub lr: u64,
    pub pc: u64,
    pub nzcv: u64,

    pub fpcr: u64,
    pub fpsr: u64,

    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    //pub cntvct_el0: u64,
    pub cntfrq_el0: u64,

    pub vregs: [u128; 32], // v0-v31
    pub predicates: [u128; 16],
    pub ffr: u128,
    pub vg: u32
}

#[cfg(target_arch = "aarch64")]
impl RegisterSnapshot {
    pub fn snap() -> Self {
        use std::mem::offset_of;
        
        let mut snapshot = RegisterSnapshot {
            gprs: [0; 31],
            sp: 0,
            lr: 0,
            pc: 0,
            nzcv: 0,
            fpcr: 0,
            fpsr: 0,
            tpidr_el0: 0,
            tpidrro_el0: 0,
            cntvct_el0: 0,
            cntfrq_el0: 0,
            vregs: [0; 32],
            predicates: [0; 16],
            ffr: 0,
            vg: 0,
        };

        let base_ptr = &mut snapshot as *mut RegisterSnapshot as *mut u8;

        unsafe {
            std::arch::asm!(
                // Save GPRs x0-x30 using calculated offsets
                "stp x0, x1, [{base}, #{gprs_offset}]",
                "stp x2, x3, [{base}, #{gprs_offset} + 16]",
                "stp x4, x5, [{base}, #{gprs_offset} + 32]",
                "stp x6, x7, [{base}, #{gprs_offset} + 48]",
                "stp x8, x9, [{base}, #{gprs_offset} + 64]",
                "stp x10, x11, [{base}, #{gprs_offset} + 80]",
                "stp x12, x13, [{base}, #{gprs_offset} + 96]",
                "stp x14, x15, [{base}, #{gprs_offset} + 112]",
                "stp x16, x17, [{base}, #{gprs_offset} + 128]",
                "stp x18, x19, [{base}, #{gprs_offset} + 144]",
                "stp x20, x21, [{base}, #{gprs_offset} + 160]",
                "stp x22, x23, [{base}, #{gprs_offset} + 176]",
                "stp x24, x25, [{base}, #{gprs_offset} + 192]",
                "stp x26, x27, [{base}, #{gprs_offset} + 208]",
                "stp x28, x29, [{base}, #{gprs_offset} + 224]",
                "str x30, [{base}, #{gprs_offset} + 240]",

                // Save SP, LR, PC
                "mov x0, sp",
                "str x0, [{base}, #{sp_offset}]",
                "str x30, [{base}, #{lr_offset}]",
                "adr x0, 1f",
                "str x0, [{base}, #{pc_offset}]",
                "1:",

                // Save NZCV
                "mrs x0, nzcv",
                "str x0, [{base}, #{nzcv_offset}]",

                // Save floating-point control/status
                "mrs x0, fpcr",
                "str x0, [{base}, #{fpcr_offset}]",
                "mrs x0, fpsr",
                "str x0, [{base}, #{fpsr_offset}]",

                // Save thread pointers
                "mrs x0, tpidr_el0",
                "str x0, [{base}, #{tpidr_el0_offset}]",
                "mrs x0, tpidrro_el0",
                "str x0, [{base}, #{tpidrro_el0_offset}]",

                // Save timer registers
                //"mrs x0, cntvct_el0",
                //"str x0, [{base}, #{cntvct_el0_offset}]",
                "mrs x0, cntfrq_el0", 
                "str x0, [{base}, #{cntfrq_el0_offset}]",

                // Save vector registers v0-v31
                "stp q0, q1, [{base}, #{vregs_offset}]",
                "stp q2, q3, [{base}, #{vregs_offset} + 32]",
                "stp q4, q5, [{base}, #{vregs_offset} + 64]",
                "stp q6, q7, [{base}, #{vregs_offset} + 96]",
                "stp q8, q9, [{base}, #{vregs_offset} + 128]",
                "stp q10, q11, [{base}, #{vregs_offset} + 160]",
                "stp q12, q13, [{base}, #{vregs_offset} + 192]",
                "stp q14, q15, [{base}, #{vregs_offset} + 224]",
                "stp q16, q17, [{base}, #{vregs_offset} + 256]",
                "stp q18, q19, [{base}, #{vregs_offset} + 288]",
                "stp q20, q21, [{base}, #{vregs_offset} + 320]",
                "stp q22, q23, [{base}, #{vregs_offset} + 352]",
                "stp q24, q25, [{base}, #{vregs_offset} + 384]",
                "stp q26, q27, [{base}, #{vregs_offset} + 416]",
                "stp q28, q29, [{base}, #{vregs_offset} + 448]",
                "stp q30, q31, [{base}, #{vregs_offset} + 480]",

                base = in(reg) base_ptr,
                gprs_offset = const offset_of!(RegisterSnapshot, gprs),
                sp_offset = const offset_of!(RegisterSnapshot, sp),
                lr_offset = const offset_of!(RegisterSnapshot, lr),
                pc_offset = const offset_of!(RegisterSnapshot, pc),
                nzcv_offset = const offset_of!(RegisterSnapshot, nzcv),
                fpcr_offset = const offset_of!(RegisterSnapshot, fpcr),
                fpsr_offset = const offset_of!(RegisterSnapshot, fpsr),
                tpidr_el0_offset = const offset_of!(RegisterSnapshot, tpidr_el0),
                tpidrro_el0_offset = const offset_of!(RegisterSnapshot, tpidrro_el0),
                cntvct_el0_offset = const offset_of!(RegisterSnapshot, cntvct_el0),
                cntfrq_el0_offset = const offset_of!(RegisterSnapshot, cntfrq_el0),
                vregs_offset = const offset_of!(RegisterSnapshot, vregs),
                out("x0") _,
                options(nostack)
            );
        }

        snapshot
    }
}

#[cfg(target_arch = "x86_64")]
pub struct RegisterSnapshot {
    pub gprs: [u64; 16], // rax-r15 + rsp + rbp
}

impl Snapshot {
    pub fn new() -> Self {
        Self {
            total_memory_usage: 0,
            max_memory_usage: 0,
            max_allowed_memory_usage: 0,
            curr_memory_usage: 0,
            registers: RegisterSnapshot::snap(),
        }
    }
}