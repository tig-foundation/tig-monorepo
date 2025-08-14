use {
    std::sync::atomic::{AtomicU64, Ordering},
    std::sync::Arc,
    crate::{__total_memory_usage, __max_memory_usage, __max_allowed_memory_usage, __curr_memory_usage},
};

#[repr(C)]
#[derive(Debug)]
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
            //cntvct_el0: 0,
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
                //cntvct_el0_offset = const offset_of!(RegisterSnapshot, cntvct_el0),
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

#[repr(C)]
#[derive(Debug)]
pub struct DeltaSnapshot {
    pub changed_regs: Vec<Registers>
}

impl DeltaSnapshot {
    pub fn delta_from(old: &Snapshot, new: &Snapshot) -> Self {
        let mut delta = DeltaSnapshot {
            changed_regs: Vec::new(),
        };

        for i in 0..31 {
            if old.registers.gprs[i] != new.registers.gprs[i] {
                delta.changed_regs.push(Registers::X(i));
            }
        }

        if old.registers.sp != new.registers.sp {
            delta.changed_regs.push(Registers::SP(new.registers.sp));
        }

        if old.registers.lr != new.registers.lr {
            delta.changed_regs.push(Registers::LR(new.registers.lr));
        }

        if old.registers.pc != new.registers.pc {
            delta.changed_regs.push(Registers::PC(new.registers.pc));
        }

        if old.registers.nzcv != new.registers.nzcv {
            delta.changed_regs.push(Registers::NZCV(new.registers.nzcv));
        }

        if old.registers.fpcr != new.registers.fpcr {
            delta.changed_regs.push(Registers::FPCR(new.registers.fpcr));
        }

        if old.registers.fpsr != new.registers.fpsr {
            delta.changed_regs.push(Registers::FPSR(new.registers.fpsr));
        }

        if old.registers.tpidr_el0 != new.registers.tpidr_el0 {
            delta.changed_regs.push(Registers::TPIDR_EL0(new.registers.tpidr_el0));
        }

        if old.registers.tpidrro_el0 != new.registers.tpidrro_el0 {
            delta.changed_regs.push(Registers::TPIDRRO_EL0(new.registers.tpidrro_el0));
        }

        if old.registers.cntfrq_el0 != new.registers.cntfrq_el0 {
            delta.changed_regs.push(Registers::CNTFRQ_EL0(new.registers.cntfrq_el0));
        }

        for i in 0..32 {
            if old.registers.vregs[i] != new.registers.vregs[i] {
                delta.changed_regs.push(Registers::V(i));
            }
        }

        for i in 0..16 {
            if old.registers.predicates[i] != new.registers.predicates[i] {
                delta.changed_regs.push(Registers::P(i));
            }
        }

        if old.registers.ffr != new.registers.ffr {
            delta.changed_regs.push(Registers::FFR(new.registers.ffr));
        }

        if old.registers.vg != new.registers.vg {
            delta.changed_regs.push(Registers::VG(new.registers.vg));
        }

        delta
    }
}

#[cfg(target_arch = "aarch64")]
#[repr(u8)]
#[derive(Debug)]
pub enum Registers {
    X0(u64) = 0,
    X1(u64) = 1,
    X2(u64) = 2,
    X3(u64) = 3,
    X4(u64) = 4,
    X5(u64) = 5,
    X6(u64) = 6,
    X7(u64) = 7,
    X8(u64) = 8,
    X9(u64) = 9,
    X10(u64) = 10,
    X11(u64) = 11,
    X12(u64) = 12,
    X13(u64) = 13,
    X14(u64) = 14,
    X15(u64) = 15,
    X16(u64) = 16,
    X17(u64) = 17,
    X18(u64) = 18,
    X19(u64) = 19,
    X20(u64) = 20,
    X21(u64) = 21,
    X22(u64) = 22,
    X23(u64) = 23,
    X24(u64) = 24,
    X25(u64) = 25,
    X26(u64) = 26,
    X27(u64) = 27,
    X28(u64) = 28,
    X29(u64) = 29,
    X30(u64) = 30,
    SP(u64) = 31,
    LR(u64) = 32,
    PC(u64) = 33,
    NZCV(u64) = 34,
    FPCR(u64) = 35,
    FPSR(u64) = 36,
    TPIDR_EL0(u64) = 37,
    TPIDRRO_EL0(u64) = 38,
    CNTVCT_EL0(u64) = 39,
    CNTFRQ_EL0(u64) = 40,
    V0(u128) = 41,
    V1(u128) = 42,
    V2(u128) = 43,
    V3(u128) = 44,
    V4(u128) = 45,
    V5(u128) = 46,
    V6(u128) = 47,
    V7(u128) = 48,
    V8(u128) = 49,
    V9(u128) = 50,
    V10(u128) = 51,
    V11(u128) = 52,
    V12(u128) = 53,
    V13(u128) = 54,
    V14(u128) = 55,
    V15(u128) = 56,
    V16(u128) = 57,
    V17(u128) = 58,
    V18(u128) = 59,
    V19(u128) = 60,
    V20(u128) = 61,
    V21(u128) = 62,
    V22(u128) = 63,
    V23(u128) = 64,
    V24(u128) = 65,
    V25(u128) = 66,
    V26(u128) = 67,
    V27(u128) = 68,
    V28(u128) = 69,
    V29(u128) = 70,
    V30(u128) = 71,
    V31(u128) = 72,
    P0(u128) = 73,
    P1(u128) = 74,
    P2(u128) = 75,
    P3(u128) = 76,
    P4(u128) = 77,
    P5(u128) = 78,
    P6(u128) = 79,
    P7(u128) = 80,
    P8(u128) = 81,
    P9(u128) = 82,
    P10(u128) = 83,
    P11(u128) = 84,
    P12(u128) = 85,
    P13(u128) = 86,
    P14(u128) = 87,
    P15(u128) = 88,
    FFR(u128) = 89,
    VG(u32) = 90,
}