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
#[derive(Debug, Default)]
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
        
        let mut snapshot = RegisterSnapshot::default();

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

                // Save predicates
                /*"str p0, [{base}, #{predicates_offset}]",
                "str p1, [{base}, #{predicates_offset} + 16]",
                "str p2, [{base}, #{predicates_offset} + 32]",
                "str p3, [{base}, #{predicates_offset} + 48]",
                "str p4, [{base}, #{predicates_offset} + 64]",
                "str p5, [{base}, #{predicates_offset} + 80]",
                "str p6, [{base}, #{predicates_offset} + 96]",
                "str p7, [{base}, #{predicates_offset} + 112]",
                "str p8, [{base}, #{predicates_offset} + 128]",
                "str p9, [{base}, #{predicates_offset} + 144]",
                "str p10, [{base}, #{predicates_offset} + 160]",
                "str p11, [{base}, #{predicates_offset} + 176]",
                "str p12, [{base}, #{predicates_offset} + 192]",
                "str p13, [{base}, #{predicates_offset} + 208]",
                "str p14, [{base}, #{predicates_offset} + 224]",
                "str p15, [{base}, #{predicates_offset} + 240]",*/

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
                //predicates_offset = const offset_of!(RegisterSnapshot, predicates),
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
                delta.changed_regs.push(Registers::X(i as u8, new.registers.gprs[i]));
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
                delta.changed_regs.push(Registers::V(i as u8, new.registers.vregs[i]));
            }
        }

        for i in 0..16 {
            if old.registers.predicates[i] != new.registers.predicates[i] {
                delta.changed_regs.push(Registers::P(i as u8, new.registers.predicates[i]));
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

    pub fn generate_restore_chunk(&self) -> Vec<u8> {
        let mut code = Vec::with_capacity(128 * 4);

        for x in self.changed_regs.iter() {
            match x {
                Registers::X(x, value) => {
                    if *value == 0 {
                        code.extend_from_slice(&((0xAA1F03E0 | (*x as u32)).to_le_bytes()));  // mov x<n>, xzr
                    } else if *value <= 0xFFFF {
                        // movz x{x}, #{value}
                        let instr = 0xD2800000 | ((*value as u32 & 0xFFFF) << 5) | (*x as u32);
                        code.extend_from_slice(&instr.to_le_bytes());
                    } else if *value <= 0xFFFFFFFF && (*value & 0xFFFF) == 0 {
                        // movz x{x}, #{value >> 16}, lsl #16 (clears lower 16 bits)
                        let instr = 0xD2A00000 | (((*value >> 16) as u32 & 0xFFFF) << 5) | (*x as u32);
                        code.extend_from_slice(&instr.to_le_bytes());
                    } else if *value <= 0xFFFFFFFF && (*value & 0xFFFF0000) == 0 {
                        // movz x{x}, #{value >> 32}, lsl #32 (clears lower 32 bits)
                        let instr = 0xD2C00000 | (((*value >> 32) as u32 & 0xFFFF) << 5) | (*x as u32);
                        code.extend_from_slice(&instr.to_le_bytes());
                    } else {
                        // For complex values, use movz + movk sequence
                        // Always start with movz to clear the register
                        let instr1 = 0xD2800000 | ((*value as u32 & 0xFFFF) << 5) | (*x as u32);
                        code.extend_from_slice(&instr1.to_le_bytes());
                        
                        if (*value >> 16) & 0xFFFF != 0 {
                            // movk x{x}, #{(value >> 16) & 0xFFFF}, lsl #16
                            let instr2 = 0xF2A00000 | ((((*value >> 16) as u32) & 0xFFFF) << 5) | (*x as u32);
                            code.extend_from_slice(&instr2.to_le_bytes());
                        }
                        
                        if (*value >> 32) & 0xFFFF != 0 {
                            // movk x{x}, #{(value >> 32) & 0xFFFF}, lsl #32
                            let instr3 = 0xF2C00000 | ((((*value >> 32) as u32) & 0xFFFF) << 5) | (*x as u32);
                            code.extend_from_slice(&instr3.to_le_bytes());
                        }
                        
                        if (*value >> 48) & 0xFFFF != 0 {
                            // movk x{x}, #{(value >> 48) & 0xFFFF}, lsl #48
                            let instr4 = 0xF2E00000 | ((((*value >> 48) as u32) & 0xFFFF) << 5) | (*x as u32);
                            code.extend_from_slice(&instr4.to_le_bytes());
                        }
                    }
                }
            }
        }

        code
    }
}

#[cfg(target_arch = "aarch64")]
#[repr(u8)]
#[derive(Debug)]
pub enum Registers {
    X(u8, u64) = 0, // x<n>, value, x0-x30
    SP(u64) = 31, // sp
    LR(u64) = 32,
    PC(u64) = 33,
    NZCV(u64) = 34,
    FPCR(u64) = 35,
    FPSR(u64) = 36,
    TPIDR_EL0(u64) = 37,
    TPIDRRO_EL0(u64) = 38,
    CNTVCT_EL0(u64) = 39,
    CNTFRQ_EL0(u64) = 40,
    V(u8, u128) = 41, // v<n>, value, v0-v31
    P(u8, u128) = 73, // p<n>, value, p0-p15
    FFR(u128) = 89,
    VG(u32) = 90,
}