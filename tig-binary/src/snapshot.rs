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

#[cfg(target_arch = "aarch64")]
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
        let mut restore_chunk = Vec::with_capacity(128 * 4);

        for x in self.changed_regs.iter() {
            match x {
                Registers::X(x, value) => {
                    let (code, size) = self.mov_GPR_IMM64(*x, *value);
                    restore_chunk.extend_from_slice(&code[..size]);
                }
                _ => {}
            }
        }

        restore_chunk
    }
}

#[cfg(target_arch = "aarch64")]
impl DeltaSnapshot {
    /// Encodes the EOR X_dst, X_dst, X_src instruction.
    /// This is the AArch64 equivalent of xor reg_dst, reg_src.
    fn xor_GPR_GPR(&self, reg_dst: u8, reg_src: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];

        // EOR Xd, Xn, Xm instruction encoding for 64-bit registers.
        // Operation: Xd = Xn EOR Xm
        // To match `xor dst, src`, we use: EOR X_dst, X_dst, X_src
        // Rd = reg_dst, Rn = reg_dst, Rm = reg_src
        let instr: u32 = 0xCA000000 
                       | ((reg_src as u32 & 0x1F) << 16) 
                       | ((reg_dst as u32 & 0x1F) << 5) 
                       | (reg_dst as u32 & 0x1F);
        
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading an 8-bit immediate into a register using MOVZ.
    fn mov_GPR_IMM8(&self, reg: u8, value: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        
        // MOVZ Xd, #imm, LSL 0
        // The register is zero-extended to 64 bits.
        let instr: u32 = 0xD2800000 | ((value as u32) << 5) | (reg as u32 & 0x1F);
        
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading a 16-bit immediate into a register using MOVZ.
    fn mov_GPR_IMM16(&self, reg: u8, value: u16) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        
        // MOVZ Xd, #imm, LSL 0
        // The register is zero-extended to 64 bits.
        let instr: u32 = 0xD2800000 | ((value as u32) << 5) | (reg as u32 & 0x1F);

        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading a 32-bit immediate using a MOVZ/MOVK sequence.
    fn mov_GPR_IMM32(&self, reg: u8, value: u32) -> ([u8; 8], usize) {
        let mut code = [0; 8];
        let mut size = 0;
        let reg = reg as u32 & 0x1F;

        let chunk0 = (value & 0xFFFF) as u32;
        let chunk1 = ((value >> 16) & 0xFFFF) as u32;

        // Helper to write an instruction to the buffer 
        let mut write_instr = |instr: u32| {
            let bytes = instr.to_le_bytes();
            code[size..size + 4].copy_from_slice(&bytes);
            size += 4;
        };
        
        if chunk0 == 0 && chunk1 != 0 {
            // If lower 16 bits are zero, we can load the upper bits in one go.
            // MOVZ Xd, #chunk1, LSL 16
            write_instr(0xD2A00000 | (chunk1 << 5) | reg);
        } else {
            // Load lower 16 bits, zeroing the register.
            // MOVZ Xd, #chunk0, LSL 0
            write_instr(0xD2800000 | (chunk0 << 5) | reg);
            
            if chunk1 != 0 {
                // Keep the lower bits and insert the upper 16 bits.
                // MOVK Xd, #chunk1, LSL 16
                write_instr(0xF2A00000 | (chunk1 << 5) | reg);
            }
        }
        
        (code, size)
    }

    /// Encodes loading a 64-bit immediate using a MOVZ/MOVK sequence.
    fn mov_GPR_IMM64(&self, reg: u8, value: u64) -> ([u8; 16], usize) {
        let mut code = [0; 16];
        let mut size = 0;
        let reg = reg as u32 & 0x1F;

        let chunks = [
            (value & 0xFFFF) as u32,
            ((value >> 16) & 0xFFFF) as u32,
            ((value >> 32) & 0xFFFF) as u32,
            ((value >> 48) & 0xFFFF) as u32,
        ];

        // Helper to write an instruction to the buffer
        let mut write_instr = |instr: u32| {
            let bytes = instr.to_le_bytes();
            code[size..size + 4].copy_from_slice(&bytes);
            size += 4;
        };

        // Always start with MOVZ to clear the upper bits of the register.
        // MOVZ Xd, #chunk0, LSL 0
        write_instr(0xD2800000 | (chunks[0] << 5) | reg);

        // MOVK for any subsequent non-zero chunks
        if chunks[1] != 0 {
            // MOVK Xd, #chunk1, LSL 16
            write_instr(0xF2A00000 | (chunks[1] << 5) | reg);
        }
        if chunks[2] != 0 {
            // MOVK Xd, #chunk2, LSL 32
            write_instr(0xF2C00000 | (chunks[2] << 5) | reg);
        }
        if chunks[3] != 0 {
            // MOVK Xd, #chunk3, LSL 48
            write_instr(0xF2E00000 | (chunks[3] << 5) | reg);
        }

        (code, size)
    }
}

#[cfg(target_arch = "x86_64")]
impl DeltaSnapshot {
    fn xor_GPR_GPR(&self, reg_dst: u8, reg_src: u8) -> ([u8; 3], usize) {
        let mut code = [0; 3];
        let mut size = 0;
    
        // REX prefix is always needed for 64-bit operations.
        let mut rex = 0x48;  // Start with REX.W (64-bit)
        if reg_src >= 8 { rex |= 0x04; }  // R bit extends the ModR/M reg field
        if reg_dst >= 8 { rex |= 0x01; }  // B bit extends the ModR/M r/m field
        code[size] = rex;
        size += 1;
    
        // XOR opcode
        code[size] = 0x31;
        size += 1;
    
        // ModR/M byte
        let modrm = 0xC0 | ((reg_src & 0x7) << 3) | (reg_dst & 0x7);
        code[size] = modrm;
        size += 1;
    
        (code, size)
    }

    fn mov_GPR_IMM8(&self, reg: u8, value: u8) -> ([u8; 3], usize) {
        let mut code = [0; 3];
        let mut size = 0;
        
        if reg >= 8 {
            code[size] = 0x41;  // REX.B for extended registers
            size += 1;
        }

        // MOV r8, imm8 (0xB0 + reg)
        code[size] = 0xB0 + (reg & 0x7);
        code[size + 1] = value;
        size += 2;

        (code, size)
    }

    fn mov_GPR_IMM16(&self, reg: u8, value: u16) -> ([u8; 5], usize) {
        let mut code = [0; 5];
        let mut size = 0;
        
        // 16-bit operand size prefix
        code[size] = 0x66;
        size += 1;
        
        if reg >= 8 {
            code[size] = 0x41;  // REX.B for extended registers
            size += 1;
        }

        // MOV r16, imm16 (0xB8 + reg)
        code[size] = 0xB8 + (reg & 0x7);
        code[size + 1] = value as u8;
        code[size + 2] = (value >> 8) as u8;
        size += 3;

        (code, size)
    }

    fn mov_GPR_IMM32(&self, reg: u8, value: u32) -> ([u8; 6], usize) {
        let mut code = [0; 6];
        let mut size = 0;
    
        if reg >= 8 {
            code[size] = 0x41;  // REX.B for extended registers
            size += 1;
        }
        
        // MOV r32, imm32 (0xB8 + reg)
        code[size] = 0xB8 + (reg & 0x7);
        code[size + 1] = value as u8;
        code[size + 2] = (value >> 8) as u8;
        code[size + 3] = (value >> 16) as u8;
        code[size + 4] = (value >> 24) as u8;
        size += 5;

        (code, size)
    }

    fn mov_GPR_IMM64(&self, reg: u8, value: u64) -> ([u8; 10], usize) {
        let mut code = [0; 10];
        let mut size = 0;

        if reg >= 8 {
            code[size] = 0x49;  // REX.W (64-bit operation) for extended registers
            size += 1;
        } else {
            code[size] = 0x48;  // REX.W (64-bit operation)
            size += 1;
        }

        // MOV r64, imm64 (0xB8 + reg)
        code[size] = 0xB8 + (reg & 0x7);
        code[size + 1] = value as u8;
        code[size + 2] = (value >> 8) as u8;
        code[size + 3] = (value >> 16) as u8;
        code[size + 4] = (value >> 24) as u8;
        code[size + 5] = (value >> 32) as u8;
        code[size + 6] = (value >> 40) as u8;
        code[size + 7] = (value >> 48) as u8;
        code[size + 8] = (value >> 56) as u8;
        size += 9;

        (code, size)
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

#[no_mangle]
extern "C" fn __create_snapshot() -> Snapshot {
    Snapshot::new()
}

#[cfg(target_arch = "aarch64")]
#[macro_export]
macro_rules! clear_registers {
    () => {
        unsafe {
            std::arch::asm!(
                //"mov x0, xzr", ; arg0
                //"mov x1, xzr", ; func_to_call
                //"mov x2, xzr", ; stack_top 
                "mov x3, xzr",
                "mov x4, xzr",
                "mov x5, xzr",
                "mov x6, xzr",
                "mov x7, xzr",
                "mov x8, xzr",
                "mov x9, xzr",
                "mov x10, xzr",
                "mov x11, xzr",
                "mov x12, xzr",
                "mov x13, xzr",
                "mov x14, xzr",
                "mov x15, xzr",
                "mov x16, xzr",
                "mov x17, xzr",
                "mov x18, xzr",
                "mov x19, xzr",
                "mov x20, xzr",
                "mov x21, xzr",
                "mov x22, xzr",
                "mov x23, xzr",
                "mov x24, xzr",
                "mov x25, xzr",
                "mov x26, xzr",
                "mov x27, xzr",
                "mov x28, xzr",
                "mov x29, xzr",
                "mov x30, xzr",

                "movi v0.16b, #0",
                "movi v1.16b, #0",
                "movi v2.16b, #0",
                "movi v3.16b, #0",
                "movi v4.16b, #0",
                "movi v5.16b, #0",
                "movi v6.16b, #0",
                "movi v7.16b, #0",
                "movi v8.16b, #0",
                "movi v9.16b, #0",
                "movi v10.16b, #0",
                "movi v11.16b, #0",
                "movi v12.16b, #0",
                "movi v13.16b, #0",
                "movi v14.16b, #0",
                "movi v15.16b, #0",
                "movi v16.16b, #0",
                "movi v17.16b, #0",
                "movi v18.16b, #0",
                "movi v19.16b, #0",
                "movi v20.16b, #0",
                "movi v21.16b, #0",
                "movi v22.16b, #0",
                "movi v23.16b, #0",
                "movi v24.16b, #0",
                "movi v25.16b, #0",
                "movi v26.16b, #0",
                "movi v27.16b, #0",
                "movi v28.16b, #0",
                "movi v29.16b, #0",
                "movi v30.16b, #0",
                "movi v31.16b, #0",

                "msr fpcr, xzr",
                "msr fpsr, xzr",

                /*"pfalse p0.b",
                "pfalse p1.b", 
                "pfalse p2.b",
                "pfalse p3.b",
                "pfalse p4.b",
                "pfalse p5.b",
                "pfalse p6.b",
                "pfalse p7.b",
                "pfalse p8.b",
                "pfalse p9.b",
                "pfalse p10.b",
                "pfalse p11.b",
                "pfalse p12.b",
                "pfalse p13.b",
                "pfalse p14.b",
                "pfalse p15.b",

                "setffr",*/
            );
        }
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TlsEntry {
    pub name: *const std::ffi::c_char,
    pub address: *mut u8,
    pub size: usize,
}

impl TlsEntry {
    pub fn name(&self) -> &str {
        unsafe { std::ffi::CStr::from_ptr(self.name) }
    }

    pub fn address(&self) -> *mut u8 {
        self.address
    }

    pub fn read(&self) -> Vec<u8> {
        let mut value = vec![0; self.size];
        std::ptr::copy_nonoverlapping(self.address, value.as_mut_ptr(), self.size);
        value
    }

    pub fn write(&self, value: &[u8]) {
        std::ptr::copy_nonoverlapping(value.as_ptr(), self.address, self.size);
    }
}

extern "C" {
    static __tls_registry: *const TlsEntry;
    static __tls_registry_size: usize;
}