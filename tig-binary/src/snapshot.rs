#![feature(naked_functions)]

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
    pub nzcv: u64,

    pub fpcr: u64,
    pub fpsr: u64,

    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    //pub cntvct_el0: u64,
    //pub cntfrq_el0: u64,

    pub vregs: [u128; 32], // v0-v31
    //pub predicates: [u128; 16],
    //pub ffr: u128,
    //pub vg: u32
}

#[cfg(target_arch = "aarch64")]
impl RegisterSnapshot {
}

#[cfg(target_arch = "x86_64")]
pub struct RegisterSnapshot {
    pub gprs: [u64; 16], // rax-r15 + rsp + rbp
}

impl Snapshot {
    /// Captures ALL registers and stores in registry atomically
    /// Uses stack to preserve working registers
    #[naked]
    pub fn capture_pristine() -> *const Snapshot {
        unsafe {
            std::arch::naked_asm!(
                // Reserve stack space for 4 registers (32 bytes)
                "sub sp, sp, #32",
                
                // Save the 4 registers we need to use for work
                "stp x27, x28, [sp, #0]",
                "stp x29, x30, [sp, #16]",
                
                // Atomic increment of snapshot count and get index
                "adrp x30, {snapshot_count}",
                "add x30, x30, :lo12:{snapshot_count}",
                "1:",
                "ldaxr x29, [x30]",           // Load-acquire exclusive (current count)
                "add x28, x29, #1",           // Increment for new count
                "stlxr w27, x28, [x30]",      // Store-release exclusive (new count)
                "cbnz w27, 1b",               // Retry if failed
                
                // Now x29 contains the index we should use (old count)
                // Load registry pointer and calculate snapshot address
                "adrp x30, {registry_ptr}",
                "add x30, x30, :lo12:{registry_ptr}",
                "ldr x30, [x30]",             // Dereference to get actual array address
                
                // Calculate offset: registry + (index * sizeof(Snapshot))
                "mov x27, #{snapshot_size}",
                "mul x29, x29, x27",          // index * sizeof(Snapshot)
                "add x30, x30, x29",          // registry[index] address
                
                // Calculate offset to registers field within Snapshot
                "add x29, x30, #{registers_offset}",
                
                // Save ALL GPRs x0-x26 (pristine values)
                "stp x0, x1, [x29, #{gprs_offset}]",
                "stp x2, x3, [x29, #{gprs_offset} + 16]", 
                "stp x4, x5, [x29, #{gprs_offset} + 32]",
                "stp x6, x7, [x29, #{gprs_offset} + 48]",
                "stp x8, x9, [x29, #{gprs_offset} + 64]",
                "stp x10, x11, [x29, #{gprs_offset} + 80]",
                "stp x12, x13, [x29, #{gprs_offset} + 96]",
                "stp x14, x15, [x29, #{gprs_offset} + 112]",
                "stp x16, x17, [x29, #{gprs_offset} + 128]",
                "stp x18, x19, [x29, #{gprs_offset} + 144]",
                "stp x20, x21, [x29, #{gprs_offset} + 160]",
                "stp x22, x23, [x29, #{gprs_offset} + 176]",
                "stp x24, x25, [x29, #{gprs_offset} + 192]",
                "str x26, [x29, #{gprs_offset} + 208]",
                
                // Restore original x27, x28, x29, x30 from stack and save them
                "ldp x27, x28, [sp, #0]",
                "stp x27, x28, [x29, #{gprs_offset} + 216]",
                "ldp x27, x28, [sp, #16]",         // x27=orig_x29, x28=orig_x30
                "stp x27, x28, [x29, #{gprs_offset} + 232]",
                
                // Save SP (original stack pointer + 32 for our reserved space)
                "mov x28, sp",
                "add x28, x28, #32",
                "str x28, [x29, #{sp_offset}]",
                
                // Save NZCV (condition flags)
                "mrs x28, nzcv",
                "str x28, [x29, #{nzcv_offset}]",
                
                // Save floating-point control/status
                "mrs x28, fpcr",
                "str x28, [x29, #{fpcr_offset}]",
                "mrs x28, fpsr", 
                "str x28, [x29, #{fpsr_offset}]",
                
                // Save thread pointers
                "mrs x28, tpidr_el0",
                "str x28, [x29, #{tpidr_el0_offset}]",
                "mrs x28, tpidrro_el0",
                "str x28, [x29, #{tpidrro_el0_offset}]",
                
                // Save ALL vector registers v0-v31 (all pristine)
                "add x28, x29, #{vregs_offset}",
                "stp q0, q1, [x28, #0]",
                "stp q2, q3, [x28, #32]",
                "stp q4, q5, [x28, #64]",
                "stp q6, q7, [x28, #96]",
                "stp q8, q9, [x28, #128]",
                "stp q10, q11, [x28, #160]",
                "stp q12, q13, [x28, #192]",
                "stp q14, q15, [x28, #224]",
                "stp q16, q17, [x28, #256]",
                "stp q18, q19, [x28, #288]",
                "stp q20, q21, [x28, #320]",
                "stp q22, q23, [x28, #352]",
                "stp q24, q25, [x28, #384]",
                "stp q26, q27, [x28, #416]",
                "stp q28, q29, [x28, #448]",
                "stp q30, q31, [x28, #480]",
                
                // Save memory usage values from globals
                "adrp x28, {total_memory}",
                "add x28, x28, :lo12:{total_memory}",
                "ldr x27, [x28]",
                "str x27, [x30, #{total_memory_offset}]",
                
                "adrp x28, {max_memory}",
                "add x28, x28, :lo12:{max_memory}",
                "ldr x27, [x28]",
                "str x27, [x30, #{max_memory_offset}]",
                
                "adrp x28, {max_allowed_memory}",
                "add x28, x28, :lo12:{max_allowed_memory}",
                "ldr x27, [x28]",
                "str x27, [x30, #{max_allowed_memory_offset}]",
                
                "adrp x28, {curr_memory}",
                "add x28, x28, :lo12:{curr_memory}",
                "ldr x27, [x28]",
                "str x27, [x30, #{curr_memory_offset}]",
                
                // Store snapshot address in x0 before restoring working registers
                "mov x0, x30",
                
                // Restore all working registers from stack
                "ldp x27, x28, [sp, #0]",
                "ldp x29, x30, [sp, #16]",
                
                // Restore stack pointer
                "add sp, sp, #32",
                
                // Return (x0 already contains the snapshot address)
                "ret",
                
                registry_ptr = sym __snapshot_registry,
                snapshot_count = sym __snapshot_count,
                total_memory = sym __total_memory_usage,
                max_memory = sym __max_memory_usage,
                max_allowed_memory = sym __max_allowed_memory_usage,
                curr_memory = sym __curr_memory_usage,
                snapshot_size = const std::mem::size_of::<Snapshot>(),
                registers_offset = const std::mem::offset_of!(Snapshot, registers),
                total_memory_offset = const std::mem::offset_of!(Snapshot, total_memory_usage),
                max_memory_offset = const std::mem::offset_of!(Snapshot, max_memory_usage),
                max_allowed_memory_offset = const std::mem::offset_of!(Snapshot, max_allowed_memory_usage),
                curr_memory_offset = const std::mem::offset_of!(Snapshot, curr_memory_usage),
                gprs_offset = const std::mem::offset_of!(RegisterSnapshot, gprs),
                sp_offset = const std::mem::offset_of!(RegisterSnapshot, sp),
                nzcv_offset = const std::mem::offset_of!(RegisterSnapshot, nzcv),
                fpcr_offset = const std::mem::offset_of!(RegisterSnapshot, fpcr),
                fpsr_offset = const std::mem::offset_of!(RegisterSnapshot, fpsr),
                tpidr_el0_offset = const std::mem::offset_of!(RegisterSnapshot, tpidr_el0),
                tpidrro_el0_offset = const std::mem::offset_of!(RegisterSnapshot, tpidrro_el0),
                vregs_offset = const std::mem::offset_of!(RegisterSnapshot, vregs),
            );
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

        /*if old.registers.lr != new.registers.lr {
            delta.changed_regs.push(Registers::LR(new.registers.lr));
        }

        if old.registers.pc != new.registers.pc {
            delta.changed_regs.push(Registers::PC(new.registers.pc));
        }*/

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

        /*if old.registers.cntfrq_el0 != new.registers.cntfrq_el0 {
            delta.changed_regs.push(Registers::CNTFRQ_EL0(new.registers.cntfrq_el0));
        }*/

        for i in 0..32 {
            if old.registers.vregs[i] != new.registers.vregs[i] {
                delta.changed_regs.push(Registers::V(i as u8, new.registers.vregs[i]));
            }
        }

        /*for i in 0..16 {
            if old.registers.predicates[i] != new.registers.predicates[i] {
                delta.changed_regs.push(Registers::P(i as u8, new.registers.predicates[i]));
            }
        }

        if old.registers.ffr != new.registers.ffr {
            delta.changed_regs.push(Registers::FFR(new.registers.ffr));
        }

        if old.registers.vg != new.registers.vg {
            delta.changed_regs.push(Registers::VG(new.registers.vg));
        }*/

        delta
    }

    /*pub fn generate_restore_chunk(&self) -> Vec<u8> {
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
    }*/
}

#[cfg(target_arch = "aarch64")]
#[repr(u8)]
#[derive(Debug)]
pub enum Registers {
    X(u8, u64) = 0, // x<n>, value, x0-x30
    SP(u64) = 31, // sp
    NZCV(u64) = 32,
    FPCR(u64) = 33,
    FPSR(u64) = 34,
    TPIDR_EL0(u64) = 35,
    TPIDRRO_EL0(u64) = 36,
    V(u8, u128) = 37, // v<n>, value, v0-v31
    P(u8, u128) = 69, // p<n>, value, p0-p15
    FFR(u128) = 84,
    VG(u32) = 85
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
                "msr nzcv, xzr",

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
pub struct BasicBlockInfo {
    pub offset_in_code: u32,
    pub size: u16, // in bytes
    pub entity_changes_count: u16, // Number of entity changes
    pub entity_changes_offset: u32,
}

impl BasicBlockInfo {
    pub unsafe fn entity_changes(&self) -> &[EntityChange] {
        std::slice::from_raw_parts(
            __entity_changes_registry.add(self.entity_changes_offset as usize),
            self.entity_changes_count as usize
        )
    }
}

#[repr(u64)]
pub enum EntityChangeFlags {
    None = 0x0,
    Global = 0x1,
    ThreadLocal = 0x2,
    Stack = 0x4,
    Heap = 0x8,
    Register = 0x10,
    U8 = 0x20,
    U16 = 0x40,
    U32 = 0x80,
    U64 = 0x100,
    F32 = 0x200,
    F64 = 0x400,
    V64 = 0x800,
    V128 = 0x1000,
    V256 = 0x2000,
    V512 = 0x4000,
    Ptr = 0x8000,
    Bytes = 0x10000,
    // aarch64 exclusive
    MOVK = 0x20000,
    MOVZ = 0x40000,
    LSL16 = 0x80000,
    LSL32 = 0x100000,
    LSL48 = 0x200000,
    // rest of bits are used for tls offset, register, other metadata
}

const METADATA_SHIFT: u64 = 22;

const REGISTER_MASK: u64 = 0x3F << METADATA_SHIFT; // 6 bits

const TLS_OFFSET_BITS: u64 = 32;
const TLS_OFFSET_MASK: u64 = ((1 << TLS_OFFSET_BITS) - 1) << METADATA_SHIFT;

const GLOBAL_OFFSET_BITS: u64 = 32;
const GLOBAL_OFFSET_MASK: u64 = ((1 << GLOBAL_OFFSET_BITS) - 1) << METADATA_SHIFT;

const STACK_OFFSET_BITS: u64 = 32;
const STACK_OFFSET_MASK: u64 = ((1 << STACK_OFFSET_BITS) - 1) << METADATA_SHIFT;

const HEAP_OFFSET_BITS: u64 = 32;
const HEAP_OFFSET_MASK: u64 = ((1 << HEAP_OFFSET_BITS) - 1) << METADATA_SHIFT;

const DATA_TYPE_BITS: u64 = 12;  // 12 bits (5-16)
const DATA_TYPE_SHIFT: u64 = 5;  // Start at bit 5
const DATA_TYPE_MASK: u64 = ((1 << DATA_TYPE_BITS) - 1) << DATA_TYPE_SHIFT;

const CHANGE_TYPE_MASK: u64 = 0x1F;

#[repr(C)]
pub union EntityChangeData {
    pub u8: u8,
    pub u16: u16,
    pub u32: u32,
    pub u64: u64,
    pub f32: f32,
    pub f64: f64,
    pub ptr: *const u8, // this will also contain vector values
}

#[repr(C)]
pub struct EntityChange {
    pub flags: u64,
    pub data: EntityChangeData,
}

impl EntityChange {
    pub fn is_global(&self) -> bool {
        self.flags & EntityChangeFlags::Global != 0
    }

    pub fn is_thread_local(&self) -> bool {
        self.flags & EntityChangeFlags::ThreadLocal != 0
    }

    pub fn is_stack(&self) -> bool {
        self.flags & EntityChangeFlags::Stack != 0
    }

    pub fn is_heap(&self) -> bool {
        self.flags & EntityChangeFlags::Heap != 0
    }

    pub fn is_register(&self) -> bool {
        self.flags & EntityChangeFlags::Register != 0
    }

    pub fn is_u8(&self) -> bool {
        self.flags & EntityChangeFlags::U8 != 0
    }

    pub fn is_u16(&self) -> bool {
        self.flags & EntityChangeFlags::U16 != 0
    }

    pub fn is_u32(&self) -> bool {
        self.flags & EntityChangeFlags::U32 != 0
    }

    pub fn is_u64(&self) -> bool {
        self.flags & EntityChangeFlags::U64 != 0
    }

    pub fn is_f32(&self) -> bool {
        self.flags & EntityChangeFlags::F32 != 0
    }

    pub fn is_f64(&self) -> bool {
        self.flags & EntityChangeFlags::F64 != 0
    }

    pub fn is_v64(&self) -> bool {
        self.flags & EntityChangeFlags::V64 != 0
    }

    pub fn is_v128(&self) -> bool {
        self.flags & EntityChangeFlags::V128 != 0
    }

    pub fn is_v256(&self) -> bool {
        self.flags & EntityChangeFlags::V256 != 0
    }

    pub fn is_v512(&self) -> bool {
        self.flags & EntityChangeFlags::V512 != 0
    }

    pub fn is_ptr(&self) -> bool {
        self.flags & EntityChangeFlags::Ptr != 0
    }

    pub fn is_bytes(&self) -> bool {
        self.flags & EntityChangeFlags::Bytes != 0
    }

    pub fn is_movk(&self) -> bool {
        self.flags & EntityChangeFlags::MOVK != 0
    }

    pub fn is_movz(&self) -> bool {
        self.flags & EntityChangeFlags::MOVZ != 0
    }

    pub fn is_lsl16(&self) -> bool {
        self.flags & EntityChangeFlags::LSL16 != 0
    }
    
    pub fn is_lsl32(&self) -> bool {
        self.flags & EntityChangeFlags::LSL32 != 0
    }

    pub fn is_lsl48(&self) -> bool {
        self.flags & EntityChangeFlags::LSL48 != 0
    }

    pub fn is_lsl(&self) -> u32 {
        if self.is_lsl16() {
            16
        } else if self.is_lsl32() {
            32
        } else if self.is_lsl48() {
            48
        } else {
            0
        }
    }

    pub fn get_stack_offset(&self) -> u32 {
        (self.flags & STACK_OFFSET_MASK) >> STACK_OFFSET_SHIFT
    }

    pub fn get_global_offset(&self) -> u32 {
        (self.flags & GLOBAL_OFFSET_MASK) >> GLOBAL_OFFSET_SHIFT
    }

    pub fn get_tls_offset(&self) -> u32 {
        (self.flags & TLS_OFFSET_MASK) >> TLS_OFFSET_SHIFT
    }

    pub fn get_register(&self) -> u32 {
        (self.flags & REGISTER_MASK) >> REGISTER_SHIFT
    }

    pub fn get_heap_offset(&self) -> u32 {
        (self.flags & HEAP_OFFSET_MASK) >> HEAP_OFFSET_SHIFT
    }
}

impl DeltaSnapshot {
    pub fn generate_restore_chunk(&self) -> Vec<u8> {
        let mut code = Vec::new();

        for change in self.changes.iter() {
            change.generate_restore_chunk(&mut code);
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl EntityChange {
    pub fn apply_change(&self) {
        let value_type = (self.flags & DATA_TYPE_MASK) >> DATA_TYPE_SHIFT;
        
        match self.flags & CHANGE_TYPE_MASK {
            x if x == EntityChangeFlags::Global as u64 => self.apply_global(value_type),
            x if x == EntityChangeFlags::ThreadLocal as u64 => self.apply_thread_local(value_type),
            x if x == EntityChangeFlags::Stack as u64 => self.apply_stack(value_type),
            x if x == EntityChangeFlags::Heap as u64 => self.apply_heap(value_type),
            x if x == EntityChangeFlags::Register as u64 => self.apply_register(value_type),
            _ => {} // ?????????
        }
    }

    pub fn apply_global(&self, value_type: u32) {
        let offset = self.get_global_offset();
    }

    pub fn apply_thread_local(&self, value_type: u32) {
        let offset = self.get_tls_offset();
    }

    pub fn apply_stack(&self, value_type: u32) {
        let offset = self.get_stack_offset();
    }
    
    pub fn apply_heap(&self, value_type: u32) {
        let offset = self.get_heap_offset();
    }

    pub fn apply_register(&self, value_type: u32) {
        let register = self.get_register();
        let is_lsl = self.is_lsl();

        match is_lsl {
            16 => self.apply_register_lsl(value_type, register, 16),
            32 => self.apply_register_lsl(value_type, register, 32),
            48 => self.apply_register_lsl(value_type, register, 48),
            _ => {
                match value_type {
                    //0 => self.apply_register_u64(register),
                    x if x == EntityChangeFlags::U64 as u64 => DeltaSnapshot::mov_GPR_IMM64(register as u8, self.data.u64),
                    _ => {}
                }
            }
        }
    }

    pub fn apply_register_lsl(&self, value_type: u32, register: u32, lsl: u32) { // keep
        if self.is_movk() {
            self.apply_register_lsl_k(self.data.u16, register, lsl)
        } else if self.is_movz() {
            self.apply_register_lsl_z(self.data.u16, register, lsl)
        }
    }

    pub fn apply_register_lsl_k(&self, value: u16, register: u32, lsl: u32) { // keep
        let (code, size) = DeltaSnapshot::movk_GPR_IMM16_LSL(register as u8, value, lsl as u8);
    }

    pub fn apply_register_lsl_z(&self, value: u16, register: u32, lsl: u32) { // zero
        let (code, size) = DeltaSnapshot::movz_GPR_IMM16_LSL(register as u8, value, lsl as u8);
    }
}

#[cfg(target_arch = "aarch64")]
impl DeltaSnapshot {
    /// Encodes the EOR X_dst, X_dst, X_src instruction.
    /// This is the AArch64 equivalent of xor reg_dst, reg_src.
    fn xor_GPR_GPR(reg_dst: u8, reg_src: u8) -> ([u8; 4], usize) {
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

    fn zero_GPR(reg: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        // MOV Xd, XZR instruction encoding
        // This is actually: ORR Xd, XZR, XZR
        let instr: u32 = 0xAA1F03E0 | (reg as u32 & 0x1F);
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading an 8-bit immediate into a register using MOVZ.
    fn mov_GPR_IMM8(reg: u8, value: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        
        // MOVZ Xd, #imm, LSL 0
        // The register is zero-extended to 64 bits.
        let instr: u32 = 0xD2800000 | ((value as u32) << 5) | (reg as u32 & 0x1F);
        
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading a 16-bit immediate into a register using MOVZ.
    fn mov_GPR_IMM16(reg: u8, value: u16) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        
        // MOVZ Xd, #imm, LSL 0
        // The register is zero-extended to 64 bits.
        let instr: u32 = 0xD2800000 | ((value as u32) << 5) | (reg as u32 & 0x1F);

        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    /// Encodes loading a 32-bit immediate using a MOVZ/MOVK sequence.
    fn mov_GPR_IMM32(reg: u8, value: u32) -> ([u8; 8], usize) {
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
    fn mov_GPR_IMM64(reg: u8, value: u64) -> ([u8; 16], usize) {
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

    fn movz_GPR_IMM16_LSL(reg: u8, value: u16, lsl: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        // MOVZ Xd, #imm16, LSL #shift
        // shift: 0, 16, 32, or 48 (encoded as 0, 1, 2, 3)
        let shift_bits = match lsl {
            0 => 0,
            16 => 1,
            32 => 2,
            48 => 3,
            _ => panic!("Invalid LSL value: {}", lsl),
        };
        let instr: u32 = 0xD2800000 
                       | ((shift_bits as u32) << 21)   // hw field (bits 22-21)
                       | ((value as u32) << 5)         // imm16 field (bits 20-5)
                       | (reg as u32 & 0x1F);          // Rd field (bits 4-0)
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }

    fn movk_GPR_IMM16_LSL(reg: u8, value: u16, lsl: u8) -> ([u8; 4], usize) {
        let mut code = [0; 4];
        // MOVK Xd, #imm16, LSL #shift
        let shift_bits = match lsl {
            0 => 0,
            16 => 1,
            32 => 2,
            48 => 3,
            _ => panic!("Invalid LSL value: {}", lsl),
        };
        let instr: u32 = 0xF2800000                    // MOVK opcode
                       | ((shift_bits as u32) << 21)   // hw field
                       | ((value as u32) << 5)         // imm16 field
                       | (reg as u32 & 0x1F);          // Rd field
        code.copy_from_slice(&instr.to_le_bytes());
        (code, 4)
    }
}

#[cfg(target_arch = "x86_64")]
impl EntityChange {
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

extern "C" {
    static __basic_blocks_registry: *const BasicBlockInfo;
    static __basic_blocks_count: usize;
    static __entity_changes_registry: *const EntityChange;
}

#[no_mangle]
static __snapshot_registry: std::sync::atomic::AtomicPtr<u8> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

#[no_mangle]
static __snapshot_count: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


// mprotect, fork
// hybrid shadow copy of registers
// alternatively handle registers and memory different
// mprotect + fork we can leverage CoW
// allocate region of memory that holds 'cards' (byte array tracking diry state of each like 128 bytes or something), each section of memory will get a bit set if its dirty and needs to be copied. 
// setting cards dirty will probably be done from llvm machine pass, if electing to go this route
// then between snapshots just memsetting to 0 should be quite fast
// i mean, i guess we dont need to use 'thread-locals' proper for writing, we can just sub something from the stack, allocate some scratch space for us to keep writing to
// i think the cards one might be the best, at least for our heap