// Read-only access to TIG's pending per-thread fuel accounting.
// The stable inline-assembly spellings avoid crate-level unstable attributes and
// preserve the initial-exec TLS model used by the measured research builds.
// No runtime counter is written here; the official instrumenter enforces Fuel.
extern "C" { static __fuel_remaining: u64; }

#[inline(always)]
pub(super) fn remaining() -> u64 {
    let used: u64;
    unsafe {
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!(
            "mov {offset}, qword ptr [rip + __thread_local_fuel_used@gottpoff]",
            "mov {used}, qword ptr fs:[{offset}]",
            offset = out(reg) _, used = out(reg) used,
            options(nostack, preserves_flags),
        );
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!(
            "mrs {thread}, tpidr_el0",
            "adrp {offset}, :gottprel:__thread_local_fuel_used",
            "ldr {offset}, [{offset}, #:gottprel_lo12:__thread_local_fuel_used]",
            "ldr {used}, [{thread}, {offset}]",
            thread = out(reg) _, offset = out(reg) _, used = out(reg) used,
            options(nostack, preserves_flags),
        );
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        compile_error!("TIG official builds target x86_64 or aarch64");
        core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)).saturating_sub(used)
    }
}
