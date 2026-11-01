pub const FUEL_STOP: u64 = 100_000_000;

extern "C" {
    static __fuel_remaining: u64;
}

#[inline]
pub fn fuel_left() -> u64 {
    unsafe { core::ptr::read_volatile(&__fuel_remaining) }
}

pub fn run_bounded<F>(
    work_cap: u64,
    guarded: &dyn Fn(&tig_challenges::job_scheduling::Solution) -> anyhow::Result<()>,
    engine: F,
) where
    F: FnOnce(
        &dyn Fn(&tig_challenges::job_scheduling::Solution) -> anyhow::Result<()>,
    ) -> anyhow::Result<()>,
{
    let fuel_start = fuel_left();
    let floor_fuel = fuel_start.saturating_sub(work_cap.min(fuel_start.saturating_sub(FUEL_STOP)));
    let bound = |s: &tig_challenges::job_scheduling::Solution| -> anyhow::Result<()> {
        guarded(s)?;
        if fuel_left() <= floor_fuel {
            return Err(anyhow::anyhow!("fuel cap"));
        }
        Ok(())
    };
    let _ = engine(&bound);
}
