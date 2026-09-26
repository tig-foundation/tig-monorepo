use super::{anyhow, Result, State, Cfg, STOPPED, REC_DELTA};

pub(super) fn stop<T>() -> Result<T> {
    STOPPED.with(|f| f.set(true));
    Err(anyhow!("run ended by the optimizer"))
}

pub(super) fn lr_at(s: &State, c: &Cfg, epoch: usize) -> f32 {
    if epoch < s.warmup_eff {
        return c.lr_max * ((epoch + 1) as f32) / (s.warmup_eff.max(1) as f32);
    }
    let denom = c.t_max.saturating_sub(s.warmup_eff).max(1) as f32;
    let p = if s.cos_fix_eff {
        ((epoch - s.warmup_eff) as f32 / denom).min(1.0)
    } else {
        (epoch as f32 / denom).min(1.0)
    };
    c.lr_min + 0.5 * (c.lr_max - c.lr_min) * (1.0 + (std::f32::consts::PI * p).cos())
}

pub(super) fn advance_epoch(s: &mut State, c: &Cfg, epoch: usize, val_loss: Option<f32>) -> Result<()> {
    if let (Some(p), Some(v)) = (&mut s.pstop, val_loss) { p.push(v); }
    if s.epoch_seen && s.batches_per_epoch == 0 {
        s.batches_per_epoch = s.batch_in_epoch;
    }
    s.batch_in_epoch = 0;
    s.epoch = epoch;
    s.epoch_seen = true;
    if let Some(val) = val_loss.filter(|v| v.is_finite()) {
        if val < s.best_val {
            s.best_val = val;
            s.plateau_stale = 0;
            s.stale_total = 0;
            if !s.cooling {
                s.lr_scale = (s.lr_scale * c.plateau_grow).min(1.0);
            }
        } else {
            s.plateau_stale += 1;
            s.stale_total += 1;
            if !s.cooling && s.plateau_stale >= c.plateau_patience {
                s.lr_scale = (s.lr_scale * c.plateau_decay).max(c.plateau_floor);
                s.plateau_stale = 0;
            }
        }
    }
    if s.cool_stale_eff > 0 && !s.cooling && s.stale_total >= s.cool_stale_eff {
        s.cooling = true;
        s.cool_start = epoch;
    }
    if s.cooling {
        s.lr_scale = (s.lr_scale * s.cool_decay_eff).max(c.cool_floor);
    }
    if let Some(val) = val_loss {
        if val < s.rec_lowest - REC_DELTA.with(|d| d.get()) {
            s.rec_lowest = val;
            s.rec_stale = 0;
        } else {
            s.rec_stale += 1;
        }
    }
    if s.pstop.as_ref().map_or(false, |p| p.should_stop(
        s.rec_lowest, s.rec_stale, REC_DELTA.with(|d| d.get())))
    {
        return stop();
    }
    if s.stop_stale_eff > 0
        && s.cooling
        && epoch - s.cool_start >= s.stop_age_eff
        && s.rec_stale >= s.stop_stale_eff
    {
        return stop();
    }
    Ok(())
}
