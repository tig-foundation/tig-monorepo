pub(super) struct Stopper {
    values: [f64; 64],
    len: usize,
    next: usize,
    w: usize,
    s0: usize,
    threshold: f64,
    patience: usize,
}

impl Stopper {
    pub(super) fn new(w: usize, s0: usize, z: f64, patience: usize) -> Self {
        Self { values: [0.0; 64], len: 0, next: 0, w, s0,
            threshold: 10.0f64.powf(-z), patience }
    }

    pub(super) fn push(&mut self, v: f32) {
        if v.is_finite() {
            self.values[self.next] = v as f64;
            self.next = (self.next + 1) % self.w;
            self.len = (self.len + 1).min(self.w);
        }
    }

    pub(super) fn should_stop(&self, lowest: f32, stale: usize, delta: f32) -> bool {
        if stale < self.s0 || self.len != self.w || !lowest.is_finite() { return false; }
        let mut ordered = [0.0; 64];
        for i in 0..self.w { ordered[i] = self.values[(self.next + i) % self.w]; }
        let mu = fsum(&ordered[..self.w]) / self.w as f64;
        for v in &mut ordered[..self.w] { let d = *v - mu; *v = d * d; }
        let sigma = (fsum(&ordered[..self.w]) / self.w as f64).sqrt();
        if !(sigma > 0.0 && sigma.is_finite()) { return false; }
        let gap = (mu - (lowest as f64 - delta as f64)) / sigma;
        if gap.is_nan() { return false; }
        let x = gap.abs() / 2.0f64.sqrt();
        let tail = 0.5 * erfc_positive(x);
        let p1 = if gap < 0.0 { 1.0 - tail } else { tail };
        let remaining = self.patience.saturating_sub(stale).saturating_sub(1).max(1);
        let pr = (remaining as f64 * p1).min(1.0);
        pr < self.threshold
    }
}

fn fsum(values: &[f64]) -> f64 {
    let mut partials = [0.0f64; 64];
    let mut n = 0;
    for &v in values {
        let mut x = v;
        let mut k = 0;
        for j in 0..n {
            let mut y = partials[j];
            if x.abs() < y.abs() { let t = x; x = y; y = t; }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 { partials[k] = lo; k += 1; }
            x = hi;
        }
        partials[k] = x;
        n = k + 1;
    }
    let mut hi = 0.0;
    let mut lo = 0.0;
    while n > 0 {
        n -= 1;
        let x = hi;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 { break; }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0)
        || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi { hi = x; }
    }
    hi
}

fn erfc_positive(x: f64) -> f64 {
    if x >= 40.0 { return 0.0; }
    let xx = x * x;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    if x < 1.0 {
        let mut term = x;
        let mut terms = [0.0; 40];
        terms[0] = term;
        for n in 1..40 {
            term *= -xx / n as f64;
            terms[n] = term / (2 * n + 1) as f64;
        }
        return 1.0 - (2.0 / sqrt_pi) * fsum(&terms);
    }
    let mut b = xx + 0.5;
    let mut c = 1e300;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=1024 {
        let n = i as f64;
        let a = -n * (n - 0.5);
        b += 2.0;
        d = a * d + b;
        if d.abs() < 1e-300 { d = 1e-300; }
        c = b + a / c;
        if c.abs() < 1e-300 { c = 1e-300; }
        d = 1.0 / d;
        let change = d * c;
        h *= change;
        if (change - 1.0).abs() <= f64::EPSILON { break; }
    }
    ((-xx).exp() * (x / sqrt_pi)) * h
}
