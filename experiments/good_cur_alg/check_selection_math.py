"""CPU checks of the selection math; these do not execute CUDA.
Run: OPENBLAS_NUM_THREADS=1 python3 experiments/good_cur_alg/check_selection_math.py"""
import numpy as np

def oracle(x, y, k):
    """Rebuild each proposed span with least squares and measure its residual."""
    chosen = []
    for _ in range(k):
        losses = []
        for j in range(x.shape[1]):
            if j in chosen:
                losses.append(np.inf)
                continue
            cols = x[:, chosen + [j]]
            pred = cols @ np.linalg.lstsq(cols, y, rcond=1e-12)[0]
            losses.append(np.linalg.norm(y - pred) ** 2)
        chosen.append(int(np.argmin(losses)))
    return chosen

def greedy(x, k, y=None):
    """Reference arithmetic for the mixed-precision greedy selection kernels."""
    x = x.astype(np.float32).copy()
    y = None if y is None else y.astype(np.float32).copy()
    chosen = []
    for _ in range(k):
        norm = (x.astype(float) ** 2).sum(axis=0)
        if y is None:
            scores = norm
        else:
            cross = y.T @ x
            scores = (cross.astype(float) ** 2).sum(axis=0) / np.maximum(norm, 1e-30)
            scores[norm <= 1e-30] = 0
        scores[chosen] = -1
        j = int(np.argmax(scores))
        chosen.append(j)
        q = (x[:, j].astype(float) / np.sqrt(norm[j])).astype(np.float32) if norm[j] > 1e-30 else np.zeros(x.shape[0], np.float32)
        for _ in range(2):
            x = (x.astype(float) - q.astype(float)[:, None] * (q.astype(float) @ x.astype(float))).astype(np.float32)
            if y is not None:
                y = (y.astype(float) - q.astype(float)[:, None] * (q.astype(float) @ y.astype(float))).astype(np.float32)
    return chosen
rng = np.random.default_rng(891)
for (dim, points, k) in [(3, 7, 1), (7, 11, 3), (12, 17, 6)]:
    for _ in range(15):
        x = rng.normal(size=(dim, points))
        y = rng.normal(size=(dim, k))
        assert greedy(x, k, y) == oracle(x, y, k)
print('45 conditional pivot sequences match exhaustive least-squares oracle')
for x in [np.zeros((5, 9)), np.ones((5, 9)), np.eye(5), np.tile(np.eye(5), (1, 2))]:
    a = greedy(x, 5)
    assert len(set(a)) == 5 and min(a) >= 0 and (max(a) < x.shape[1])
    assert a == greedy(x, 5)
print('zero, rank-deficient, square and duplicate-column cases return unique deterministic indices')
for _ in range(20):
    (m, n, r, k) = (35, 43, 12, 5)
    u = np.linalg.qr(rng.normal(size=(m, r)))[0]
    v = np.linalg.qr(rng.normal(size=(n, r)))[0]
    s = np.geomspace(1, 0.001, r)
    a = u * s @ v.T
    cols = rng.choice(n, k, False)
    rows = rng.choice(m, k, False)
    qc = np.linalg.qr(a[:, cols])[0]
    qr = np.linalg.qr(a[rows, :].T)[0]
    c = s[:, None] * v.T[:, cols]
    rr = s[:, None] * u.T[:, rows]
    qcc = np.linalg.qr(c)[0]
    qrr = np.linalg.qr(rr)[0]
    target = s[:, None] * qcc
    captured = np.linalg.norm(qrr.T @ target) ** 2
    exact = np.linalg.norm(qc.T @ a @ qr) ** 2
    assert np.isclose(captured, exact, rtol=1e-10, atol=1e-12)
print('20 compressed conditional objectives match the full CUR projection objective')
