const K: usize = 64;

fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Graph {
    m: usize,
    n: usize,
    he_off: Vec<u32>,
    he_nodes: Vec<u32>,
    nd_off: Vec<u32>,
    nd_edges: Vec<u32>,
}

fn km1(g: &Graph, part: &[u8]) -> u64 {
    let mut total = 0u64;
    for e in 0..g.m {
        let mut f = 0u64;
        for &v in &g.he_nodes[g.he_off[e] as usize..g.he_off[e + 1] as usize] {
            f |= 1u64 << part[v as usize];
        }
        total += f.count_ones() as u64 - 1;
    }
    total
}

fn affinity(g: &Graph, part: &[i32]) -> Vec<f64> {
    let mut w = vec![0f64; K * K];
    let mut bs = [0usize; K];
    for e in 0..g.m {
        let mut f = 0u64;
        for &v in &g.he_nodes[g.he_off[e] as usize..g.he_off[e + 1] as usize] {
            f |= 1u64 << part[v as usize];
        }
        let lam = f.count_ones() as usize;
        if lam < 2 {
            continue;
        }
        let x = 1.0 / (lam - 1) as f64;
        let mut c = 0;
        for b in 0..K {
            if f >> b & 1 == 1 {
                bs[c] = b;
                c += 1;
            }
        }
        for i in 0..c {
            for j in i + 1..c {
                w[bs[i] * K + bs[j]] += x;
                w[bs[j] * K + bs[i]] += x;
            }
        }
    }
    w
}

fn bisect(w: &[f64], set: &[usize], rng: &mut u64, children: &mut Vec<[usize; 2]>) -> usize {
    if set.len() == 1 {
        return set[0];
    }
    let h = set.len() / 2;
    let mut best_cut = f64::INFINITY;
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..30 {
        let mut perm = set.to_vec();
        for i in (1..perm.len()).rev() {
            let j = (mix(rng) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        let mut side = [false; K];
        for &x in &perm[..h] {
            side[x] = true;
        }
        loop {
            let mut d = [0f64; K];
            for &x in set {
                for &y in set {
                    if x != y {
                        d[x] += if side[x] != side[y] { w[x * K + y] } else { -w[x * K + y] };
                    }
                }
            }
            let mut gain = 1e-12;
            let mut pick = None;
            for &x in set {
                if !side[x] {
                    continue;
                }
                for &y in set {
                    if side[y] {
                        continue;
                    }
                    let gxy = d[x] + d[y] - 2.0 * w[x * K + y];
                    if gxy > gain {
                        gain = gxy;
                        pick = Some((x, y));
                    }
                }
            }
            match pick {
                Some((x, y)) => {
                    side[x] = false;
                    side[y] = true;
                }
                None => break,
            }
        }
        let mut cut = 0.0;
        for &x in set {
            for &y in set {
                if side[x] && !side[y] {
                    cut += w[x * K + y];
                }
            }
        }
        if cut < best_cut {
            best_cut = cut;
            best = set.iter().copied().filter(|&x| side[x]).collect();
        }
    }
    let rest: Vec<usize> = set.iter().copied().filter(|x| !best.contains(x)).collect();
    let a = bisect(w, &best, rng, children);
    let b = bisect(w, &rest, rng, children);
    children.push([a, b]);
    K + children.len() - 1
}

fn greedy(g: &Graph, deg: &[u32], key: &[u32], left: &[bool], flags: &mut [u8], touched: &mut Vec<u32>) -> u64 {
    let n = g.n;
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        deg[b as usize]
            .cmp(&deg[a as usize])
            .then(key[a as usize].cmp(&key[b as usize]))
    });
    let mut part: Vec<u32> = (0..n).map(|v| if left[v] { 1 } else { 2 }).collect();
    for level in 1..6u32 {
        let np = 1usize << level;
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); np];
        for &v in &order {
            buckets[part[v as usize] as usize + 1 - np].push(v);
        }
        for (b, bucket) in buckets.iter().enumerate() {
            let count = bucket.len();
            if count == 0 {
                continue;
            }
            let p = (np + b - 1) as u32;
            let size_left = count / 2;
            let size_right = count - size_left;
            let (mut lc, mut rc) = (0usize, 0usize);
            for &e in touched.iter() {
                flags[e as usize] = 0;
            }
            touched.clear();
            for &v in bucket {
                let vi = v as usize;
                let edges = &g.nd_edges[g.nd_off[vi] as usize..g.nd_off[vi + 1] as usize];
                let go_left = if lc >= size_left {
                    false
                } else if rc >= size_right {
                    true
                } else {
                    let (mut cl, mut cr) = (0u32, 0u32);
                    for &e in edges {
                        let f = flags[e as usize];
                        cl += (f & 1) as u32;
                        cr += (f >> 1) as u32;
                    }
                    if cl == cr {
                        lc < rc
                    } else {
                        cl > cr
                    }
                };
                let bit = if go_left {
                    part[vi] = 2 * p + 1;
                    lc += 1;
                    1u8
                } else {
                    part[vi] = 2 * p + 2;
                    rc += 1;
                    2u8
                };
                for &e in edges {
                    if flags[e as usize] == 0 {
                        touched.push(e);
                    }
                    flags[e as usize] |= bit;
                }
            }
        }
    }
    let p8: Vec<u8> = part.iter().map(|&x| (x - 63) as u8).collect();
    km1(g, &p8)
}

pub fn estimate(he_off: &[i32], he_nodes: &[i32], n: usize, part: &[i32], samples: usize, seed: u64) -> f64 {
    let m = he_off.len() - 1;
    let mut deg = vec![0u32; n];
    for &v in he_nodes {
        deg[v as usize] += 1;
    }
    let mut nd_off = vec![0u32; n + 1];
    for v in 0..n {
        nd_off[v + 1] = nd_off[v] + deg[v];
    }
    let mut nd_edges = vec![0u32; he_nodes.len()];
    let mut pos = nd_off[..n].to_vec();
    for e in 0..m {
        for &v in &he_nodes[he_off[e] as usize..he_off[e + 1] as usize] {
            nd_edges[pos[v as usize] as usize] = e as u32;
            pos[v as usize] += 1;
        }
    }
    let g = Graph {
        m,
        n,
        he_off: he_off.iter().map(|&x| x as u32).collect(),
        he_nodes: he_nodes.iter().map(|&x| x as u32).collect(),
        nd_off,
        nd_edges,
    };
    let w = affinity(&g, part);
    let mut rng = seed;
    let mut children = Vec::new();
    let all: Vec<usize> = (0..K).collect();
    let root = bisect(&w, &all, &mut rng, &mut children);
    let mut flags = vec![0u8; m];
    let mut touched = Vec::new();
    let mut total = 0f64;
    for _ in 0..samples {
        let mut rank = [0u32; K];
        let mut lside = [false; K];
        let mut next = 0u32;
        let mut stack = vec![(root, 2u8)];
        while let Some((node, side)) = stack.pop() {
            if node < K {
                rank[node] = next;
                next += 1;
                lside[node] = side == 1;
                continue;
            }
            let mut ch = children[node - K];
            if mix(&mut rng) & 1 == 1 {
                ch.swap(0, 1);
            }
            let s0 = if side == 2 { 1 } else { side };
            let s1 = if side == 2 { 0 } else { side };
            stack.push((ch[1], s1));
            stack.push((ch[0], s0));
        }
        let mut tie: Vec<(u32, u64, u32)> = (0..n as u32)
            .map(|v| (rank[part[v as usize] as usize], mix(&mut rng), v))
            .collect();
        tie.sort_unstable();
        let mut key = vec![0u32; n];
        for (i, t) in tie.iter().enumerate() {
            key[t.2 as usize] = i as u32;
        }
        let left: Vec<bool> = (0..n).map(|v| lside[part[v] as usize]).collect();
        total += greedy(&g, &deg, &key, &left, &mut flags, &mut touched) as f64;
    }
    total / samples as f64
}
