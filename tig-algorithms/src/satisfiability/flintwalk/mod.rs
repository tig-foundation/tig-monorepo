// flintwalk — clean-room SLS-Loeser fuer TIG c001 (satisfiability), mit Regimeerkennung.
// Gemessen: n=100000/ratio=4150 100/100 (offiziell, volle Buendelgroesse, Vorversion),
//           n=100000/ratio=4200 25/25 (Fuel-Messstand, volles Budget).
// Gegenueber der Vorversion ausschliesslich die drei heissen Schleifen umgebaut; die Suche ist
// unveraendert (Lauf fuer Lauf bitgleiche Loesung, gleiche Flip- und Freebie-Zahl).
//
// Kern: WalkSAT/SKC (Selman, Kautz, Cohen 1994) mit Freebie-Zug: aus einer zufaelligen
// unerfuellten Klausel die Variable mit break=0 (frei), sonst mit Wahrscheinlichkeit p eine
// zufaellige, sonst die mit minimalem break (Gleichstand zufaellig).
// Rauschen adaptiv (Hoos 2002: keine Verbesserung ueber theta*m Schritte -> p hoch, jede
// Verbesserung -> p runter) + eskalierende Zufallskicks bei langer Stagnation.
// Startbelegung: Literalzaehl-Bias (np/nn), Pure-Literal, Unit-Propagation.
// Datenlayout fuer den Fuel-Zaehler (Instruktionen): literal-indizierte Occurrence-CSR
// (Vorkommen von Literal l = 2v+pol zusammenhaengend), Klauselzustand u8, u32-Indizes,
// Delta-Updates, Lazy-Deletion der Unsat-Liste, Lemire-Multiplikation statt Modulo,
// get_unchecked in den heissen Schleifen (Grenzen per Debug-Assertion belegt, s. BLOCK).
// Tempo: das Netz fuehrt die fuel-instrumentierte Binaerdatei aus, die je BASISBLOCK verbucht.
// Deshalb zaehlen die drei heissen Schleifen einen festen, sprungfreien BLOCK von acht Eintraegen
// und maskieren den Ueberhang, statt je Eintrag zu springen: mehr Zugriffe (1,35x im Mittel),
// aber ein Viertel bis ein Drittel weniger Wanduhr. Vorkommenslisten sind im Mittel 3m/2n lang,
// bei den Netzspuren (ratio >= 4150) also gut sechs Eintraege.
// Deterministisch: eigener xoshiro256++ aus challenge.seed.
// Alle Bausteine publiziert; keine Zeile aus fremdem Qualifizierer-Code.

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

pub fn help() {
    println!("flintwalk: WalkSAT/SKC + Freebie, adaptives Rauschen, Bias-Init. \
              hp: max_flips_per_var (0=unbegrenzt, Fuel entscheidet), noise (0.5), \
              theta (0.1667), phi (0.0), kick_window_m (40), init (bias|rand), stats (0/1), \
              pick (skc|probsat|probsat_poly), cb (2.06 exponentiell / 2.38 polynomiell), eps (0.9, nur polynomiell), trace (Flips je Zwischenstand, 0=aus), \
              kick_len0 (8), kick_max (4096), restart (0=aus; 1=vom besten Punkt neu ansetzen), \
              kick_reset (0=aus), best_keep (64), \
              try_flips_per_var (0=ein Versuch; sonst Grundeinheit Flips je Versuch), \
              luby (0=feste Laenge, 1=Versuchslaengen nach der Luby-Folge)");
}

fn hp_f64(hp: &Option<Map<String, Value>>, k: &str, d: f64) -> f64 {
    hp.as_ref().and_then(|m| m.get(k)).and_then(|v| v.as_f64()).unwrap_or(d)
}
fn hp_str(hp: &Option<Map<String, Value>>, k: &str, d: &str) -> String {
    hp.as_ref().and_then(|m| m.get(k)).and_then(|v| v.as_str()).unwrap_or(d).to_string()
}

/// xoshiro256++ (Blackman & Vigna, public domain), Saat via SplitMix64.
struct Rng([u64; 4]);
impl Rng {
    fn new(seed: &[u8; 32]) -> Self {
        let mut x = u64::from_le_bytes(seed[0..8].try_into().unwrap())
            ^ u64::from_le_bytes(seed[8..16].try_into().unwrap()).rotate_left(17)
            ^ u64::from_le_bytes(seed[16..24].try_into().unwrap()).rotate_left(34)
            ^ u64::from_le_bytes(seed[24..32].try_into().unwrap()).rotate_left(51);
        let mut s = [0u64; 4];
        for w in s.iter_mut() {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *w = z ^ (z >> 31);
        }
        if s.iter().all(|&w| w == 0) { s[0] = 1; }
        Rng(s)
    }
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    /// Gleitkomma in [0,1)
    #[inline(always)]
    fn f64(&mut self) -> f64 { (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64) }
}

/// Luby-Folge (Luby, Sinclair, Zuckerman 1993): 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
/// Nachweislich optimal fuer Neustart-Abstaende bei schwerschwaenzigen Laufzeiten, wenn man
/// die Verteilung nicht kennt. Eine FESTE Versuchslaenge ist messbar schlechter: gemessen
/// 14.09.2026 auf n=100000/ratio=4200 zerschneidet sie die Nonces, die in einem langen Anlauf
/// durchkaemen (7 und 16 brauchten ohne Schleife einen Zug, mit fester Laenge 12 bzw. 15
/// Versuche), waehrend sie den festgefahrenen hilft.
#[inline]
fn luby(i: u64) -> u64 {
    let mut i = i + 1;              // Folge ist 1-basiert
    let mut k: u32 = 1;
    loop {
        let p = (1u64 << k) - 1;    // 1, 3, 7, 15, ...
        if i == p { return 1u64 << (k - 1); }
        if i < p { i -= (1u64 << (k - 1)) - 1; k = 1; continue; }
        k += 1;
        if k >= 63 { return 1u64 << 62; }
    }
}

/// Width of the fixed, branch-free block used in the three hot loops. `occ` carries exactly this
/// many reserve slots at the end, so a block may always read (and harmlessly rewrite) past the end
/// of one occurrence list. Widening this constant without widening the reserve is out of bounds.
const BLOCK: usize = 8;

/// Lemire: gleichverteilter Index in 0..len aus 32 Zufallsbits, ohne Division.
#[inline(always)]
fn idx32(r32: u64, len: usize) -> usize { ((r32 & 0xFFFF_FFFF) * len as u64 >> 32) as usize }

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let n = challenge.num_variables;
    // Platzhalter zuerst: bricht der Fuel-Zaehler ab, bleibt sonst gar keine Ausgabe.
    save_solution(&Solution { variables: vec![false; n] })?;

    // Zwei Regime, die genau Gegenteiliges verlangen. Das Verhaeltnis Klauseln zu Variablen sagt,
    // in welchem wir sind, und das weiss der Loeser ohne jede Zusatzinformation:
    //   ratio < 4,25  -> unter dem Phasenuebergang. Die Netzspuren dort sind riesig (100.000
    //     Variablen) und praktisch immer loesbar; gemessen 20.09. auf ratio 4200, gepaart je Nonce:
    //     feste Versuchslaenge 74 / 447 / 93 s, mit Luby-Neustart 649 / 628 / 240 s, mit kuerzeren
    //     Versuchen teils gar nicht geloest. Also: WalkSAT/SKC, lange Versuche, kein Neustart.
    //   ratio >= 4,25 -> am Phasenuebergang. Dort sind die Instanzen klein (5.000 bis 10.000) und
    //     zu einem guten Teil unerfuellbar. Gemessen 21.09. auf n_vars=5000/ratio=4267, 10 Nonces,
    //     50.000 Flips je Variable, Mass ist der kleinste erreichte Rest unerfuellter Klauseln:
    //       probSAT polynomiell, feste Versuchslaenge   4,5 (loest 1 Nonce)
    //       dasselbe mit Neustart bei Stagnation       39,5
    //       WalkSAT mit Neustart bei Stagnation        20,6
    //       WalkSAT mit Alter als Stichentscheid      135,8
    //     Also: nur die polynomielle Bruchfunktion umschalten, Versuchsdisziplin und Stichentscheid
    //     bleiben wie unter dem Uebergang. Luby und Alter sind gemessen schaedlich.
    let ratio_cm = challenge.clauses.len() as f64 / n.max(1) as f64;
    let am_uebergang = ratio_cm >= hp_f64(hyperparameters, "ratio_split", 4.25);

    // Aufgabegrenze. Gemessen 21.09.2026 auf der Spur n_vars=5000/ratio=4267: die Fuel-Grenze der
    // Laufzeitumgebung wird in einer engen Rechenschleife **nicht** erzwungen. Gleiches Budget,
    // gleicher Seed, nur mit einer periodischen Ausgabe im Takt: dann bricht der Lauf sauber bei
    // erschoepftem Fuel ab (Status 87, 0,27 s); ohne sie rechnet er unbegrenzt weiter (>100 s,
    // abgeschnitten). Auf einer unerfuellbaren Instanz laeuft der Loeser also endlos, statt
    // aufzugeben — fuer einen Benchmarker mit 100 Nonces je Buendel unbrauchbar, und die am Ende
    // gemeldete Fuel-Zahl koennte das Budget ueberschreiten. Der Loeser begrenzt sich daher selbst:
    //   am Phasenuebergang 50.000 Flips je Variable (bei 5.000 Variablen 2,5e8 Flips, rund 80 s
    //     und 2,5 % des Budgets) — dort ist ein guter Teil der Instanzen unerfuellbar,
    //   darunter 80.000 Flips je Variable als reines Sicherheitsnetz: gebraucht werden dort
    //     gemessen 340 Flips je Variable, die Schranke bindet also nie, garantiert aber, dass
    //     kein Lauf das Fuel-Budget ueberschreiten kann.
    let mfpv = hp_f64(
        hyperparameters,
        "max_flips_per_var",
        if am_uebergang { 50_000.0 } else { 80_000.0 },
    );
    let max_flips: u64 = if mfpv <= 0.0 { u64::MAX } else { (mfpv * n as f64) as u64 };
    let p0 = hp_f64(hyperparameters, "noise", 0.5).clamp(0.0, 1.0);
    let theta = hp_f64(hyperparameters, "theta", 1.0 / 6.0);
    // phi = 0 -> adaptives Rauschen AUS. Gemessen 13.09.2026 auf n=100000/ratio=4150:
    // mit Anpassung (phi 0,2) zieht p auf 0,110 und der Loeser bleibt bei 116 unerfuellten
    // Klauseln stehen; mit festem p=0,5 loesen 12 von 12 Nonces bei <=0,52 % des Fuel.
    // Das Optimum ist scharf: 0,40 -> 45 unerfuellt, 0,50 -> geloest, 0,57 -> 2727.
    let phi = hp_f64(hyperparameters, "phi", 0.0);
    let kick_window_m = hp_f64(hyperparameters, "kick_window_m", 40.0);
    // Kickzuruecksetzung und Neuansatz vom besten Punkt - BEIDE STANDARDMAESSIG AUS.
    // Vermutung war: der Loeser erreicht auf n=5000/ratio=4267 best=1 und driftet auf
    // unsat=158 ab, weil kick_len einmalig auf 4096 eskaliert und nie zurueckgesetzt
    // wird; ein Neuansatz vom besten Punkt muesste das heilen. Gemessen 13.09.2026
    // (je zwei Nonces, 1e11 Fuel, Rauschen 0,45), best_nun kleiner ist besser:
    //   ohne Neuansatz, klen 4096 (Ausgangsstand): 1 / 3
    //   ohne Neuansatz, klen 8:                    3 / 2
    //   mit  Neuansatz, klen 512:                  4 / 2
    //   mit  Neuansatz, klen 64:                   5 / 5
    //   mit  Neuansatz, klen 8:                    6 / 4
    // Der Neuansatz hilft nicht, er schadet. Die Spannweite 1-6 ueber zwei Nonces ist
    // ohnehin Streuung, nicht Signal (s. Regel: unter 25 Nonces misst man Rauschen).
    // Die Knoepfe bleiben fuer spaetere Versuche, die Vorgaben bilden den verifizierten
    // Zustand ab (100/100 offiziell auf n=100000/ratio=4150).
    let kick_len0 = hp_f64(hyperparameters, "kick_len0", 8.0).max(1.0) as u64;
    let kick_max = hp_f64(hyperparameters, "kick_max", 4096.0).max(1.0) as u64;
    let restart_best = hp_f64(hyperparameters, "restart", 0.0) > 0.0;
    let kick_reset = hp_f64(hyperparameters, "kick_reset", 0.0) > 0.0;
    // Beste Belegung nur nahe der Loesung sichern: fern davon waere das Kopieren von
    // n Bytes je neuem Minimum reine Fuel-Verschwendung ohne Nutzen.
    let best_keep = hp_f64(hyperparameters, "best_keep", 64.0).max(0.0) as usize;
    // Flips je Versuch, dann frische Zufallsbelegung. 0 = ein einziger Versuch.
    // Gemessen 14.09.2026 auf n=100000/ratio=4200, 25 Nonces, volles Budget:
    //   ohne Versuchsschleife   21/25 - vier Nonces stehen bei best 1..4 und verbrennen
    //                           12,5 Mrd. Flips im selben lokalen Minimum
    //   5000 Flips/Var je Versuch  25/25 - Fuel Median 7,4 %, Maximum 90,8 %
    //   20000 Flips/Var je Versuch 3 von 4 harten Nonces (auf den vier Harten gemessen)
    // Auf n=100000/ratio=4150 loest der erste Versuch (500 Mio. Flips) ohnehin alles
    // (20-65 Mio. Flips gebraucht), ein Neustart wird dort nie ausgeloest - gegengeprueft,
    // Nonce 0/1/2 liefern bitgleich flips=31164021/27473214/65237317.
    let tfpv = hp_f64(hyperparameters, "try_flips_per_var", 5000.0);
    let try_flips: u64 = if tfpv <= 0.0 { 0 } else { (tfpv * n as f64) as u64 };
    // luby=1: try_flips ist die GRUNDEINHEIT, die tatsaechliche Laenge ist Einheit x luby(k).
    // Unter dem Uebergang AUS (dort gemessen schaedlich, s. oben), am Uebergang AN.
    let luby_an = hp_f64(hyperparameters, "luby", 0.0) > 0.0;
    let init_bias = hp_str(hyperparameters, "init", "bias") == "bias";
    let stats = hp_f64(hyperparameters, "stats", 0.0) > 0.0;
    let pick = hp_str(hyperparameters, "pick", if am_uebergang { "probsat_poly" } else { "skc" });
    let probsat = pick == "probsat" || pick == "probsat_poly";
    // Balint & Schoening 2012 geben fuer 3-SAT zwei Bruchfunktionen an: exponentiell cb^-b mit
    // cb = 2,06 und polynomiell (eps + b)^-cb mit cb = 2,38 / eps = 0,9. Auf 3-SAT ist die
    // polynomielle in ihrer Messung die etwas bessere; beide sind hier eine Tabelle bis b = 31.
    let poly = pick == "probsat_poly";
    let cb = hp_f64(hyperparameters, "cb", if poly { 2.38 } else { 2.06 }).max(1.01);
    let eps = hp_f64(hyperparameters, "eps", 0.9).max(1.0e-6);
    let trace = hp_f64(hyperparameters, "trace", 0.0) as u64;
    let mut next_trace: u64 = if trace > 0 { trace } else { u64::MAX };
    // probSAT (Balint & Schoening 2012), exponentielle Break-Gewichte cb^-b, Tabelle bis 31.
    let pow: [f64; 32] = {
        let mut t = [0.0; 32];
        for k in 0..32 {
            t[k] = if poly { (eps + k as f64).powf(-cb) } else { cb.powi(-(k as i32)) };
        }
        t
    };
    let mut rng = Rng::new(&challenge.seed);

    // ---- Klauseln normalisieren: Literal l = 2*var + pol (pol 1 = positiv).
    // Der Generator zieht mit Zuruecklegen: [3,3,7] -> [3,7]; [3,-3,7] ist Tautologie -> weg.
    let m0 = challenge.clauses.len();
    let mut lits: Vec<u32> = Vec::with_capacity(3 * m0);
    let mut co: Vec<u32> = Vec::with_capacity(m0 + 1);
    co.push(0);
    for cl in challenge.clauses.iter() {
        let mut buf = [0u32; 3];
        let mut len = 0usize;
        let mut taut = false;
        for &l in cl {
            let v = l.unsigned_abs() - 1;
            let e = (v << 1) | (l > 0) as u32;
            let mut dup = false;
            for j in 0..len {
                if buf[j] >> 1 == v { if buf[j] != e { taut = true; } dup = true; break; }
            }
            if taut { break; }
            if !dup && len < 3 { buf[len] = e; len += 1; }
        }
        if taut || len == 0 { continue; }
        lits.extend_from_slice(&buf[..len]);
        co.push(lits.len() as u32);
    }
    let m = co.len() - 1;

    // ---- Occurrence-CSR je Literal: occ[ls[l]..ls[l+1]] = Klauseln, die Literal l enthalten.
    let nl = 2 * n;
    let mut ls: Vec<u32> = vec![0; nl + 1];
    for &l in &lits { ls[l as usize + 1] += 1; }
    for l in 0..nl { ls[l + 1] += ls[l]; }
    // 8 Reserveplaetze am Ende: der Zaehlblock unten liest immer 8 Eintraege und maskiert den
    // Ueberhang weg, statt je Eintrag zu springen.
    let mut occ: Vec<u32> = vec![0; lits.len() + BLOCK];
    {
        let mut fill = ls.clone();
        for c in 0..m {
            for j in co[c] as usize..co[c + 1] as usize {
                let l = lits[j] as usize;
                occ[fill[l] as usize] = c as u32;
                fill[l] += 1;
            }
        }
    }

    // ---- Puffer, ueber alle Versuche wiederverwendet
    let mut assign: Vec<u8> = vec![0; n];
    let mut num_true: Vec<u8> = vec![0; m];
    let mut unsat: Vec<u32> = Vec::with_capacity(m / 8 + 16);
    let mut nun: usize = 0;
    let mut init_unsat: usize = 0;
    let mut versuche: u64 = 0;

    // ---- Suche
    let mut p = p0;
    let mut p_thr: u32 = (p * 4294967295.0) as u32;
    let theta_m: u64 = ((theta * m as f64) as u64).max(1);
    let kick_window: u64 = ((kick_window_m * m as f64) as u64).max(theta_m);
    let mut ref_nun: usize = 0;   // Bezug fuer adaptives Rauschen, je Versuch gesetzt
    let mut last_change: u64 = 0;
    let mut best_nun: usize = usize::MAX;  // Bezug fuer Kicks, je Versuch gesetzt
    let mut best_alle: usize = usize::MAX; // bestes unsat ueber ALLE Versuche (nur Statistik)
    let mut last_best: u64 = 0;
    let mut kick_len: u64 = kick_len0;
    let mut kicks_done: u64 = 0;
    let mut restarts_done: u64 = 0;
    let mut best_assign: Vec<u8> = Vec::new();   // Belegung zum besten gesehenen nun
    let mut best_saved = false;
    let mut flips: u64 = 0;
    let mut freebies: u64 = 0;

    debug_assert!(lits.iter().all(|&l| (l as usize) < nl));
    debug_assert!(occ[..lits.len()].iter().all(|&c| (c as usize) < m));
    debug_assert!(m > 0 && occ.len() >= lits.len() + BLOCK);
    debug_assert!(ls.iter().all(|&s| s as usize + BLOCK <= occ.len()));


    // ---- Aeussere Versuchsschleife (WalkSAT/SKC im Original: MAX_TRIES x MAX_FLIPS).
    // Der Loeser lief bisher mit genau EINEM Versuch. Auf n=100000/ratio=4200 bleiben so
    // 4 von 25 Nonces bei best=1..7 stehen und verbrennen 12,5 Mrd. Flips in demselben
    // lokalen Minimum, waehrend das Feld dort 98-100 % loest. Eine frische Zufallsbelegung
    // ist das klassische Gegenmittel: jeder Versuch faellt in ein anderes Becken.
    // try_flips_per_var = 0 -> ein einziger Versuch (Verhalten wie bisher).
    'versuchsschleife: loop {
        // ---- Startbelegung: Bias aus Literalzaehlung, Pure-Literal exakt, dann Unit-Propagation.
        assign.iter_mut().for_each(|a| *a = 0);
        for v in 0..n {
            let nn = (ls[2 * v + 1] - ls[2 * v]) as f64;
            let np = (ls[2 * v + 2] - ls[2 * v + 1]) as f64;
            let p_true = if nn == 0.0 && np > 0.0 { 1.0 }
                else if np == 0.0 && nn > 0.0 { 0.0 }
                else if init_bias { (np + 1.0) / (np + nn + 2.0) }
                else { 0.5 };
            assign[v] = (rng.f64() < p_true) as u8;
        }
        unit_propagate(&lits, &co, &mut assign);

        // ---- Klauselzustand + Unsat-Liste
    
        unsat.clear();
        for c in 0..m {
            let mut t = 0u8;
            for j in co[c] as usize..co[c + 1] as usize {
                let l = lits[j];
                if assign[(l >> 1) as usize] == (l & 1) as u8 { t += 1; }
            }
            num_true[c] = t;
            if t == 0 { unsat.push(c as u32); }
        }
        nun = unsat.len();
        if versuche == 0 { init_unsat = nun; }


        // Suchzustand je Versuch zuruecksetzen
        p = p0; p_thr = (p * 4294967295.0) as u32;
        ref_nun = nun; best_nun = nun;
        last_change = flips; last_best = flips;
        kick_len = kick_len0; best_saved = false;
        let versuch_ende: u64 = if try_flips == 0 {
            u64::MAX
        } else if luby_an {
            flips.saturating_add(try_flips.saturating_mul(luby(versuche)))
        } else {
            flips.saturating_add(try_flips)
        };
        unsafe {
            while nun > 0 && flips < max_flips && flips < versuch_ende {
                let r = rng.next();
                // Unsat-Klausel ziehen; erfuellte Eintraege lazy entfernen.
                let c = loop {
                    let len = unsat.len();
                    debug_assert!(len > 0);
                    let i = idx32(r >> 32, len);
                    let c = *unsat.get_unchecked(i) as usize;
                    if *num_true.get_unchecked(c) == 0 { break c; }
                    unsat.swap_remove(i);
                };
                let cs = *co.get_unchecked(c) as usize;
                let ce = *co.get_unchecked(c + 1) as usize;

                // break-Werte: Klauseln im Vorkommen des aktuell wahren Literals mit num_true==1.
                let mut best = u32::MAX;
                let mut cand = [0u32; 3];
                let mut ncand = 0usize;
                let mut brk = [0u32; 3];
                for j in cs..ce {
                    let l = *lits.get_unchecked(j);
                    let v = l >> 1;
                    let tl = (v << 1) | *assign.get_unchecked(v as usize) as u32;
                    let os = *ls.get_unchecked(tl as usize) as usize;
                    let oe = *ls.get_unchecked(tl as usize + 1) as usize;
                    // Kein vorzeitiger Abbruch und keine Verzweigung im Rumpf: der Fuel-Zaehler
                    // schlaegt je Basisblock zu, deshalb ist die kurze, sprungfreie Schleife
                    // billiger als das Sparen einzelner Durchlaeufe. Die Entscheidungen bleiben
                    // gleich, weil ein abgebrochener Zaehler ohnehin nur "groesser als best" war.
                    // Vorkommenslisten sind im Mittel ~6 Eintraege lang. Ein fester, sprungfreier
                    // 8er-Block mit Maske ist deshalb billiger als Schleife samt Rest-Schleife;
                    // die acht Lesezugriffe liegen dank der Reserveplaetze immer im Array.
                    let mut b = 0u32;
                    for t in 0..BLOCK {
                        let k = os + t;
                        let oc = *occ.get_unchecked(k) as usize;
                        b += ((*num_true.get_unchecked(oc) == 1) & (k < oe)) as u32;
                    }
                    let mut k = os + BLOCK;
                    while k < oe {
                        let oc = *occ.get_unchecked(k) as usize;
                        b += (*num_true.get_unchecked(oc) == 1) as u32;
                        k += 1;
                    }
                    *brk.get_unchecked_mut(j - cs) = b;
                    if b < best { best = b; cand[0] = v; ncand = 1; }
                    else if b == best { *cand.get_unchecked_mut(ncand) = v; ncand += 1; }
                }
                // Auswahlbits: eigener Zug (32 Bit), unabhaengig von Klausel-Index und Rauschen.
                // (Vorher: 16-Bit-Hash von r -> idx32 lieferte immer 0 = kein Zufall in der Auswahl.)
                let r2 = rng.next() >> 32;
                let v = if probsat {
                    // probSAT: Variable j mit Wahrscheinlichkeit ~ cb^-break_j
                    let k = ce - cs;
                    let mut w = [0.0f64; 3];
                    let mut sum = 0.0;
                    for j in 0..k { let x = *pow.get_unchecked((*brk.get_unchecked(j)).min(31) as usize); *w.get_unchecked_mut(j) = x; sum += x; }
                    let mut x = rng.f64() * sum;
                    let mut j = 0usize;
                    while j + 1 < k { x -= *w.get_unchecked(j); if x <= 0.0 { break; } j += 1; }
                    if best == 0 { freebies += 1; }
                    *lits.get_unchecked(cs + j) >> 1
                } else if best == 0 || (r as u32) >= p_thr {
                    if best == 0 { freebies += 1; }
                    *cand.get_unchecked(idx32(r2, ncand))
                } else {
                    *lits.get_unchecked(cs + idx32(r2, ce - cs)) >> 1
                } as usize;

                // Flip: bisher wahres Literal wird falsch (Zaehler runter), das andere wahr (hoch).
                let a = *assign.get_unchecked(v) as u32;
                let tl = ((v as u32) << 1) | a;
                let fl = tl ^ 1;
                let (ts, te) = (*ls.get_unchecked(tl as usize) as usize, *ls.get_unchecked(tl as usize + 1) as usize);
                // Vier Klauseln je Durchlauf; der haeufige Fall (keine Klausel wird leer) kostet
                // dann einen Sprung statt vier. Reihenfolge der Eintraege in unsat bleibt gleich.
                // Fester, sprungfreier 8er-Block: ausserhalb des Bereichs wird derselbe Wert
                // zurueckgeschrieben (unschaedlich, die Reserveplaetze zeigen auf gueltige
                // Klauseln), und die leer gewordenen Klauseln sammelt eine Bitmaske. Im haeufigen
                // Fall (keine leer) kostet der ganze Block genau einen Sprung.
                let mut maske = 0u32;
                for t in 0..BLOCK {
                    let kk = ts + t;
                    let oc = *occ.get_unchecked(kk) as usize;
                    let drin = (kk < te) as u8;
                    let tv = *num_true.get_unchecked(oc) - drin;
                    *num_true.get_unchecked_mut(oc) = tv;
                    maske |= (((tv == 0) & (drin == 1)) as u32) << t;
                }
                if maske != 0 {
                    for t in 0..BLOCK {
                        if maske >> t & 1 == 1 { unsat.push(*occ.get_unchecked(ts + t)); }
                    }
                    nun += maske.count_ones() as usize;
                }
                let mut k = ts + BLOCK;
                while k < te {
                    let oc = *occ.get_unchecked(k) as usize;
                    let t = *num_true.get_unchecked(oc) - 1;
                    *num_true.get_unchecked_mut(oc) = t;
                    if t == 0 { unsat.push(oc as u32); nun += 1; }
                    k += 1;
                }
                let (fs, fe) = (*ls.get_unchecked(fl as usize) as usize, *ls.get_unchecked(fl as usize + 1) as usize);
                // Hier braucht es keinen Sprung: nun sinkt um die Anzahl der zuvor leeren Klauseln.
                let mut nullen = 0u32;
                for t in 0..BLOCK {
                    let kk = fs + t;
                    let oc = *occ.get_unchecked(kk) as usize;
                    let drin = (kk < fe) as u8;
                    let tv = *num_true.get_unchecked(oc);
                    *num_true.get_unchecked_mut(oc) = tv + drin;
                    nullen += ((tv == 0) & (drin == 1)) as u32;
                }
                nun -= nullen as usize;
                let mut k = fs + BLOCK;
                while k < fe {
                    let oc = *occ.get_unchecked(k) as usize;
                    let t = *num_true.get_unchecked(oc);
                    *num_true.get_unchecked_mut(oc) = t + 1;
                    nun -= (t == 0) as usize;
                    k += 1;
                }
                *assign.get_unchecked_mut(v) = (a ^ 1) as u8;
                flips += 1;
                if flips >= next_trace {
                    eprintln!("trace flips={} unsat={} best={} p={:.3} kicks={} restarts={} klen={} v={} bestalle={}", flips, nun, best_nun, p, kicks_done, restarts_done, kick_len, versuche, best_alle);
                    next_trace += trace;
                }

                // Adaptives Rauschen (Hoos 2002)
                if nun < ref_nun {
                    ref_nun = nun; last_change = flips;
                    p -= p * phi * 0.5; p_thr = (p * 4294967295.0) as u32;
                    if nun < best_nun {
                        best_nun = nun; last_best = flips;
                    if nun < best_alle { best_alle = nun; }
                        // Der letzte Kick hat gewirkt -> Laenge zurueck auf den Grundwert.
                        // Gemessen wirkungslos, daher standardmaessig aus (s. oben).
                        if kick_reset { kick_len = kick_len0; }
                        if restart_best && nun <= best_keep {
                            if best_assign.is_empty() { best_assign = assign.clone(); }
                            else { best_assign.copy_from_slice(&assign); }
                            best_saved = true;
                        }
                    }
                } else if flips - last_change > theta_m {
                    ref_nun = nun; last_change = flips;
                    p += (1.0 - p) * phi; p_thr = (p * 4294967295.0) as u32;
                    if unsat.len() > 4 * nun + 4096 {
                        let nt = &num_true;
                        unsat.retain(|&c| nt[c as usize] == 0);
                    }
                    // Stagnations-Kick: lange kein neues Minimum -> eskalierende Zufallsschritte
                    if flips - last_best > kick_window && nun > 0 {
                        // Vom besten bekannten Punkt neu ansetzen statt von der Stelle,
                        // an die der Zufallslauf abgedriftet ist. Buchfuehrung dafuer neu
                        // aufbauen: O(3m), gegenueber kick_window = 40*m Flips vernachlaessigbar.
                        if restart_best && best_saved && nun > best_nun {
                            assign.copy_from_slice(&best_assign);
                            unsat.clear();
                            nun = 0;
                            for c in 0..m {
                                let mut t = 0u8;
                                for j in *co.get_unchecked(c) as usize..*co.get_unchecked(c + 1) as usize {
                                    let l = *lits.get_unchecked(j);
                                    if *assign.get_unchecked((l >> 1) as usize) == (l & 1) as u8 { t += 1; }
                                }
                                *num_true.get_unchecked_mut(c) = t;
                                if t == 0 { unsat.push(c as u32); nun += 1; }
                            }
                            restarts_done += 1;
                        }
                        for _ in 0..kick_len {
                            let r = rng.next();
                            let c = loop {
                                let len = unsat.len();
                                if len == 0 { break usize::MAX; }
                                let i = idx32(r >> 32, len);
                                let c = *unsat.get_unchecked(i) as usize;
                                if *num_true.get_unchecked(c) == 0 { break c; }
                                unsat.swap_remove(i);
                            };
                            if c == usize::MAX { break; }
                            let cs = *co.get_unchecked(c) as usize;
                            let ce = *co.get_unchecked(c + 1) as usize;
                            let v = (*lits.get_unchecked(cs + idx32(r & 0xFFFF_FFFF, ce - cs)) >> 1) as usize;
                            let a = *assign.get_unchecked(v) as u32;
                            let tl = ((v as u32) << 1) | a;
                            let fl = tl ^ 1;
                            for k in *ls.get_unchecked(tl as usize) as usize..*ls.get_unchecked(tl as usize + 1) as usize {
                                let oc = *occ.get_unchecked(k) as usize;
                                let t = *num_true.get_unchecked(oc) - 1;
                                *num_true.get_unchecked_mut(oc) = t;
                                if t == 0 { unsat.push(oc as u32); nun += 1; }
                            }
                            for k in *ls.get_unchecked(fl as usize) as usize..*ls.get_unchecked(fl as usize + 1) as usize {
                                let oc = *occ.get_unchecked(k) as usize;
                                let t = *num_true.get_unchecked(oc);
                                *num_true.get_unchecked_mut(oc) = t + 1;
                                if t == 0 { nun -= 1; }
                            }
                            *assign.get_unchecked_mut(v) = (a ^ 1) as u8;
                            flips += 1;
                        }
                        kicks_done += 1;
                        kick_len = (kick_len * 2).min(kick_max);
                        last_best = flips; ref_nun = nun;
                    }
                }
            }
        }

        versuche += 1;
        if nun == 0 || flips >= max_flips || try_flips == 0 { break 'versuchsschleife; }
    }

    if stats {
        eprintln!("flintwalk: n={} m={} init_unsat={} flips={} unsat={} best={} p={:.3} freebies={} kicks={} restarts={} versuche={} bestalle={}",
                  n, m, init_unsat, flips, nun, best_nun, p, freebies, kicks_done, restarts_done, versuche, best_alle);
    }
    // Immer speichern: die letzte Speicherung zaehlt, und rt-c001 erfasst fuel_consumed nur
    // beim save_solution-Aufruf (Fuel-Messung auch fuer ungeloeste Laeufe).
    if nun == 0 { debug_assert!(num_true.iter().all(|&t| t > 0)); }
    save_solution(&Solution { variables: assign.iter().map(|&a| a == 1).collect() })?;
    Ok(())
}

/// Unit-Propagation ueber Klauseln, die (nach Normalisierung) nur ein Literal haben.
/// Auf Zufalls-3-SAT praktisch nie aktiv, aber billig und korrekt.
fn unit_propagate(lits: &[u32], co: &[u32], assign: &mut [u8]) {
    let m = co.len() - 1;
    let mut forced: Vec<i8> = vec![-1; assign.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for c in 0..m {
            let (cs, ce) = (co[c] as usize, co[c + 1] as usize);
            let mut sat = false;
            let mut open = 0usize;
            let mut last = 0u32;
            for j in cs..ce {
                let l = lits[j];
                let v = (l >> 1) as usize;
                match forced[v] {
                    -1 => { open += 1; last = l; }
                    f => { if f as u32 == (l & 1) { sat = true; break; } }
                }
            }
            if !sat && open == 1 && forced[(last >> 1) as usize] < 0 {
                forced[(last >> 1) as usize] = (last & 1) as i8;
                changed = true;
            }
        }
    }
    for v in 0..assign.len() {
        if forced[v] >= 0 { assign[v] = forced[v] as u8; }
    }
}
