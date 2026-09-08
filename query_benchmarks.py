import psycopg2
from collections import defaultdict

CHALLENGE_NAMES = {
    1000: "SAT",
    2000: "Vehicle Routing",
    3000: "Knapsack",
    4000: "Vector Search",
    5000: "Hypergraph",
    6000: "Neural Net",
    7000: "Scheduling",
}

conn = psycopg2.connect(
    host="qlj1we4j5n.d4fbgnw7hn.tsdb.cloud.timescale.com",
    database="tsdb",
    user="readonly",
    password="aebdbd2a72388b6b8c5f4f022470ad33",
    port=30101,
    sslmode="require"
)

with conn.cursor() as cur:
    cur.execute("""
    SELECT
        A.player_id::TEXT,
        A.challenge_id,
        A.track_id,
        COUNT(*) AS n,
        COUNT(*) FILTER (WHERE B.stopped) AS n_stopped,
        COUNT(*) FILTER (WHERE C.benchmark_id IS NOT NULL) AS n_fraud  
    FROM precommit A
    INNER JOIN benchmark B
        ON A.benchmark_id = B.id
    LEFT JOIN fraud C
        ON A.benchmark_id = C.benchmark_id
    WHERE A.block_started >= 1086891 - 120
    GROUP BY A.player_id, A.challenge_id, A.track_id;
    """)
    results = cur.fetchall()

def short_id(hex_str):
    clean = hex_str.replace("\\x", "")
    return clean[:8] + "…"

players = defaultdict(lambda: defaultdict(list))
for player_id, challenge_id, track_id, n, n_stopped, n_fraud in results:
    players[player_id][(challenge_id, track_id)] = (n, n_stopped, n_fraud)

sorted_players = sorted(players.keys())

for player_id in sorted_players:
    tracks = players[player_id]
    total = sum(v[0] for v in tracks.values())
    total_stopped = sum(v[1] for v in tracks.values())
    total_fraud = sum(v[2] for v in tracks.values())

    print(f"\n{'='*80}")
    print(f"  Player: {short_id(player_id)}    Total: {total} benchmarks, {total_stopped} stopped, {total_fraud} fraud")
    print(f"{'='*80}")

    by_challenge = defaultdict(list)
    for (cid, track), (n, ns, nf) in tracks.items():
        by_challenge[cid].append((track, n, ns, nf))

    for cid in sorted(by_challenge.keys()):
        cname = CHALLENGE_NAMES.get(cid, f"Challenge {cid}")
        entries = sorted(by_challenge[cid], key=lambda x: -x[1])
        subtotal = sum(e[1] for e in entries)
        sub_stopped = sum(e[2] for e in entries)
        sub_fraud = sum(e[3] for e in entries)

        print(f"\n  {cname} ({cid})  —  {subtotal} total, {sub_stopped} stopped, {sub_fraud} fraud")
        print(f"  {'─'*60}")
        print(f"  {'Track':<35} {'Count':>6} {'Stopped':>8} {'Fraud':>6}")
        print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*6}")
        for track, n, ns, nf in entries:
            flag = ""
            if nf > 0:
                flag = " ⚠"
            if ns > 0 and ns == n:
                flag += " [all stopped]"
            elif ns > 0:
                flag += f" [{ns} stopped]"
            print(f"  {track:<35} {n:>6} {ns:>8} {nf:>6}{flag}")

print(f"\n{'='*80}")
print(f"  SUMMARY: {len(sorted_players)} players, {sum(sum(v[0] for v in p.values()) for p in players.values())} total benchmarks")
print(f"{'='*80}")




