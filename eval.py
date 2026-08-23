"""Does the style space actually recommend well?

Everything in winemap.py was tuned against internal proxies -- variance
explained, taster share, same-colour neighbour rate. None of those measure the
product claim, which is "if you like X you'll like Y". This asks that directly:
for pairs that wine writing already treats as settled substitutions, where does
the model rank the intended partner among all 510 other cells?

Run:  python3 eval.py        (add `demo` for the self-check)
"""
import sys

import numpy as np

from winemap import MIN_REVIEWS, N_DIMS, cells, descriptor_vocab, load, project

SHOWN = 6  # how many neighbours the map actually offers a user

# (query cell) -> (cell a wine list would offer as the substitute).
# Written against real (variety, province) cells: "Cabernet Sauvignon/Bordeaux"
# looks obvious and does not exist, because Bordeaux is cellled as a blend.
PAIRS = [
    # Same grape, different place: does the space cross regions?
    (("Sauvignon Blanc", "Loire Valley"), ("Sauvignon Blanc", "Marlborough")),
    (("Riesling", "Mosel"), ("Riesling", "Alsace")),
    (("Riesling", "Mosel"), ("Riesling", "South Australia")),
    (("Chardonnay", "Burgundy"), ("Chardonnay", "California")),
    (("Chenin Blanc", "Loire Valley"), ("Chenin Blanc", "Stellenbosch")),
    (("Pinot Noir", "Burgundy"), ("Pinot Noir", "Oregon")),
    (("Malbec", "Mendoza Province"), ("Malbec", "Southwest France")),
    (("Syrah", "Rhône Valley"), ("Syrah", "Washington")),
    (("Viognier", "Rhône Valley"), ("Viognier", "California")),
    (("Vermentino", "Sicily & Sardinia"), ("Vermentino", "Tuscany")),
    (("Merlot", "Bordeaux"), ("Merlot", "Washington")),
    # Same grape under another name -- the model has never been told they match.
    (("Syrah", "Rhône Valley"), ("Shiraz", "South Australia")),
    (("Grenache", "South Australia"), ("Garnacha", "Northern Spain")),
    (("Primitivo", "Southern Italy"), ("Zinfandel", "California")),
    (("Albariño", "Galicia"), ("Alvarinho", "Vinho Verde")),
    (("Pinot Grigio", "Northeastern Italy"), ("Pinot Gris", "Alsace")),
    (("Monastrell", "Levante"), ("Mourvèdre", "California")),
    # Genuinely different grapes that drinkers swap for one another.
    (("Nebbiolo", "Piedmont"), ("Sangiovese", "Tuscany")),
    (("Aglianico", "Southern Italy"), ("Nebbiolo", "Piedmont")),
    (("Nero d'Avola", "Sicily & Sardinia"), ("Primitivo", "Southern Italy")),
    (("Cabernet Sauvignon", "California"), ("Bordeaux-style Red Blend", "Bordeaux")),
    (("Carmenère", "Maipo Valley"), ("Merlot", "Bordeaux")),
    (("Tempranillo", "Northern Spain"), ("Touriga Nacional", "Douro")),
    (("Mencía", "Northern Spain"), ("Gamay", "Beaujolais")),
    (("Grüner Veltliner", "Wachau"), ("Riesling", "Wachau")),
    (("Verdejo", "Northern Spain"), ("Sauvignon Blanc", "Loire Valley")),
    (("Champagne Blend", "Champagne"), ("Sparkling Blend", "Catalonia")),
]


def rank_of(score, query, target):
    """Rank of target when candidates are sorted by score, best (lowest) first.

    Ties are averaged rather than broken arbitrarily, so the metadata baselines
    -- which score every same-grape cell identically -- get the rank they
    actually deserve instead of one flattered by list order.
    """
    s = np.delete(score, query)
    t = score[target]
    better = int((s < t).sum())
    tied = int((s == t).sum()) - 1          # others equal to it, excluding itself
    return better + 1 + tied / 2


def report(name, ranks, n):
    ranks = np.array(ranks, dtype=float)
    print(f"  {name:22} median rank {np.median(ranks):6.1f} / {n}"
          f"   recall@{SHOWN} {(ranks <= SHOWN).mean():5.1%}"
          f"   MRR {(1 / ranks).mean():.3f}")


def main():
    df = load([f"data/{s}.parquet" for s in ("train", "test", "validation")])
    c = cells(df, MIN_REVIEWS)
    coords, _ = project(c["doc"].to_list(), descriptor_vocab(df), n_components=N_DIMS)
    idx = {(v, p): i for i, (v, p) in enumerate(zip(c["variety"], c["province"]))}

    missing = [q for pair in PAIRS for q in pair if q not in idx]
    if missing:
        sys.exit(f"pairs reference cells that do not exist: {missing}")

    variety = np.array(c["variety"])
    province = np.array(c["province"])
    n = len(idx)
    rng = np.random.default_rng(0)

    got = {"model": [], "same grape": [], "same region": [], "random": []}
    cross = []
    for (qv, qp), (tv, tp) in PAIRS:
        q, t = idx[(qv, qp)], idx[(tv, tp)]
        scored = {
            "model": np.linalg.norm(coords - coords[q], axis=1),
            "same grape": (variety != variety[q]).astype(float),
            "same region": (province != province[q]).astype(float),
            "random": rng.random(n),
        }
        for k, s in scored.items():
            got[k].append(rank_of(s, q, t))
        if qv != tv:
            cross.append(got["model"][-1])

    print(f"{len(PAIRS)} pairs over {n} cells, neighbours in {N_DIMS}-D\n")
    for k in got:
        report(k, got[k], n)
    print()
    report(f"model, {len(cross)} cross-grape", cross, n)

    worst = sorted(zip(got["model"], PAIRS))[-5:]
    print("\n  worst misses:")
    for r, ((qv, qp), (tv, tp)) in worst:
        print(f"    {r:5.0f}  {qv} ({qp})  ->  {tv} ({tp})")


def demo():
    """rank_of: best score ranks 1, and ties average instead of guessing."""
    s = np.array([0.0, 5.0, 1.0, 2.0])          # query 0; target 2 is closest
    assert rank_of(s, 0, 2) == 1, rank_of(s, 0, 2)
    assert rank_of(s, 0, 1) == 3, rank_of(s, 0, 1)
    flat = np.array([0.0, 1.0, 1.0, 1.0])        # query 0, all others tied
    assert rank_of(flat, 0, 1) == 2, rank_of(flat, 0, 1)
    print("demo ok")


if __name__ == "__main__":
    demo() if sys.argv[1:2] == ["demo"] else main()
