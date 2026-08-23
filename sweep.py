"""Sweep the pipeline's knobs against the pair-ranking eval.

Each knob in winemap.py was set by judgement plus internal proxies. This scores
them on the only measure that reflects the product: where a known substitute
lands among all other cells.

Run:  python3 sweep.py
"""
import numpy as np

from eval import PAIRS, SHOWN, rank_of
from winemap import cells, load, place_names, project, term_concentration

MAX_DIMS = 50


def score(coords, idx, dims):
    ranks = []
    for (qv, qp), (tv, tp) in PAIRS:
        if (qv, qp) not in idx or (tv, tp) not in idx:
            continue
        q, t = idx[(qv, qp)], idx[(tv, tp)]
        d = np.linalg.norm(coords[:, :dims] - coords[q, :dims], axis=1)
        ranks.append(rank_of(d, q, t))
    r = np.array(ranks, dtype=float)
    return np.median(r), (r <= SHOWN).mean(), (1 / r).mean(), len(r)


def pct(median, n_cells):
    """Median rank as a share of the candidate pool.

    Raw rank is not comparable across settings that change the number of cells:
    36th of 287 is a worse result than 36th of 511, not a better one.
    """
    return median / n_cells


def run(df, terms, share, max_share, min_reviews, dims_list):
    vocab = sorted(terms[share <= max_share])
    c = cells(df, min_reviews)
    top = min(MAX_DIMS, len(vocab) - 1, c.height - 1)
    if top < max(dims_list):
        dims_list = [d for d in dims_list if d <= top]
    if not dims_list:
        return
    coords, _ = project(c["doc"].to_list(), vocab, n_components=top)
    idx = {(v, p): i for i, (v, p) in enumerate(zip(c["variety"], c["province"]))}
    for dims in dims_list:
        med, rec, mrr, n = score(coords, idx, dims)
        print(f"  share {max_share:<4} minrev {min_reviews:<3} dims {dims:<3}"
              f" cells {c.height:<4} vocab {len(vocab):<5}"
              f" median {med:6.1f} ({pct(med, c.height):5.1%})"
              f"  recall@{SHOWN} {rec:5.1%}  MRR {mrr:.3f}  (n={n})")


def main():
    df = load([f"data/{s}.parquet" for s in ("train", "test", "validation")])
    terms, share = term_concentration(df)
    print(f"{len(terms)} candidate terms, {len(PAIRS)} pairs\n")

    print("dims (share 0.30, minrev 40) -- how much of the space to use:")
    run(df, terms, share, 0.30, 40, [2, 5, 10, 15, 20, 30, 50])

    print("\nvocabulary concentration threshold (dims 15, minrev 40):")
    for ms in (0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00):
        run(df, terms, share, ms, 40, [15])

    print("\ncell size (share 0.30, dims 15) -- fewer, fatter cells or more, thinner:")
    for mr in (20, 30, 40, 60, 100):
        run(df, terms, share, 0.30, mr, [15])

    # Place names were stripped on the theory they are noise. The eval can test
    # that: keeping them is a strictly larger vocabulary.
    print("\nkeeping place names in the vocabulary (share 0.30, dims 15, minrev 40):")
    keep = set(place_names(df))
    print(f"  ({sum(t in keep for t in terms)} of the candidate terms are place names,"
          f" already excluded upstream)")


if __name__ == "__main__":
    main()
