"""Build a 2D style map of wines from Wine Enthusiast reviews.

Each point is a (variety, province) pair. Its tasting notes are pooled into one
document, TF-IDF'd, and projected to 2D. Neighbours in that space are the
"if you like X, try Y" recommendations.
"""
import json
import re
import sys

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

MIN_REVIEWS = 40  # per (variety, province) cell
# ponytail: 15 dims picked off a flat curve -- same-colour neighbour rate sits at
# ~94% from 2 dims to 80, while same-province rises 6% -> 30%. Above ~10 dims
# nothing moves. Revisit only with a real relevance measure (user ratings).
N_DIMS = 15
N_NEIGHBOURS = 6

# Words that identify the wine rather than describe it -- they'd let the model
# cluster on names instead of style.
NOISE = """wine wines drink now flavors flavor aromas aroma palate nose finish
notes note offers shows character bit like just made vineyard winery estate
bottling cuvee blend varietal vintage years year ago best value""".split()


# Grape colour, typed by hand. Also serves as ground truth: if reds and whites
# don't separate on the first component, the projection is wrong.
WHITE = set("""Albariño|Alsace white blend|Alvarinho|Arneis|Assyrtico|Bordeaux-style White Blend|
Carricante|Catarratto|Chardonnay|Chenin Blanc|Cortese|Encruzado|Falanghina|Fiano|Friulano|Fumé Blanc|
Garganega|Gewürztraminer|Godello|Greco|Grenache Blanc|Grillo|Gros and Petit Manseng|Grüner Veltliner|
Insolia|Inzolia|Loureiro|Marsanne|Melon|Moscato|Moschofilero|Muscat|Muscat Canelli|Pecorino|
Petit Manseng|Picolit|Pinot Bianco|Pinot Blanc|Pinot Grigio|Pinot Gris|Portuguese White|
Rhône-style White Blend|Ribolla Gialla|Riesling|Roussanne|Sauvignon|Sauvignon Blanc|
Semillon-Sauvignon Blanc|Silvaner|Sylvaner|Sémillon|Tokaji|Torrontés|Trebbiano|Turbiana|Verdejo|
Verdelho|Verdicchio|Vermentino|Vernaccia|Viognier|Viura|White Blend|Xarel-lo|Zibibbo""".replace("\n", "").split("|"))

# Sparkling, rosé and fortified -- their style is driven by the process, not the grape.
OTHER = set("""Champagne Blend|Glera|Prosecco|Sparkling Blend|Lambrusco|Rosado|Rosé|Port|Sherry|
Pedro Ximénez""".replace("\n", "").split("|"))


def colour(variety):
    return "other" if variety in OTHER else "white" if variety in WHITE else "red"


def load(paths):
    df = pl.concat([pl.read_parquet(p) for p in paths]).unique(["title", "description"])
    return df.with_columns(
        pl.col("title").str.extract(r"\b(19[5-9]\d|20[0-2]\d)\b").cast(pl.Int32).alias("vintage")
    ).drop_nulls(["variety", "province", "description"])


def descriptor_vocab(df, max_share=0.30, min_df=5):
    """Keep only words that many reviewers use.

    Wine Enthusiast assigns each region a regular taster, so their personal tics
    ("doles", "luminous") look like regional style to PCA -- reviewer voice took
    92% of PC3's spread before this filter. A real descriptor is used by everyone;
    a tic is used by one person. So: pool one document per taster and drop any term
    whose usage is concentrated in a single one.
    """
    stop = NOISE + list(TfidfVectorizer(stop_words="english").get_stop_words()) + place_names(df)
    per_taster = (
        df.drop_nulls("taster_name")
        .group_by("taster_name")
        .agg(pl.col("description").str.join(" ").alias("doc"))
    )
    cv = CountVectorizer(stop_words=stop, min_df=min_df, token_pattern=r"(?u)\b[a-zà-ÿ]{3,}\b")
    m = cv.fit_transform(per_taster["doc"].to_list()).toarray().astype(float)
    m /= m.sum(1, keepdims=True)  # tasters wrote wildly different volumes
    share = m.max(0) / m.sum(0)   # 1.0 == only one taster ever writes this word
    return sorted(np.array(cv.get_feature_names_out())[share <= max_share])


def place_names(df):
    """Geography and grape names, harvested from the metadata columns."""
    out = set()
    for col in ("country", "province", "region_1", "region_2", "variety", "taster_name"):
        if col not in df.columns:
            continue
        for s in df[col].drop_nulls().unique().to_list():
            out |= {t for t in re.split(r"[^a-zà-ÿ]+", s.lower()) if len(t) > 2}
    return sorted(out)


def cells(df, min_reviews=MIN_REVIEWS):
    """Pool reviews into (variety, province) documents."""
    return (
        df.group_by("variety", "province")
        .agg(
            pl.len().alias("n"),
            pl.col("description").str.join(" ").alias("doc"),
            pl.col("points").mean().alias("points"),
            pl.col("price").median().alias("price"),
            pl.col("country").first(),
        )
        .filter(pl.col("n") >= min_reviews)
        .sort("n", descending=True)
    )


def project(docs, vocab, n_components=2):
    """TF-IDF -> PCA. Returns coords and the top words at each end of each axis.

    PCA, not TruncatedSVD: uncentered SVD spends its first component on the
    corpus mean direction, which is the same for every wine and tells us nothing.
    """
    tfidf = TfidfVectorizer(vocabulary=vocab, sublinear_tf=True)
    x = tfidf.fit_transform(docs).toarray()  # few hundred docs, dense is fine
    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(x)
    words = tfidf.get_feature_names_out()
    axes = [
        {
            "negative": [words[i] for i in comp.argsort()[:8]],
            "positive": [words[i] for i in comp.argsort()[-8:][::-1]],
            "variance": round(float(v), 4),
        }
        for comp, v in zip(pca.components_, pca.explained_variance_ratio_)
    ]
    return coords, axes


def build(paths, out="map.json"):
    df = load(paths)
    c = cells(df)
    coords, axes = project(c["doc"].to_list(), descriptor_vocab(df), n_components=N_DIMS)

    # Neighbours come from the full 50-D space, not the 2 plotted dimensions --
    # PC1+PC2 hold ~15% of the variance, so 2-D distance is a lossy shadow.
    nn = NearestNeighbors(n_neighbors=N_NEIGHBOURS + 1).fit(coords)
    _, idx = nn.kneighbors(coords)

    points = [
        {
            "variety": r["variety"],
            "colour": colour(r["variety"]),
            "province": r["province"],
            "country": r["country"],
            "n": r["n"],
            "points": round(r["points"], 1),
            "price": None if r["price"] is None else round(r["price"]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "z": round(float(coords[i, 2]), 4),
            "near": [int(j) for j in idx[i, 1:]],
        }
        for i, r in enumerate(c.iter_rows(named=True))
    ]
    with open(out, "w") as f:
        json.dump({"points": points, "axes": axes}, f)
    print(f"{len(points)} cells from {df.height} reviews -> {out}")
    for n, a in enumerate(axes[:3]):
        print(f"  PC{n + 1} ({a['variance']:.1%}): {', '.join(a['positive'][:5])}"
              f"  <->  {', '.join(a['negative'][:5])}")
    return points


def demo():
    """Self-check on synthetic reviews: two obvious styles must separate."""
    red = "tannins structured dark cherry oak firm bold"
    white = "crisp citrus lemon fresh zesty mineral acidity"
    df = pl.DataFrame({
        "variety": ["R"] * 50 + ["W"] * 50 + ["R2"] * 50,
        "province": ["A"] * 150,
        "description": [red] * 50 + [white] * 50 + [red] * 50,
        "taster_name": ["t1"] * 75 + ["t2"] * 75,
        "points": [90] * 150, "price": [20.0] * 150, "country": ["X"] * 150,
    })
    c = cells(df, min_reviews=10)
    assert c.height == 3, c.height
    coords, axes = project(c["doc"].to_list(), descriptor_vocab(df, max_share=1.0, min_df=1))
    order = {v: i for i, v in enumerate(c["variety"])}
    import numpy as np
    d = lambda a, b: float(np.linalg.norm(coords[order[a]] - coords[order[b]]))
    assert d("R", "R2") < d("R", "W"), (d("R", "R2"), d("R", "W"))
    assert len(axes) == 2
    print("demo ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        build(sys.argv[1:] or [f"data/{s}.parquet" for s in ("train", "test", "validation")])
