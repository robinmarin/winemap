# TODO

Roughly in order of how much they would change what we believe.

## Make the eval honest

- [ ] **Held-out pairs.** ~80 configurations were scored against 27 pairs and
      the winner kept, so the current numbers are in-sample. Write ~15 fresh
      pairs *without* looking at the model's output, then score the chosen
      config against them exactly once. Until then the improvement is fitted,
      not measured.
- [ ] **More pairs generally.** 27 is small enough that recall@6 moves in steps
      of 3.7 percentage points — one pair. Differences smaller than that are
      noise and should not be used to pick anything.
- [ ] **Pairs that should *not* match.** Only positives are tested now. A few
      known non-substitutions (Mosel Riesling → Barolo) would catch a space
      that ranks everything close to everything.

## Make the recommendations better

- [ ] **Chase the worst misses.** `Tempranillo (Northern Spain) → Touriga
      Nacional (Douro)` ranks 432/811, by far the worst. Iberian reds sharing
      almost no vocabulary suggests national language, not style, still drives
      those cells. Read the actual documents before theorising.
- [ ] **Same grape, different border.** `Albariño (Galicia) → Alvarinho (Vinho
      Verde)` ranks 186. Literally the same grape. This is the clearest single
      diagnostic of leftover regional vocabulary.
- [ ] **Residualise region out of the vocabulary.** Subtract each province's
      mean TF-IDF vector before projecting. Risky — region genuinely does drive
      style, so this may remove signal along with noise. The eval can referee.
- [ ] **CellarTracker ratings** (McAuley/UCSD, ~2M reviews with user ids). The
      only route to real "people who liked X also liked Y" instead of "gets
      described similarly". Check the research-use terms first.

## Data

- [ ] **Vintage.** Recoverable for ~116k of 218k reviews (`title` is null on
      45%). Not a style axis — treat as a quality/ripeness modifier, e.g. a
      per-cell "good years" annotation, not another PCA dimension.
- [ ] **A second critic.** Everything here is one publication's palate.

## Interface

- [ ] Redraw on OS theme change — `css()` reads the variables once at load, so
      switching light/dark needs a reload.
- [ ] Keep the sidebar's numbers derived from `map.json`, never typed. The cell
      count and the variance caveat both read from the data now; a hardcoded
      "511" survived two rounds of tuning before anyone noticed.
- [ ] Consider showing *why* two wines are neighbours: the shared terms that
      pulled them together would make a recommendation arguable rather than
      oracular.

## Deliberately not doing

- **3D rotating plot.** Tried it, reverted. PC3 holds 4.6% of the variance and
  the cube cost far more in interaction bugs than it returned in insight. Two
  linked 2D panels show the same thing and stay clickable. See CLAUDE.md.
- **A charting library.** Plotly worked but was 3.5MB and its WebGL scene made
  clicks unreliable. Plain SVG circles with DOM handlers have never once
  misbehaved.
