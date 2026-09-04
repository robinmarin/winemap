# CLAUDE.md

Working notes for this repo. Read before changing the pipeline or the map.

## What this is

218k Wine Enthusiast reviews → 811 (variety, province) style cells → TF-IDF →
PCA → nearest neighbours. `map.html` draws it and lists recommendations.

```
python3 winemap.py          # build map.json   (~30s)
python3 eval.py             # score it
python3 sweep.py            # sweep knobs against eval.py
python3 winemap.py demo     # self-checks (also: eval.py demo)
./serve.sh                  # map.html needs HTTP, not file:// (frees :8777 first)
```

`data/*.parquet` and `map.json` are generated and untracked — see README for
the download command.

## Change nothing about the pipeline without running eval.py

Every knob here (`MAX_SHARE`, `N_DIMS`, `MIN_REVIEWS`) was originally set by
judgement plus internal proxies — variance explained, taster share,
same-colour neighbour rate. **Two of those judgements were later shown wrong by
the eval**, including a confident "the curve is flat above 10 dimensions" that
came from a proxy blind to the effect. Internal metrics are for diagnosing;
`eval.py` is for deciding.

When tuning, remember the eval has 27 pairs. One pair is 3.7 points of
recall@6. Prefer settings that sit on a plateau with a monotone gradient
either side over the single best cell, and record when a result is in-sample.

## Traps already fallen into

- **PCA, not TruncatedSVD.** Uncentered SVD burns its first component on the
  corpus mean direction — same for every wine, 0.5% variance, junk loadings.
- **Reviewer voice looks like regional style.** Each region has a regular
  taster; their tics (`doles`, `luminous`) once took 92% of PC3. Handled by
  `term_concentration` / `MAX_SHARE`. Do not remove this filter without
  re-running the eval.
- **Taster η² overstates leakage.** Tasters are assigned by region, and region
  genuinely drives style. The two are not separable with this data. Do not
  quote η² as if it were pure contamination.
- **Rank is not comparable across pool sizes.** 36th of 287 is worse than 36th
  of 511. `sweep.py` reports a percentage for this reason.
- **Place names were not the problem.** Stripping 2,227 geography tokens
  changed the taster contamination by nothing. Verify before attributing.

## The map is plain SVG on purpose

`map.html` has no dependencies and draws each mark as an `<svg:circle>` with an
ordinary DOM `onclick`. This is deliberate and hard-won:

- A hand-rolled 3D orbit view was built and reverted. Auto-spin fought the
  user, and unthrottled re-renders fell behind the event stream.
- Plotly `scatter3d` was tried and reverted. 3.5MB, and clicks on its WebGL
  scene resolve through a pick buffer that proved unreliable — translucent
  markers degrade it further.
- PC3 is shown as a **second linked 2D panel**, not a third axis. It holds 4.6%
  of the variance; it does not justify a rotating cube.

If a plot change is needed, prefer SVG and DOM events. Do not reach for a 3D
library without a reason the two panels cannot serve.

## Do not debug the browser through automation

Synthetic mouse events drive SVG fine but drive WebGL scenes unreliably. During
the Plotly work, identical clicks produced contradictory probe results, and
several "bugs" chased on that basis did not exist. Read the code first. If the
page genuinely misbehaves, ask the user for the console output rather than
inferring from screenshots — they can see it and you cannot.

Also: `python3 -m http.server` reads from disk per request and never needs
restarting after an edit. A stale page is the browser cache — hard reload.

## Conventions

- Polars, not pandas.
- Non-trivial logic leaves one runnable assert-based `demo()` behind. The one
  in `eval.py` caught a real tie-averaging bug on its first run.
- Comments explain *why*, especially where a simpler-looking option was
  rejected for a measured reason.
- No `Co-Authored-By` trailers in commits.
