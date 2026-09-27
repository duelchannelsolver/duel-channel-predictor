# Duel Channel Predictor

Static site: paste a Duel Channel screenshot, it tries to detect the enemies
on each side, then predicts the win probability using a logistic-regression
strength model trained (in-browser, on page load) from `data/matches.json`.

## Setup

1. Run `download_duel_channel_sprites.py` (from earlier) and copy the
   downloaded `.png` files into `sprites/` here, named exactly as they
   appear in your training data (e.g. `sprites/Big Snowball Thrower.png`).
   Only enemies that appear in `data/matches.json` are used as templates --
   the model can't usefully weigh in on an enemy it hasn't seen a match for
   anyway.
2. Whenever you add more logged matches, regenerate `data/matches.json`
   from your Excel workbook (reuse the `_parse_side`/loader logic from
   `duel_channel_model.py`) and commit the updated file.
3. Push this folder to a GitHub repo and enable GitHub Pages (Settings ->
   Pages -> deploy from branch). No build step needed -- it's static
   HTML/JS.

## How it works

- **Model training** happens client-side, on every page load, straight from
  `data/matches.json`. With ~100 rows and a few dozen enemies this takes
  well under a second, so there's no need to precompute or cache weights.
- **Enemy detection** slides a template-match window across a wide search
  band on each side (not a fixed slot box) and keeps local best matches via
  non-max suppression, since icon count and position shift depending on how
  many enemies are in a round -- a fixed box only worked for the exact
  screenshot it was measured from. Each kept detection's "xN" badge (just
  below the icon) is read with Tesseract.js (loaded from a CDN, so the page
  needs internet access on first load).

## Known limitations (read before trusting this)

- **Search bands, icon size, and OCR crop are estimated from ~3 example
  screenshots**, not a large verified set. If detection still misses
  icons, widen the left/right band boxes or adjust "icon size %"; if it
  finds false positives (e.g. matching the round-number fire graphic),
  narrow the bands or raise the match threshold.
- **Performance**: sliding a window across a band at every stride, against
  every known enemy template, is O(positions x templates) -- fine for ~100
  enemies and a modest band, but if it feels slow, shrink the band size or
  increase the stride divisor in `detectSide()` (currently `iconSize / 3`).
- **Sprite art vs. queue-icon art.** The wiki sprites should be a closer
  match to these queue icons than to in-motion battlefield sprites, but
  match quality is still unverified against a broad set of screenshots --
  treat detections as suggestions to confirm/edit, not ground truth.
- **No held-out evaluation wired in yet.** The page reports how many
  matches it trained on, not how well it's predicting -- keep evaluating
  model quality offline (with `evaluate()` from `duel_channel_model.py`)
  as your dataset grows.
