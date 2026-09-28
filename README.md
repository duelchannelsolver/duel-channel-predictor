# Duel Channel Predictor

Static site: paste a Duel Channel screenshot, it tries to detect the enemies
on each side, then predicts the win probability using a logistic-regression
strength model trained (in-browser, on page load) from `data/matches.json`.

## Setup

1. Run `download_duel_channel_sprites.py`, then move the downloaded PNGs into
   a folder called `sprites_full/` here (it's git-ignored, so the big files
   never get committed).
2. Run `pip install pillow` then `python make_small_sprites.py`. This writes
   64x64 copies into `sprites/`, which is what the site actually loads and
   what you commit. Full-size sprites made the page download and decode far
   more data than the 24x24 comparison ever uses.
   Sprite filenames must match the enemy names in your training data
   exactly (e.g. `sprites/Big Snowball Thrower.png`). Only enemies present
   in `data/matches.json` are used as templates.
3. Whenever you add more logged matches, regenerate `data/matches.json`
   from your Excel workbook and commit the updated file.
4. Push this folder to a GitHub repo and enable GitHub Pages (Settings ->
   Pages -> deploy from branch). No build step needed.

## Performance notes

- Sprite templates load in parallel and are cached; the OCR engine is one
  shared worker. Both start warming up as soon as the page loads.
- Open the browser console (F12) after pasting a screenshot: it logs how
  many templates loaded (`templates loaded: X of Y` -- if X is much smaller
  than Y, sprite filenames don't match your enemy names) and total detection
  time.

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

- **Matching now uses color, mean-centered, with border cropping** to fix a
  specific failure mode: raw-grayscale cosine similarity was dominated by
  the circular border every queue icon shares (which the wiki sprite
  templates don't have), causing very different-looking enemies to all
  match the same wrong template. If misclassifications persist, try
  adjusting the `innerCropFrac` values in `detect.js` (currently 0.6 for
  candidate windows, 0.85 for templates) or increasing `THUMB` (currently
  24px) for more detail at the cost of speed.
- **Search bands, icon size, and OCR crop are estimated from a handful of
  example screenshots**, not a large verified set. If detection still
  misses icons, widen the left/right band boxes or adjust "icon size %"; if
  it finds false positives, narrow the bands or raise the match threshold.
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
