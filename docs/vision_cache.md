# The Vision Cache

`bird_crop.py`'s crop cache (`cache/crops/`) is PeakPic's shared Computer
Vision infrastructure layer: the one place every CV consumer - AI-model
training, Classic Vision (SuperAnimal-Bird and EyePose-v0 both), and any
future vision module (species classification, sharpness analysis,
composition, feather detail, a future head-detection model, ...) - reads
image data from, so a RAW is decoded and its subject detected **once**, ever,
no matter how many times or how many different modules later need that same
image.

This document records the investigation that led to it, the real numbers
behind the defaults, and how it differs from the (deliberately separate)
Preview Cache.

## Background: what changed and why

Before this, the cache had a hardcoded `max_side=1024` cap and was stored as
lossless PNG - both chosen for AI-model training throughput and disk
footprint, never re-evaluated once the same cache became the direct input to
Classic Vision's eye detectors *and* its sharpness metrics.

A real audit of this project's own cache (74,450 crops, measured directly -
see the investigation this document accompanies) found:

- **71.5% of crops were actually being downscaled** by the 1024px cap -
  not an edge case, the majority.
- Among those, the average pre-cap resolution was **3175px** (a **3.1×**
  downscale, ~9.6× fewer pixels than were available), with some crops
  downscaled as much as **8.1×** (64× fewer pixels).
- The eye-sharpness metric (`ranking.metrics.region_focus_measure`, Classic
  Vision's default 50%-weighted score component) reads the cached crop
  **directly**, then further resamples an ~8%-of-crop eye region to a fixed
  128px canonical size. On a capped crop, that eye region was often smaller
  than 128px - meaning the "resample to canonical size" step was an
  *upscale* (pure interpolation, no new information) instead of a genuine
  high-resolution downscale (real detail) - precisely on the best-composed,
  frame-filling shots.

Training was unaffected by resolution loss in the same way: `RawImageLoader`
already does its own resize-to-`output_size` at load time
(`raw_io.RawImageLoader._letterbox`), independent of whatever the cache
stores - it was already designed the way this whole cache now is.

## Design

```
Original RAW/JPEG
  -> decode once (rawpy.postprocess, full resolution, no half_size)
  -> detect once (BirdDetector, full frame)
  -> crop (true sub-rectangle, no distortion)
  -> Vision Cache (JPEG q98 by default, ORIGINAL crop resolution by default)
  -> every consumer reads from here, resizing further only if and how IT needs to
```

- **`CropParams.max_side: int | None = None`** - no cap by default. The
  crop is cached at its own resolution. Set an explicit pixel value to trade
  detail for disk space (e.g. for a disk-constrained machine or a quick
  experiment); the capping code path (`bird_crop.downscale_long_side`) is
  unchanged, only the default changed.
- **`CropParams.image_format: str = "jpeg"`**, **`jpeg_quality: int = 98`** -
  JPEG instead of lossless PNG. Measured on this project's own real cache:
  JPEG q98 runs about **3× smaller than PNG** for the same pixels, and a
  measured mean-absolute round-trip error under ~4/255 on structured
  synthetic content (see `tests/test_bird_crop.py`'s
  `ConfigurableFormatAndQualityTests`) - visually and numerically close to
  lossless. `image_format="png"` remains available for a use case that
  genuinely wants zero compression loss and can afford the disk cost.
- **A consumer that wants a smaller input still resizes itself, at load
  time** - this was never Classic Vision's job and isn't a new pattern:
  `RawImageLoader` already worked this way. Nothing downstream needed to
  change to benefit from the cache no longer being a resolution ceiling.

### Real disk-size math (measured, not guessed)

| | Current (1024px PNG) | Vision Cache (uncapped, JPEG q98) |
| --- | --- | --- |
| Avg crop size | ~1 MB | median ~1.05 MB, mean ~3.3 MB (skewed by a long tail of very large source frames) |
| 74,450 crops, actual/estimated | 71.1 GB (real, on disk) | ~78 GB (median-based) to ~245 GB (mean-based) |
| 20,000 images | ~19 GB | ~21-65 GB |
| 100,000 images | ~97 GB | ~105-326 GB |

JPEG's ~3× better compression than PNG roughly offsets the resolution
increase for a *typical* image - the median case lands close to today's
footprint. The mean is pulled up by real, large source frames (up to
~8280px long side measured in this archive).

## Versioning and backward compatibility

`CropParams` is a plain dataclass, and `build_cache` already compared the
stored `crop_params.json` against the requested params on every run,
refusing (`SystemExit`, "pass --force to rebuild") rather than silently
reusing a mismatch - this predates the Vision Cache work and needed no new
mechanism, just new fields (`image_format`, `jpeg_quality`, and `max_side`'s
new default) to participate in the comparison that already existed.

`CROP_CACHE_VERSION` was bumped to `v5` (`bird_crop.py`'s own module
docstring lists what each version means) to mark the crop *pixel* format as
having changed, independent of the crop *selection* policy (which did not
change). `crop_cache_path`'s file extension also depends on `image_format`,
so an old PNG cache entry is not merely flagged as stale - a reader
configured for the new default (`.jpg`) simply never finds it in the first
place, and rebuilds fresh without ever risking a wrong-format read. The old
files are never deleted automatically (see "Never silently delete" below) -
they sit alongside the new ones as harmless, orphaned bytes until a future
Cache Manager (or a manual `rm`) cleans them up.

**Migrating an existing archive**: the next `preprocess`/`rank`/Classic
Vision run against an existing v4-or-earlier cache directory will refuse
with a clear message; pass `--force` (CLI) or `force_preprocess=True`
(`rank.rank_folder`/`ClassicVisionStrategy.rank_folder`) to rebuild. This
re-decodes and re-detects every image (the detection *record* itself is
still valid, but today's pipeline does not yet split "re-crop from a known
detection" from "re-detect" - see "Future work" below) - budget for that the
same way a first-time preprocessing run is budgeted for.

## Preview Cache vs Vision Cache - deliberately separate

`review/thumbnails.py`'s `REVIEW_PREVIEW_CACHE` (a JPEG q92 cache with its
own 20 GB budget and LRU eviction, `_enforce_cache_budget`) is a **UI**
optimization for the Loupe/Lightbox - it decodes the *original* image
directly (`contactsheets.load_source_image`), never the crop cache, and
nothing in this change touched it. The two caches answer different
questions ("what does a photographer see" vs "what does a Computer Vision
model receive") and must keep answering them independently - see
`tests/test_review_thumbnails.py`'s `PreviewCacheIsIndependentOfTheVisionCacheTests`.

## Future work (not implemented - by design)

- **A Cache Manager** - per-cache statistics/location/size, rebuild, clear,
  and a "the configured limit was reached, what would you like to do"
  prompt (Continue / clear manually / enable automatic cleanup), generalizing
  the Preview Cache's own already-working `_enforce_cache_budget` pattern to
  the Vision Cache. The application must never silently delete analysis
  data - the existing `build_cache` mismatch check already honors this by
  refusing rather than overwriting; a Cache Manager would turn that refusal
  into a UI prompt instead of a raised exception.
- **Migration without re-detection.** `bird_crop.save_detections` already
  persists `expanded_box` in full-frame coordinates - a version bump that
  changes only the crop's pixel format (not the selection policy) could, in
  principle, re-crop from a fresh decode using the *existing* detection
  record instead of re-running the subject detector. `preprocess.build_cache`
  does not yet make this distinction (a missing/stale crop today always
  re-runs full detection); decoupling it is future work, not required for
  this change to be correct.
