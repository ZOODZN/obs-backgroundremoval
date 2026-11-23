# OBS Background Removal (ZOODESIGN build)

A slimmed-down fork of [royshil/obs-backgroundremoval](https://github.com/royshil/obs-backgroundremoval) 1.4.1 for our OBS training sessions on Mac. It cuts the presenter out of the camera so the overlay's background shows through, or blurs the room behind them.

Everything we don't use has been removed: other platforms, other models, the Enhance portrait filter, focal blur and the update checker. Git history keeps all of it.

## What it does

| Setting | Options | Default |
| --- | --- | --- |
| Blur background | 0 = transparent, 1–20 = blur | 0 |
| Segmentation model | Robust Video Matting, MediaPipe | Robust Video Matting |
| RVM quality | Fast (320×192), Balanced (640 px), High (1024 px) | Balanced |
| Inference device | CoreML GPU (full precision), CPU | CoreML GPU |
| # CPU threads | 0–8 (CPU only) | 4 |
| Threshold | How sure the model must be before a pixel counts as you | 0.50 |
| Mask expansion | Grow (+) or shrink (–) the cut-out, in pixels | 0 |
| Calculate every X frame | Work out a new cut-out every X frames | 1 |

Everything except Blur background sits under **Advanced settings**.

How the cut-out is made, on every frame:

1. **Robust Video Matting at the camera's real shape.** The model runs at the frame's aspect ratio and uses its own edge refiner, instead of squashing the frame to 320×192.
2. **Stable mask.** Asymmetric smoothing over time, hysteresis thresholding, keeping only the main subject, filling small holes in the face or shirt, and a guided filter that snaps the edge to hair and shoulders.
3. **Off OBS's video thread.** The cut-out is worked out on its own thread, so slow processing lowers the mask rate instead of stalling OBS.
4. **Memory safeguard.** If OBS's memory grows by more than 3 GB, the filter rebuilds the model, then switches to CPU, then pauses (the camera passes through unchanged). Each step is written to the OBS log.

Scenes saved with the full plugin keep working: a removed model switches to Robust Video Matting, and a removed inference device (plain CoreML, CUDA and so on) switches to CoreML GPU.

## Install

Use the installer and setup guide in the training-assets repo: `obs-plugins/README.md`.

## Build

GitHub Actions builds the Mac installer on every push to `rvm-fix` or `main`, or from **Actions › Build macOS › Run workflow**. A build takes about 1–2 hours because it compiles OBS and ONNX Runtime from source. Download `pluginbinary-macOS-universal-…` from the run's **Artifacts**.

The plugin is ad-hoc signed during packaging. Without that, macOS refuses to load it and OBS quits on start-up.

## Layout

```
src/
  background-filter.cpp     the filter: settings, worker thread, stable mask, memory safeguard, rendering
  models/ModelRVM.hpp       Robust Video Matting at the frame's shape, recurrent state
  models/ModelMediapipe.hpp MediaPipe fallback
  ort-utils/                ONNX Runtime session (CPU or CoreML GPU)
  required_operators_and_types.with_runtime_opt.config   operators ONNX Runtime is built with
data/
  models/                   the two models (ONNX Runtime format)
  effects/                  blur and mask shaders
  locale/en-US.ini          labels
models/                     the two models' ONNX sources
.github/workflows/          build-macos.yml (trigger), plugin-build-macos.yml, ci-config.yml
```

If you add a model back, regenerate the operator config from all the shipped `.ort` models with ONNX Runtime's `create_config_from_models` (type reduction on), then keep the `com.microsoft;1;FusedConv` line. The saved runtime optimisations create FusedConv at load time, so the generator doesn't see it.

## Credits and licence

Original plugin by Roy Shilkrot and Kaito Udagawa. GPL-3.0-or-later; see `LICENSE` and `LICENSES/`. Models keep their own licences beside each file.
