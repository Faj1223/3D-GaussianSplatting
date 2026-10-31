# SynthaDrive — 3D Gaussian Splatting for driving scenes (nuScenes)

> **Status:** Phase 1 complete (analysis of vanilla 3DGS on driving data). Phase 2 in progress: dynamic-scene methods (OmniRe / drivestudio).
> The notebooks are exploratory research code; a cleaned-up version is planned.

## Goal

Reconstruct real driving scenes from camera and lidar logs, then:

1. **Novel view synthesis** — render the scene from viewpoints the car never occupied (e.g. shifted lanes).
2. **Edge-case generation** — edit the reconstructed scene (move, remove or add road users) to create rare and dangerous situations for testing autonomous driving systems.

This repository documents Phase 1: how far the original 3D Gaussian Splatting (3DGS) goes on driving data, where and why it fails, measured with a careful evaluation protocol.

## Setup

| | |
|---|---|
| Data | nuScenes `v1.0-mini`, `scene-0061` (Singapore, ≈ 20 s) |
| Sensors | `CAM_FRONT` (later + `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`), `LIDAR_TOP` |
| Images | keyframes (2 Hz) **and** sweeps (≈ 12 Hz): 224 images per camera |
| Method | `splatfacto` (nerfstudio 1.1.5), gsplat 1.4.0 |
| Hardware | Kaggle, NVIDIA T4 (16 GB) |
| Metrics | PSNR, SSIM, LPIPS |

### Evaluation protocol

- **Time-block split.** One keyframe out of five goes to *test*, another one out of five to *validation*; sweeps taken within ≈ 0.25 s of those keyframes follow them. Near-duplicate frames therefore never leak from training into evaluation.
- **Model selection on validation only** (checkpoint, number of iterations, initialisation); the test split is used once at the end.
- **Same images, same code** whenever two models are compared.
- **Multi-camera split:** each side-camera image takes the split of the nearest-in-time front image, so all cameras share the same held-out time blocks.

## Experiments and results

| Exp. | Question | Key result |
|---|---|---|
| 1–3 | Does adding sweeps (≈ 12 Hz instead of 2 Hz) help? | Yes: better on 49 of 52 test images |
| 4 | Lidar initialisation vs. default initialisation | Better LPIPS (val −0.045), similar PSNR |
| 5 | How many iterations? (30k run, checkpoints every 3k) | Validation plateaus around 18k iterations; strong overfitting |
| 6 | Which lidar cloud? raw / moving objects removed / + sweeps | Cleaner and denser cloud gives a better start (val PSNR at 2k: 18.2 → 19.0 → 19.6 dB) but the same final quality (≈ 20 dB); no extra training cost |
| 7 | Train on the 3 front cameras | Side views are learned, but the front view drops: 19.7 → 15.5 dB (same 40 validation images) |
| 7b | Correct per-camera exposure using lidar correspondences | +1 dB on the front view (16.5 dB) |
| 7c | Same as 7b, 3× longer with a 3× stretched schedule | 17.6 dB: still ≈ 2.4 dB below the single-camera model |
| Shifts | Move the camera −3 m … +3 m sideways | The 3-camera model holds up much better when shifting left; everything degrades beyond ≈ 2 m |

### Main findings

- **Measurement pitfall.** splatfacto resets all opacities every 3,000 iterations; checkpoints taken exactly at a reset underestimate the model. Saving every 2,500 (or 10,000) iterations avoids it.
- **Exposure conflict between cameras.** Each camera runs its own auto-exposure. Using lidar points seen by two cameras at the same instant, the front-left camera sees the same surfaces **≈ 5× brighter in linear light (up to ≈ 9×)** than the front camera. A per-image correction recovers about one quarter of the multi-camera gap.
- **Multi-camera trade-off.** Adding side cameras widens the reconstructed area and improves sideways view synthesis, but degrades the front view, even on its own training images at equal training budget (≈ 24.5 vs ≈ 27.6 dB). The remaining cause is open (residual cross-camera conflicts, dynamic objects).
- **Floaters and dynamic objects.** The car driving ahead becomes a large blur, and Gaussians close to the camera ("floaters") reproduce training images but hide the scene from any new viewpoint.

## Why vanilla 3DGS struggles on driving data

Original 3DGS (Kerbl et al., SIGGRAPH 2023) assumes a **static** scene observed from **many viewpoints**. Driving logs break both assumptions:

- road users move, and a static model cannot represent them;
- the car moves forward along its viewing direction, so each surface is seen from very similar angles and depth is poorly constrained;
- lidar is only used for initialisation, so nothing keeps Gaussians at the right depth afterwards;
- several cameras with independent exposure, and a sky at infinity.

## Next steps (Phase 2)

Move to **drivestudio** with **OmniRe** (Chen et al., ICLR 2025), which models the scene as a Gaussian scene graph: static background, one node per vehicle following its 3D box, deformable humans (SMPL), a separate sky model, lidar depth supervision during training and per-camera appearance. Each road user becomes editable, which is the basis for edge-case generation.

Roadmap:

- [x] Phase 1 — vanilla 3DGS: protocol, lidar initialisation, multi-camera, lateral shifts
- [ ] Phase 2 — OmniRe on the same scene and split, comparison with Phase 1
- [ ] Phase 3 — editing road users to generate edge cases
- [ ] Phase 4 — diffusion models to repair large viewpoint changes

## Repository structure

```
notebooks/   Kaggle notebooks (exploratory)
results/     metrics per checkpoint (JSON)
figures/     plots and renders (PNG)
docs/        detailed write-up of each experiment (in French)
```

## Reproducing

The notebooks run on Kaggle with a T4 GPU. nuScenes data is **not** included: download `v1.0-mini` from [nuscenes.org](https://www.nuscenes.org/) and accept its terms of use. Model checkpoints are not included because of their size.

## Limitations

- One scene, and one training run per configuration (no variance estimate).
- Sideways shifts have no ground truth and are judged visually.
- Exposure correction is a single gain per image, measured in linear light.

## References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Ye et al., *gsplat: An Open-Source Library for Gaussian Splatting*, 2024.
- Tancik et al., *Nerfstudio: A Modular Framework for Neural Radiance Field Development*, SIGGRAPH 2023.
- Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*, CVPR 2020.
- Chen et al., *OmniRe: Omni Urban Scene Reconstruction*, ICLR 2025.
- Yan et al., *Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting*, ECCV 2024.

## Author

Adbias Houessou — M2 student.
<!-- Optional: "Code written with the help of an AI assistant; experiment design and analysis are my own." -->
