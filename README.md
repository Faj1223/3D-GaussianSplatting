# SynthaDrive: reconstructing and editing driving scenes with Gaussian Splatting (nuScenes)

> Status: Phase 1 (vanilla 3DGS) and Phase 2 (OmniRe) done; Phase 3 (editing road users to create edge cases) in progress.
> The notebooks are exploratory research code; a cleaned-up version is planned.

## Goal

Reconstruct real driving scenes from camera and lidar logs, then:

1. Novel view synthesis: render the scene from viewpoints the car never occupied (shifted lanes, turned cameras).
2. Edge-case generation: edit the reconstructed scene (move, remove or add road users) to create rare and dangerous situations for testing autonomous driving systems, without having to record them in real life.

## Setup

| | |
|---|---|
| Data | nuScenes `v1.0-mini`, `scene-0061` (Singapore, ≈ 20 s) |
| Sensors | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `LIDAR_TOP` |
| Hardware | Kaggle, NVIDIA T4 (16 GB) |
| Metrics | PSNR, SSIM, LPIPS |

## Phase 1: vanilla 3DGS (splatfacto)

`splatfacto` (nerfstudio 1.1.5, gsplat 1.4.0), keyframes (2 Hz) and sweeps (≈ 12 Hz): 224 images per camera.

### Evaluation protocol

Time-block split: one keyframe out of five goes to *test*, another one out of five to *validation*, and sweeps within ≈ 0.25 s follow their keyframe, so near-duplicate frames never leak into evaluation. Model selection is done on validation only, and two models are always compared on the same images with the same code. With several cameras, each side image takes the split of the nearest-in-time front image.

| Exp. | Question | Key result |
|---|---|---|
| 1–3 | Does adding sweeps (≈ 12 Hz instead of 2 Hz) help? | Yes: better on 49 of 52 test images |
| 4 | Lidar initialisation vs. default initialisation | Better LPIPS (val −0.045), similar PSNR |
| 5 | How many iterations? (30k run) | Validation plateaus around 18k iterations; strong overfitting |
| 6 | Which lidar cloud? raw / moving objects removed / with sweeps | Better start with a clean, dense cloud (val PSNR at 2k: 18.2, 19.0, 19.6 dB), same final quality (≈ 20 dB) |
| 7 | Train on the 3 front cameras | Side views are learned, but the front view drops from 19.7 to 15.5 dB (same 40 validation images) |
| 7b | Per-camera exposure correction from lidar correspondences | +1 dB on the front view (16.5 dB) |
| 7c | Same as 7b, 3× longer with a 3× stretched schedule | 17.6 dB: still ≈ 2.4 dB below the single-camera model |
| Shifts | Move the camera sideways, from −3 m to +3 m | The 3-camera model holds up better when shifting left; everything degrades beyond ≈ 2 m |

### Main findings

- Measurement pitfall: splatfacto resets all opacities every 3,000 iterations, so checkpoints taken exactly at a reset underestimate the model.
- Exposure conflict between cameras: each camera runs its own auto-exposure. On the same surfaces (lidar points seen by two cameras at the same instant), the front-left camera is about 5× brighter than the front camera in linear light, and up to 9×.
- Multi-camera trade-off: side cameras widen the reconstructed area but degrade the front view, even on its own training images at equal training budget.
- Floaters and dynamic objects: the car driving ahead becomes a large blur, and Gaussians close to the camera hide the scene from any new viewpoint.

### Why vanilla 3DGS struggles here

3DGS (Kerbl et al., SIGGRAPH 2023) assumes a static scene seen from many viewpoints. Driving logs break both assumptions. Road users move, and the car moves along its viewing direction, so depth is poorly constrained. On top of that, lidar only initialises the Gaussians, cameras have independent exposures, and the sky is at infinity.

## Phase 2: OmniRe (drivestudio)

OmniRe (Chen et al., ICLR 2025) represents the scene as a Gaussian scene graph: a static background, one node per vehicle moved by its 3D box, pedestrians bound to the SMPL body model, deformable nodes for other actors, and a separate sky model. Lidar supervises depth during the whole training, and a colour correction is learned for each image. Each road user is a separate, editable node.

### Setup

I installed drivestudio's official requirements (Python 3.9, PyTorch 2.0 with CUDA 11.7) in a separate micromamba environment on Kaggle, and compiled `gsplat` 1.3.0, `pytorch3d` and `nvdiffrast` once for the T4. nuScenes is converted to drivestudio's format at 10 Hz (keyframes and interpolated annotations). Sky masks are computed with SegFormer-B5 (Cityscapes), the model used by drivestudio, through Hugging Face `transformers`. The pedestrian poses are the pre-computed ones shared by the drivestudio authors.

### Training

3 front cameras, every 10th frame held out for testing (drivestudio's protocol), 30,000 iterations, images loaded at half resolution (800 × 450). Training takes about 4 h on a T4, while the authors report about 1 h per scene on an RTX 4090. The scene contains 18 pedestrians modelled with SMPL.

### Test results

| Region | PSNR (dB) | SSIM |
|---|---|---|
| Full image | 22.92 | 0.662 |
| Non-sky | 22.86 | 0.653 |
| Dynamic regions | 21.14 | 0.551 |
| Pedestrians | 21.75 | 0.536 |
| Vehicles | 20.58 | 0.539 |

LPIPS (full image): 0.231.

### Novel views

- Lateral shifts (front camera moved from −3 m to +3 m) and rotations (turned from −60° to +60°, tilted from −20° to +20°) are rendered from the trained model.
- Turning left or right up to about 60° stays clean, thanks to the side cameras. Tilting the camera up by more than about 10° reaches regions no camera has seen.
- Moving vehicles and pedestrians are rendered as objects, instead of the blurs obtained with splatfacto.

### Caveat

OmniRe and splatfacto were not evaluated with the same test protocol (every 10th frame vs. time blocks, 10 Hz vs. raw sweeps, half vs. full resolution). The gap (≈ 22.9 vs. ≈ 17.6 dB on the front camera) is indicative, not a controlled comparison.

### Engineering notes

- Long trainings run with Kaggle's *Save & Run All*: interactive sessions are stopped after inactivity, and files written during interactive sessions were not reliably kept.
- drivestudio does not support resuming training from a checkpoint as is. Two small patches were needed (SMPL neighbour lists and empty optimizer states), and the final model was trained from scratch.

## Roadmap

- [x] Phase 1, vanilla 3DGS: protocol, lidar initialisation, multi-camera, lateral shifts
- [x] Phase 2, OmniRe on the same scene: training, test metrics, lateral shifts and rotations
- [ ] Controlled comparison of OmniRe and splatfacto (same split, same resolution)
- [ ] Phase 3: editing road users (move, remove, add) to generate edge cases
- [ ] Phase 4: diffusion models to repair large viewpoint changes

## Repository structure

```
notebooks/   Kaggle notebooks (Phase 1: splatfacto; Phase 2: OmniRe training, novel views)
results/     metrics (JSON / CSV)
figures/     plots and renders (PNG)
docs/        detailed write-up of each experiment (in French)
```

## Reproducing

All notebooks run on Kaggle with a T4 GPU. nuScenes data is not included: download `v1.0-mini` from [nuscenes.org](https://www.nuscenes.org/) and accept its terms of use. Model checkpoints are not included because of their size.

For Phase 2:
1. Run the OmniRe training notebook with Save Version, then Save & Run All (about 5 to 6 h on a T4 the first time, including compilation). Its output contains the processed scene, the trained model (`checkpoint_final.pth`) and the compiled wheels. An SMPL model (registration on the SMPL website) must be attached as a dataset.
2. Attach that output as an input of the same notebook, or of another one, to render novel views without retraining.

## Limitations

- One scene, one training run per configuration (no variance estimate).
- Novel views have no ground truth and are judged visually.
- nuScenes annotations exist only at 2 Hz; drivestudio interpolates them to 10 Hz.

## References

- Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Chen et al., *OmniRe: Omni Urban Scene Reconstruction*, ICLR 2025.
- Yan et al., *Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting*, ECCV 2024.
- Tancik et al., *Nerfstudio: A Modular Framework for Neural Radiance Field Development*, SIGGRAPH 2023.
- Ye et al., *gsplat: An Open-Source Library for Gaussian Splatting*, 2024.
- Loper et al., *SMPL: A Skinned Multi-Person Linear Model*, SIGGRAPH Asia 2015.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Caesar et al., *nuScenes: A Multimodal Dataset for Autonomous Driving*, CVPR 2020.

## Author

Adbias Houessou, M2 student, personal project.
<!-- Optional: "Code written with the help of an AI assistant; experiment design and analysis are my own." -->

Phase 1 notebooks are written in French, Phase 2 notebooks in English.
