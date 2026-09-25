<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/nnInteractive_header.png"  width="1200">

# nnInteractive: Redefining 3D Promptable Segmentation

This repository contains the napari plugin for nnInteractive. Check out the
[python backend](https://github.com/MIC-DKFZ/nnInteractive) and [MITK integration](https://www.mitk.org/MITK-nnInteractive) for more.

## What is nnInteractive?

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> \*: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

##### Abstract:

Accurate and efficient 3D segmentation is essential for both clinical and research applications.
While foundation models like SAM have revolutionized interactive segmentation, their 2D design and domain shift limitations make them ill-suited for 3D medical images.
Current adaptations address some of these challenges but remain limited, either lacking volumetric awareness, offering restricted interactivity, or supporting only a small set of structures and modalities.
Usability also remains a challenge, as current tools are rarely integrated into established imaging platforms and often rely on cumbersome web-based interfaces with restricted functionality.
We introduce nnInteractive, the first comprehensive 3D interactive open-set segmentation method.
It supports diverse prompts—including points, scribbles, boxes, and a novel lasso prompt—while leveraging intuitive 2D interactions to generate full 3D segmentations.
Trained on 120+ diverse volumetric 3D datasets (CT, MRI, PET, 3D Microscopy, etc.), nnInteractive sets a new state-of-the-art in accuracy, adaptability, and usability.
Crucially, it is the first method integrated into widely used image viewers (e.g., Napari, MITK), ensuring broad accessibility for real-world clinical and research applications.
Extensive benchmarking demonstrates that nnInteractive far surpasses existing methods, setting a new standard for AI-driven interactive 3D segmentation.

<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/figure1_method.png" width="1200">

## Demo Videos

<a href="https://www.youtube.com/watch?v=H_L6LL0FRoo">
    <img src="https://img.youtube.com/vi/H_L6LL0FRoo/0.jpg" width="270">
</a>
<a href="https://www.youtube.com/watch?v=YoMZ7Xv7gKI">
    <img src="https://img.youtube.com/vi/YoMZ7Xv7gKI/0.jpg" width="270">
</a>
<a href="https://www.youtube.com/watch?v=V0rqPYA3sjA">
    <img src="https://img.youtube.com/vi/V0rqPYA3sjA/0.jpg" width="270">
</a>

## Installation

### Prerequisites

You need a Linux or Windows computer with a Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with \<6GB.

##### 1. Create a virtual environment:

nnInteractive supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n nnInteractive python=3.12
conda activate nnInteractive
```

##### 2. Install the correct PyTorch for your system

Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the right configuration.
Note that since recently PyTorch needs to be installed via pip. This is fine to do within your conda environment.

For Ubuntu with a Nvidia GPU, pick 'stable', 'Linux', 'Pip', 'Python', 'CUDA12.6' (if all drivers are up to date, otherwise use and older version):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### 3. Install this repository + dependencies via

Either install via pip:

```bash
pip install napari-nninteractive
```

Or clone and install this repository:

```bash
git clone https://github.com/MIC-DKFZ/napari-nninteractive
cd napari-nninteractive
pip install -e .
```

**Note:** Model weights are automatically downloaded on first use. This can take up to a couple of minutes depending on your internet connection

## Getting Started

Use one of these three options to start napari and activate the plugin.
Afterward, Drag and drop your images into napari.

\***Note if getting asked which plugin to use for opening .nii.gz files use napari-nifti.**

a) Start napari, then Plugins -> nnInteractive.

```
napari
```

b) Run this to start napari with the plugin already started.

```
napari -w napari-nninteractive
```

c) Run this to start napari with the plugin and open an image directly

```
napari demo_data/liver_145_0000.nii.gz -w napari-nninteractive
```

### Launching this fork's timing build

This fork adds the Prompt Placement Timing build (Resume / Complete / Abandon
gating, preset dropdown, per-case notes) on the `feature/timing-instrumentation`
branch. `launch_timing.sh` at the repo root starts it:

```bash
bash launch_timing.sh
```

It activates the `napari-nninteractive-fork-dev` conda env, re-points that env's
editable install at this checkout (`pip install --no-deps -e .`), then runs
`napari -w napari-nninteractive`.

Things to know:

- **It does not switch branches.** It runs whatever is checked out in this
  working directory, so check out `feature/timing-instrumentation` first.
- **Machine-specific.** The script hardcodes the path to `conda.sh` and expects
  the `napari-nninteractive-fork-dev` env to already exist.
- **Timing build dependency.** It imports `presets` and `interaction_log` from the
  separate `front-end-timing` package, which must be installed in the same env.
- **Model weights.** If a `checkpoints/nnInteractive_v1.0` folder exists in this
  checkout, the Model Selection field defaults to it. Clear that field to
  download the weights from HuggingFace instead.

#### Where the timing logs are written

The timing build writes its records next to the config you load, not into this
repo. For a config at `<output>/<dataset>/<experiment>/config.json` (the layout
`export_napari_config.py --output <output>` produces), records go to
`<output>/<dataset>/timing/<experiment>/<fe_experiment>/`: a `<model>_NN.jsonl`
per preset (plus `_play_around` variants) and a `_conditions.json` manifest.

Nothing in the code stops that path landing inside a git checkout, so keep it out:

- Give `export_napari_config.py --output` a directory **outside** every repo
  checkout (this fork, napari-clopa, `front-end-timing`, CLoPA). Pointing it
  inside one puts both the config and the logs in that repo's working tree.
- If you do end up inside a repo, add the export folder and `timing/` to that
  repo's `.gitignore`, and run `git status` before committing to check that no
  `.jsonl` slipped in.
- The flip side: a directory outside git has no history and no backup. Copy the
  timing folders somewhere safe once real data is being collected.
- **Cleaning up a dataset's exports can delete its logs.** The logs sit in that
  dataset's own folder, at `<output>/<dataset>/timing/<experiment>/`, beside the
  experiment's config folder rather than inside it. Deleting only the config
  folder leaves them, but deleting `<dataset>/timing/`, one `timing/<experiment>/`
  or the whole `<dataset>/` folder removes them. So before clearing out or
  regenerating a dataset's exports (for example after correcting its data), copy
  its `timing/` folder somewhere safe. Other datasets' logs are unaffected.

# How to use

**Note:** To open Nifti (.nii.gz, .nii) files we recommend to select napari-nifti.

<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/gui_instuctions.png" width="1200">

## Citation

When using nnInteractive, please cite the following paper:

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> \*: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

# License

Note that while this repository is available under Apache-2.0 license (see [LICENSE](./LICENSE)), the [model checkpoint](https://huggingface.co/nnInteractive/nnInteractive) is `Creative Commons Attribution Non Commercial Share Alike 4.0`!

______________________________________________________________________

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

[copier]: https://copier.readthedocs.io/en/stable/
[napari]: https://github.com/napari/napari
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
