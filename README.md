# UREP GSD models

Inference weights and architecture configurations for UREP, Regression Tree CNN (RT-CNN), and ResNet-18 across NYSDOP, rebuilt DOTA, fMoW10k, and rebuilt Mixed8. The index contains 36 main-comparison checkpoint configurations and 20 additional ablation, output-range, and supplementary configurations. A1 weights are shared with the NYSDOP main comparison.

## Installation

Use Python 3.10 and a matching PyTorch/torchvision installation. `requirements-tested.txt` records the validation environment. From the repository root, install the package before running the examples:

```bash
pip install .
```

## Checkpoint downloads and inference

Download the required architecture/benchmark bundle from [UREP v1.0.0](https://github.com/PPandyLu/UREP/releases/tag/v1.0.0), together with the release's `LICENSE.txt`, `README.md` (weight license guide), `asset_index.json` and `checksums.sha256`. Retain the license alongside the downloaded bundle. The weights are for non-commercial academic research, evaluation and teaching only; see [License](#license).

For the NYSDOP UREP example, place `urep_nysdop.tar.gz` and `asset_index.json` in the repository root. Verify the bundle against the asset index, then extract it in that directory:

```bash
python -c "import hashlib,json; from pathlib import Path; p=Path('urep_nysdop.tar.gz'); a=next(x for x in json.loads(Path('asset_index.json').read_text())['assets'] if x['file']==p.name); assert p.stat().st_size==a['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'], 'Bundle verification failed'; print('Bundle verified')"
tar -xzf urep_nysdop.tar.gz
python scripts/verify_models.py --model nysdop_urep_a1_s43
```

Extraction creates `weights/urep/nysdop/` inside the repository. The loader defaults to the repository's `weights/` directory. `--model` selects one downloaded checkpoint; omit it only when all indexed weights are present. `checksums.sha256` also lists hashes for the release's companion files.

Replace `example.png` below with an existing 256 x 256 RGB input image, then run:

```bash
python scripts/infer.py --model nysdop_urep_a1_s43 --image example.png
```

Per-checkpoint `download_url` values in `models/model_index.json` refer to the bundle containing that checkpoint. Prediction artifact URLs remain null where those results are not published. The index binds every weight to its configuration, source-checkpoint SHA-256, selected epoch, dataset, unit and paper role. Weight files contain only PyTorch state dictionaries and can be read with `weights_only=True`; the loader checks configuration and weight hashes.

### Server directory layout

For a separate server layout where `release_assets/weights/` is a sibling of the code directory, append `--weights-dir ../release_assets/weights` to the commands above. This override is unnecessary for bundles extracted inside the cloned repository.

## Data and interpretation

Dataset processing scripts, fixed splits and separate build indices are available at [ModelScope PPLandy/UREPdataset](https://modelscope.cn/datasets/PPLandy/UREPdataset). Only the fMoW10k image archives are hosted there. NYSDOP, rebuilt DOTA and all eight Mixed8 sources must be acquired from the upstream links in their dataset-specific `UPSTREAM_DOWNLOADS.md` files, then processed with the original experiment scripts in each dataset's scripts/reference/ directory. Upstream access and use terms apply; WiDS2019 requires applicable additional permission for competition-external research. Inputs are 256 × 256 RGB crops, with independent per-channel 2nd–98th percentile stretching. Inference does not apply augmentation or ImageNet normalization. Dataset-specific crop and resampling rules are in the corresponding data cards; arbitrary resizing changes GSD.

NYSDOP outputs are in **ft/px**; multiply by 0.3048 for m/px. The other three benchmarks use m/px. RT-CNN uses an unrestricted mantissa head and has no sigmoid clipping. UREP and ResNet-18 use the ranges recorded in their configurations; the supplementary Softplus model has no upper bound.

See [model documentation](docs/models.md), [training recipes](docs/training_recipes.md), and [dependencies and attribution](THIRD_PARTY.md). A3 seed42 uses the successful repeat reported in the main ablation; the original collapsed run is described separately, with its own identity, in the supplementary model metadata.

## License

Project-authored code, configurations and trained weights are available solely for non-commercial academic research, evaluation and teaching under the [UREP Academic Non-Commercial License](LICENSE). Commercial use requires separate written permission from the relevant rights holders. Redistribution and modifications must retain the license and applicable notices. Third-party dependencies and source data retain their own terms; this project license does not authorize additional upstream-data uses.

This is a source-available academic non-commercial release. The project license is not an OSI-approved open-source license because it restricts commercial use.
