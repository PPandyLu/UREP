# Model architectures and checkpoint configurations

Three shared architecture implementations serve all experiments: UREP (`patch_model.py`, `fpn_source.py`), RT-CNN (`rtcnn.py`), and ResNet-18 (torchvision). The output-range helper changes the UREP readout through configuration and does not duplicate its encoder. Each experiment contributes only a JSON configuration and a checkpoint. A1 shares its checkpoint with the NYSDOP main comparison.

| Checkpoint scope | Count |
|---|---:|
| Main comparison: three methods, four datasets, three seeds | 36 |
| Additional NYSDOP A2–A5 configurations, three seeds | 12 |
| NYSDOP seed43 output-range configurations | 5 |
| Supplementary NYSDOP RT-CNN 32-epoch configurations | 3 |
| Total distinct checkpoint entries | 56 |

A1: local NLL + learned fusion; A2: local NLL + uniform fusion; A3: no local loss + learned fusion; A4: no local loss + uniform fusion; A5: local log-MSE + learned fusion.

NYSDOP RT-CNN main weights use paired 100-epoch training. The 32-epoch weights are supplementary. All DOTA weights refer to the rebuilt 1,850-image test protocol. fMoW UREP uses its reported [0.05, 3.2] m/px range. Output-range alternatives are separate configurations bound to their own weights, even when their architecture and state-dictionary keys match.

The original A3 seed42 boundary-saturation result is retained in `models/supplementary_original_A3_seed42.json`. Its weights are not included; the main ablation uses the successful repeat, with an independent checkpoint identity. These records must not be combined into one score.

The weights are inference state dictionaries. Optimizer, scheduler, RNG state, paths, and training-resume state are excluded. They are not resumable training checkpoints. The source and exported SHA-256 values differ because the containers contain different information. The loader checks configuration and weight hashes and requires strict state-dictionary compatibility.
