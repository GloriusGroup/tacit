<div align="center">

# TACIT

**TAckling Chemistry through ImpliciT negative learning**

Nick Wierich\*, Niklas Dobberstein\*, Frank Glorius†

*Organisch-Chemisches-Institut, Universität Münster, 48149 Münster, Germany*

<sub>\* These authors contributed equally. &nbsp;·&nbsp; † Corresponding author.</sub>

<img src="docs/static/images/pbox.png" alt="TACIT overview" width="800">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23669984.svg)](https://doi.org/10.5281/zenodo.23669984)
[![Project page](https://img.shields.io/badge/project-page-blue.svg)](https://gloriusgroup.github.io/tacit/)

[Project page](https://gloriusgroup.github.io/tacit/) · [Models and data (Zenodo)](https://doi.org/10.5281/zenodo.23669984) · [Paper](#citation)

</div>

---

## Abstract

Machine learning models for reaction product prediction have largely been
trained and evaluated on datasets of successful reactions. While this setting
allows reaching high accuracy scores, these models generally struggle to
discriminate between successful and unsuccessful transformations due to the
lack of negative training data. This critically limits their practical use, as
reliably recognizing unproductive substrate combinations *in silico* is
required to accelerate experimental campaigns and save resources in the
laboratory. While the availability of negative reaction data remains limited to
date, we show that even purely positive datasets implicitly contain a
significant amount of information about cases in which no reaction occurs. To
exploit this, we introduce TACIT (TAckling Chemistry through ImpliciT negative
learning). While trained on positive-only data, TACIT learns to distinguish
between productive and unproductive transformations. On two published
high-throughput experimentation datasets, it outperforms all evaluated reaction
outcome prediction models in separating successful from failed reactions.
Crucially, TACIT's accuracy on the popular USPTO-MIT benchmark remains
competitive, despite its added ability to predict negative reaction outcomes.

## Quickstart

Install, download the checkpoints and data, and evaluate TACIT on the USPTO-MIT test set:

```bash
git clone https://github.com/GloriusGroup/tacit.git && cd tacit

uv sync

uv run python download_zenodo.py

uv run python src/bond_match/evaluate_ranking.py --config runs/tacit/config.yaml --checkpoint runs/tacit/checkpoints/tacit.pt --data data/uspto_mit/test.jsonl --all-entries --save-csv tacit_uspto.csv
```

## Results

| Model | USPTO-MIT top-1 [%] | [Chemlex](https://doi.org/10.1038/s41467-025-59812-0) ROC-AUC / PR-AUC | [AIChemEco](https://doi.org/10.1039/d5sc03364k) ROC-AUC / PR-AUC |
|---|---|---|---|
| TACIT | 77.8 | **0.80** / **0.56** | **0.61** / **0.69** |
| [MEGAN](https://doi.org/10.1021/acs.jcim.1c00537)<sup>a</sup> | 80.5 | 0.69 / 0.38 | 0.47 / 0.56 |
| [FlowER](https://doi.org/10.1038/s41586-025-09426-9)<sup>b</sup> | 42.8 | 0.50 / 0.26 | 0.51 / 0.60 |
| [Molecular Transformer](https://doi.org/10.1021/acscentsci.9b00576)<sup>a,d</sup> | 81.7 | 0.68 / 0.42 | 0.51 / 0.61 |
| [LocalTransform](https://doi.org/10.1038/s42256-022-00526-z)<sup>a,d</sup> | 80.4 | 0.64 / 0.37 | 0.51 / 0.59 |
| [ReactionT5](https://doi.org/10.1186/s13321-025-01075-4)<sup>a,c,d</sup> | **95.1**<sup>e</sup> | 0.69 / 0.41 | 0.47 / 0.58 |

USPTO-MIT: all 40,000 test reactions. HTE datasets: all 11,669 (Chemlex) and
47,015 (AIChemEco) wells. A well is positive if its conversion or yield is at
least 20%, and negative otherwise.

<sup>a</sup> Retrained on the blockspace-filtered USPTO-MIT training set used for TACIT.<br>
<sup>b</sup> Published checkpoint. Not retrained, since a mechanistic dataset is required for training.<br>
<sup>c</sup> Pretrained on the Open Reaction Database.<br>
<sup>d</sup> No stop action. The confidence of the top-1 prediction is used as the score.<br>
<sup>e</sup> 34,552 of the 40,000 test reactions (86.4%) are contained in the ORD pretraining data. The top-1 accuracy is 97.5% on these and 80.4% on the remaining 5,448 reactions.

## Installation

Requires **Python 3.11** and a CUDA 12.4 environment. The project uses
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GloriusGroup/tacit.git && cd tacit

uv sync
```

## Models and data

The checkpoints and the dataset are archived on
[Zenodo](https://doi.org/10.5281/zenodo.23669984). Download them into the
right folders with

```bash
uv run python download_zenodo.py
```

or place them manually:

| File | Content | Place at |
|---|---|---|
| `tacit.pt` | TACIT weights (base model) | `runs/tacit/checkpoints/` |
| `tacit_no-stop.pt` | TACIT<sub>no-stop</sub> weights (trained without the auxiliary stop loss) | `runs/tacit_no-stop/checkpoints/` |
| `uspto_mit_blockspace.zip` | USPTO-MIT train, validation and test splits with blockspace decompositions | `data/uspto_mit/` (and unzip) |

### Recreating the training data

The data is computed from the original USPTO-MIT splits of
[Jin et al. (2017)](https://github.com/wengong-jin/nips17-rexgen/tree/master/USPTO).

```bash
wget https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip -O uspto.zip

unzip -j uspto.zip -d uspto && (echo rxn_smiles; cut -d' ' -f1 uspto/train.txt) > uspto/train.csv

uv run python src/bond_match/blockspace/batch.py uspto/train.csv -c rxn_smiles -o uspto/train.jsonl --strict --mixed --remove_stereo --clean-coo -w 32 -t 10
```

The Zenodo record also contains the retrained baseline models (MEGAN, Molecular
Transformer, LocalTransform and ReactionT5), each with its upstream license and
a note on how it was trained.

## Usage

### USPTO-MIT

```bash
uv run python src/bond_match/evaluate_ranking.py --config runs/tacit/config.yaml --checkpoint runs/tacit/checkpoints/tacit.pt --data data/uspto_mit/test.jsonl --all-entries --save-csv tacit_uspto.csv
```

### HTE datasets

The HTE datasets are not redistributed here. Download them from the original
authors and convert each into a `wells.csv`:

| Dataset | Source file | License |
|---|---|---|
| Chemlex ([Zhong et al., *Nat. Commun.* 2025](https://doi.org/10.1038/s41467-025-59812-0)) | [`Chemlex_Acidamine_Wetlab_Data.xlsx`](https://zenodo.org/records/17596563/files/Chemlex_Acidamine_Wetlab_Data.xlsx) (Zenodo, version of Nov 2025) | CC BY-NC 4.0 |
| AIChemEco ([Zhang et al., *Chem. Sci.* 2025](https://doi.org/10.1039/d5sc03364k)) | [`all_HTE_with_condition.csv`](https://github.com/aichemeco/amide_coupling/raw/fa29d55/data/all_HTE_with_condition.csv) (GitHub, commit `fa29d55`) | MIT |

```bash
wget https://zenodo.org/records/17596563/files/Chemlex_Acidamine_Wetlab_Data.xlsx

wget https://github.com/aichemeco/amide_coupling/raw/fa29d55/data/all_HTE_with_condition.csv

uv run python src/bond_match/hte_wells.py chemlex Chemlex_Acidamine_Wetlab_Data.xlsx chemlex.csv

uv run python src/bond_match/hte_wells.py aichemeco all_HTE_with_condition.csv aichemeco.csv
```

Then score the wells:

```bash
uv run python src/bond_match/score_hte.py --config runs/tacit/config.yaml --checkpoint runs/tacit/checkpoints/tacit.pt --wells chemlex.csv --label-column conversion_pct --out tacit_chemlex.csv

uv run python src/bond_match/score_hte.py --config runs/tacit/config.yaml --checkpoint runs/tacit/checkpoints/tacit.pt --wells aichemeco.csv --label-column yield_pct --out tacit_aichemeco.csv
```

### Interactive prediction

```bash
uv run python src/bond_match/evaluate_ranking.py --config runs/tacit/config.yaml --checkpoint runs/tacit/checkpoints/tacit.pt
```

```txt
=== Ranking Model Eval Menu ===
1. Interactive SMILES Prediction (Top 5)
2. Evaluate on Test Set (test.jsonl)
3. Random Sample from Test Set (Visual)
q. Quit
Select option: 3

Chosen Reaction: [Br:1][c:2]1[cH:3][cH:4][c:5]([F:22])[c:6]([CH:8]([C:9]([CH2:10][O:11][Si:12]([CH3:13])([CH3:14])[C:15]([CH3:16])([CH3:17])[CH3:18])([F:19])[F:20])[OH:21])[cH:7]1.[Cl:44][CH2:45][Cl:46].[Cr:23]([O:24][Cr:25]([O-:26])(=[O:27])=[O:28])([O-:29])(=[O:30])=[O:31].[nH+:32]1[cH:33][cH:34][cH:35][cH:36][cH:37]1.[nH+:38]1[cH:39][cH:40][cH:41][cH:42][cH:43]1>>[Br:1][c:2]1[cH:3][cH:4][c:5]([F:22])[c:6]([C:8]([C:9]([CH2:10][O:11][Si:12]([CH3:13])([CH3:14])[C:15]([CH3:16])([CH3:17])[CH3:18])([F:19])[F:20])=[O:21])[cH:7]1

--- Initial Step Analysis for: CC(C)(C)[Si](C)(C)OCC(F)(F)C(O)c1cc(Br)ccc1F.ClCCl.[O]=[Cr](=[O])([O-])[O][Cr](=[O])(=[O])[O-].c1cc[nH+]cc1.c1cc[nH+]cc1 ---
Rank  Score    Prob     Action                    Match?
----------------------------------------------------------------------
1     25.2775  99.91    H#0-O:21 | H#29-C:8       ✅ 0 (0: 96%, 1: 4%)
2     18.2333  0.09     is_over                   ❌
3     1.7181   0.00     H#0-O:21 | H#18-C:16      ❌ 0 (0: 89%, 1: 11%)
4     0.8446   0.00     H#0-O:21 | O:21-C:8       ❌ 0 (0: 90%, 1: 10%)
5     0.7310   0.00     H#0-O:21 | C:8-C:9        ❌ 0 (0: 82%, 1: 18%)

>>> Continuing to step 2 from current product: [H][H].[H][n+:32]1[c:33]([H])[c:34]([H])[c:35]([H])[c:36]([H])[c:37]1[H].[H][n+:38]1[c:39]([H])[c:40]([H])[c:41]([H])[c:42]([H])[c:43]1[H].[H][c:3]1[c:2]([Br:1])[c:7]([H])[c:6]([C:8]([C:9]([C:10]([H])([H])[O:11][Si:12]([C:13]([H])([H])[H])([C:14]([H])([H])[H])[C:15]([C:16]([H])([H])[H])([C:17]([H])([H])[H])[C:18]([H])([H])[H])([F:19])[F:20])=[O:21])[c:5]([F:22])[c:4]1[H].[H][C:45]([H])([Cl:44])[Cl:46].[Cr:23]([O:24][Cr:25]([O-:26])(=[O:27])=[O:28])([O-:29])(=[O:30])=[O:31]

   --- Iterative Step 2 ---
   Rank  Score    Prob     Action                    Match?
   ----------------------------------------------------------------------
   1     20.3534  100.00   is_over                   ✅
   2     0.0574   0.00     O:11-Si:12 | Cl:44-LP     ❌ 1 (0: 39%, 1: 61%)
   3     0.0528   0.00     H#30-C:13 | Cl:44-LP      ✅ 0 (0: 55%, 1: 45%)
   4     0.0422   0.00     H#14-N:32 | Cl:44-LP      ✅ 0 (0: 52%, 1: 48%)
   5     0.0421   0.00     H#0-H#29 | Cl:44-LP       ✅ 1 (0: 44%, 1: 56%)
   >>> Stopping: Model selected is_over

--- Final Result ---
Steps applied: 1
Stop reason: Model selected is_over stop action
Pred Product: [Cr:23]([O:24][Cr:25]([O-:26])(=[O:27])=[O:28])([O-:29])(=[O:30])=[O:31].[H][C:45]([H])([Cl:44])[Cl:46].[H][c:3]1[c:2]([Br:1])[c:7]([H])[c:6]([C:8]([H])([C:9]([C:10]([H])([H])[O:11][Si:12]([C:13]([H])([H])[H])([C:14]([H])([H])[H])[C:15]([C:16]([H])([H])[H])([C:17]([H])([H])[H])[C:18]([H])([H])[H])([F:19])[F:20])[O:21][H])[c:5]([F:22])[c:4]1[H].[H][n+:32]1[c:33]([H])[c:34]([H])[c:35]([H])[c:36]([H])[c:37]1[H].[H][n+:38]1[c:39]([H])[c:40]([H])[c:41]([H])[c:42]([H])[c:43]1[H]>>[H][H].[H][n+:32]1[c:33]([H])[c:34]([H])[c:35]([H])[c:36]([H])[c:37]1[H].[H][n+:38]1[c:39]([H])[c:40]([H])[c:41]([H])[c:42]([H])[c:43]1[H].[H][c:3]1[c:2]([Br:1])[c:7]([H])[c:6]([C:8]([C:9]([C:10]([H])([H])[O:11][Si:12]([C:13]([H])([H])[H])([C:14]([H])([H])[H])[C:15]([C:16]([H])([H])[H])([C:17]([H])([H])[H])[C:18]([H])([H])[H])([F:19])[F:20])=[O:21])[c:5]([F:22])[c:4]1[H].[H][C:45]([H])([Cl:44])[Cl:46].[Cr:23]([O:24][Cr:25]([O-:26])(=[O:27])=[O:28])([O-:29])(=[O:30])=[O:31]
Top-1 Correct? True
```

### Training

```bash
uv run bash run_train_multigpu.sh runs/tacit/config.yaml 4
```

## Repository layout

```
src/bond_match/
├── train_ranking.py        # training entry point (training/ package, DDP and single GPU)
├── evaluate_ranking.py     # evaluation entry point (evaluation/ package, batch and interactive)
├── hte_wells.py            # converts the published HTE files into wells.csv
├── score_hte.py            # HTE well scoring and metrics
├── training/               # trainer, EMA, DDP, loss and optimizer helpers
├── evaluation/             # evaluator, BE-step operations, inference data, SMILES utilities
├── model/                  # BondGAT encoder, Transformer, heads, scorers
├── modules/                # MoE, attention residuals, LoRA, Muon optimizer, mHC
├── losses/                 # pair-ranking and energy-based losses
├── blockspace/             # Bond-Electron matrix algebra, reaction decomposition, batch.py
├── dataset/                # sharded dataset: molecular graphs, reaction processing, collate
├── config.py               # RankingConfig schema and YAML loader
└── sampler.py              # shard-aware and weighted sampling
runs/                       # config.yaml per model (checkpoints/ from Zenodo)
data/uspto_mit/             # dataset files (from Zenodo)
docs/                       # project page (GitHub Pages)
```

## Citation

```bibtex
@article{tacit,
  title  = {Implicit Learning: Reaction Outcome Prediction from Positive-Only Data},
  author = {Wierich, Nick and Dobberstein, Niklas and Glorius, Frank},
  year   = {2026},
  note   = {TACIT: TAckling Chemistry through ImpliciT negative learning}
}
```

## License

The code, the TACIT checkpoints and the data on Zenodo are released under the
MIT License (see `LICENSE`). The USPTO-MIT splits are by
[Jin et al. (2017)](https://github.com/wengong-jin/nips17-rexgen) and were
released under the MIT License. The retrained baseline models keep their
upstream licenses.
