<div align="center">

**English** | [简体中文](./README_zh.md)

</div>

<h1 align="center">Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization</h1>

<p align="center">
  <a href="https://www.ijcai.org/proceedings/2024/609">
    <img src="https://img.shields.io/badge/Paper-IJCAI_2024-red" alt="Paper">
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="https://github.com/zxlml/BilevelOptimization">
    <img src="https://img.shields.io/github/stars/zxlml/BilevelOptimization?style=social" alt="GitHub Stars">
  </a>
  <img src="https://img.shields.io/github/last-commit/zxlml/BilevelOptimization?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>

<h5 align="center"> ⭐ If you like our project, please give us a star on GitHub for the latest updates! ⭐ </h5>

---

**BilevelOptimization** is the official implementation of the IJCAI 2024 paper *"Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization"*. It provides first-order stochastic bilevel optimization (SBO) solvers together with an empirical validation framework that connects the **on-average argument stability** of an algorithm with its **generalization gap**, reproducing all key findings of the paper on hyperparameter optimization as data reweighting.

* 🧮 **Three Algorithms from the Paper**
  <br> SSGD (Algorithm 1), TSGD with warm start (Algorithm 2), and UD with re-initialization (Algorithm 3), all following the paper's convention (K = outer iterations, T = inner iterations).

* 📈 **Generalization Gap Assessment**
  <br> Full-dataset validation / testing errors and their divergence (Eq. 2), tracked over training for different sample sizes, iteration budgets, and learning rates.

* 🔬 **Empirical Stability Measurement**
  <br> On-average argument stability (Definition 5) measured by re-running the algorithm with one validation sample replaced, validating the O(K/m1) theory (Table 1).

* ⚙️ **Paper-faithful Experiment Pipeline**
  <br> 50% label-corrupted MNIST reweighting, 784/256/10 MLP, batch size 8, step sizes 0.01 (inner) / 5 (outer), multi-seed averaging with CSV + curve outputs.

<span id='news'/>

## 📢 News

- **[2026-09-24]**: 🚀 Major code refactoring released: paper-faithful K/T iteration convention, SSGD/TSGD/UD implementations, fixed generalization-gap estimation, empirical stability measurement, multi-seed experiment driver, and a 10-case unit test suite.
- **[2024-08]**: 🎉🎉🎉 The paper has been accepted by **IJCAI 2024** (proceedings pages 5508–5516).
- **[2024]**: 🚀 Initial code release for hyperparameter optimization (data reweighting) and meta learning.

<span id='contents'/>

## 📑 Table of Contents

* <a href='#algorithms'>🧮 Algorithms</a>
* <a href='#installation'>🔧 Installation</a>
* <a href='#quickstart'>⚡ Quick Start</a>
* <a href='#structure'>📁 Project Structure</a>
* <a href='#results'>📊 Experimental Results</a>
* <a href='#tests'>🧪 Unit Tests</a>
* <a href='#todo'>☑️ Todo List</a>
* <a href='#citation'>📚 Citation</a>
* <a href='#acknowledgement'>🔗 Acknowledgement</a>

<span id='algorithms'/>

## 🧮 Algorithms

The paper considers the stochastic bilevel formulation

```
min_{x}  R(x) = F(x, y*(x)) = E_{ξ}[f(x, y*(x); ξ)]
s.t.     y*(x) = argmin_{y} G(x, y) = E_{ζ}[g(x, y; ζ)]
```

and analyzes first-order gradient methods. Three solvers are provided (`core/ablo.py`):

| Algorithm | Paper | Key Property | Setting |
| :--- | :--- | :--- | :--- |
| **SSGD** | Algorithm 1 | Single timescale, one inner + one outer step per iteration (`T = 1`) | `ho_algo='ssgd'` / `T=1` |
| **TSGD** | Algorithm 2 | Two timescale: `T` inner steps per outer step, **warm start** `y0_{k+1} = y^T_k` | `ho_algo='tsgd'` |
| **UD** | Algorithm 3 (Bao et al., 2021) | Two timescale with **re-initialization** `y0_k = y0` before each outer loop | `ho_algo='ud'` |

The generalization gap (Eq. 2) is estimated by the divergence between the validation error and the testing error, and the empirical `l2` on-average argument stability (Definition 5) is measured via paired runs in which one validation sample is replaced by an independent draw from `D1`.

<span id='installation'/>

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zxlml/BilevelOptimization.git
cd BilevelOptimization

# Create virtual environment
conda create -n sbo python=3.10 -y
conda activate sbo

# Install dependencies
pip install torch torchvision matplotlib pytest
```

The MNIST dataset is downloaded automatically by `torchvision` on the first run (cached under `workspace/datasets/mnist`).

<span id='quickstart'/>

## ⚡ Quick Start

All experiments are driven by `Hyperparameter_Optimization.py` (paper settings by default):

```bash
# 1. Fast end-to-end sanity check (a few seconds)
python Hyperparameter_Optimization.py --mode smoke

# 2. Fig.1 analog: vary inner iterations T (1, 32) with K = 5000, 3 seeds
python Hyperparameter_Optimization.py --mode fig1 --K_max 5000 --seeds 42 52 82

# 3. Fig.2 analog: SSGD (T = 1) with varying validation size m1 (500, 2000), 5 seeds
python Hyperparameter_Optimization.py --mode fig2 --K_max 5000 --seeds 42 52 62 72 82

# 4. Learning-rate sensitivity: lr_l x lr_h grid on SSGD
python Hyperparameter_Optimization.py --mode lr --K_max 3000 --seeds 42 52 62

# 5. Empirical on-average argument stability (Definition 5)
python Hyperparameter_Optimization.py --mode stability --K_max 2000 --seeds 42

# Single run with a custom configuration (also supports --mode all)
python core/run.py  # see Hyperparameter_Optimization.default_config() for the full arg list
```

Every mode writes averaged curves (`*_summary.csv`) and per-curve plots into `workspace/exp/<mode>_<timestamp>/`. Curves are evaluated every `--eval_every` outer iterations on the full validation / test sets.

<span id='structure'/>

## 📁 Project Structure

```
BilevelOptimization/
├── Hyperparameter_Optimization.py   # Experiment driver (smoke / fig1 / fig2 / lr / stability)
├── Meta_Learning.py                 # Omniglot one-shot learning (l2l, SSGD vs TSGD adaptation steps)
├── core/
│   ├── ablo.py                      # SSGD / TSGD / UD solvers + stability measurement
│   ├── datasets.py                  # Corrupted MNIST (50% label noise) & Omniglot generators
│   ├── loss.py                      # ReweightingMLP / MLPFeatureLearning bilevel losses
│   ├── mlp.py                       # MLP parameter factory & forward
│   ├── random_search.py             # Random-search baseline
│   ├── run.py                       # Single-run entry (logging, record saving, plots)
│   └── utils.py                     # Seed / loader / CSV export / record averaging utilities
├── func/                            # Autograd helpers (unrolled differentiation, tensor ops)
├── tests/                           # Unit tests (pytest)
├── results/                         # Curves from our paper-faithful reproduction runs
└── workspace/                       # (git-ignored) datasets, logs, run outputs
```

<span id='results'/>

## 📊 Experimental Results

Reproduction curves (averaged over seeds, matching paper Sec. 5 settings) are provided in the [`results/`](./results/) directory as summary CSVs, without any result numbers repeated here. Run the commands in [Quick Start](#quickstart) to regenerate all curves and summaries.

<span id='tests'/>

## 🧪 Unit Tests

```bash
python -m pytest tests/ -v
```

10 test cases cover: single-step gradient correctness, hypergradient flow to the outer variable, UD re-initialization semantics vs warm start, bilevel convergence, gap consistency, NaN early-stopping, multi-seed record averaging, 50% MNIST label corruption, and a real-MNIST convergence smoke test.

<span id='todo'/>

## ☑️ Todo List

- [ ] GPU acceleration and mixed-precision support
- [ ] Meta-learning (Omniglot) experiment driver with the same multi-seed pipeline
- [ ] More bilevel applications (data cleaning, Coreset selection)
- [ ] Automatic step-size scheduling beyond the constant setting

<span id='citation'/>

## 📚 Citation

If you are interested in this work, please refer to [the IJCAI 2024 proceedings](https://www.ijcai.org/proceedings/2024/609) and cite as:

```bibtex
@inproceedings{ijcai2024p609,
  title     = {Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization},
  author    = {Zhang, Xuelin and Chen, Hong and Gu, Bin and Gong, Tieliang and Zheng, Feng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {5508--5516},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/609},
  url       = {https://doi.org/10.24963/ijcai.2024/609},
}
```

<span id='acknowledgement'/>

## 🔗 Acknowledgement

We thank the authors of the following projects from which part of the codes and data pipelines are adapted:

* [stability_ho](https://github.com/baofff/stability_ho) — Stability and generalization of bilevel programming in hyperparameter optimization (NeurIPS 2021)
* [stocBiO](https://github.com/JunjieYang97/stocBiO) — Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms (ICML 2021)
* [MNIST](http://yann.lecun.com/exdb/mnist/) and [Omniglot](https://github.com/brendenlake/omniglot) datasets
