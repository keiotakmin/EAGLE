# EAGLE Optimizer

EAGLE (Early Approximated Gradient-based Learning-rate Estimator) is a novel optimizer designed to accelerate deep learning convergence by estimating local curvature from gradient differences. It adaptively switches between the EAGLE update rule and Adam for stable and efficient training.

---

## Features

- **Curvature Estimation**: Approximates second-order information using only first-order gradients.
- **Adaptive Switching**: Dynamically branches to Adam when gradient differences are too small or local curvature is non-convex.
- **Lightweight**: No Hessian computation; minimal memory and compute overhead relative to second-order methods.
- **Plug-and-Play**: Drop-in replacement for SGD/Adam in PyTorch training loops.

---

## Installation

Install from source:

```bash
git clone https://github.com/keiotakmin/EAGLE.git
cd EAGLE
pip install -(as needed) .
```

---

## Quick Start
refer ↑

---

## API Reference

### `EAGLE(params, lr, threshold=1e-4, alpha=5e-3, tau_min=1e-5, tau_max=1e-2)`

- **params**: Iterable of model parameters to optimize.
- **lr**: Base learning rate (η) for EAGLE update.
- **threshold**: Initial gradient-difference threshold (τ₀).
- **alpha**: Scaling factor (α) for adaptive threshold calculation.
- **tau_min**, **tau_max**: Minimum and maximum bounds for τ.

**Behavior:** Uses the EAGLE update rule when conditions permit; otherwise falls back to Adam with default β₁=0.9, β₂=0.999, ε=1e-8.

---

## Experimental Results

| Task                     | Baseline   | Steps to Final Loss | Speedup   |
|--------------------------|------------|---------------------|-----------|
| GPT-2 on SST-2 (6.83×)   | SGD_Mom    | 4,313 / 29,469      | 6.83×     |
|                          | Adam       | 4,355 / 29,469      | 6.77×     |
| ViT-B/16 on CIFAR-10     | SGD_Mom    | 1,607 / 5,473       | 3.41×     |
|                          | Adam       | 829 / 5,473         | 6.60×     |

For full details, refer to the NeurIPS 2025 paper and `appendix/`.
