# ISD+: Interpretable Structural Distillation for Lightweight Ship Detection in UHR Remote Sensing

This is the official repository for the paper **"ISD+: Interpretable Structural Distillation for Lightweight Ship Detection in UHR Remote Sensing"**.

## 1. Algorithm Overview

The overall training pipeline of our **ISD+** framework is outlined below:

```text
Algorithm 1: Interpretable Structural Distillation (ISD+) Framework
--------------------------------------------------------------------------------
Input:  Training batch B = {(I, Y)}; Pretrained frozen teacher f^T; 
        Lightweight student f^S with parameters Θ_S; 
        Hyperparameters λ_sep, λ_SCM, μ; Tucker ranks R = (r1, r2, r3, r4).
Output: Optimized student parameters Θ_S*.
--------------------------------------------------------------------------------
1: Initialize student parameters Θ_S and distillation loss weights;
2: while not converged do
3:     Sample a mini-batch (I, Y) ~ B;
4:     Extract multi-scale feature maps: F^T = f^T(I) and F^S = f^S(I);
5:     Compute task-driven predictive relevance fields R^T and R^S;
6:     // Phase 1: Structural Tensor Construction (STC)
7:     Construct 4th-order structural tensors T^T and T^S across K scales;
8:     // Phase 2: Structural Tensor Separation (STS)
9:     Decouple T^T into components {T_s^T, T_b^T, T_n^T} via low-rank and sparse optimization;
10:    // Phase 3: Predictive Basis Consistent Distillation (PCSD)
11:    Project tensors into core tensor manifolds via Tucker decomposition with rank R;
12:    Compute distillation loss: L_SCM = L_v + μ_n * L_n + μ_R * L_R;
13:    // Optimization Step
14:    Update student weights: Θ_S ← Θ_S - η * ∇_{Θ_S} (L_det + λ_sep * L_sep + λ_SCM * L_SCM);
15: end while
16: return Θ_S* ← Θ_S.
```

## 2. Main Results

Our lightweight model achieves state-of-the-art trade-offs between precision and efficiency on both Ships-UHR and HRSC2016E benchmarks:

* **ISD+L**: 83.5% AP50 on Ships-UHR, 8.6M Params, 47.5 FPS.
* **ISD+S**: 83.0% AP50 on Ships-UHR, 5.1M Params, 61.5 FPS.

## 3. Environment Setup

```bash
pip install -r requirements.txt
```