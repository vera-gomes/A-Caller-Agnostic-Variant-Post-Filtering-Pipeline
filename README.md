# Caller-Agnostic-Variant-Post-Filtering-Pipeline

<div align="center">

<p>This repository accompanies the paper:</p>

<p>
  Pinto V, Sousa L, Silva C<br>
  <b>A Caller-Agnostic Variant Post-Filtering Pipeline</b>
</p>

<p>
  This project presents a transparent, caller-agnostic framework for post-filtering genomic variant calls using generative and discriminative machine-learning models, with per-variant audit outputs, fixed-sensitivity tranche analyses, and bootstrap confidence intervals.
</p>

</div>

---

## Overview

Raw variant calls often require post-filtering to remove artefacts while preserving true variants for downstream interpretation. This repository contains the code used to implement and evaluate a caller-agnostic post-filtering workflow on **NA12878 / HG001**, using **GRCh38 (hg38)** and a **GIAB high-confidence truth set**. The framework compares multiple post-filtering models across different upstream variant callers and supports both aggregate and per-variant evaluation. :contentReference[oaicite:0]{index=0}

The pipeline was designed to:

- compare **Gaussian Mixture Model (GM)**, **Bayesian Gaussian Mixture (BGM)**, **Logistic Regression (LR)**, **Random Forest (RF)**, **LightGBM (LGB)**, and **Bayes-optimized LightGBM**
- operate across multiple upstream callers in a **caller-agnostic** way
- export **per-variant TP / FP / FN / TN audit tables**
- evaluate **fixed-sensitivity tranche thresholds**
- quantify uncertainty with **95% bootstrap confidence intervals** :contentReference[oaicite:1]{index=1}

---


