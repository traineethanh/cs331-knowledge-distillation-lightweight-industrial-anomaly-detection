---
title: Industrial Anomaly Detection via Knowledge Distillation
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: mit
---

# Knowledge Distillation-Based Industrial Anomaly Detection

**Course:** CS331 — Advanced Computer Vision  
**Authors:** Thành (Researcher A) & Luân (Researcher B)  

## Overview
This interactive demonstration showcases real-time unsupervised anomaly detection and localization on industrial surfaces using Teacher–Student Knowledge Distillation.

### How It Works:
1. **Teacher–Student Distillation:** A frozen ImageNet-pretrained Teacher network extracts multi-scale feature pyramids from industrial images.
2. **One-Class Modeling:** A compact Student network is trained exclusively on defect-free normal products to match teacher representations.
3. **Anomaly Localization:** When presented with a defective query, the student fails to reproduce anomalous teacher features, producing sharp localized discrepancies across spatial scales.

### Features:
- Real-time CPU inference ($<60\text{ ms}$ per image).
- Multi-scale anomaly heatmap visualization with alpha overlay.
- Diagnostic decision output (PASS / DEFECT) based on calibrated category thresholds.
