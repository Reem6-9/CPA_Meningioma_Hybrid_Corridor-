# CPA Meningioma Hybrid Corridor Selection

**Dynamic surgical decision-making using XGBoost and Graph Neural Networks with IONM integration**

## Overview
AI framework for cerebellopontine angle (CPA) meningioma surgical corridor selection, achieving 85% accuracy in predicting optimal hybrid vs standard approaches.

## Key Results
| Metric | Hybrid Approach | Standard Approach | p-value |
|--------|----------------|-------------------|---------|
| Gross Total Resection | 72% | 58% | 0.03 |
| Blood Loss (mL) | 320 ± 120 | 450 ± 150 | 0.01 |
| Model Accuracy | 85% (AUC=0.89) | - | - |

## Quick Start
```bash
pip install -r requirements.txt
python reproduce_results.py
Models

cpa_corridor_classifier.py: XGBoost classifier for corridor selection (SHAP=0.41 for vascular encasement)
gnn_ionm_framework.py: Graph Neural Network with real-time IONM integration
Citation
bibtex@article{fatima2025cpa,
  title={Dynamic surgical decision-making in resection of meningiomas of the cerebellopontine angle},
  author={Fatima, Tahreem and Di Ieva, Antonio and others},
  journal={Journal of Neurosurgery},
  year={2025}
}
Note: This repository contains model implementations only. No patient data is included.
