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
