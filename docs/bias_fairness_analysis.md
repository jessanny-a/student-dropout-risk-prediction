# Bias & Fairness Analysis

The final XGBoost model was evaluated on an untouched 726-record holdout set at a frozen threshold of 0.48.

## Explainability

SHAP, LIME, PDP, and ICE were used to examine model behavior. SHAP probability reconstruction passed with a maximum error of approximately `3.58e-07`. Semester 1 completion rate was the strongest global predictor.

Explanations describe model behavior and associations; they do not establish causal effects.

## Formal Fairness Metrics

| Audit dimension | Demographic parity gap | Disparate impact ratio | Equal opportunity gap | Equalised odds gap |
|---|---:|---:|---:|---:|
| Financial risk | 51.1 pp | 0.357 | 20.1 pp | 20.1 pp |
| Debtor status | 42.5 pp | 0.451 | 16.2 pp | 16.2 pp |
| Age | 47.0 pp | 0.373 | 15.1 pp | 15.1 pp |
| Scholarship status | 33.5 pp | 0.300 | 11.4 pp | 11.4 pp |
| Gender | 23.2 pp | 0.568 | 3.6 pp | 6.0 pp |
| Displacement | 15.3 pp | 0.683 | 4.9 pp | 4.9 pp |
| Tuition status | Not comparable | Not comparable | Not comparable | Not comparable |

Demographic parity and disparate impact are screening measures and are interpreted alongside base-rate differences, support, and operational context.

## Key Limitations

- moderate class imbalance;
- temporal leakage risk if Semester 1 variables are used prematurely;
- nonlinear-model overfitting risk;
- 36 remaining false negatives;
- subgroup performance disparities;
- subgroup sample-size constraints;
- dependence on observable institutional signals;
- explainability is not causality;
- LIME operates in transformed feature space;
- external validity and model drift remain future-cohort risks;
- model is suitable only for supportive, human-reviewed use.
