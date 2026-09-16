# Responsible AI and Governance

## Approved Configuration
- Final model: XGBoost
- Frozen threshold: 0.48
- Prediction horizon: End of Semester 1
- Recommendation: **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

## Required Controls
- Human review of model alerts
- Human override documentation and monitoring
- No autonomous adverse decisions
- Alternative referral pathway for non-alerted students
- Recurring subgroup fairness monitoring
- Program-level performance monitoring
- Probability calibration monitoring
- Privacy and access controls
- Audit logging and model-version tracking

## Fairness Monitoring Priorities
1. Financial-risk group — 20.11 pp recall gap
2. Debtor group — 16.22 pp
3. Age group — 15.12 pp
4. Scholarship group — 11.45 pp
5. Displacement group — 4.89 pp
6. Gender group — 3.58 pp

These are monitoring priorities, not overall fairness scores.

## Governance Principle

A high-risk prediction may create an additional opportunity for support. A low-risk prediction must never remove a student's opportunity to receive support.

Future mitigation experiments require new development and validation data.
