# Data

This project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

## Source and License

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI: `10.24432/C5MC89`.

- UCI dataset ID: 697
- Year: 2021
- License: **Creative Commons Attribution 4.0 (CC BY 4.0)**
- Instances: 4,424
- Predictor variables: 36
- Target classes: Dropout / Enrolled / Graduate
- Total columns: 37

The licensed source CSV is included in this repository at:

`data/raw/Predict Student Dropout.csv`

See `DATA_LICENSE_AND_ATTRIBUTION.md` for attribution details.

## Modeling Population

The primary binary modeling task uses resolved outcomes only:

- `Dropout = 1`
- `Graduate = 0`
- `Enrolled` is excluded because the final outcome is unresolved.

This leaves **3,630 resolved-outcome records**.

## Prediction Horizon

The final model is explicitly defined for **End of Semester 1**. Semester 2 variables are excluded from the final model. Semester 1 academic variables must not be used for predictions made before those data are genuinely available.

## Privacy

The repository uses the public UCI research dataset. Do not add private institutional student records, direct identifiers, credentials, or restricted operational data to this public repository.
