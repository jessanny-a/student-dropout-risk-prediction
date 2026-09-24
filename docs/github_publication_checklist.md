# GitHub Publication Checklist

- [ ] Rename the repository if desired.
- [ ] Add the final cleaned notebook under `notebooks/`.
- [ ] Refactor reusable notebook code into `src/`.
- [ ] Confirm the dataset source and redistribution license.
- [ ] Decide whether raw data may be committed.
- [ ] Pin exact package versions.
- [ ] Add the final report under `reports/`.
- [x] Select an open-source license — **MIT**.
- [ ] Run a secret/PII scan.
- [ ] Verify all notebook cells run top-to-bottom.
- [ ] Verify final metrics match the README.
- [ ] Verify governance language matches the final Step 5 conclusion.
- [x] Create the GitHub repository.
- [ ] Upload any remaining notebook and binary artifacts.
- [ ] Verify repository topics/tags and concise description.
- [ ] Add the final GitHub URL to the report and presentations if required.

## Current repository

`https://github.com/jessanny-a/student-dropout-risk-prediction`

## Final publication checks

Before considering the repository fully published, confirm that the following files are physically present in GitHub rather than only referenced in README files:

- `notebooks/01_student_dropout_capstone.ipynb`
- `notebooks/01_student_dropout_capstone_no_outputs.ipynb`
- `notebooks/student_dropout_technical_jupyter_slides_updated.ipynb`
- `presentations/student_dropout_business_deck.pptx`
- `reports/final_report.pdf`
- `reports/final_report.docx`
- `models/final_student_dropout_xgboost.joblib` if you choose to publish the serialized model

Large or binary artifacts should be reviewed for privacy, licensing, and repository-size implications before upload.
