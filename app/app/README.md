# FastAPI forward-prediction platform

This folder contains the self-contained web application used for formulation-level
forward prediction of viscosity (VC), solids content (SC), ethylene content (EC),
and free formaldehyde (HCHO).

## Run on Windows

1. Create/activate a Python environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Start the application:

   `python run.py`

The interface is served at `http://127.0.0.1:8000`.

The application contains its own frozen model bundle and feature assets so that it can
run independently of the top-level research scripts. The raw reference table in
`feature_assets/VAE_formulation_dataset.xlsx` is the corrected HCHO-consistent version
stored with the application. The platform-specific frozen master reference is stored as
`feature_assets/X_reduced_final_features_only.xlsx`.

## Repository packaging

A Python virtual environment is not distributed in this repository. `app/.venv/` is an installation-local folder and is ignored by Git; create it on your own system before starting the server. The platform contains separate frozen artifacts and may use different ingredient identifiers from the English-standardized modeling workbooks.

## Smoke test (verified)

From the repository root, after installing `app/requirements.txt` (including `httpx`), run:

```bash
python app/smoke_test.py
```

This checks HTTP 200 for the frontend, model and feature readiness, the input schema, a raw example, four-target predictions from raw inputs, four-target predictions from a 40/39-feature vector, and the expected rejection of missing process inputs. The shipped example returns approximately VC 328.813 mPa·s, SC 49.910 wt%, EC 22.448 wt%, HCHO 23.065 ppm. The test uses FastAPI TestClient; a separate graphical-browser UI test has not been completed. Models were originally serialized with scikit-learn 1.7.0; the repository requirements list that version. Unpickling a model from an incompatible version may trigger warnings or fail.
