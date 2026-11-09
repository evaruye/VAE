Frozen surrogate-model bundle for four-property forward prediction:
- EC_final_model.joblib   (CatBoost; 39 inputs)
- SC_final_model.joblib   (XGBoost; 39 inputs)
- VC_model.joblib         (CatBoost; 39 inputs; model output on log1p VC scale)
- HCHO_final_model.joblib (MLP regression pipeline; 40 inputs; includes initiator_feed_semi_continuous)

The application loads these frozen model artifacts for forward prediction.
