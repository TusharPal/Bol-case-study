# Bot traffic classification (Bol.com case study)

## Ideas
- Use time as a feature
- Use session ids with multiple entries to study what legit vs non-legit user activity looks like
- Certain countries/regions may have more bot traffic coming from them
- Distribution of classes between train, val and test needs to be similiar. 
- Weighting for classes, when calculating loss.
- Compare KNN, Catboost, XGBoost and LightGBM
- All entries for a session id should belong to only train, val or test.
- Verify if a session id has entries from different sources.