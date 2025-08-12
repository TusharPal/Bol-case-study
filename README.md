# Bot traffic classification (Bol.com case study)

# Setup
- Clone the repo, and navigate to the root of its directory
- Create the python environment from the `requirements.txt` file, and activate it
- The notebooks for data EDA, KNeighborsClassifier (baseline), XGBoost and CatboostClassifier are in `research`
- Data is in the `data` directory

# Exploratory data analysis
- Browser Webview rows have been been rolled into Browser.
- Robot Mobile rows have been rolled into Robot. 
- Labels are then broadly Browser (human), Robot (search bot), Hacker (scraper/potential competitor/coupon aggregator) and Special(no idea what it is yet). The other labels (Cloud application, Mobile App) are quite less, and so have been filtered out.
- Hacker and Special rows are quite less, compared to Browser and Human. (2% and 0.2%)
- The urls needed to be split up, to extract category, product, listing, and search parameters. These help the models perform better in correctly classifying Hackers

# Experiment design 
- KNN, XGBoost and Catboost classifiers were trained on features from the cleaned data.  
- Data was stratified to maintain similiar distribution of `ua_agent_class` between train, val and test.
- KNN could not handle the high cardinality of the mostly categorical features. Catboost and XGBoost handle categorical features natively, and thus mitigate this problem.
- Macro Precision, Recall and F1 scores are evaluated for comparison, as n_samples per class weighted average Precision/Recall/F1 score would drastically diminish the effect of misclassification for rarer classes (Hacker, Special)

## 

# Future work
- Use time as a feature
- Use session ids with multiple entries to study what legit vs non-legit user activity looks like
- Certain countries/regions may have more bot traffic coming from them
- Weighting for classes, when calculating loss.
- Compare LightGBM as well
- All entries for a session id should belong to only train or test, not both.