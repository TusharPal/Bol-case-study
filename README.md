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
- The urls needed to be split up, to extract product and listing info, as well as search parameters. These help the models perform better in correctly classifying Hackers

# Experiment design 
- Distribution of classes between train, val and test needs to be similiar. 

# Future work
- Use time as a feature
- Use session ids with multiple entries to study what legit vs non-legit user activity looks like
- Certain countries/regions may have more bot traffic coming from them
- Weighting for classes, when calculating loss.
- Compare LightGBM as well
- All entries for a session id should belong to only train, val or test.