import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def parse_url(url, prefix = 'https://www.bol.com/nl/'):
    url_components = url.removeprefix(prefix).split('/')
    row = {
        'url_function': '',
        'category': '',
        'category_id': '',
        'category_filters': [],
        'n_category_filters': 0,
        'attribute_filters': [],
        'n_attribute_filters': 0,
        'search_type': '',
        'search_text': '',
        'search_context': '',
        'Nty': '',
        'product_id': '',
        'other': '',
        'tracking_id': ''
    }

    if url_components[0] == 'c':
        row['url_function'] = url_components[0]

        if url_components[1] == 'ajax':
            row['other'] = url_components[1]

        else:
            row['category'] = url_components[1]

            if url_components[2].isdigit():
                row['category_id'] = url_components[2]

            else:
                row['category'] = row['category'] + '/' + url_components[2]

                if url_components[3].isdigit():
                    row['category_id'] = url_components[3]

        if 'N' in url_components:
            index = url_components.index('N')
            row['category_filters'] = url_components[index + 1].split('+')

        if 'sc' in url_components:
            index = url_components.index('sc')
            row['search_context'] = url_components[index + 1]

        if 'filter_N' in url_components:
            index = url_components.index('filter_N')
            row['attribute_filters'] = url_components[index + 1].split('+')

    elif url_components[0] == 'checkout':
        row['url_function'] = url_components[0]
        row['other'] = url_components[1]

    elif url_components[0] == 'l':
        row['url_function'] = url_components[0]

        if url_components[1] == 'ajax':
            row['other'] = url_components[1]

        else:
            row['category'] = url_components[1]

        if 'N' in url_components:
            index = url_components.index('N')
            row['category_filters'] = url_components[index + 1].split('+')

        if 'filter_N' in url_components:
            index = url_components.index('filter_N')
            row['attribute_filters'] = url_components[index + 1].split('+')

    elif url_components[0] == 'order':
        row['url_function'] = url_components[0]
        row['other'] = url_components[1]

    elif url_components[0] == 'p':
        row['url_function'] = url_components[0]
        row['category'] = url_components[1]

        if url_components[2].isdigit():
            row['product_id'] = url_components[2]

    elif url_components[0] == 's':
        row['url_function'] = url_components[0]

        if url_components[1].isdigit():
            row['category_id'] = url_components[1]
        
        else:
            row['category'] = url_components[1]

        if 'N' in url_components:
            index = url_components.index('N')
            row['category_filters'] = url_components[index + 1].split('+')

        if 'Ntt' in url_components:
            index = url_components.index('Ntt')
            row['search_text'] = url_components[index + 1]

        if 'Nty' in url_components:
            index = url_components.index('Nty')
            row['Nty'] = url_components[index + 1]

        if 'sc' in url_components:
            index = url_components.index('sc')
            row['search_context'] = url_components[index + 1]

        if 'filter_N' in url_components:
            index = url_components.index('filter_N')
            row['attribute_filters'] = url_components[index + 1].split('+')

        if 'ajax' in url_components:
            row['other'] = 'ajax'

    elif url_components[0] == 'w':
        row['url_function'] = url_components[0]

        if url_components[1] == 'ajax':
            row['other'] = url_components[1]

        else:
            row['category'] = url_components[1]

            if url_components[2].isdigit():
                row['tracking_id'] = url_components[2]

            else:
                row['category'] = row['category'] + '/' + url_components[2]
                row['tracking_id'] = url_components[3]

            if 'N' in url_components:
                index = url_components.index('N')
                row['category_filters'] = url_components[index + 1].split('+')

            if 'filter_N' in url_components:
                index = url_components.index('filter_N')
                row['attribute_filters'] = url_components[index + 1].split('+')              

    row['n_category_filters'] = len(row['category_filters'])
    row['n_attribute_filters'] = len(row['attribute_filters'])

    return pd.Series(row)


def load_data():
    data = pd.read_csv('../data/clickdata.csv')

    return data


def preprocess_data(data):
    # Add stratification label
    data['stratification'] = data['visitor_recognition_type'] + '_' + data['ua_agent_class']

    # Filter out labels with less than min samples
    data = data.loc[~data['stratification'].str.contains('RECOGNIZED_Hacker'), :]
    data = data.loc[~data['ua_agent_class'].isin(['Cloud Application', 'Mobile App']), :]

    # Filling in missing values
    data.loc[data['country_by_ip_address'].isna(), 'country_by_ip_address'] = 'UNK'
    data.loc[data['region_by_ip_address'].isna(), 'region_by_ip_address'] = 'UNK'
    data.loc[data['referrer_without_parameters'].isna(), 'referrer_without_parameters'] = ''

    # Splitting class into class and source
    data.loc[data['ua_agent_class'] == 'Browser Webview', 'ua_source'] = 'Webview'
    data.loc[data['ua_agent_class'] == 'Browser Webview', 'ua_agent_class'] = 'Browser'
    data.loc[data['ua_agent_class'] == 'Robot Mobile', 'ua_source'] = 'Mobile'
    data.loc[data['ua_agent_class'] == 'Robot Mobile', 'ua_agent_class'] = 'Robot'

    url_features = ['url_function',
                    'category',
                    'category_id',
                    'category_filters',
                    'n_category_filters',
                    'attribute_filters',
                    'n_attribute_filters',
                    'search_type',
                    'search_text',
                    'search_context',
                    'Nty',
                    'product_id',
                    'other',
                    'tracking_id']

    data[url_features] = data['url_without_parameters'].apply(lambda url: parse_url(url))

    return data

def create_features(data):
    features = ['country_by_ip_address', 
                'region_by_ip_address', 
                'visitor_recognition_type',
                'url_function',
                'category',
                'category_id',
                'n_category_filters',
                'n_attribute_filters',
                'search_type',
                'search_text',
                'search_context',
                'Nty',
                'product_id',
                'other',
                'tracking_id']

    cat_features = ['country_by_ip_address', 
                    'region_by_ip_address', 
                    'visitor_recognition_type',
                    'url_function',
                    'category',
                    'category_id',
                    'search_type',
                    'search_text',
                    'search_context',
                    'Nty',
                    'product_id',
                    'other',
                    'tracking_id']

    numerical_features = ['n_category_filters',
                            'n_attribute_filters']

    X = pd.concat([data[numerical_features],
                    data[cat_features].astype('category')], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(data['ua_agent_class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=data['stratification'], random_state=42)

    return le, X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter search space
    param_dist = {
        # Learning rate and trees
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],

        # Tree structure
        "max_depth": [2, 4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 7, 9],

        # Sampling
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],

        # Regularization
        "gamma": [0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0, 0.01, 0.05, 0.1, 1, 10],  # L1
        "reg_lambda": [0.1, 0.5, 1, 5, 10],       # L2
    }

    # TODO: Causes an error. Needs custom implementation
    # Required for stratification on multi-column ua_agent_class + visitor_recognition_type
    # cv_splitter = skf.split(X, data['stratification'])

    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted")
    }

    clf = XGBClassifier(tree_method="hist", enable_categorical=True)

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=1000,
        scoring=scoring,
        refit='f1_macro',
        cv=skf,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search_results = search.fit(X_train, y_train)

    return search_results

def save_train_results(search_results):
    joblib.dump(search_results, '../checkpoints/XGBoost/search/search_results.pkl')
    
def print_classification_report(clf, X_test, y_test, le):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0.0))

def main():
    data = load_data()
    data = preprocess_data(data)

    le, X_train, X_test, y_train, y_test = create_features(data)

    search_results = train_model(X_train, y_train)

    save_train_results(search_results)

    print_classification_report(search_results.best_estimator_, X_test, y_test, le)

if __name__ == "__main__":
    main()