import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import warnings
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, VotingClassifier, VotingRegressor, AdaBoostClassifier,
                              AdaBoostRegressor)
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)


################ Explonatory Data Analysis

def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe: pd.DataFrame, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and
                   dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                   dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations (Gözlem Birimleri): {dataframe.shape[0]}")
    print(f"Variables (Değişkenler): {dataframe.shape[1]}")
    print(f"cat_cols (kategorik_değişkenler): {len(cat_cols)} - {cat_cols}")
    print(f"num_cols (numerik_değişkenler): {len(num_cols)} - {num_cols}")
    print(f"cat_but_car (kategorik_ama_kardinal): {len(cat_but_car)} - {cat_but_car}")
    print(f"num_but_cat (numerik_ama_kategorik): {len(num_but_cat)} - {num_but_cat}")

    return cat_cols, num_cols, cat_but_car


################ Explonatory Data Analysis

################ Summaries

def cat_summary(dataframe, categorical_col, plot=False):
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}))
    print("#####################################")
    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe: pd.DataFrame, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col + '_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)


################ Summaries

################ Correlation Summaries

def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show(block=True)


def df_corr(dataframe, annot=True):
    sns.heatmap(dataframe.corr(), annot=annot, linewidths=.2, cmap='Reds', square=True)
    plt.show(block=True)


def high_correlated_cols(dataframe, head=10):
    corr_matrix = dataframe.corr().abs()
    corr_cols = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                   .astype(bool)).stack().sort_values(ascending=False)).head(head)
    return corr_cols


################ Correlation Summaries

################ Outliers
def outlier_thresholds(dataframe: pd.DataFrame, value: str, q1=0.25, q3=0.75):
    q1 = dataframe[value].quantile(q1)
    q3 = dataframe[value].quantile(q3)
    iqr = q3 - q1
    up = q3 + iqr * 1.5
    low = q1 - iqr * 1.5

    return up, low


def check_outlier(dataframe: pd.DataFrame, value: str, q1=0.25, q3=0.75):
    up, low = outlier_thresholds(dataframe, value, q1=q1, q3=q3)
    if dataframe[(dataframe[value] > up) | (dataframe[value] < low)].any(axis=None):
        return True
    else:
        return False


def show_outliers(dataframe: pd.DataFrame, col_name: str, q1=0.25, q3=0.75, index=False):
    up, low = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].head())
    else:
        print(dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)])

    if index:
        return dataframe[(dataframe[col_name] > up) | (dataframe[col_name] < low)].index


def remove_outlier(dataframe: pd.DataFrame, col_name, q1=0.25, q3=0.75):
    up, low = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    return df_without_outliers


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75, low_threshold=False):
    up_limit, low_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    if low_threshold:
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit


############# Outliers

############# Missing Values

def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_cols


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n")


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir
    temp_target = data[target]
    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
    data[target] = temp_target
    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")
    return data


############# Missing Values

############# Scaling the Numbers

def robust_scaler(dataframe: pd.DataFrame, num_col):
    dataframe[num_col] = RobustScaler().fit_transform(dataframe[[num_col]])
    return dataframe


def min_max_scaler(dataframe: pd.DataFrame, num_col):
    dataframe[num_col] = MinMaxScaler().fit_transform(dataframe[[num_col]])
    return dataframe


def standard_scaler(dataframe: pd.DataFrame, num_col):
    dataframe[num_col] = StandardScaler().fit_transform(dataframe[[num_col]])
    return dataframe


############# Scaling the Numbers

############# Encoding the Categorical Variables

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n")


def rare_encoder(dataframe: pd.DataFrame, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and
                    (temp_df.value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


############# Encoding the Categorical Variables


############# Machine Learning Pipeline

def base_models(X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], is_classifier=True):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Base Models....")
    if is_classifier:
        models = {
            "GNB":GaussianNB(), #Multinomial Naive Bayes,  Complement Naive Bayes, Bernoulli Naive Bayes, Categorical Naive Bayes
            "SGD":SGDClassifier(),
            "LR": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "SVC": SVC(),
            "CART": DecisionTreeClassifier(),
            "RF": RandomForestClassifier(),
            "Adaboost": AdaBoostClassifier(),
            "GBM": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(verbose=-1),# HistGradientBoostingClassifier
            "Catboost": CatBoostClassifier(verbose=False)
        }
    else:
        models = {
            "SGD": SGDRegressor(),
            "LR": LogisticRegression(),
            "KNN": KNeighborsRegressor(),
            "SVC":SVC(),
            "CART": DecisionTreeRegressor(),
            "RF": RandomForestRegressor(),
            "Adaboost": AdaBoostRegressor(),
            "GBM": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMRegressor(verbose=-1),
            "Catboost": CatBoostRegressor(verbose=False)
        }

    for name, model in models.items():
        print(f"################## {name} ################## ")
        for score_param in scoring:
            cv_results = cross_validate(model, X, y, cv=cv, scoring=score_param)
            print(f"{score_param}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")
        print()


def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc", is_classifier=True, is_grid_search=True):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Hyperparameter Optimization.....")
    knn_params = {"n_neighbors": (2, 50)}
    cart_params = {"max_depth": range(1, 20),
                   "min_samples_split": range(2, 30)}
    rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "sqrt"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}
    xboost_params = {"learning_rate": [0.1, 0.01],
                     "max_depth": [5, 8],
                     "n_estimators": [100, 200],
                     "colsample_bytree": [0.5, 1]}
    lightgbm_params = {"learning_rate": [0.01, 0.1],
                       "n_estimators": [300, 500],
                       "colsample_bytree": [0.7, 1]}

    cat_params = {'learning_rate': [0.09, 0.1, 0.12, 0.13],
                "max_depth": [3, 4, 5, 6],
                "n_estimators": [200,250,259, 260, 261]}
    lr_params = {
             'max_iter': [100, 200, 250, 300, 350],
             #'solver': ['lbfgs', 'liblinear', 'sag','saga', "newton-sg"],
             'penalty':[None, 'l2','l1' ],
             'C':[1.0, 2.5, 1.5, 3.0],
             'tol':[0.0001, 0.1, 1, 10],
             'class_weight':['balanced',None],
             'multi_class':['auto','ovr','multinomial']}

    svc_params = {
             'max_iter': [-1, 5, 10],
             'cache_size': [100,200,300],
              #'kernel':['rbf'],
             'degree': [1,2,3,4],
             'C':[1.0, 2.5, 1.5, 3.0],
             'tol':[0.0001, 0.1, 1, 10],
             'class_weight':['balanced',None],
             'decision_function_shape':['ovr','ovo']}

    ada_params = {
    'n_estimators': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 20, 30, 50],
    'learning_rate': [(0.97 + x / 100) for x in range(0, 20)],
    'algorithm': ['SAMME', 'SAMME.R']
}

    gbm_params = {
             'learning_rate': [0.09,0.1, 0.085, 0.08],
             'loss': ['log_loss',"exponential"],
             'max_depth': [3,4,5,6],
             'max_features': [2,3,4, None],
             'max_leaf_nodes': [2,3, None],
             'n_estimators': [100,200,250]}


    if is_classifier:
        models = [
            ("LR", LogisticRegression(), lr_params),
            #("SVC", SVC(), svc_params),
            ("Adaboost", AdaBoostClassifier(), ada_params),
            ("GBM", GradientBoostingClassifier(), gbm_params),
            ("Catboost", CatBoostClassifier(verbose=False), cat_params),
            ("KNN", KNeighborsClassifier(), knn_params),
            ("CART", DecisionTreeClassifier(), cart_params),
            ("RF", RandomForestClassifier(), rf_params),
            ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xboost_params),
            ("LightGBM", LGBMClassifier(verbose=-1), lightgbm_params)
        ]
    else:
        models = [
            ("LR", LogisticRegression(), lr_params),
            #("SVC", SVC(), svc_params),
            ("Adaboost", AdaBoostRegressor(), ada_params),
            ("GBM", GradientBoostingRegressor(), gbm_params),
            ("Catboost", CatBoostRegressor(verbose=False), cat_params),
            ("KNN", KNeighborsRegressor(), knn_params),
            ("CART", DecisionTreeRegressor(), cart_params),
            ("RF", RandomForestRegressor(), rf_params),
            ("XGBoost", XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xboost_params),
            ("LightGBM", LGBMRegressor(verbose=-1), lightgbm_params)
        ]
    best_models = {}
    for name, model, params in models:
        print(f"########### {name} ###########")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        if is_grid_search:
            s_best = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        else:
            s_best = RandomizedSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = model.set_params(**s_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {s_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models


def voting_model(best_models, X, y, cv=5, is_classifier=True):
    if is_classifier:
        print(f"Voting Model: Classifier...")
        voting = VotingClassifier(estimators=list(best_models.items()), voting="soft").fit(X, y)
    else:
        print(f"Voting Model: Regression...")
        voting = VotingRegressor(estimators=list(best_models.items()), voting="soft").fit(X, y)

    cv_results = cross_validate(voting, X, y, cv=cv, scoring=["accuracy","precision", "recall", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting

############# Machine Learning Pipeline


############# Importance Table

def plot_importance(model, features, len_of_x, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len_of_x])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

############# Importance Table

### by Hüseyin Battal ###
## https://www.linkedin.com/in/h%C3%BCseyin-battal-31433116b/ ##