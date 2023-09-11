import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, VotingClassifier, VotingRegressor, AdaBoostClassifier,
                              AdaBoostRegressor)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score as f1_score_, roc_auc_score as roc_auc_score_

from required_functions import *
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

df_ = pd.read_csv("datasets/high_diamond_ranked_10min.csv")

corr_list = df_[df_.columns[1:]].apply(lambda x: x.corr(df_['blueWins']))
cols = []
for col in corr_list.index:
    if (corr_list[col] > 0.2 or corr_list[col] < -0.2):
        cols.append(col)
cols

"""
blueWins(1) = redWins(0) blueFirstBlood(1) = redWins(0)

blueWardsPlaced blueWardsDestroyed  blueKills blueDeaths blueAssists 
blueDragons  blueHeralds  blueTowersDestroyed blueTotalMinionsKilled blueTotalJungleMinionsKilled blueCSPerMin

redWardsPlaced  redWardsDestroyed redKills  redDeaths  redAssists 
redDragons  redHeralds redTowersDestroyed redTotalMinionsKilled  redTotalJungleMinionsKilled redCSPerMin

drop edilecekler: gameId redGoldDiff  redExperienceDiff  redCSPerMin  redGoldPerMin blueGoldDiff  blueExperienceDiff  blueCSPerMin  blueGoldPerMin, 
 blue-redAvgLevel, blueTotalGold, blue-redEliteMonsters, blueDragon ve blueHerald olmalı sadece, ilk 10 dakikada sadece bir defa çıkıyorlar,
"""
# Data Preprocessing
df = df_.drop(["gameId","blueAvgLevel","blueTotalGold", "blueTotalExperience","redFirstBlood", "redGoldDiff",
               "redTotalExperience",  "redExperienceDiff","redAvgLevel", "redTotalGold",  "redGoldPerMin",
               "redEliteMonsters", "blueGoldDiff",  "blueExperienceDiff",  "blueGoldPerMin","blueEliteMonsters",
               "redDragons","redHeralds"],axis=1)
#
# Explonatory Data Analysis
check_df(df)
red, blue = [col for col in df.columns if "red" in col], [col for col in df.columns if "blue" in col]
df.columns
df["redHeralds"].unique()
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=3)

cat_cols = ['blueWins','blueFirstBlood']
num_cols.pop(num_cols.index("blueFirstBlood"))
num_cols.pop(num_cols.index("blueWins"))
num_cols

# Summary
for col in cat_cols:
    cat_summary(df,col,True)

for col in num_cols:
    num_summary(df, col, True)

for col in cat_cols:
    target_summary_with_cat(df, "blueWins", col, True)

for col in num_cols:
    target_summary_with_num(df, "blueWins",col, True)


# Correlation Summary
correlation_matrix(df, cat_cols)

correlation_matrix(df, num_cols)

high_correlated_cols(df)

# Outliers
for col in num_cols:
    print(col, check_outlier(df, col, .01, .99))

for col in num_cols:
    show_outliers(df, col,0.01, 0.99)

check_df(df)
dff = df.copy()

# Removing outliers
for col in num_cols:
    dff = remove_outlier(dff, col,0.01, 0.99)


cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=10)
for col in num_cols:
    dff = robust_scaler(dff,col)

y = dff["blueWins"]
X = dff.drop("blueWins",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20)

base_models(X, y, cv=3) # LogisticRegression() 8093

best_models = hyperparameter_optimization(X, y, 3)
dff_voting = voting_model(best_models, X, y,cv=3)



########################### Model With Preprocessor
df2 = df_.drop(["gameId","blueAvgLevel","blueTotalGold", "blueTotalExperience","redFirstBlood", "redGoldDiff",
               "redTotalExperience",  "redExperienceDiff","redAvgLevel", "redTotalGold",  "redGoldPerMin",
               "redEliteMonsters", "blueGoldDiff",  "blueExperienceDiff",  "blueGoldPerMin","blueEliteMonsters",
               "redDragons","redHeralds"],axis=1)
cat_cols, num_cols, cat_but_car = grab_col_names(df2,cat_th=3)
df2.columns

for col in num_cols:
    remove_outlier(df2,col, q1=.99, q3=.01)

y = df2["blueWins"]
X = df2.drop("blueWins",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20,random_state=42)
best_models = hyperparameter_optimization(X_train, y_train, 3)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
    ],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor',VotingClassifier(voting="soft",estimators=list(best_models.items())))
]).fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test,y_pred) # 0.2768218623481781
acc_score = accuracy_score(y_test, y_pred) # 0.7231781376518218
f1_score = f1_score_(y_test, y_pred) # 0.7241553202218859
roc_auc_score = roc_auc_score_(y_test, y_pred) # 0.7231787312817393

random_user1 = X_test.sample(100, random_state=50)
pipeline_samp1 = pipeline.predict(random_user1)
"""
array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0,
       1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
       1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
       0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1], dtype=int64)
"""

joblib.dump(pipeline, "lol_prediction.joblib")

model = joblib.load("lol_prediction.joblib")

model.predict(random_user1) #0






