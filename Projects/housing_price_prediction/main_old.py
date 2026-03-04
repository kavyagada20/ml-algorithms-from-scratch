from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

# 1. load the dataset
import pandas as pd
import numpy as np
housing = pd.read_csv("housing.csv")

# 2.Create a stratified test set
housing['income_cat'] = pd.cut(housing['median_income'],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat',axis=1)
    strat_test_set = housing.loc[test_index].drop('income_cat',axis=1)

# we will work on the the train data
housing = strat_train_set.copy()

# 3.Separate the predictors and the labels
housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value',axis=1)

# print(housing, housing_labels)

# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
# print(num_attribs
#       , cat_attribs)

# 5. Create piplelines for numerical columns
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",StandardScaler() )
    ]
)

# 6. Create piplelines for categorical columns
cat_pipeline = Pipeline(
    [
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# construct the full pipeline
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ]
)

# 6. Transform the data 
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)
# print(housing_prepared.shape)
# print(type(housing_prepared))
# it is a sparse matrix, convert it to a dense matrix
# housing_prepared = housing_prepared.toarray()
# print(housing_prepared)
# print(housing_prepared.shape) 
# print(type(housing_prepared))

# 7. Now housing_prepared and housing_labels are ready to be used in any ML model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print("Linear Regression RMSE :", lin_rmse)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print("Linear Regression RMSE:", lin_rmse)

# Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared , housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
# tree_rsme = root_mean_squared_error(housing_labels, housing_predictions)
tree_rsme = -cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_root_mean_squared_error",
                            cv=10) # 10 cross validation set will be formed
# print(pd.Series(tree_rsme).describe())
print("Decision Tree RMSE :", tree_rsme.mean())

# Random Forest
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared , housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
# forest_rmse = root_mean_squared_error(housing_labels, housing_predictions)
forest_rmse_scores = -cross_val_score(forest_reg, housing_prepared, housing_labels,
                                      scoring="neg_root_mean_squared_error", cv=10)
print("Random Forest Cross-Validated RMSE:", forest_rmse_scores.mean())


# Note: The RMSE values on the training set are likely to be very low, especially for Decision Tree and Random Forest, indicating overfitting.
# To properly evaluate the models, you should use cross-validation or a separate validation set.    
# For example, using cross-validation for Random Forest:

 # cross_validation means splitting the training set into 10 different sets and training the model on 9 sets and testing it on the 10th set.
 # This process is repeated 10 times, each time with a different set as the test set
# forest_rmse_scores = -cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                       scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmse_scores).describe())
# print("Random Forest Cross-Validated RMSE:", forest_rmse_scores.mean())

# The cross-validated RMSE gives a better estimate of the model's performance on unseen data.
# 8. You can try other models as well





