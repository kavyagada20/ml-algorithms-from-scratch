import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE ="model.pkl"
PIPELINE_FILE ="pipeline.pkl"

def build_pipeline(nums_attribs,cat_attribs):
    #For numerical columns
    num_pipeline =Pipeline([
        ("Imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])

    #For categorical colums
    cat_pipeline =Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    # construct the full pipeline 
    full_pipeline = ColumnTransformer([ 
        ("num",num_pipeline,nums_attribs),
        ('cat',cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Let's train the model
    housing = pd.read_csv("housing.csv")

    #create a stratified test set
    housing['income_cat'] = pd.cut(housing["median_income"],
                                   bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                                   labels=[1,2,3,4,5] )

    split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)
        test_set  = housing.loc[test_index].drop("income_cat", axis=1)

    housing = train_set     # keep training data in housing
    testing = test_set      # keep test data in testing
    testing.to_csv("input.csv", index=False)


    housing_labels = housing["median_house_value"].copy() 
    housing_features = housing.drop("median_house_value",axis=1)

    nums_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs  = ["ocean_proximity"]

    pipeline = build_pipeline(nums_attribs,cat_attribs)
    housing_prepared  = pipeline.fit_transform(housing_features)
    print(housing_prepared)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained ! Congrats")
else:
    # let's do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input) 
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv",index=False)
    print("Inference is complete, results saved to output.csv . Enjoy!")
    
        

        










    
    
    

