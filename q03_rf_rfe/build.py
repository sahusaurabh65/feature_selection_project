# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(df):
    X = df.drop('SalePrice',axis=1)
    y = df['SalePrice']
    
    model = RandomForestClassifier()
    rfe = RFE(model,n_features_to_select=len(X.columns)/2)
    rfe = rfe.fit(X,y)
    
    return list(X.columns[rfe.support_])



rf_rfe(data)


