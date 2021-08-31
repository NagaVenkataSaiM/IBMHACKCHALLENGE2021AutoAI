import pandas as pd
from sklearn.tree import DecisionTreeRegressor

datafile_path='datafile.csv'
datafile_data=pd.read_csv(datafile_path)
print(datafile_data.columns)
datafile_data=datafile_data.dropna(axis=0)
y=datafile_data.Supportprice
datafile_features=['CostofCultivation','CostofCultivation2','CostofProduction','Yield']
X=datafile_data[datafile_features]
X.describe()
print(X.head())
datafile_model=DecisionTreeRegressor(random_state=0)
datafile_model.fit(X,y)
print("Making prediction of 5 states:")
print(X.head())
print("The predictions are")
print(datafile_model.predict(X.head()))
