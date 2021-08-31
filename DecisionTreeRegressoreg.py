import pandas as pd
from sklearn.tree import DecisionTreeRegressor

datafile_path='datafile.csv'
datafile_data=pd.read_csv(datafile_path)
print(datafile_data.columns)
datafile_data=datafile_data.dropna(axis=0)
y=datafile_data.Supportprice
datafile_features=['Crop','CostofCultivation','CostofCultivation2','CostofProduction','Yield']
X=datafile_data[datafile_features]
X.describe()
print("The top values for prediction validation are:")
print(X.head())
datafile_features=['CostofCultivation','CostofCultivation2','CostofProduction','Yield']
X=datafile_data[datafile_features]
X.describe()
datafile_model=DecisionTreeRegressor(random_state=0)
datafile_model.fit(X,y)
print("The predictions are")
print(datafile_model.predict(X.head()))
