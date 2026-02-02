from pygam import LinearGAM, s, f
import pandas as pd


data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")


data['Timestamp'] = pd.to_datetime(data['Timestamp'])

x = data[['year', 'month', 'day', 'hour']]
y = data['trips']

model = LinearGAM(s(0) + s(1) + f(2) + f(3))
modelFit = model.gridsearch(x.values, y)


next_year = data['year'].max()
jan_data = data[(data['year']==next_year) & (data['month']==1)]
X_jan = jan_data[['year','month','day','hour']].values

pred = modelFit.predict(X_jan)
