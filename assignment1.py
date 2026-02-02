import pandas as pd
from prophet import Prophet


data = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")

data = data[['Timestamp', 'trips']].rename(columns={'Timestamp': 'ds', 'trips': 'y'}).assign(ds=lambda df: pd.to_datetime(df['ds']))

model = Prophet()
model.add_seasonality(name='weekly', period=7, fourier_order=5)
model.add_seasonality(name='daily', period=1, fourier_order=11)
modelFit = model.fit(data)

future = model.make_future_dataframe(periods=744)
forecast = model.predict(future)

plt = model.plot(forecast)
comp = model.plot_components(forecast)

predictions = forecast["predict"].values[-744:]
