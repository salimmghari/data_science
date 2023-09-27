import re
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from selenium import webdriver
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


data = {
    'dates': [],
    'temperatures': []
}
page_source = ''
days_mapping = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}
current_date = datetime.now()
current_weekday = current_date.weekday()


class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


driver = webdriver.Chrome()
driver.get('https://forecast.weather.gov/MapClick.php?lat=41.884250000000065&lon=-87.63244999999995#.XtpdeOfhXIX')
driver.implicitly_wait(2)
page_source = driver.page_source
driver.quit()

soup = BeautifulSoup(page_source, 'html.parser')
items = soup.find_all('div', class_='tombstone-container')
for item in items: 
    date = item.find(class_='period-name').text.lower()    
    if 'night' in date: continue
    elif 'today' in date: 
        date = current_date.strftime('%Y-%m-%d')
    else:
        date = (current_date + timedelta(days=(days_mapping[date] - current_weekday) % 7)).strftime('%Y-%m-%d')
    data['dates'].append(date)
    data['temperatures'].append(float(re.search(r'[-+]?\d*\.\d+|\d+', item.find(class_='temp').text).group()))

plt.figure(figsize=(10, 6))
plt.plot(data['dates'], data['temperatures'], marker='o')
plt.title('Temperature Over Time')
plt.xlabel('Dates')
plt.ylabel('Temperatures')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data_frame = pd.DataFrame(data)
data_frame['dates'] = pd.to_datetime(data['dates'], format='%Y-%m-%d')
X = data_frame['dates'].values.reshape(-1, 1)
y = data_frame['temperatures'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=100, verbose=0)
y_pred = model.predict(X_test_scaled)
tf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'TensorFlow RMSE: {tf_rmse:.2f}')
future_dates = pd.date_range(start=data_frame['dates'].max(), periods=5, freq='D')
future_dates = future_dates.to_numpy().astype(int) // 10**9
future_dates_scaled = scaler.transform(future_dates.reshape(-1, 1))
future_temperatures_tf = model.predict(future_dates_scaled).flatten()
for date, temp in zip(future_dates, future_temperatures_tf):
    print(f'Date: {pd.to_datetime(date, unit="s")}, TensorFlow Predicted Temperature: {temp:.2f}')
    
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()
pytorch_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'PyTorch RMSE: {pytorch_rmse:.2f}')
future_dates = pd.date_range(start=data_frame['dates'].max(), periods=5, freq='D')
future_dates = future_dates.to_numpy().astype(int) // 10**9
future_dates_scaled = scaler.transform(future_dates.reshape(-1, 1))
future_temperatures_pytorch = model(torch.tensor(future_dates_scaled, dtype=torch.float32)).detach().numpy().flatten()
for date, temp in zip(future_dates, future_temperatures_pytorch):
    print(f'Date: {pd.to_datetime(date, unit="s")}, PyTorch Predicted Temperature: {temp:.2f}')

rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_model.fit(X_train_scaled, y_train)
rf_pred = rf_reg_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f'Random Forest RMSE: {rf_rmse:.2f}')
future_temperatures_rf = rf_reg_model.predict(future_dates_scaled)
for date, temp in zip(future_dates, future_temperatures_rf):
    print(f'Date: {pd.to_datetime(date, unit="s")}, Random Forest Predicted Temperature: {temp:.2f}')
