import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Fetch stock data
def get_stock_data(ticker, start="2015-01-01", end="2024-01-01"):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

# Prepare data for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    return X, y, scaler

# Define LSTM model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  
        return out

# Train the model
def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Predict stock prices
def predict_stock(model, data, scaler, time_step=60):
    model.eval()
    
    test_inputs = data["Close"].values[-time_step:].reshape(-1, 1)
    test_inputs = scaler.transform(test_inputs)

    test_inputs = torch.tensor(test_inputs, dtype=torch.float32).unsqueeze(0)  # Shape: (1, time_step, 1)

    predicted_price = model(test_inputs).detach().numpy()
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
    
    return predicted_price[0, 0]


# Main execution
if __name__ == "__main__":
    stock_symbol = input("Enter the Stock Symbol (Eg : AAPL)")  # Change to any stock ticker
    stock_data = get_stock_data(stock_symbol)

    X_train, y_train, scaler = prepare_data(stock_data)
    model = StockLSTM()
    train_model(model, X_train, y_train, epochs=50)

    predicted_price = predict_stock(model, stock_data, scaler)
    print(f"Predicted stock price for next day: ${predicted_price:.2f}")
