import math
import csv
import numpy as np
from keras.layers import LSTM, Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

# USER INPUT
amazon = "csv_items/Amazon.csv"
apple = "csv_items/Apple.csv"
facebook = "csv_items/Facebook.csv"
google = "csv_items/Google.csv"
netflix = "csv_items/Netflix.csv"

file = [amazon, apple, facebook, google, netflix]
company = ["Amazon", "Apple", "Facebook", "Google", "Netflix"]

# SETTING DEFAULT VALUES FOR PLACEHOLDERS
number = 0
iterations = 1
size = 0.80
df = pd.read_csv("csv_items/Amazon.csv")

# REAL TIME QUERIES & SECURITY MEASURES
# Exception handling to ensure the correct values are input.
print("\nWelcome to the Stock Prediction Program. \n"
      "\nPlease enter the company you would like to predict using integer values")
try:
    number = input("Options include Amazon(0), Apple(1), Facebook(2), Google(3), Netflix(4): ")
    df = pd.read_csv(file[int(number)])
except ValueError or IndexError:
    print("Invalid input. Value must be from 0-4")
    exit()
try:
    iterations = int(input(
        "\nHow many times would you like the model to train with the data? More iterations yields a better "
        "prediction at the cost of time. \nRecommended values include:\n1 (up to ten seconds)\n5 (up to 25 "
        "seconds)\n10 (up to 35 seconds)\n"))
except ValueError:
    print("Invalid input.")
    exit()
try:
    size = int(
        input("\nFinally, how much of the data would you like the model to train with? More data also yields a better "
              "prediction.\nEnter value as a percentage of 100. (Values between 60 and 90 are recommended)\n"))
except ValueError:
    print("Invalid input.")
    exit()
Size = int(size) / 100
print("You selected '" + company[int(number)] + "' Please wait while the prediction data is calculated.")

# LOAD IN THE DATA
plt.style.use('fivethirtyeight')
# df = pd.read_csv("csv_items/amazon.csv")
# print(df.shape)

# DATA VISUALIZATION
# Descriptive method 1
plt.figure(figsize=(12, 6))
plt.title('Original Close Price History')
plt.plot(df['Close'])
plt.xlabel('Time')
plt.ylabel('Close Price US ($)')
# plt.show()

# Descriptive method 2
plt.figure(figsize=(12, 6))
plt.title('Close Price Histogram')
plt.hist(df['Close'])
plt.xlabel('Price')
plt.ylabel('Frequency')


# GET CLOSING DATA TO USE WITH MODEL
# DATA CLEANING
main_data = df.filter(['Close'])
data_set = main_data.values
training_set = math.ceil(len(data_set) * Size)
# print(training_set)

# DATA SCALING / PREPROCESSING
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_set)
# print(data_scaled)

# CREATE TRAINING SET
training_data = data_scaled[0:training_set, :]
x_train = []
y_train = []

for i in range(60, len(training_data)):
    x_train.append(training_data[i - 60:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)

# BUILD LONG SHORT TERM MEMORY (LSTM) MODEL
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# MODEL COMPILATION
model.compile(optimizer='adam', loss='mean_squared_error')

# BEGIN TRAINING THE MODEL
# ADAPTIVE ELEMENT - user can select how much they want the model to train and improve predictions.
# fit the training data to our model
model.fit(x_train, y_train, batch_size=3, epochs=int(iterations))

# CREATING THE TEST DATA
testing_data = data_scaled[training_set - 60:, :]
x_test = []
y_test = data_set[training_set:, :]

for i in range(60, len(testing_data)):
    x_test.append(testing_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# MAKE PREDICTIONS
# reverse the scaling to return the values to normal
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)
root_mean = np.sqrt(np.mean(prediction - y_test) ** 2)
print("The calculation has finished...")
print("View the provided graphs to see more information.\nFigure 3 has the prediction results.\n")

# Descriptive method 3 - Displaying root mean squared error
# ACCURACY VALIDATION
print("The Root Mean Squared value is: " + str(root_mean) + ". The closer the value is to 0, the more accurate the "
                                                            "prediction was.")

# MAKE PLOTS
train_plot = main_data[0:training_set]
validation_plot = main_data[training_set:]
# validation_plot['Prediction'] = prediction
validation_plot.insert(loc=0, column='Prediction', value=prediction)

# Non-Descriptive method. prediction results.
print("Below is a printed version of the prediction graph to see the prediction vs. actual results.\n")
print(validation_plot)

# Non-Descriptive method. prediction results.
plt.figure(figsize=(12, 6))
plt.title(f"{company[int(number)]} prediction model")
plt.xlabel('Time')
plt.ylabel('Close Price US ($)')
plt.plot(train_plot['Close'])
plt.plot(validation_plot[['Close', 'Prediction']])
plt.legend(['Training', 'Actual', 'Prediction'], loc='lower right')
plt.show()
