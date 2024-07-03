import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
from itertools import cycle
import io
import warnings
warnings.filterwarnings("ignore")

st.title("Stock Price Prediction using LSTM")

st.sidebar.header("Upload your file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == "csv":
        reliance = pd.read_csv(uploaded_file, index_col='Date', parse_dates=True)
    elif file_type == "xlsx":
        reliance = pd.read_excel(uploaded_file, index_col='Date', parse_dates=True)

    # Check for NaN values and handle them
    if reliance.isnull().values.any():
        st.write("## Handling Missing Data")
        st.write("Data contains NaN values. Filling NaN values with forward fill method.")
        reliance.fillna(method='ffill', inplace=True)
        if reliance.isnull().values.any():
            st.write("Some NaN values still exist. Filling remaining NaN values with backward fill method.")
            reliance.fillna(method='bfill', inplace=True)
    
    st.write("## Dataset Preview")
    st.write(reliance.head())

    st.write("## Data Description")
    st.write(reliance.describe())

    st.write("## Data Information")
    buffer = io.StringIO()
    reliance.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("## Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(reliance.corr(), annot=True)
    st.pyplot(plt.gcf())

    st.write("## Pairplot")
    sns.pairplot(reliance)
    st.pyplot(plt.gcf())

    st.write("## Time Series Plots")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    reliance['Open'].plot(ax=axes[0, 0], title='Open', color='green')
    reliance['Close'].plot(ax=axes[0, 1], title='Close', color='red')
    reliance['High'].plot(ax=axes[1, 0], title='High', color='blue')
    reliance['Low'].plot(ax=axes[1, 1], title='Low', color='cyan')
    st.pyplot(fig)

    st.write("## Volume Plot")
    plt.figure(figsize=(15, 6))
    plt.plot(reliance['Volume'])
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Date vs Volume')
    st.pyplot(plt.gcf())

    st.write("## Moving Averages")
    reliance_ma = reliance.copy()
    reliance_ma['30-day MA'] = reliance['Close'].rolling(window=30).mean()
    reliance_ma['100-day MA'] = reliance['Close'].rolling(window=100).mean()

    plt.figure(figsize=(15, 6))
    plt.plot(reliance_ma['Close'], label='Original data')
    plt.plot(reliance_ma['30-day MA'], label='30-MA')
    plt.legend()
    plt.title('Stock Price vs 30-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt.gcf())

    plt.figure(figsize=(15, 6))
    plt.plot(reliance_ma['Close'], label='Original data')
    plt.plot(reliance_ma['100-day MA'], label='100-MA')
    plt.legend()
    plt.title('Stock Price vs 100-day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(plt.gcf())

    st.write("## Model Training")
    close_df = pd.DataFrame(reliance['Close']).reset_index()
    close_stock = close_df.copy()
    close_df.drop(columns=['Date'], inplace=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    closedf = scaler.fit_transform(close_df.values.reshape(-1,1))

    training_size = int(len(closedf) * 0.86)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[0:training_size], closedf[training_size:]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 13
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    st.write("### Model Summary")
    model.summary(print_fn=lambda x: st.text(x))

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)
            st.success("Model trained successfully!")

            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)
            original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
            original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

            st.write("### Performance Metrics")
            st.write("Train data RMSE: ", mean_squared_error(original_ytrain, train_predict, squared=False))
            st.write("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
            st.write("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))
            st.write("Test data RMSE: ", mean_squared_error(original_ytest, test_predict, squared=False))
            st.write("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
            st.write("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))
            st.write("Train data explained variance score:", explained_variance_score(original_ytrain, train_predict))
            st.write("Test data explained variance score:", explained_variance_score(original_ytest, test_predict))
            st.write("Train data R2 score:", r2_score(original_ytrain, train_predict))
            st.write("Test data R2 score:", r2_score(original_ytest, test_predict))

            look_back = time_step
            trainPredictPlot = np.empty_like(closedf)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

            testPredictPlot = np.empty_like(closedf)
            testPredictPlot[:, :] = np.nan
            start_index = len(closedf) - len(test_predict)
            testPredictPlot[start_index:, :] = test_predict

            names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])
            plotdf = pd.DataFrame({
                'Date': close_stock['Date'],
                'original_close': close_stock['Close'],
                'train_predicted_close': trainPredictPlot.reshape(-1).tolist(),
                'test_predicted_close': testPredictPlot.reshape(-1).tolist()
            })

            fig = px.line(plotdf, x='Date', y=['original_close', 'train_predicted_close', 'test_predicted_close'],
                          labels={'value': 'Stock price', 'Date': 'Date'})
            fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                              plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
            fig.for_each_trace(lambda t: t.update(name=next(names)))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig)

            st.write("### Future Prediction")
            x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
            temp_input = list(x_input[0])
            lst_output = []

            pred_days = st.slider("Number of days to predict", min_value=1, max_value=365, value=30, step=1)
            if st.button("Predict"):
                with st.spinner("Predicting future values..."):
                    i = 0
                    while i < pred_days:
                        if len(temp_input) > time_step:
                            x_input = np.array(temp_input[1:])
                            x_input = x_input.reshape(1, -1)
                            x_input = x_input.reshape((1, time_step, 1))
                            yhat = model.predict(x_input, verbose=0)
                            temp_input.extend(yhat[0].tolist())
                            temp_input = temp_input[1:]
                            lst_output.extend(yhat.tolist())
                            i += 1
                        else:
                            x_input = x_input.reshape((1, time_step, 1))
                            yhat = model.predict(x_input, verbose=0)
                            temp_input.extend(yhat[0].tolist())
                            lst_output.extend(yhat.tolist())

                    last_days = np.arange(1, time_step + 1)
                    day_pred = np.arange(time_step + 1, time_step + pred_days + 1)

                    temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
                    temp_mat[:] = np.nan
                    temp_mat = temp_mat.reshape(1, -1).tolist()[0]

                    last_original_days_value = temp_mat
                    next_predicted_days_value = temp_mat

                    last_original_days_value[0:time_step + 1] = scaler.inverse_transform(
                        closedf[len(closedf) - time_step:]).reshape(-1).tolist()
                    next_predicted_days_value[time_step + 1:] = scaler.inverse_transform(
                        np.array(lst_output).reshape(-1, 1)).reshape(-1).tolist()

                    new_pred_plot = pd.DataFrame({
                        'last_original_days_value': last_original_days_value,
                        'next_predicted_days_value': next_predicted_days_value
                    })

                    names = cycle(['Last 15 days close price', 'Predicted next 30 days close price'])
                    fig = px.line(new_pred_plot, labels={'value': 'Stock price', 'index': 'Timestamp'})
                    fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
                    fig.for_each_trace(lambda t: t.update(name=next(names)))
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid(False))
                    st.plotly_chart(fig)

                    lstmdf = closedf.tolist()
                    lstmdf.extend(np.array(lst_output).reshape(-1, 1).tolist())
                    lstmdf = scaler.inverse_transform(lstmdf).reshape(-1).tolist()

                    names = cycle(['Close price'])
                    fig = px.line(lstmdf, labels={'value': 'Stock price', 'index': 'Timestamp'})
                    fig.update_layout(title_text='Plotting whole closing stock price with prediction',
                                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Stock')
                    fig.for_each_trace(lambda t: t.update(name=next(names)))
                    fig.update_xaxes(showgrid(False))
                    fig.update_yaxes(showgrid(False))
                    st.plotly_chart(fig)
