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
    st.write("## File Uploaded Successfully")
    file_type = uploaded_file.name.split('.')[-1]
    try:
        if file_type == "csv":
            reliance = pd.read_csv(uploaded_file, index_col='Date', parse_dates=True)
        elif file_type == "xlsx":
            reliance = pd.read_excel(uploaded_file, index_col='Date', parse_dates=True)

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
        plt.figure(figsize=(10, 8))
        sns.heatmap(reliance.corr(), annot=True)
        st.pyplot(plt.gcf())

        st.write("## Pairplot")
        sns.pairplot(reliance)
        st.pyplot(plt.gcf())

        st.write("## Time Series Plots")
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        reliance['Open'].plot(ax=axes[0, 0], title='Open', color='green')
        reliance['Close'].plot(ax=axes[0, 1], title='Close', color='red')
        reliance['High'].plot(ax=axes[1, 0], title='High', color='blue')
        reliance['Low'].plot(ax=axes[1, 1], title='Low', color='c')
        for ax in axes.flatten():
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
        st.pyplot(fig)

        st.write("## Volume Plot")
        plt.figure(figsize=(20, 8))
        plt.plot(reliance['Volume'])
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.title('Date vs Volume')
        st.pyplot(plt.gcf())

        st.write("## Moving Averages")
        reliance_ma = reliance.copy()
        reliance_ma['30-day MA'] = reliance['Close'].rolling(window=30).mean()
        reliance_ma['100-day MA'] = reliance['Close'].rolling(window=100).mean()

        plt.figure(figsize=(20, 7))
        plt.plot(reliance_ma['Close'], label='Original data')
        plt.plot(reliance_ma['30-day MA'], label='30-MA')
        plt.legend()
        plt.title('Stock Price vs 30-day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(plt.gcf())

        plt.figure(figsize=(20, 7))
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

        epochs = 20
        batch_size = 32

        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
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

                fig = px.line(plotdf, x=plotdf['Date'], y=[plotdf['original_close'], plotdf['train_predicted_close'], plotdf['test_predicted_close']],
                              labels={'value': 'Stock price', 'Date': 'Date'})
                fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                                  plot_bgcolor='white')
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid(False))
                st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV or Excel file.")
