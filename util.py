import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
import tensorflow as tf

def set_datetime_index(df):
    """
    Set date as datetime-index and sort the index
    """
    df = df.copy()
    # Convert Date to Datetime-format
    df["date"] = pd.to_datetime(df["date"])

    # Set as index and drop the column
    df = df.set_index("date", drop=True)

    # Sort the index
    df = df.sort_index()

    return df


def fill_missing_dates_and_values(df):
    """
    Look for missing dates and impute all values
    """
    # Ensure the DataFrame has a continuous date range index
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)

    # Identify columns with all missing values
    cols_with_all_missing = df.columns[df.isna().all()].tolist()

    # Handle columns with all missing values by filling with a default value (e.g., 0) or dropping
    df[cols_with_all_missing] = df[cols_with_all_missing].fillna(0)

    # Impute the missing dates
    imputer = KNNImputer(n_neighbors=2)
    imputed_data = imputer.fit_transform(df)

    # Create a DataFrame with the imputed data
    imputed_df = pd.DataFrame(data=imputed_data, columns=df.columns[:imputed_data.shape[1]], index=df.index)

    # Identify which column was dropped
    missing_cols = set(df.columns) - set(imputed_df.columns)

    # If a column was dropped, add it back with the mean
    for col in missing_cols:
        imputed_df[col] = df[col].mean()

    # Reorder columns to match the original DataFrame
    result_df = imputed_df[df.columns]

    return result_df

def df_to_X_y(df, window_size=7):
    """
    Converts a DataFrame into sequences of features (X) and corresponding labels (y) using a specific window size.

    """
    index_of_snow = df.columns.get_loc('snow')

    df_as_np = df.to_numpy()
    X = []
    y = []

    # Loop through the data to create input sequences and corresponding target values
    for i in range(len(df_as_np) - window_size):
        # Create a window of data
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)

        # The target value is the 'snow' column value after the current window
        label = df_as_np[i + window_size][index_of_snow]
        y.append(label)

    # Convert lists to numpy arrays
    return np.array(X), np.array(y)

def split_data(X, y, df):
    """
    Split data into train, val, test
    """
    # Fractions for Train, Validation und Test Sets
    train_frac = 0.6
    val_frac = 0.2

    # Calculation of the indices
    train_end = int(train_frac * len(df))
    val_end = train_end + int(val_frac * len(df))

    # Slicing of the DataFrames X, y and the dates
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    train_dates = df.index[:train_end]
    val_dates = df.index[train_end:val_end]
    test_dates = df.index[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test, train_dates, val_dates, test_dates


def preprocess(X, temp_training_mean, temp_training_std, i):
    """
    Standadize the data using the training mean and std
    """
    # z = (x - μ) / σ
    X[:, :, i] = (X[:, :, i] - temp_training_mean) / temp_training_std

    return X


def build_and_compile_model(window_size, n_features):
    """
    Build and compile a LSTM model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer((window_size, n_features)))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, 'relu'))
    model.add(tf.keras.layers.Dense(32, 'relu'))
    model.add(tf.keras.layers.Dense(16, 'relu'))
    model.add(tf.keras.layers.Dense(1, 'sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

    return model

def fit_model(model, X_train, X_val, y_train, y_val):
    """
    Fit the model on the training data with validation
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    model.fit(X_train, y_train,
           validation_data=(X_val, y_val),
           epochs=100,
           callbacks=early_stopping,
           verbose=False)

    return model
