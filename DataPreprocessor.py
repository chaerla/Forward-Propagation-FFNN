import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, file_path: str = None):
        self.data = None
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

        if file_path is not None:
            self.load_data(file_path)

    def load_data(self, file_path: str):
        self.data = pd.read_csv(file_path)

    def preprocess(self, target_column: str):
        """
        Preprocesses the data by splitting it, standardizing the features, and encoding the labels.

        :param target_column: The name of the column to use as the target (label).

        :return: The standardized features and encoded labels for the training and testing data.
        """
        X_train, X_test, y_train, y_test = self.__split_data(0.2, target_column)
        X_train_standardized, X_test_standardized = self.__standardize_data(X_train, X_test)
        y_train_encoded, y_test_encoded = self.__encode_data(y_train, y_test)

        return X_train_standardized, X_test_standardized, y_train_encoded, y_test_encoded

    def __split_data(self, test_size: float = 0.2, stratify_column_label: str = None):
        """
        Splits the data into training and testing sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param stratify_column_label: The name of the column to use for stratification.

        :return: The training and testing data and labels.
        """
        X = self.data.drop(columns=[stratify_column_label])
        y = self.data[stratify_column_label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            stratify=y if stratify_column_label else None)

        return X_train, X_test, y_train, y_test

    def __standardize_data(self, X_train, X_test):
        """
        Standardizes the features in the training and testing data.

        :param X_train: The training data.
        :param X_test: The testing data.

        :return: The standardized training and testing data.
        """
        X_train_standardized = self.scaler.fit_transform(X_train)
        X_test_standardized = self.scaler.transform(X_test)

        return X_train_standardized, X_test_standardized

    def __encode_data(self, y_train, y_test):
        """
        Encodes the labels in the training and testing data.

        :param y_train: The labels for the training data.
        :param y_test: The labels for the testing data.

        :return: The encoded labels for the training and testing data.
        """
        y_train_encoded = self.encoder.fit_transform(y_train)
        y_test_encoded = self.encoder.transform(y_test)

        return y_train_encoded, y_test_encoded
