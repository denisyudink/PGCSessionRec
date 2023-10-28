# Import libraries
import numpy as np
import pandas as pd

from math import sqrt
from ordered_set import OrderedSet

class ItemKNN:
    # Initialize the class
    def __init__(self, k=20, session_column='SessionId', item_column='ItemId', timestamp_column='timestamp'):
        """
        Initialize the ItemKNN class with hyperparameters and data column names.

        Args:
            k (int): Length of the recommended item list.
            session_column (str): Name of the column containing session identifiers.
            item_column (str): Name of the column containing item identifiers.
            timestamp_column (str): Name of the column containing timestamps.
        """
        self.k = k
        self.session_column = session_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column

        self.item_sessions = {}  # Dictionary to store item sessions

    def fit(self, train_data):
        """
        Fit the ItemKNN model on the training data.

        Args:
            train_data (DataFrame): Training data containing session, item, and timestamp columns.
        """
        # Iterate over train data to store in self.item_sessions dictionary
        for _, row in train_data.iterrows():
            item = row[self.item_column]
            session = row[self.session_column]
            if item not in self.item_sessions:
                self.item_sessions[item] = []
            self.item_sessions[item].append(session)

    def predict_next(self, current_session, current_item, predict_next_items):
        """
        Predict the next items for a given session and current item.

        Args:
            current_session: Identifier of the current session.
            current_item: Identifier of the current item.
            predict_next_items: List of items to consider for prediction.

        Returns:
            List of recommended item identifiers.
        """
        # Create similarities array
        similarities = []

        # List of items to predict in the train set
        predict_next_items = list(self.item_sessions.keys())

        # If the current item is not in the predict items, return a list of recommendations as null
        if current_item not in predict_next_items:
            return similarities
        
        sessions_current_item = self.item_sessions.get(current_item)
        
        # Get the possible neighbors
        possible_neighbors = {key: value for key, value in self.item_sessions.items() if any(item in value for item in sessions_current_item)}

        # Calculate the similarity between the current item and items in the train set
        for item, vector in possible_neighbors.items():
            similarity = self.cosine(self.item_sessions.get(current_item), vector)
            similarities.append((item, similarity))

        # Remove all items where similarity <= 0
        similarities_zero = [(id, value) for id, value in similarities if value != 0.0]

        # Sort the items by similarity descending, then by the most recent session, and then by session ID ascending
        similarities_zero.sort(key=lambda x: x[1], reverse=True)

        # Get the first 20 items on the list of recommendations
        recommendations = [item for item, _ in similarities_zero[:self.k]]
        return recommendations

    def cosine(self, array1, array2):
        """
        Calculate the cosine similarity between two arrays.

        Args:
            array1: First array.
            array2: Second array.

        Returns:
            Cosine similarity between the two arrays.
        """
        # Remove duplicate sessions from the current_item array
        current_item_set = list(dict.fromkeys(array1))
        # Remove duplicate sessions from the past_item array
        past_item_set = list(dict.fromkeys(array2))

        # Calculate the intersection between the current_item array and past_item array
        intersection = len(self.intersection(current_item_set, past_item_set))

        # Calculate the Euclidean distance of the current_item array
        dist_current = sqrt(len(current_item_set))

        # Calculate the Euclidean distance of the past_item array
        dist_past = sqrt(len(past_item_set))

        # Calculate the similarity
        similarity = intersection / (dist_current * dist_past)

        return similarity
    
    def intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3