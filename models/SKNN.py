import numpy as np
import pandas as pd

from math import sqrt
from ordered_set import OrderedSet

class SKNN:
    # Initialize the class
    def __init__(self, k=500, sample_size=1000, list_length=20, session_column='SessionId', item_column='ItemId', timestamp_column='timestamp'):
        """
        Initialize the SKNN class with hyperparameters and data column names.

        Args:
            k (int): Number of nearest neighbors to consider.
            sample_size (int): Number of recent sessions to sample from.
            list_length (int): Length of the recommended item list.
            session_column (str): Name of the column containing session identifiers.
            item_column (str): Name of the column containing item identifiers.
            timestamp_column (str): Name of the column containing timestamps.
        """
        self.k = k
        self.sample_size = sample_size
        self.list_length = list_length
        self.session_column = session_column
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        
        self.session = -1 # Current session
        self.session_items = list()  # List to store current session items
        
        self.session_item_map = dict()  # Dictionary to store sessions and their respective items
        self.session_time = dict()  # Dictionary to store the time of the last event of each session
        
    def fit(self, train_data):
        """
        Fit the SKNN model on the training data.

        Args:
            train_data (DataFrame): Training data containing session, item, and timestamp columns.
        """
        # Initialize variables
        session_current = -1
        current_session_items = OrderedSet()
        
        # Iterate over the training data to store in self.session_item_map and self.session_time dictionary
	# Most of this section of the code is derived from: https://github.com/rn5l/session-rec.
        for _, row in train_data.iterrows():
            if row[self.session_column] != session_current:
                if len(current_session_items) > 0:
                    self.session_item_map.update({session_current: current_session_items})
                    self.session_time.update({session_current: time})
                session_current = row[self.session_column]
                current_session_items = OrderedSet()
            time = row[self.timestamp_column]
            current_session_items.add(row[self.item_column])

        # Last session of the training data
        self.session_item_map.update({session_current: current_session_items})
        self.session_time.update({session_current: time})
                     
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
        # Change the current session
        if self.session != current_session:
            self.session = current_session
            self.session_items = list()
        
        # Append to the current session
        self.session_items.append(current_item)
        
        # Get the possible neighbors
        possible_neighbors = {key: value for key, value in self.session_item_map.items() if any(item in value for item in self.session_items)}

        if not possible_neighbors:
            return []
        
        # Filter session time dictionary by possible neighbors session ids
        filtered_session_time = {key: self.session_time[key] for key in possible_neighbors.keys()}
        
        # Sort sessions by most recent
        session_time_sorted = dict(sorted(filtered_session_time.items(), key=lambda item: item[1], reverse=True))
        
        # Get the IDs of the 500 most recent sessions
        sample_sessions = dict(list(session_time_sorted.items())[:self.sample_size])

        # Filter and create a dictionary that contains the most recent sessions and their respective arrays of items
        filtered_sample_sessions = {key: self.session_item_map[key] for key in sample_sessions.keys()}

        # Calculate the similarity between the current session and filtered_sample_sessions
        similarities = list()
        for session, vector in filtered_sample_sessions.items():
            similarity = self.cosine(self.session_items, vector)
            similarities.append((session, similarity))

        # Remove all sessions where similarity <= 0
        similarities_zero = [(id, value) for id, value in similarities if value != 0.0]
        
        # Sort the sessions by similarity descending
        similarities_zero.sort(key=lambda x: x[1], reverse=True)

        # Get the k nearest neighbor sessions
        nearest = similarities_zero[:self.k]
        
        # If nearest is empty, return an empty list
        if not nearest:
            return []
        
        # Calculate the score of the sessions items
        scores = self.calc_scores(nearest)
        
        # Remove all items where scores <= 0
        scores_zero = {key: value for key, value in scores.items() if value > 0}
        
        # If scores_zero is empty, return an empty list
        if not scores_zero:
            return []
            
        # Sort the items by score descending
        sorted_items = sorted(scores_zero.items(), key=lambda x: x[1], reverse=True)
        
        # Get the item IDs
        sorted_keys = [item[0] for item in sorted_items]
        
        # Get the first 20 items on the list of recommendations
        items_recommendations = sorted_keys[:self.list_length]
        
        return items_recommendations
    
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
        current_session_set = OrderedSet(array1)
        # Remove duplicate sessions from the past_item array
        past_session_set = OrderedSet(array2)
        
        # Calculate the intersection between the current_item array and past_item array
        intersec = len(current_session_set & past_session_set)

        # Calculate the Euclidean distance of the current_item array
        dist_current = sqrt(len(current_session_set))

        # Calculate the Euclidean distance of the past_item array
        dist_past = sqrt(len(past_session_set))

        # Calculate the similarity
        similarity = intersec / (dist_current * dist_past)
        
        return similarity

    
    def calc_scores(self, recommendations):
        """
        Calculate scores for items based on the similarity of nearest neighbor sessions.

        Args:
            recommendations: List of nearest neighbor sessions and their similarities.

        Returns:
            Dictionary of item scores based on session similarities.
        """
        # Initialize dictionary to store scores
        scores = dict()

        # Iterate over nearest neighbor sessions
	# Most of this section of the code is derived from: https://github.com/rn5l/session-rec.
        for session in recommendations:
            items = self.session_item_map.get(session[0])  # Items of the nearest neighbors
            # Iterate over items of the neighbor
            for item in items:
                score_current = scores.get(item)  # Get the current score of the item in the dictionary
                score_updated = session[1]  # Update the score in the dictionary with the similarity of the session

                if score_current is None:
                    scores.update({item: score_updated})  # Update the score of the item in the dictionary
                else:
                    score_updated = score_current + score_updated  # Sum the current score with the updated score
                    scores.update({item: score_updated})  # Update the score of the item in the dictionary

        return scores