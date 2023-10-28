import pandas as pd
import numpy as np

# Enhanced evaluation with more frequent model updates.

def evaluate(df, item_column_name, session_column_name, timestamp_column, window_length_hours, n, n_window, model, limit):
    """
    Evaluate the dataset using window-based session splitting and recommendation system models.

    Args:
        df: DataFrame of the dataset.
        item_column_name: Name of the column containing item information in the DataFrame.
        session_column_name: Name of the column containing session information in the DataFrame.
        timestamp_column: Name of the column containing timestamp information in the DataFrame.
        n: Parameter for dividing the window_length_hours for more frequent updates.
        window_length_hours: Length of each evaluation window in hours.
        model: Specific number indicating the model utilized.
	limit: Boolean flag indicating whether to limit the number to 10.000 interactions per window.

    Returns:
        Tuple containing MRR (Mean Reciprocal Rank) and Hit Rate of the recommendations.
    """

    # Calculate the number of time windows based on the window length
    num_windows = int(np.ceil((df[timestamp_column].max() - df[timestamp_column].min()).total_seconds() / (window_length_hours * 3600)))
    print('num_windows: ', num_windows)
    # Initialize metrics
    mrr_scores = []
    hit_rates = []
    
    if i == 1:
    	# Initialize the ItemKNN model
    	knn = ItemKNN(k=20, session_column=session_column_name, item_column=item_column_name, timestamp_column=timestamp_column)
    elif model == 2:
	# Initialize the SKNN model
    	knn = SKNN(k=500, sample_size=1000, list_length=20, session_column=session_column_name, item_column=item_column_name, timestamp_column=timestamp_column)
    elif model == 3:
	# Initialize the V-SKNN model
    	knn = VSKNN(k=500, sample_size=1000, list_length=20, session_column=session_column_name, item_column=item_column_name, timestamp_column=timestamp_column)
    elif model == 4:
	# Initialize the S-SKNN model
    	knn = SSKNN(k=500, sample_size=1000, list_length=20, session_column=session_column_name, item_column=item_column_name, timestamp_column=timestamp_column)
    elif model == 5:
	# Initialize the SFSKNN model
    	knn = SFSKNN(k=500, sample_size=1000, list_length=20, session_column=session_column_name, item_column=item_column_name, timestamp_column=timestamp_column)

    # Iterate through each time window
    for i in range(1, n_window):
        print('iteration:', i)
        print('waiting for evaluation...')

        # Initialize metrics window
        mrr_window = [] # Mean Reciprocal Rank for the current window
        hit_window = [] # Hit Rates for the current window
        
        # Split sessions into training and testing based on time windows
        if i == 1:
	    # Training and testing data for the first window
            
            train_split = df[df[timestamp_column] <= df[timestamp_column].min() + pd.Timedelta(hours= i * (window_length_hours))]

            unique_sessions = train_split[session_column_name].unique()

            df_right = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours= i * (window_length_hours)))]

            filtered_df = df_right[df_right[session_column_name].isin(unique_sessions)]

            train_append = pd.concat([train_split, filtered_df])

            df_left = df[(df[timestamp_column] < df[timestamp_column].min() + pd.Timedelta(hours= (i-1) * (window_length_hours)))]

            unique_sessions_left = df_left[session_column_name].unique()

            train_sessions = train_append[~train_append[session_column_name].isin(unique_sessions_left)]
            
            # test sessions for i = 1
            test_split = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours= ((i*n)) * (window_length_hours / n))) &
                               (df[timestamp_column] <= df[timestamp_column].min() + pd.Timedelta(hours= ((i*n)+1) * (window_length_hours / n)))]

            unique_sessions_test = test_split[session_column_name].unique()

            df_right_test = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours=((i*n)+1)* (window_length_hours / n)))]

            filtered_df_test = df_right_test[df_right_test[session_column_name].isin(unique_sessions_test)]

            test_append = pd.concat([test_split, filtered_df_test])

            test_sessions = test_append[~test_append[session_column_name].isin(unique_sessions)]
        
        else: 
            # Training and testing data for subsequent windows
            train_split = df[df[timestamp_column] <= df[timestamp_column].min() + pd.Timedelta(hours=(((i+n-1)) * (window_length_hours / n)))]

            unique_sessions = train_split[session_column_name].unique()

            df_right = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours=(((i+n-1)) * (window_length_hours / n))))]

            filtered_df = df_right[df_right[session_column_name].isin(unique_sessions)]

            train_append = pd.concat([train_split, filtered_df])

            df_left = df[(df[timestamp_column] <= df[timestamp_column].min() + pd.Timedelta(hours=(((i+n-2)) * (window_length_hours / n))))]

            unique_sessions_left = df_left[session_column_name].unique()

            train_sessions = train_append[~train_append[session_column_name].isin(unique_sessions_left)]
                                                                                              
            # train_sessions for i > 1                                                                                   
            test_split = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours=((i+n-1) * (window_length_hours / n)))) &
                               (df[timestamp_column] <= df[timestamp_column].min() + pd.Timedelta(hours= ((i+n) * (window_length_hours / n))))]

            unique_sessions_test = test_split[session_column_name].unique()

            df_right_test = df[(df[timestamp_column] > df[timestamp_column].min() + pd.Timedelta(hours=((i+n)* (window_length_hours / n))))]

            filtered_df_test = df_right_test[df_right_test[session_column_name].isin(unique_sessions_test)]

            test_append = pd.concat([test_split, filtered_df_test])

            test_sessions = test_append[~test_append[session_column_name].isin(unique_sessions)]                                                                                   
        
        if limit == True
            print('Size of train sessions antes: ', len(train_sessions))
            min_timestamp = df[timestamp_column].min()
            print(min_timestamp)

            # Convert the minimum timestamp to the specified format
            min_timestamp = np.array([min_timestamp], dtype='datetime64[ns]')

            # Check if the minimum timestamp exists in the DataFrame
            if min_timestamp in train_sessions[timestamp_column].values:
               # Convert min_timestamp back to the previous format
               min_timestamp = min_timestamp[0].astype('datetime64[ns]')
               print(min_timestamp)

               # Convert min_timestamp to a pandas datetime object
               min_timestamp = pd.to_datetime(min_timestamp)
               print(min_timestamp)

               # Format min_timestamp to the desired format
               min_timestamp = min_timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00")
               print(min_timestamp)

               print(f"{min_timestamp} is present in the 'result' DataFrame.")
               train_sessions = train_sessions[train_sessions[timestamp_column] != min_timestamp]
               print('Size of train sessions excl: ', len(train_sessions))
            else:
               print(f"{min_timestamp} is not present in the 'result' DataFrame.") 
        
        train_sessions.sort_values([session_column_name, timestamp_column], inplace=True)
        test_sessions.sort_values([session_column_name, timestamp_column], inplace=True)

        train_sessions = train_sessions.reset_index(drop=True)  # Train sessions
        test_sessions = test_sessions.reset_index(drop=True)  # Test sessions

        print('Size of train sessions: ', len(train_sessions))
        print('Number of sessions in train: ', train_sessions[session_column_name].nunique())
        print('Size of test sessions: ', len(test_sessions))
        print('Number of sessions in test: ', test_sessions[session_column_name].nunique())                                                                       
                                                                                           
        if (len(train_sessions)) > 0:
            print('entrou no if do fit....')
            knn.fit(train_sessions)

        # Items to predict
        items_to_predict = train_sessions[item_column_name].unique()

        # Sessions length evolution
        offset_sessions = np.zeros(test_sessions[session_column_name].nunique() + 1, dtype=np.int32)

        # Length of sessions
        length_session = np.zeros(test_sessions[session_column_name].nunique(), dtype=np.int32)

        # Populate sessions length evolution
        offset_sessions[1:] = test_sessions.groupby(session_column_name).size().cumsum()

        # Populate length of sessions
        length_session[0:] = test_sessions.groupby(session_column_name).size()

        # Initialize variables to loop
        current_session_idx = 0
        pos = offset_sessions[current_session_idx]
        position = 0
        finished = 0

        # Iterate over events of sessions
        while finished == 0:
            if (len(test_sessions)) == 0:
                break
            current_item = test_sessions[item_column_name][pos]  # current item id
            current_session = test_sessions[session_column_name][pos]  # current session id
            next_item = test_sessions[item_column_name][pos + 1]  # next_item

            # generate recommendation list
            recommendation_list = knn.predict_next(current_session, current_item, items_to_predict)

            if len(recommendation_list) == 0:  # in case recommendation is empty
                mrr = 0
                hr = 0
            else:
                mrr = mean_reciprocal_rate(next_item, recommendation_list)  # Calculate MRR
                hr = hit_rate_atk(next_item, recommendation_list)  # Calculate Hit Rate

            # Append results to arrays
            mrr_window.append(mrr)
            hit_window.append(hr)

            # Plus on position
            pos += 1
            position += 1

            # Change session
            if pos + 1 == offset_sessions[current_session_idx] + length_session[current_session_idx]:
                current_session_idx += 1

                if current_session_idx == test_sessions[session_column_name].nunique():
                    finished = 1

                pos = offset_sessions[current_session_idx]
                position = 0

        print('Result for window: ', i)
        # Calculate mean of window
        mean_mrr_window = np.nanmean(mrr_window)
        mean_hr_window = np.nanmean(hit_window)
        print('MRR Window: ', mean_mrr_window)
        print('Hit Rate Window: ', mean_hr_window)

        # Append result of window to the array of all windows
        mrr_scores.append(mean_mrr_window)
        hit_rates.append(mean_hr_window)
        #print('MRR Array: ', mrr_scores)
        #print('HR Array: ', hit_rates)

        # Result of iteration
        print('Result for iteration: ', i)
        mean_mrr_iter = np.nanmean(mrr_scores)
        hit_rate_iter = np.nanmean(hit_rates)
        print('MRR: ', mean_mrr_iter)
        print('Hit Rate: ', hit_rate_iter)

    # Calculate mean MRR and Hit Rate
    mean_mrr = np.nanmean(mrr_scores)
    hit_rate = np.nanmean(hit_rates)

    print('Final result (MRR, Hit Rate): ')
    return mean_mrr, hit_rate


def mean_reciprocal_rate(next_item, recommendations):
    """
    Calculate MRR (Mean Reciprocal Rank) of the next item in the list of recommendations.

    Args:
        next_item: Next item of the session to predict.
        recommendations: List of items generated by the recommendation algorithm.

    Returns:
        MRR of the item.
    """
    if next_item in recommendations:
        rank = recommendations.index(next_item) + 1  # Rank of the item in the recommendation list
        reciprocal_rank = 1 / rank  # Reciprocal rank
        return reciprocal_rank
    else:
        return 0.0  # Return zero if the item is not in the list


def hit_rate_atk(next_item, recommendations):
    """
    Calculate Hit Rate of the item.

    Args:
        next_item: Next item of the session to predict.
        recommendations: List of items generated by the recommendation algorithm.

    Returns:
        Hit Rate of the item.
    """
    if next_item in recommendations:
        return 1.0  # Return 1 if the item is in the list
    else:
        return 0.0  # Return 0 otherwise