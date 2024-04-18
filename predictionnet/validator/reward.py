# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from typing import List
import bittensor as bt
from predictionnet.protocol import Challenge
import time
from datetime import datetime, timedelta
import yfinance as yf
from pytz import timezone
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# def get_rmse(challenge: List[Challenge], close_price: float) -> float:
#     if challenge.prediction is None:
#         raise ValueError("Where is my prediction bro.?")
#     prediction_arr = np.array([c.prediction for c in challenge])
#     squared_error = (prediction_arr - close_price) ** 2
#     rmse = squared_error ** 0.5
#     return rmse

INTERVAL = 30

def get_direction_accuracy(close_price_array, prediction_array):
    # Calculate the direction of changes (up: 1, down: -1)
    actual_directions = [1 if close_price_array[i] > close_price_array[i - 1] else -1 for i in range(1, len(close_price_array))]
    predicted_directions = [1 if prediction_array[i] > prediction_array[i - 1] else -1 for i in range(1, len(prediction_array))]

    # Calculate the number of times the predicted direction matches the actual direction
    correct_predictions = sum(1 for i in range(len(actual_directions)) if actual_directions[i] == predicted_directions[i])

    # Calculate directional accuracy
    directional_accuracy = (correct_predictions/len(actual_directions)-1)*100

    return directional_accuracy

def reward(response: Challenge, close_price: float) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    prediction_array = np.array(response.prediction)
    close_price_array = np.array(close_price)

    mse = mean_squared_error(prediction_array, close_price_array)
    directional_accuracy = get_direction_accuracy(close_price_array, prediction_array)

    # subtracting dir accuracy from 100 because the goal is to reward those that make quality predictions for longer durations
    # If the reward function gives a higher value, the weights will be
    # lower since the result from this is subtracted from 1 subsequently
    return 0.5*(mse**0.5 + (100 - directional_accuracy))

# Query prob editied to query: Protocol defined synapse
# For this method mostly should defer to Rahul/Tommy
def get_rewards(
    self,
    query: Challenge,
    responses: List[Challenge],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[Challenge]): A list of responses from the miner.
    
    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """

    if len(responses) == 0:
        bt.logging.info("Got no responses. Returning reward tensor of zeros.")
        return [], torch.zeros_like(0).to(self.device)  # Fallback strategy: Log and return 0.

    # Prepare to extract close price for this timestamp
    ticker_symbol = '^GSPC'
    ticker = yf.Ticker(ticker_symbol)

    timestamp = query.timestamp
    timestamp = datetime.fromisoformat(timestamp)

    # Round up current timestamp and then wait until that time has been hit
    rounded_up_time = timestamp - timedelta(minutes=timestamp.minute % INTERVAL,
                                    seconds=timestamp.second,
                                    microseconds=timestamp.microsecond) + timedelta(minutes=INTERVAL + 5, seconds=30)
    
    ny_timezone = timezone('America/New_York')

    while (datetime.now(ny_timezone) < rounded_up_time - timedelta(minutes=4, seconds=30)):
        bt.logging.info(f"Waiting for next {INTERVAL}m interval...")
        time.sleep(15)

    current_time_adjusted = rounded_up_time - timedelta(minutes=INTERVAL + 5)
    print(rounded_up_time, rounded_up_time.hour, rounded_up_time.minute, current_time_adjusted)
    
    data = yf.download(tickers=ticker_symbol, period='1d', interval='5m')
    bt.logging.info("Procured data from yahoo finance.")

    bt.logging.info(data.iloc[-7:-1])
    close_price = data['Close'].iloc[-7:-1].tolist()
    close_price_revealed = ' '.join(str(price) for price in close_price)

    bt.logging.info(f"Revealing close prices for this interval: {close_price_revealed}")

    # Get all the reward results by iteratively calling your reward() function.
    scoring = [reward(response, close_price) if response.prediction != None else 0 for response in responses]
    worst_loss = max(scoring)
    bt.logging.debug(worst_loss)
    scoring = [1 - (score / worst_loss) if score != 0 else 0 for score in scoring]
    return torch.FloatTensor(scoring)

    #scaler = MinMaxScaler(feature_range=(0,1))
    #return torch.FloatTensor(scaler.fit_transform(np.array(scoring).reshape(-1, 1)).flatten()).to(self.device)


