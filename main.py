"""
File: main.py
Author: Chuncheng Zhang
Date: 2025-05-08
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Main enter.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-08 ------------------------
# Requirements and constants
import time
from data_collector.neuroscan import Client


# %% ---- 2025-05-08 ------------------------
# Function and class
class StopWatch:
    # The window length to acquire in seconds.
    window_length: float = 4
    # The tick interval in seconds.
    interval: float = 1
    # The total length of the experiment.
    total: float = 100


client_kwargs = dict(
    host='192.168.31.79',
    port=4000
)

# %% ---- 2025-05-08 ------------------------
# Play ground
if __name__ == '__main__':
    client = Client(**client_kwargs)
    client.start_receiving_thread()

    sw = StopWatch()
    for i in range(int(sw.total / sw.interval+1)):
        time.sleep(sw.interval)
        data = client.fetch_data(sw.window_length)
        # TODO: Do something with the data.
        print(data)

    # Wait for enter to be pressed.
    input('Press Enter to Escape.')
    client.stop_receiving_thread()

# %% ---- 2025-05-08 ------------------------
# Pending


# %% ---- 2025-05-08 ------------------------
# Pending
