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
from data_collector.neuroscan import Client


# %% ---- 2025-05-08 ------------------------
# Function and class


# %% ---- 2025-05-08 ------------------------
# Play ground
if __name__ == '__main__':
    client = Client()
    client.start_receiving_thread()

    # Wait for enter to be pressed.
    input('Press Enter to Escape.')
    client.stop_receiving_thread()

# %% ---- 2025-05-08 ------------------------
# Pending


# %% ---- 2025-05-08 ------------------------
# Pending
