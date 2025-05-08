"""
File: neuroscan.py
Author: Chuncheng Zhang
Date: 2025-05-08
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Collect EEG data from neuroscan device.

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
import socket
import numpy as np

from threading import Thread
from loguru import logger

host = '192.168.31.79'
port = 4000
logger.add('log/debug.log', level='DEBUG', rotation='5 MB')
logger.add('log/info.log', level='INFO', rotation='5 MB')

# %% ---- 2025-05-08 ------------------------
# Function and class


class BasicClient:
    host: str = host
    port: int = port
    channels: int = 67
    # The resolution of the received EEG data
    fResolution = 150e-6  # 0.00015 Volts
    # How many packages are received per second.
    packages_per_second: int = 25
    # How many seconds do the package span.
    seconds_per_package: float = 1/packages_per_second
    timeout: float = 1  # Seconds

    basicInfoType = np.dtype([
        ('dwSize', np.uint32),
        ('nEegChan', np.int32),
        ('nEvtChan', np.int32),
        ('nBlockPnts', np.int32),
        ('nRate', np.int32),
        ('nDataSize', np.int32),
        ('fResolution', np.float32)])

    headType = np.dtype([
        ('IDString', '>S4'),
        ('Code', '>u2'),
        ('Request', '>u2'),
        ('BodySize', '>u4')])

    def __init__(self):
        def _format_head(IDString, Code, Request, BodySize):
            return np.array([(IDString, Code, Request, BodySize)], dtype=self.headType)

        self.commandDict = {
            'Request_for_Version': _format_head('CTRL', 1, 1, 0),
            'Closing_Up_Connection': _format_head('CTRL', 1, 2, 0),
            'Start_Acquisition': _format_head('CTRL', 2, 1, 0),
            'Stop_Acquisition': _format_head('CTRL', 2, 2, 0),
            'Start_Impedance': _format_head('CTRL', 2, 3, 0),
            'Change_Setup': _format_head('CTRL', 2, 4, 0),
            'DC_Correction': _format_head('CTRL', 2, 5, 0),
            'Request_for_EDF_Header': _format_head('CTRL', 3, 1, 0),
            'Request_for_AST_Setup_File': _format_head('CTRL', 3, 2, 0),
            'Request_to_Start_Sending_Data': _format_head('CTRL', 3, 3, 0),
            'Request_to_Stop_Sending_Data': _format_head('CTRL', 3, 4, 0),
            'Request_for_basic_info': _format_head('CTRL', 3, 5, 0),
            'Neuroscan_16bit_Raw_Data': _format_head('Data', 2, 1, 0),
            'Neuroscan_32bit_Raw_Data': _format_head('Data', 2, 2, 0)
        }

        logger.info(f'Initialize {self}')


class Client(BasicClient):
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                logger.info(f'Set {k}={v}')

        try:
            self.connect()
        except TimeoutError as err:
            logger.exception(err)
            logger.error(f'Can not connect to server {self.host}:{self.port}.')
            raise err

        logger.info(f'Initialize {self}')

    def connect(self):
        ''' Connect to the server. '''
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))
        self.socket.setblocking(True)
        logger.info(f'Connected to {self.socket}')

    def start_receiving_thread(self):
        ''' Start the receiving loop. '''
        if not hasattr(self, '_recv_thread'):
            self._recv_thread = Thread(
                target=self.receiving_loop, daemon=False)
            self._recv_thread.start()
            logger.info('Receiving loop starts')
        return

    @logger.catch
    def stop_receiving_thread(self):
        self.is_receiving = False
        if hasattr(self, '_recv_thread'):
            logger.info(f'Waiting for receiving thread stops.')
            self._recv_thread.join()
        logger.info('Stopped receiving.')

    def receiving_loop(self):
        try:
            # ----------------------------------------
            # ---- Request basic info ----
            logger.info('Request for basic info')
            self._send_command(self.commandDict['Request_for_basic_info'])
            head, buff = self.receive_single()
            logger.debug(f'Got head: {head}')
            logger.debug(f'Got buff: {buff[:8]}...{buff[-8:]} | {len(buff)}')

            # TODO: It is incorrect to decode the basic info, don't know why.
            # basicInfo = np.array(buff, self.basicInfoType)
            # logger.debug(f'BasicInfo: {basicInfo}')

            # ----------------------------------------
            # ---- Request read-time EEG data ----
            logger.info('Receiving loop starts')
            self._send_command(
                self.commandDict['Request_to_Start_Sending_Data'])
            self.is_receiving = True
            self.data = []
            self.times = []
            self.n = 0

            while self.is_receiving:
                head, buff = self.receive_single()

                # Convert into (40 x self.channels) matrix
                matrix = np.frombuffer(
                    buff, '<i4').reshape(-1, self.channels)
                # buffer_array[:, :-1] *= fResolution

                print(np.max(matrix), np.min(matrix))
                print(head, len(buff), matrix.shape, matrix.dtype)
                self.data.append(matrix)
                self.times.append(time.time())
                self.n += 1
                if self.n % 1000 == 0:
                    self.report_current_data_length()
                time.sleep(0.001)

            self._send_command(
                self.commandDict['Request_to_Stop_Sending_Data'])
            logger.info('Receiving loop stops')
        except Exception as err:
            logger.exception(err)
        finally:
            # ----------------------------------------
            # ---- Close connection ----
            self._send_command(self.commandDict['Closing_Up_Connection'])
            logger.info('Closed connection')
            return

    def report_current_data_length(self):
        logger.debug(
            f'Data length: {self.n} | {self.n * self.seconds_per_package:0.2f} seconds.')

    @logger.catch
    def fetch_data(self, length: float = 1.0):
        # How many packages are required
        n = length / self.seconds_per_package
        if n > self.n:
            logger.warning(f'Not have enough data: {self.n} < {n}')
        return np.concatenate(self.data[-n:], axis=0)

    @logger.catch
    def receive_single(self):
        recv = self.socket.recv(12)
        assert len(
            recv) == 12, f'Received illegal head length: {len(recv)} is not 12.'
        head = np.frombuffer(recv, self.headType)

        n_received = 0
        total = int(head[0]['BodySize'])
        rec_buff = []  # 只包含data body,不含header
        while n_received < total:
            n_buffer = min(4096, total - n_received)
            this_buffer = self.socket.recv(n_buffer)
            rec_buff.append(this_buffer)
            n_received += len(this_buffer)

        assert n_received == total, f'Incorrect bodySize {n_received}, but supposed to be {total}.'

        buff = b''.join(rec_buff)

        return head, buff

    def _send_command(self, command):
        self.socket.sendall(command.tobytes())

# %% ---- 2025-05-08 ------------------------
# Play ground


# %% ---- 2025-05-08 ------------------------
# Pending


# %% ---- 2025-05-08 ------------------------
# Pending
