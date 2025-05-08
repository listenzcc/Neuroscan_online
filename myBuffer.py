# coding: utf-8

import numpy as np
from myClientNew import ScanClient


class Buffer():
    def __init__(self, nchan, windowLength, srate=1000, offline=False):
        self.is_on = False
        self.record = False
        self.offline = offline

        self.nchan = nchan
        # self.nchan = 68
        self.windowLength = windowLength
        self.srate = srate
        self.npoints = self.srate * self.windowLength
        self.all_buffer = np.zeros((self.nchan+1, self.npoints))
        self.currentPtr = 0

    def record_buffer(self, buffer):
        if self.record:
            # self.all_buffer = np.concatenate((self.all_buffer, buffer), axis=1)
            # print('  ', self.all_buffer.shape)
            n = buffer.shape[1]
            fill_in_place = np.mod(np.arange(self.currentPtr, self.currentPtr + n), self.npoints)
            self.all_buffer[:, fill_in_place] = buffer
            self.currentPtr = np.mod(self.currentPtr + n - 1, self.npoints) + 1
        else:
            pass

    def on(self, IP, port):
        if self.is_on:
            print('Connected: IP: %s port: %d' % (IP, port))
            return 0

        # No connection when offline testing
        if self.offline:
            return 0

        self.client = ScanClient(IP, port)
        self.client.register_receive_callback(self.record_buffer)
        self.record = False
        self.client.start_receive_thread(self.nchan+1)
        self.client.start_sending_data()
        self.is_on = True
        print('Connection established: IP: %s port: %d' % (IP, port))

    def off(self):
        if not self.is_on:
            print('Unconnected, do nothing.')
            return 0
        print('Disconnect, connection can not be recovered.')
        try:
            self.client.stop_receive_thread(stop_measurement=True)
            self.client._send_command(self.client.commandDict['Request_to_Stop_Sending_Data'])
            self.client._close_connect()
            self.client._sock.close()
        except:
            pass

    def start(self):
        if not self.is_on:
            return 0
        # self.all_buffer = np.empty((self.nchan, 0))
        self.all_buffer = np.zeros((self.nchan+1, self.npoints))
        self.record = True

    def stop(self):
        if not self.is_on:
            return 0
        self.record = False

    def output(self):
        data = np.hstack((self.all_buffer[:, self.currentPtr:], self.all_buffer[:, :self.currentPtr]))
        return data


