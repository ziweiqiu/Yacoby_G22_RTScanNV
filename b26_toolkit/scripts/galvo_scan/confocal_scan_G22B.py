"""
    This file is part of b26_toolkit, a PyLabControl add-on for experiments in Harvard LISE B26.
    Copyright (C) <2016>  Arthur Safira, Jan Gieseler, Aaron Kabcenell

    Foobar is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import time
import math
from scipy import ndimage
from pyanc350.v2 import Positioner

from b26_toolkit.instruments import NI6733, NI6602, NI6220, NI6210,G22BPulseBlaster
from b26_toolkit.plotting.plots_2d import plot_fluorescence_new, update_fluorescence
from pylabcontrol.core import Script, Parameter
from b26_toolkit.plotting.plots_1d import update_counts_vs_pos, plot_counts_vs_pos, \
    plot_counts_vs_pos_multilines
from b26_toolkit.scripts.set_laser import SetScannerXY_gentle

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from b26_toolkit.scripts.qm_scripts.Configuration import config

import smtplib, ssl
receiver_email = "ziweiqiu@g.harvard.edu"
def send_email(receiver_email, message):
    port = 465  # For SSL
    password = "diamond2020"
    sender_email = "nv.scanning.alert@gmail.com"
    # Create a secure SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        server.quit()

class ObjectiveScan(Script):

    """
        Objective scan x, y and z. After scan, the objective will stay at the last point.
        - Ziwei Qiu 7/24/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('scan_axes', 'xy', ['xy', 'xz', 'yz', 'x', 'y', 'z'], 'Choose 2D or 1D confocal scan to perform'),
        Parameter('point_a',
                  [Parameter('x', 0, float, 'x-coordinate [V]'),
                   Parameter('y', 0, float, 'y-coordinate [V]'),
                   Parameter('z', 5, float, 'z-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                   Parameter('y', 1.0, float, 'y-coordinate [V]'),
                   Parameter('z', 10.0, float, 'z-coordinate [V]')
                   ]),
        Parameter('RoI_mode', 'center', ['corner', 'center'], 'mode to calculate region of interest.\n \
                                                           corner: pta and ptb are diagonal corners of rectangle.\n \
                                                           center: pta is center and pta is extend or rectangle'),
        Parameter('num_points',
                  [Parameter('x', 125, int, 'number of x points to scan'),
                   Parameter('y', 125, int, 'number of y points to scan'),
                   Parameter('z', 51, int, 'number of z points to scan')
                   ]),
        Parameter('time_per_pt',
                  [Parameter('xy', .01, [.002, .005, .01, .015, .02, 0.05, 0.1, 0.2],
                             'time in s to measure at each point'),
                   Parameter('z', .5, [.25, .5, 1.], 'time in s to measure at each point for 1D z-scans only'),
                   ]),
        Parameter('settle_time',
                  [Parameter('xy', .001, [.0002, .0005, .001, .002, .005, .01, .05, .1, .25],
                             'wait time between points to allow objective to settle'),
                   Parameter('z', .05, [.01, .05, .1, .25],
                             'settle time for objective z-motion (measured for oil objective to be ~10ms, in reality appears to be much longer)'),
                   ]),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao5', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao6', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('z_ao_channel', 'ao7', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for z voltage analog output'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        # Parameter('ending_behavior', 'return_to_start', ['return_to_start', 'return_to_origin', 'leave_at_corner'], 'return to the corn'),
        # Parameter('daq_type', 'PCI', ['PCI', 'cDAQ'], 'Type of daq to use for scan')
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'PB': G22BPulseBlaster}
    _SCRIPTS = {}

    def __init__(self, instruments, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes ConfocalScan script for use in gui

        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data

        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, log_function=log_function,
                        data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in = self.instruments['NI6602']['instance']
        self.daq_out = self.instruments['NI6733']['instance']

    def _function(self):
        """
        Executes threaded galvo scan
        """

        # update_time = datetime.datetime.now()

        # self._plot_refresh = True

        # self._plotting = True

        # turn on laser and apd_switch
        print('Turned on laser and APD readout channel.')
        self.instruments['PB']['instance'].update({'laser': {'status': True}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': True}})

        def scan2D():
            self._recording = False

            self.clockAdjust = int(
                (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) /
                self.settings['settle_time']['xy'])

            if self.clockAdjust % 2 == 1:
                self.clockAdjust += 1

            self.var1_array = np.repeat(
                np.linspace(self.var1range[0], self.var1range[1], self.settings['num_points'][self.var1],
                            endpoint=True),
                self.clockAdjust)
            self.var2_array = np.linspace(self.var2range[0], self.var2range[1], self.settings['num_points'][self.var2],
                                          endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var2],
                                                 self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1],
                                    self.var2range[0], self.var2range[1]]}
            self.data['extent'] = [self.var1range[0], self.var1range[1], self.var2range[1], self.var2range[0]]
            self.data['varlbls'] = [self.var1 + ' [V]', self.var2 + ' [V]']

            # objective takes longer to settle after big jump, so give it time before starting scan:
            if self.settings['scan_axes'] == 'xz' or self.settings['scan_axes'] == 'yz':
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.var2_array[0]})
                time.sleep(0.4)

            for var2Num in range(0, len(self.var2_array)):

                if self._abort:
                    break

                # set galvo to initial point of next line
                self.initPt = [self.var1_array[0], self.var2_array[var2Num]]
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.initPt[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.initPt[1]})

                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(
                    self.settings['DAQ_channels']['counter_channel'],
                    len(self.var1_array) + 1)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array, ctrtask)

                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                var1LineData, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(var1LineData)
                summedData = np.zeros(int(len(self.var1_array) / self.clockAdjust))

                for i in range(0, int((len(self.var1_array) / self.clockAdjust))):
                    pxarray = diffData[(i * self.clockAdjust + 1):(i * self.clockAdjust + self.clockAdjust - 1)]
                    normalization = len(pxarray) / self.sample_rate / 0.001
                    summedData[i] = np.sum(pxarray) / normalization

                # summedData = np.flipud(summedData)

                # also normalizing to kcounts/sec
                # self.data['image_data'][var2Num] = summedData * (.001 / self.settings['time_per_pt']['galvo'])
                self.data['image_data'][var2Num] = summedData
                self.progress = float(var2Num + 1) / len(self.var2_array) * 100
                self.updateProgress.emit(int(self.progress))

        def scan1D(var1):
            self._recording = False
            if var1 == 'z':
                nsamples = int(
                    (self.settings['time_per_pt'][var1] + self.settings['settle_time'][var1]) * self.sample_rate)
            else:
                nsamples = int(
                    (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) * self.sample_rate)
            if nsamples % 2 == 1:
                nsamples += 1
            self.var1_array = np.linspace(self.var1range[0],
                                          self.var1range[1],
                                          self.settings['num_points'][self.var1], endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1]]}
            self.data['varlbls'] = self.var1 + ' [V]'

            # objective takes longer to settle after a big jump, so give it time before starting scan:
            self.daq_out.set_analog_voltages({self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0]})
            time.sleep(0.5)

            for var1Num in range(0, len(self.var1_array)):
                if self._abort:
                    break
                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(self.settings['DAQ_channels']['counter_channel'], nsamples)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array[var1Num] * np.ones(nsamples), ctrtask)
                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                samparray, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(samparray)

                # sum and normalize to kcounts/sec
                # self.data['image_data'][var1Num] = np.sum(diffData) * (.001 / self.settings['time_per_pt']['z-piezo'])
                normalization = len(diffData) / self.sample_rate / 0.001
                self.data['image_data'][var1Num] = np.sum(diffData) / normalization

                self.progress = float(var1Num + 1) / len(self.var1_array) * 100
                self.updateProgress.emit(int(self.progress))

        # if self.settings['daq_type'] == 'PCI':
        # initial_position = self.daq_out.get_analog_voltages(
        #     [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel'], self.settings['DAQ_channels']['z_ao_channel']])
        #
        # print('initial positions are:')
        # print(initial_position)
        [self.xVmin, self.xVmax, self.yVmin, self.yVmax, self.zVmin, self.zVmax] = self.pts_to_extent(
            self.settings['point_a'],
            self.settings['point_b'],
            self.settings['RoI_mode'])

        self.sample_rate = float(1) / self.settings['settle_time']['xy']

        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['z_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate

        # print('ready for scanning')

        # depending to axes to be scanned, assigns the correct channels to be scanned and scan ranges, then starts the 2D or 1D scan
        self.var1range = 0
        self.var2range = 0
        if self.settings['scan_axes'] == 'xy':
            self.var1 = 'x'
            self.var2 = 'y'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'y_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.yVmin, self.yVmax]
            # self.varinitialpos = [initial_position[0],initial_position[1]]
            scan2D()
        elif self.settings['scan_axes'] == 'xz':
            self.var1 = 'x'
            self.var2 = 'z'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.zVmin, self.zVmax]
            # self.varinitialpos = [initial_position[0],initial_position[2]]
            scan2D()
        elif self.settings['scan_axes'] == 'yz':
            self.var1 = 'y'
            self.var2 = 'z'
            self.var1channel = 'y_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            self.var2range = [self.zVmin, self.zVmax]
            # self.varinitialpos = [initial_position[1],initial_position[2]]
            scan2D()
        elif self.settings['scan_axes'] == 'x':
            self.var1 = 'x'
            self.var1channel = 'x_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            # self.varinitialpos = initial_position[0]
            scan1D(self.var1)
        elif self.settings['scan_axes'] == 'y':
            self.var1 = 'y'
            self.var1channel = 'y_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            # self.varinitialpos = initial_position[1]
            scan1D(self.var1)
        elif self.settings['scan_axes'] == 'z':
            self.var1 = 'z'
            self.var1channel = 'z_ao_channel'
            self.var1range = [self.zVmin, self.zVmax]
            # self.varinitialpos = initial_position[2]
            scan1D(self.var1)

        # self.daq_out.set_analog_voltages(
        #     {self.settings['DAQ_channels']['x_ao_channel']: initial_position[0],
        #     self.settings['DAQ_channels']['y_ao_channel']: initial_position[1],
        #     self.settings['DAQ_channels']['z_ao_channel']: initial_position[2]})
        # print('voltage returned to initial values')

        # turn off laser and apd_switch
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': False}})
        print('laser and APD readout is off.')

    # def get_confocal_location(self):
    #     """
    #     Returns the current position of the galvo. Requires a daq with analog inputs internally routed to the analog
    #     outputs (ex. NI6353. Note that the cDAQ does not have this capability).
    #     Returns: list with two floats, which give the x and y position of the galvo mirror
    #     """
    #     confocal_position = self.daq_out.get_analog_voltages([
    #         self.settings['DAQ_channels']['x_ao_channel'],
    #         self.settings['DAQ_channels']['y_ao_channel'],
    #         self.settings['DAQ_channels']['z_ao_channel']]
    #     )
    #     return confocal_position

    def set_confocal_location(self, confocal_position):
        """
        sets the current position of the confocal
        confocal_position: list with three floats, which give the x, y, z positions of the confocal (galvo mirrors and objective)
        """
        print('\t'.join(map(str, confocal_position)))
        if confocal_position[0] > 10 or confocal_position[0] < -10 or confocal_position[1] > 10 or confocal_position[
            1] < -10:
            raise ValueError('The script attempted to set the galvo position to an illegal position outside of +- 10 V')
        if confocal_position[2] > 10 or confocal_position[2] < 0:
            raise ValueError(
                'The script attempted to set the objective position to an illegal position outside of 0-10 V')

        pt = confocal_position
        # daq API only accepts either one point and one channel or multiple points and multiple channels
        pt = np.transpose(np.column_stack((pt[0], pt[1], pt[2])))
        pt = (np.repeat(pt, 3, axis=1))

        task = self.daq_out.setup_AO(
            [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel'],
             self.settings['DAQ_channels']['z_ao_channel']], pt)
        self.daq_out.run(task)
        self.daq_out.waitToFinish(task)
        self.daq_out.stop(task)

    @staticmethod
    def pts_to_extent(pta, ptb, roi_mode):
        """

        Args:
            pta: point a
            ptb: point b
            roi_mode:   mode how to calculate region of interest
                        corner: pta and ptb are diagonal corners of rectangle.
                        center: pta is center and ptb is extend or rectangle

        Returns: extend of region of interest [xVmin, xVmax, yVmax, yVmin]

        """
        if roi_mode == 'corner':
            xVmin = min(pta['x'], ptb['x'])
            xVmax = max(pta['x'], ptb['x'])
            yVmin = min(pta['y'], ptb['y'])
            yVmax = max(pta['y'], ptb['y'])
            zVmin = min(pta['z'], ptb['z'])
            zVmax = max(pta['z'], ptb['z'])
        elif roi_mode == 'center':
            xVmin = pta['x'] - float(ptb['x']) / 2.
            xVmax = pta['x'] + float(ptb['x']) / 2.
            yVmin = pta['y'] - float(ptb['y']) / 2.
            yVmax = pta['y'] + float(ptb['y']) / 2.
            zVmin = pta['z'] - float(ptb['z']) / 2.
            zVmax = pta['z'] + float(ptb['z']) / 2.
        return [xVmin, xVmax, yVmin, yVmax, zVmin, zVmax]

    def plot(self, figure_list):
        # Choose whether to plot results in top or bottom figure
        # print('plot')
        if 'image_data' in self.data.keys() is not None:
            if np.ndim(self.data['image_data']) == 2:
                super(ObjectiveScan, self).plot([figure_list[0]])
            elif np.ndim(self.data['image_data']) == 1:
                super(ObjectiveScan, self).plot([figure_list[1]])

    def _plot(self, axes_list, data=None):
        """
        Plots the confocal scan image
        Args:
            axes_list: list of axes objects on which to plot the galvo scan on the first axes object
            data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        # print('_plot')
        # print('np.ndim(data')

        if data is None:
            data = self.data
        # plot_fluorescence_new(data['image_data'], data['extent'], self.data['varcalib'], self.data['varlbls'], self.data['varinitialpos'], axes_list[0], min_counts=self.settings['min_counts_plot'], max_counts=self.settings['max_counts_plot'])
        # print(np.ndim(data['image_data']))
        if np.ndim(data['image_data']) == 2:
            # plot_fluorescence_new(data['image_data'], data['extent'], self.data['varlbls'], self.data['varinitialpos'], axes_list[0], min_counts=self.settings['min_counts_plot'], max_counts=self.settings['max_counts_plot'])
            plot_fluorescence_new(data['image_data'], data['extent'], axes_list[0],
                                  max_counts=self.settings['max_counts_plot'],
                                  min_counts=self.settings['min_counts_plot'], axes_labels=self.settings['scan_axes'])
        elif np.ndim(data['image_data']) == 1:
            # plot_counts(axes_list[0], data['image_data'], axes_labels=self.data['varlbls'])
            plot_counts_vs_pos(axes_list[0], data['image_data'],
                               np.linspace(data['bounds'][0], data['bounds'][1], len(data['image_data'])),
                               x_label=data['varlbls'])

    def _update_plot(self, axes_list):
        """
        updates the galvo scan image
        Args:
            axes_list: list of axes objects on which to plot plots the esr on the first axes object
        """
        # print('_update_plot')
        if np.ndim(self.data['image_data']) == 2:
            # update_fluorescence(self.data['image_data'], axes_list[0], self.settings['min_counts_plot'], self.settings['max_counts_plot'])
            update_fluorescence(self.data['image_data'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                                min_counts=self.settings['min_counts_plot'])
        elif np.ndim(self.data['image_data']) == 1:
            # plot_counts(axes_list[0], self.data['image_data'], np.linspace(self.data['bounds'][0],self.data['bounds'][1],len(self.data['image_data'])), self.data['varlbls'])
            update_counts_vs_pos(axes_list[0], self.data['image_data'],
                                 np.linspace(self.data['bounds'][0], self.data['bounds'][1],
                                             len(self.data['image_data'])))

    def get_axes_layout(self, figure_list):
        """
        returns the axes objects the script needs to plot its data
        the default creates a single axes object on each figure
        This can/should be overwritten in a child script if more axes objects are needed
        Args:
            figure_list: a list of figure objects
        Returns:
            axes_list: a list of axes objects

        """

        # only pick the first figure from the figure list, this avoids that get_axes_layout clears all the figures
        return super(ObjectiveScan, self).get_axes_layout([figure_list[0]])


class ObjectiveScan_qm(Script):
    """
        Objective scan x, y and z. After scan, the objective will return back to the initial locations.
        - Ziwei Qiu 7/24/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('scan_axes', 'xy', ['xy', 'xz', 'yz', 'x', 'y', 'z'], 'Choose 2D or 1D confocal scan to perform'),
        Parameter('point_a',
                  [Parameter('x', 5, float, 'x-coordinate [V]'),
                   Parameter('y', 5, float, 'y-coordinate [V]'),
                   Parameter('z', 5, float, 'z-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 7.0, float, 'x-coordinate [V]'),
                   Parameter('y', 7.0, float, 'y-coordinate [V]'),
                   Parameter('z', 10.0, float, 'z-coordinate [V]')
                   ]),
        Parameter('RoI_mode', 'center', ['corner', 'center'], 'mode to calculate region of interest.\n \
                                                           corner: pta and ptb are diagonal corners of rectangle.\n \
                                                           center: pta is center and pta is extend or rectangle'),
        Parameter('num_points',
                  [Parameter('x', 125, int, 'number of x points to scan'),
                   Parameter('y', 125, int, 'number of y points to scan'),
                   Parameter('z', 51, int, 'number of z points to scan')
                   ]),
        Parameter('time_per_pt',
                  [Parameter('xy', .005, [.002, .005, .01, .015, .02, 0.05, 0.1, 0.2, .25, .5, 1.],
                             'time in s to measure at each point'),
                   Parameter('z', .5, [.05, .1, .2, .25, .5, 1.], 'time in s to measure at each point for 1D z-scans only'),
                   ]),
        Parameter('settle_time',
                  [Parameter('xy', .001, [.0002, .0005, .001, .002, .005, .01, .05, .1, .25],
                             'wait time between points to allow objective to settle'),
                   Parameter('z', .05, [.005, .01, .05, .1, .25],
                             'settle time for objective z-motion (measured for oil objective to be ~10ms, in reality appears to be much longer)'),
                   ]),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao5', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao6', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('z_ao_channel', 'ao7', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for z voltage analog output'),
                   Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring z voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1'], 'Daq channel used for counter')
                   ])
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'PB': G22BPulseBlaster}
    _SCRIPTS = {}

    def __init__(self, instruments, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes ConfocalScan script for use in gui

        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data

        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, log_function=log_function,
                        data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in = self.instruments['NI6602']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.qm_connect()

    def qm_connect(self):
        if self.settings['IP_address'] == 'automatic':
            try:
                self.qmm = QuantumMachinesManager()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def turn_on_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as laser_on:
                with infinite_loop_():
                    play('trig', 'laser', duration=3000)

            self.qm.execute(laser_on)
            print('Laser is on.')

    def turn_off_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as job_stop:
                play('trig', 'laser', duration=10)

            self.qm.execute(job_stop)
            print('Laser is off.')

    def _function(self):
        """
        Executes threaded galvo scan
        """

        # turn on laser
        self.turn_on_laser()

        def scan2D():
            self._recording = False

            self.clockAdjust = int(
                (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) /
                self.settings['settle_time']['xy'])

            if self.clockAdjust % 2 == 1:
                self.clockAdjust += 1

            self.var1_array = np.repeat(
                np.linspace(self.var1range[0], self.var1range[1], self.settings['num_points'][self.var1],
                            endpoint=True),
                self.clockAdjust)
            self.var2_array = np.linspace(self.var2range[0], self.var2range[1], self.settings['num_points'][self.var2],
                                          endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var2],
                                                 self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1],
                                    self.var2range[0], self.var2range[1]]}
            self.data['extent'] = [self.var1range[0], self.var1range[1], self.var2range[1], self.var2range[0]]
            self.data['varlbls'] = [self.var1 + ' [V]', self.var2 + ' [V]']

            # objective takes longer to settle after big jump, so give it time before starting scan:
            if self.settings['scan_axes'] == 'xz' or self.settings['scan_axes'] == 'yz':
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.var2_array[0]})
                time.sleep(0.4)

            for var2Num in range(0, len(self.var2_array)):

                if self._abort:
                    break

                # set galvo to initial point of next line
                self.initPt = [self.var1_array[0], self.var2_array[var2Num]]
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.initPt[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.initPt[1]})

                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(
                    self.settings['DAQ_channels']['counter_channel'],
                    len(self.var1_array) + 1)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array, ctrtask)

                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                var1LineData, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(var1LineData)
                summedData = np.zeros(int(len(self.var1_array) / self.clockAdjust))

                for i in range(0, int((len(self.var1_array) / self.clockAdjust))):
                    pxarray = diffData[(i * self.clockAdjust + 1):(i * self.clockAdjust + self.clockAdjust - 1)]
                    normalization = len(pxarray) / self.sample_rate / 0.001
                    summedData[i] = np.sum(pxarray) / normalization

                # summedData = np.flipud(summedData)

                # also normalizing to kcounts/sec
                # self.data['image_data'][var2Num] = summedData * (.001 / self.settings['time_per_pt']['galvo'])
                self.data['image_data'][var2Num] = summedData
                self.progress = float(var2Num + 1) / len(self.var2_array) * 100
                self.updateProgress.emit(int(self.progress))

        def scan1D(var1):
            self._recording = False
            if var1 == 'z':
                nsamples = int(
                    (self.settings['time_per_pt'][var1] + self.settings['settle_time'][var1]) * self.sample_rate)
            else:
                nsamples = int(
                    (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) * self.sample_rate)
            if nsamples % 2 == 1:
                nsamples += 1
            self.var1_array = np.linspace(self.var1range[0],
                                          self.var1range[1],
                                          self.settings['num_points'][self.var1], endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1]]}
            self.data['varlbls'] = self.var1 + ' [V]'

            # objective takes longer to settle after a big jump, so give it time before starting scan:
            self.daq_out.set_analog_voltages({self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0]})
            time.sleep(0.5)

            last_refresh_time = time.time()
            for var1Num in range(0, len(self.var1_array)):
                if self._abort:
                    break
                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(self.settings['DAQ_channels']['counter_channel'], nsamples)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array[var1Num] * np.ones(nsamples), ctrtask)
                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                samparray, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(samparray)

                # sum and normalize to kcounts/sec
                # self.data['image_data'][var1Num] = np.sum(diffData) * (.001 / self.settings['time_per_pt']['z-piezo'])
                normalization = len(diffData) / self.sample_rate / 0.001
                self.data['image_data'][var1Num] = np.sum(diffData) / normalization

                if time.time() - last_refresh_time > 0.8:
                    self.progress = float(var1Num + 1) / len(self.var1_array) * 100
                    self.updateProgress.emit(int(self.progress))
                    last_refresh_time = time.time()
                # else:
                #     print('No refresh!!!')

        initial_position = self.daq_in_AI.get_analog_voltages(
            [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
             self.settings['DAQ_channels']['z_ai_channel']])

        print('initial positions are:')
        print(initial_position)

        [self.xVmin, self.xVmax, self.yVmin, self.yVmax, self.zVmin, self.zVmax] = self.pts_to_extent(
            self.settings['point_a'],
            self.settings['point_b'],
            self.settings['RoI_mode'])

        self.sample_rate = float(1) / self.settings['settle_time']['xy']

        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['z_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate

        # print('ready for scanning')

        # depending to axes to be scanned, assigns the correct channels to be scanned and scan ranges, then starts the 2D or 1D scan
        self.var1range = 0
        self.var2range = 0
        if self.settings['scan_axes'] == 'xy':
            self.var1 = 'x'
            self.var2 = 'y'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'y_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.yVmin, self.yVmax]
            # self.varinitialpos = [initial_position[0],initial_position[1]]
            scan2D()
        elif self.settings['scan_axes'] == 'xz':
            self.var1 = 'x'
            self.var2 = 'z'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.zVmin, self.zVmax]
            # self.varinitialpos = [initial_position[0],initial_position[2]]
            scan2D()
        elif self.settings['scan_axes'] == 'yz':
            self.var1 = 'y'
            self.var2 = 'z'
            self.var1channel = 'y_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            self.var2range = [self.zVmin, self.zVmax]
            # self.varinitialpos = [initial_position[1],initial_position[2]]
            scan2D()
        elif self.settings['scan_axes'] == 'x':
            self.var1 = 'x'
            self.var1channel = 'x_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            # self.varinitialpos = initial_position[0]
            scan1D(self.var1)
        elif self.settings['scan_axes'] == 'y':
            self.var1 = 'y'
            self.var1channel = 'y_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            # self.varinitialpos = initial_position[1]
            scan1D(self.var1)
        elif self.settings['scan_axes'] == 'z':
            self.var1 = 'z'
            self.var1channel = 'z_ao_channel'
            self.var1range = [self.zVmin, self.zVmax]
            # self.varinitialpos = initial_position[2]
            scan1D(self.var1)

        # turn off laser
        self.turn_off_laser()

        self.daq_out.set_analog_voltages(
            {self.settings['DAQ_channels']['x_ao_channel']: initial_position[0],
            self.settings['DAQ_channels']['y_ao_channel']: initial_position[1],
            self.settings['DAQ_channels']['z_ao_channel']: initial_position[2]})
        print('voltage returned to initial values')

    def get_confocal_location(self):
        """
        Returns the current position of the galvo. Requires a daq with analog inputs internally routed to the analog
        outputs (ex. NI6353. Note that the cDAQ does not have this capability).
        Returns: list with two floats, which give the x and y position of the galvo mirror
        """
        confocal_position = self.daq_out.get_analog_voltages([
            self.settings['DAQ_channels']['x_ao_channel'],
            self.settings['DAQ_channels']['y_ao_channel'],
            self.settings['DAQ_channels']['z_ao_channel']]
        )
        return confocal_position

    def set_confocal_location(self, confocal_position):
        """
        sets the current position of the confocal
        confocal_position: list with three floats, which give the x, y, z positions of the confocal (galvo mirrors and objective)
        """
        print('\t'.join(map(str, confocal_position)))
        if confocal_position[0] > 10 or confocal_position[0] < -10 or confocal_position[1] > 10 or confocal_position[
            1] < -10:
            raise ValueError('The script attempted to set the galvo position to an illegal position outside of +- 10 V')
        if confocal_position[2] > 10 or confocal_position[2] < 0:
            raise ValueError(
                'The script attempted to set the objective position to an illegal position outside of 0-10 V')

        pt = confocal_position
        # daq API only accepts either one point and one channel or multiple points and multiple channels
        pt = np.transpose(np.column_stack((pt[0], pt[1], pt[2])))
        pt = (np.repeat(pt, 3, axis=1))

        task = self.daq_out.setup_AO(
            [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel'],
             self.settings['DAQ_channels']['z_ao_channel']], pt)
        self.daq_out.run(task)
        self.daq_out.waitToFinish(task)
        self.daq_out.stop(task)

    @staticmethod
    def pts_to_extent(pta, ptb, roi_mode):
        """

        Args:
            pta: point a
            ptb: point b
            roi_mode:   mode how to calculate region of interest
                        corner: pta and ptb are diagonal corners of rectangle.
                        center: pta is center and ptb is extend or rectangle

        Returns: extend of region of interest [xVmin, xVmax, yVmax, yVmin]

        """
        if roi_mode == 'corner':
            xVmin = min(pta['x'], ptb['x'])
            xVmax = max(pta['x'], ptb['x'])
            yVmin = min(pta['y'], ptb['y'])
            yVmax = max(pta['y'], ptb['y'])
            zVmin = min(pta['z'], ptb['z'])
            zVmax = max(pta['z'], ptb['z'])
        elif roi_mode == 'center':
            xVmin = pta['x'] - float(ptb['x']) / 2.
            xVmax = pta['x'] + float(ptb['x']) / 2.
            yVmin = pta['y'] - float(ptb['y']) / 2.
            yVmax = pta['y'] + float(ptb['y']) / 2.
            zVmin = pta['z'] - float(ptb['z']) / 2.
            zVmax = pta['z'] + float(ptb['z']) / 2.
        return [xVmin, xVmax, yVmin, yVmax, zVmin, zVmax]

    def plot(self, figure_list):
        # Choose whether to plot results in top or bottom figure
        # print('plot')
        if 'image_data' in self.data.keys() is not None:
            if np.ndim(self.data['image_data']) == 2:
                super(ObjectiveScan_qm, self).plot([figure_list[0]])
            elif np.ndim(self.data['image_data']) == 1:
                super(ObjectiveScan_qm, self).plot([figure_list[1]])

    def _plot(self, axes_list, data=None):
        """
        Plots the confocal scan image
        Args:
            axes_list: list of axes objects on which to plot the galvo scan on the first axes object
            data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        # print('_plot')
        # print('np.ndim(data')

        if data is None:
            data = self.data
        # plot_fluorescence_new(data['image_data'], data['extent'], self.data['varcalib'], self.data['varlbls'], self.data['varinitialpos'], axes_list[0], min_counts=self.settings['min_counts_plot'], max_counts=self.settings['max_counts_plot'])
        # print(np.ndim(data['image_data']))
        axes_list[0].clear()
        if 'image_data' in data.keys():
            if np.ndim(data['image_data']) == 2:
                # plot_fluorescence_new(data['image_data'], data['extent'], self.data['varlbls'], self.data['varinitialpos'], axes_list[0], min_counts=self.settings['min_counts_plot'], max_counts=self.settings['max_counts_plot'])
                plot_fluorescence_new(data['image_data'], data['extent'], axes_list[0],
                                      max_counts=self.settings['max_counts_plot'],
                                      min_counts=self.settings['min_counts_plot'], axes_labels=self.settings['scan_axes'])
            elif np.ndim(data['image_data']) == 1:
                # plot_counts(axes_list[0], data['image_data'], axes_labels=self.data['varlbls'])
                plot_counts_vs_pos(axes_list[0], data['image_data'],
                                   np.linspace(data['bounds'][0], data['bounds'][1], len(data['image_data'])),
                                   x_label=data['varlbls'])

    def _update_plot(self, axes_list):
        """
        updates the galvo scan image
        Args:
            axes_list: list of axes objects on which to plot plots the esr on the first axes object
        """
        # print('_update_plot')
        # self._plot(axes_list)
        try:

            if np.ndim(self.data['image_data']) == 1:
                # update_counts_vs_pos(axes_list[0], self.data['image_data'],
                #                      np.linspace(self.data['bounds'][0], self.data['bounds'][1],
                #                                  len(self.data['image_data'])))
                axes_list[0].lines[0].set_ydata(self.data['image_data'])
                axes_list[0].relim()
                axes_list[0].autoscale_view()
            elif np.ndim(self.data['image_data']) == 2:
                # update_fluorescence(self.data['image_data'], axes_list[0], self.settings['min_counts_plot'], self.settings['max_counts_plot'])
                update_fluorescence(self.data['image_data'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                                    min_counts=self.settings['min_counts_plot'])

        except Exception as e:
            print('!! ** ATTENTION in confocal scan update plot **')
            print(e)
            self._plot(axes_list)

    def get_axes_layout(self, figure_list):
        """
        returns the axes objects the script needs to plot its data
        the default creates a single axes object on each figure
        This can/should be overwritten in a child script if more axes objects are needed
        Args:
            figure_list: a list of figure objects
        Returns:
            axes_list: a list of axes objects

        """

        # only pick the first figure from the figure list, this avoids that get_axes_layout clears all the figures
        # return super(ObjectiveScan_qm, self).get_axes_layout([figure_list[0]])
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(111))

        else:
            axes_list.append(figure_list[0].axes[0])

        return axes_list


class ObjectiveScanNoLaser(Script):
    """
        Objective scan x, y and z. Laser is turned on beforehand so no laser is involved in this script.
        After scan, the objective will return back to the initial locations.
        This script is made for tracking NV in QM pulsed experiments.
        - Ziwei Qiu 9/27/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('scan_axes', 'xy', ['xy', 'xz', 'yz', 'x', 'y', 'z'], 'Choose 2D or 1D confocal scan to perform'),
        Parameter('point_a',
                  [Parameter('x', 0, float, 'x-coordinate [V]'),
                   Parameter('y', 0, float, 'y-coordinate [V]'),
                   Parameter('z', 5, float, 'z-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                   Parameter('y', 1.0, float, 'y-coordinate [V]'),
                   Parameter('z', 10.0, float, 'z-coordinate [V]')
                   ]),
        Parameter('RoI_mode', 'center', ['corner', 'center'], 'mode to calculate region of interest.\n \
                                                           corner: pta and ptb are diagonal corners of rectangle.\n \
                                                           center: pta is center and pta is extend or rectangle'),
        Parameter('num_points',
                  [Parameter('x', 200, int, 'number of x points to scan'),
                   Parameter('y', 200, int, 'number of y points to scan'),
                   Parameter('z', 51, int, 'number of z points to scan')
                   ]),
        Parameter('time_per_pt',
                  [Parameter('xy', .005, [.002, .005, .01, .015, .02, 0.05, 0.1, 0.2, .25, .5, 1.],
                             'time in s to measure at each point'),
                   Parameter('z', .5, [.05, .1, .2, .25, .5, 1.], 'time in s to measure at each point for 1D z-scans only'),
                   ]),
        Parameter('settle_time',
                  [Parameter('xy', .001, [.0002, .0005, .001, .002, .005, .01, .05, .1, .25],
                             'wait time between points to allow objective to settle'),
                   Parameter('z', .05, [.005, .01, .05, .1, .25],
                             'settle time for objective z-motion (measured for oil objective to be ~10ms, in reality appears to be much longer)'),
                   ]),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao5', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao6', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('z_ao_channel', 'ao7', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for z voltage analog output'),
                   Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring z voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1'], 'Daq channel used for counter')
                   ])
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220}
    _SCRIPTS = {}

    def __init__(self, instruments, name=None, settings=None, log_function=None, data_path=None):
        '''
            Initializes ConfocalScan script for use in gui

            Args:
                instruments: list of instrument objects
                name: name to give to instantiated script object
                settings: dictionary of new settings to pass in to override defaults
                log_function: log function passed from the gui to direct log calls to the gui log
                data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, log_function=log_function,
                        data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in = self.instruments['NI6602']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']


    def _function(self):
        """
        Executes threaded galvo scan
        """
        def scan2D():
            self._recording = False

            self.clockAdjust = int(
                (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) /
                self.settings['settle_time']['xy'])

            if self.clockAdjust % 2 == 1:
                self.clockAdjust += 1

            self.var1_array = np.repeat(
                np.linspace(self.var1range[0], self.var1range[1], self.settings['num_points'][self.var1],
                            endpoint=True),
                self.clockAdjust)
            self.var2_array = np.linspace(self.var2range[0], self.var2range[1], self.settings['num_points'][self.var2],
                                          endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var2],
                                                 self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1],
                                    self.var2range[0], self.var2range[1]]}
            self.data['extent'] = [self.var1range[0], self.var1range[1], self.var2range[1], self.var2range[0]]
            self.data['varlbls'] = [self.var1 + ' [V]', self.var2 + ' [V]']

            # objective takes longer to settle after big jump, so give it time before starting scan:
            if self.settings['scan_axes'] == 'xz' or self.settings['scan_axes'] == 'yz':
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.var2_array[0]})
                time.sleep(0.4)

            for var2Num in range(0, len(self.var2_array)):

                if self._abort:
                    break

                # set galvo to initial point of next line
                self.initPt = [self.var1_array[0], self.var2_array[var2Num]]
                self.daq_out.set_analog_voltages(
                    {self.settings['DAQ_channels'][self.var1channel]: self.initPt[0],
                     self.settings['DAQ_channels'][self.var2channel]: self.initPt[1]})

                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(
                    self.settings['DAQ_channels']['counter_channel'],
                    len(self.var1_array) + 1)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array, ctrtask)

                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                var1LineData, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(var1LineData)
                summedData = np.zeros(int(len(self.var1_array) / self.clockAdjust))

                for i in range(0, int((len(self.var1_array) / self.clockAdjust))):
                    pxarray = diffData[(i * self.clockAdjust + 1):(i * self.clockAdjust + self.clockAdjust - 1)]
                    normalization = len(pxarray) / self.sample_rate / 0.001
                    summedData[i] = np.sum(pxarray) / normalization

                # summedData = np.flipud(summedData)

                # also normalizing to kcounts/sec
                # self.data['image_data'][var2Num] = summedData * (.001 / self.settings['time_per_pt']['galvo'])
                self.data['image_data'][var2Num] = summedData
                self.progress = float(var2Num + 1) / len(self.var2_array) * 100
                self.updateProgress.emit(int(self.progress))

        def scan1D(var1):
            self._recording = False
            if var1 == 'z':
                nsamples = int(
                    (self.settings['time_per_pt'][var1] + self.settings['settle_time'][var1]) * self.sample_rate)
            else:
                nsamples = int(
                    (self.settings['time_per_pt']['xy'] + self.settings['settle_time']['xy']) * self.sample_rate)
            if nsamples % 2 == 1:
                nsamples += 1
            self.var1_array = np.linspace(self.var1range[0],
                                          self.var1range[1],
                                          self.settings['num_points'][self.var1], endpoint=True)

            self.data = {'image_data': np.zeros((self.settings['num_points'][self.var1])),
                         'bounds': [self.var1range[0], self.var1range[1]]}
            self.data['varlbls'] = self.var1 + ' [V]'

            # objective takes longer to settle after a big jump, so give it time before starting scan:
            self.daq_out.set_analog_voltages({self.settings['DAQ_channels'][self.var1channel]: self.var1_array[0]})
            time.sleep(0.5)

            last_refresh_time = time.time()
            for var1Num in range(0, len(self.var1_array)):
                if self._abort:
                    break
                # initialize APD thread
                ctrtask = self.daq_in.setup_counter(self.settings['DAQ_channels']['counter_channel'], nsamples)
                aotask = self.daq_out.setup_AO([self.settings['DAQ_channels'][self.var1channel]],
                                               self.var1_array[var1Num] * np.ones(nsamples), ctrtask)
                # start counter and scanning sequence
                self.daq_out.run(aotask)
                self.daq_in.run(ctrtask)
                self.daq_out.waitToFinish(aotask)
                self.daq_out.stop(aotask)
                samparray, _ = self.daq_in.read(ctrtask)
                self.daq_in.stop(ctrtask)
                diffData = np.diff(samparray)

                # sum and normalize to kcounts/sec
                # self.data['image_data'][var1Num] = np.sum(diffData) * (.001 / self.settings['time_per_pt']['z-piezo'])
                normalization = len(diffData) / self.sample_rate / 0.001
                self.data['image_data'][var1Num] = np.sum(diffData) / normalization

                if time.time() - last_refresh_time > 0.8:
                    self.progress = float(var1Num + 1) / len(self.var1_array) * 100
                    self.updateProgress.emit(int(self.progress))
                    last_refresh_time = time.time()

        initial_position = self.daq_in_AI.get_analog_voltages(
            [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
             self.settings['DAQ_channels']['z_ai_channel']])

        [self.xVmin, self.xVmax, self.yVmin, self.yVmax, self.zVmin, self.zVmax] = self.pts_to_extent(
            self.settings['point_a'],
            self.settings['point_b'],
            self.settings['RoI_mode'])

        self.sample_rate = float(1) / self.settings['settle_time']['xy']

        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['z_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate


        # depending to axes to be scanned, assigns the correct channels to be scanned and scan ranges, then starts the 2D or 1D scan
        self.var1range = 0
        self.var2range = 0
        if self.settings['scan_axes'] == 'xy':
            self.var1 = 'x'
            self.var2 = 'y'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'y_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.yVmin, self.yVmax]
            scan2D()

        elif self.settings['scan_axes'] == 'xz':
            self.var1 = 'x'
            self.var2 = 'z'
            self.var1channel = 'x_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            self.var2range = [self.zVmin, self.zVmax]
            scan2D()

        elif self.settings['scan_axes'] == 'yz':
            self.var1 = 'y'
            self.var2 = 'z'
            self.var1channel = 'y_ao_channel'
            self.var2channel = 'z_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            self.var2range = [self.zVmin, self.zVmax]
            scan2D()

        elif self.settings['scan_axes'] == 'x':
            self.var1 = 'x'
            self.var1channel = 'x_ao_channel'
            self.var1range = [self.xVmin, self.xVmax]
            scan1D(self.var1)

        elif self.settings['scan_axes'] == 'y':
            self.var1 = 'y'
            self.var1channel = 'y_ao_channel'
            self.var1range = [self.yVmin, self.yVmax]
            scan1D(self.var1)

        elif self.settings['scan_axes'] == 'z':
            self.var1 = 'z'
            self.var1channel = 'z_ao_channel'
            self.var1range = [self.zVmin, self.zVmax]
            scan1D(self.var1)

        self.daq_out.set_analog_voltages(
            {self.settings['DAQ_channels']['x_ao_channel']: initial_position[0],
            self.settings['DAQ_channels']['y_ao_channel']: initial_position[1],
            self.settings['DAQ_channels']['z_ao_channel']: initial_position[2]})


    def get_confocal_location(self):
        """
        Returns the current position of the galvo. Requires a daq with analog inputs internally routed to the analog
        outputs (ex. NI6353. Note that the cDAQ does not have this capability).
        Returns: list with two floats, which give the x and y position of the galvo mirror
        """
        confocal_position = self.daq_out.get_analog_voltages([
            self.settings['DAQ_channels']['x_ao_channel'],
            self.settings['DAQ_channels']['y_ao_channel'],
            self.settings['DAQ_channels']['z_ao_channel']]
        )
        return confocal_position


    def set_confocal_location(self, confocal_position):
        """
        sets the current position of the confocal
        confocal_position: list with three floats, which give the x, y, z positions of the confocal (galvo mirrors and objective)
        """
        print('\t'.join(map(str, confocal_position)))
        if confocal_position[0] > 10 or confocal_position[0] < -10 or confocal_position[1] > 10 or confocal_position[
            1] < -10:
            raise ValueError('The script attempted to set the galvo position to an illegal position outside of +- 10 V')
        if confocal_position[2] > 10 or confocal_position[2] < 0:
            raise ValueError(
                'The script attempted to set the objective position to an illegal position outside of 0-10 V')

        pt = confocal_position
        # daq API only accepts either one point and one channel or multiple points and multiple channels
        pt = np.transpose(np.column_stack((pt[0], pt[1], pt[2])))
        pt = (np.repeat(pt, 3, axis=1))

        task = self.daq_out.setup_AO(
            [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel'],
             self.settings['DAQ_channels']['z_ao_channel']], pt)
        self.daq_out.run(task)
        self.daq_out.waitToFinish(task)
        self.daq_out.stop(task)

    @staticmethod
    def pts_to_extent(pta, ptb, roi_mode):
        """
            Args:
                pta: point a
                ptb: point b
                roi_mode:   mode how to calculate region of interest
                            corner: pta and ptb are diagonal corners of rectangle.
                            center: pta is center and ptb is extend or rectangle

            Returns: extend of region of interest [xVmin, xVmax, yVmax, yVmin]
        """
        if roi_mode == 'corner':
            xVmin = min(pta['x'], ptb['x'])
            xVmax = max(pta['x'], ptb['x'])
            yVmin = min(pta['y'], ptb['y'])
            yVmax = max(pta['y'], ptb['y'])
            zVmin = min(pta['z'], ptb['z'])
            zVmax = max(pta['z'], ptb['z'])
        elif roi_mode == 'center':
            xVmin = pta['x'] - float(ptb['x']) / 2.
            xVmax = pta['x'] + float(ptb['x']) / 2.
            yVmin = pta['y'] - float(ptb['y']) / 2.
            yVmax = pta['y'] + float(ptb['y']) / 2.
            zVmin = pta['z'] - float(ptb['z']) / 2.
            zVmax = pta['z'] + float(ptb['z']) / 2.
        return [xVmin, xVmax, yVmin, yVmax, zVmin, zVmax]

    def plot(self, figure_list):
        # Choose whether to plot results in top or bottom figure
        if 'image_data' in self.data.keys() is not None:
            if np.ndim(self.data['image_data']) == 2:
                super(ObjectiveScanNoLaser, self).plot([figure_list[0]])
            elif np.ndim(self.data['image_data']) == 1:
                super(ObjectiveScanNoLaser, self).plot([figure_list[1]])

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data

        if np.ndim(data['image_data']) == 2:
            plot_fluorescence_new(data['image_data'], data['extent'], axes_list[0],
                                  max_counts=self.settings['max_counts_plot'],
                                  min_counts=self.settings['min_counts_plot'], axes_labels=self.settings['scan_axes'])

        elif np.ndim(data['image_data']) == 1:
            plot_counts_vs_pos(axes_list[0], data['image_data'],
                               np.linspace(data['bounds'][0], data['bounds'][1], len(data['image_data'])),
                               x_label=data['varlbls'])

    def _update_plot(self, axes_list):
        """
            updates the galvo scan image
            Args:
                axes_list: list of axes objects on which to plot plots the esr on the first axes object
        """
        if np.ndim(self.data['image_data']) == 2:
            update_fluorescence(self.data['image_data'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                                min_counts=self.settings['min_counts_plot'])

        elif np.ndim(self.data['image_data']) == 1:
            update_counts_vs_pos(axes_list[0], self.data['image_data'],
                                 np.linspace(self.data['bounds'][0], self.data['bounds'][1],
                                             len(self.data['image_data'])))

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            the default creates a single axes object on each figure
            This can/should be overwritten in a child script if more axes objects are needed
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """

        # only pick the first figure from the figure list, this avoids that get_axes_layout clears all the figures
        return super(ObjectiveScanNoLaser, self).get_axes_layout([figure_list[0]])


class SampleScan1D_Single(Script):
    """
        AFM 1D scan by specifying the start and end points. Only allow one-way scan.
        - Ziwei Qiu 7/30/2020

    """
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'x-coordinate [V]'),
                   Parameter('y', 0, float, 'y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0.2, float, 'x-coordinate [V]'),
                   Parameter('y', 0.2, float, 'y-coordinate [V]')
                   ]),
        Parameter('resolution', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        Parameter('refresh_per_N_pt', 2, [2, 3, 4, 5, 6, 7, 8, 9, 10], 'refresh the data plot per N samples'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending bahavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'PB': G22BPulseBlaster}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    # def __init__(self, instruments, name=None, settings=None, log_function=None, data_path=None):
    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes SampleScan1D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        # Script.__init__(self, name, settings=settings, instruments=instruments, log_function=log_function,
        #                 data_path=data_path)
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """

        self.scripts['SetScanner'].update({'to_do': 'set'})
        self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['SetScanner'].settings['step_size'] = self.settings['resolution']

        # turn on laser and apd_switch
        print('turn on laser and APD readout channel.')
        self.instruments['PB']['instance'].update({'laser': {'status': True}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': True}})

        # Get initial positions
        self.varinitialpos = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
        self.log('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

        # move to point_a
        self.scripts['SetScanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['SetScanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['SetScanner'].run()

        scan_pos_1d, dist_array, dt, N, T_tot = self._get_scan_params()
        self.data['scan_pos_1d'] = scan_pos_1d
        self.data['dist_array'] = dist_array
        self.data['data_ctr'] = np.zeros(N)
        self.data['data_analog'] = np.zeros(N)

        self.sample_rate = float(1) / dt
        # Set the corresponding sample rates for all the DAQ channels
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_DI.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

        # ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], N+1)
        # aotask = self.daq_out.setup_AO(
        #     [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], scan_pos_1d,
        #     ctrtask)
        # aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
        #                                                       N, continuous=False, clk_source=ctrtask)

        refresh_N = self.settings['refresh_per_N_pt']
        ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                               continuous_acquisition=True)
        aotask = self.daq_out.setup_AO(
            [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], scan_pos_1d,
            ctrtask)
        aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                         refresh_N, continuous=True, clk_source=ctrtask)

        # start counter and scanning sequence
        if not self.is_at_point_a():
            print('**ATTENTION** Scanner is NOT at point_a --> no scanning started.')
        elif self.is_at_point_a():
            print('Scanner is now at point_a --> Start 1D scanning')
            self.daq_out.run(aotask)
            self.daq_in_AI.run(aitask)
            self.daq_in_DI.run(ctrtask)

            # raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)
            # print('num_read_ctr', num_read_ctr)
            # diffData = np.diff(raw_data_ctr)
            # normalized_data_ctr = diffData / dt / 0.001  # convert to kcounts/sec
            # self.data['data_ctr'][0:len(normalized_data_ctr)] = normalized_data_ctr
            #
            # raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
            # print('num_read_analog', num_read_analog)
            # self.data['data_analog'][0:len(raw_data_analog)] = raw_data_analog

            self.current_index = 0
            self.last_value = 0
            normalization = dt / .001  # convert to kcounts/sec

            while True:
                if self.current_index >= len(dist_array):  # if the maximum time is hit
                    self._abort = True  # tell the script to abort

                if self._abort:
                    break

                self.progress = self.current_index * 100. / len(dist_array)
                self.updateProgress.emit(int(self.progress))

                raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                # Analog data
                if self.current_index == 0:
                    # throw the first data point
                    raw_data_analog = raw_data_analog[1:]
                if self.current_index + len(raw_data_analog) < N:
                    self.data['data_analog'][
                    self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                else:
                    self.data['data_analog'][
                    self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                    0:N - self.current_index]

                # Counter data
                for value in raw_data_ctr:
                    # print('self.last_value', self.last_value)
                    if self.current_index >= N:
                        break
                    new_val = ((float(value) - self.last_value) / normalization)
                    if self.last_value != 0:
                        self.data['data_ctr'][self.current_index] = new_val
                        self.current_index += 1
                    # print('new_val', new_val)
                    self.last_value = value

                # self.data['accu_data_ctr'][self.current_index :self.current_index + refresh_N ] = raw_data_ctr
                #
                # diffData = np.diff(self.data['accu_data_ctr'])
                # normalized_data_ctr = diffData * 0.001 / dt   # convert to kcounts/sec
                # self.data['data_ctr'][1:] = normalized_data_ctr
                # self.current_index = self.current_index +  refresh_N
                # print('self.current_index:', self.current_index)

            # self.daq_out.waitToFinish(aotask)
            self.daq_out.stop(ctrtask)
            self.daq_out.stop(aitask)
            self.daq_out.stop(aotask)

        # Return the scanner to certain positions
        if self.settings['ending_behavior'] == 'return_to_initial':
            self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
            self.scripts['SetScanner'].run()
            print('Sample scanner returned to the initial position.')
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.scripts['SetScanner'].settings['point']['x'] = 0.0
            self.scripts['SetScanner'].settings['point']['y'] = 0.0
            self.scripts['SetScanner'].run()
            print('Sample scanner returned to the origin.')
        else:
            print('Sample scanner is left at the last point.')

        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('--> scanner is at Vx={:.4}V and Vy={:.4}V'.format(current_position[0], current_position[1]))
        self.log('Scanner is at Vx={:.4}V and Vy={:.4}V'.format(current_position[0], current_position[1]))

        # turn off laser and apd_switch
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': False}})
        print('laser and APD readout is off.')

    def _get_scan_params(self, verbose=False):
        '''
        Returns an array of points to go to in the 1D scan.
        '''
        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        T_tot = dist / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']
        N = int(np.ceil(dist * ptspervolt / 2) * 2)  # number of sample
        dt = T_tot / N  # time to stay at each point
        scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
        dist_array = np.linspace(0, dist, N, endpoint=True)

        return scan_pos_1d, dist_array, dt, N, T_tot

    def is_at_point_a(self):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(self.settings['point_a']['x'] - current_position[0]) < 0.01 and np.abs(
                self.settings['point_a']['y'] - current_position[1]) < 0.01:
            return True
        else:
            return False

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data
        plot_counts_vs_pos(axes_list[0], data['data_ctr'], data['dist_array'], x_label='Position [V]',
                           title='NV-AFM 1D Scan\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                               self.settings['point_a']['x'], self.settings['point_a']['y'],
                               self.settings['point_b']['x'], self.settings['point_b']['y']))

        plot_counts_vs_pos(axes_list[1], data['data_analog'], data['dist_array'], x_label='Position [V]',
                           y_label='Z_out [V]')

    def _update_plot(self, axes_list):
        update_counts_vs_pos(axes_list[0], self.data['data_ctr'], self.data['dist_array'])
        update_counts_vs_pos(axes_list[1], self.data['data_analog'], self.data['dist_array'])

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])

        return axes_list


class AFM1D(Script):
    """
        AFM 1D scan by specifying the start and end points. It allows multiple rounds.
        - Ziwei Qiu 7/30/2020

    """
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'x-coordinate [V]'),
                   Parameter('y', 0, float, 'y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0.2, float, 'x-coordinate [V]'),
                   Parameter('y', 0.2, float, 'y-coordinate [V]')
                   ]),
        Parameter('resolution', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        Parameter('num_of_rounds', 2, int,
                  'define number of rounds of scan. Odd (oven) number represents forward (backward) scan. -1 means infinite roudns, until user click stop.'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending bahavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'PB': G22BPulseBlaster}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes AFM1D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """
        self.scripts['SetScanner'].update({'to_do': 'set'})
        self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['SetScanner'].settings['step_size'] = self.settings['resolution']

        # turn on laser and apd_switch
        print('Turned on laser and APD readout channel.')
        self.instruments['PB']['instance'].update({'laser': {'status': True}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': True}})

        # Get initial positions
        self.varinitialpos = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
        # self.log('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

        # move to point_a
        self.scripts['SetScanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['SetScanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['SetScanner'].run()

        scan_pos_1d, scan_pos_1d_flipped, dist_array, dt, N, T_tot = self._get_scan_params()
        refresh_N = self._find_refresh_N(dt, N)

        self.sample_rate = float(1) / dt
        # Set the corresponding sample rates for all the DAQ channels
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_DI.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

        # refresh_N = self.settings['refresh_per_N_pt']

        self.data['scan_pos_1d'] = scan_pos_1d
        self.data['dist_array'] = dist_array
        self.data['data_ctr'] = np.array([np.zeros(N)])
        self.data['data_analog'] = np.array([np.zeros(N)])
        self.current_round = 1  # odd means forward scan, even means backward scan

        while True:
            if self._abort == True:
                break

            # do scan
            if self.current_round % 2 == 1:  # odd number: forward scan
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d,
                    ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)

                # start counter and scanning sequence
                if not self.is_at_point_a():
                    print('**ATTENTION** Scanner is NOT at point_a --> No scanning started.')
                elif self.is_at_point_a():
                    print('Scanner is at point_a --> Forward scan (ETA={:.2f}s).'.format(dt * (N + 1)))
                    self.daq_out.run(aotask)
                    self.daq_in_AI.run(aitask)
                    self.daq_in_DI.run(ctrtask)

                    self.current_index = 0
                    self.last_value = 0
                    normalization = dt / .001  # convert to kcounts/sec

                    while True:
                        if self.current_index >= len(dist_array):  # if the maximum time is hit
                            # self._abort = True  # tell the script to abort
                            break
                        if self._abort:
                            break

                        self.progress = self.current_index * 100. / len(dist_array)
                        self.updateProgress.emit(int(self.progress))

                        # print('read_ai')
                        # daq_in_AI needs to be before daq_in_DI
                        raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                        # print('read_ctr')
                        raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                        # store analog data
                        if self.current_index == 0:
                            # throw the first data point
                            raw_data_analog = raw_data_analog[1:]
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog'][self.current_round - 1][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                        else:
                            self.data['data_analog'][self.current_round - 1][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                            0:N - self.current_index]

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr'][self.current_round - 1][self.current_index] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                        # print('self.current_index:', self.current_index)

                    # self.daq_out.waitToFinish(aotask)
                    self.daq_out.stop(ctrtask)
                    self.daq_out.stop(aitask)
                    self.daq_out.stop(aotask)

            else:  # even number: backward scan
                # print('scan_pos_1d_flipped',scan_pos_1d_flipped)
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d_flipped,
                    ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)

                # start counter and scanning sequence
                if not self.is_at_point_b():
                    print('**ATTENTION** Scanner is NOT at point_b --> No scanning started.')
                elif self.is_at_point_b():
                    print('Scanner is at point_b --> Backward scan (ETA={:.2f}s).'.format(dt * (N + 1)))
                    self.daq_out.run(aotask)
                    self.daq_in_AI.run(aitask)
                    self.daq_in_DI.run(ctrtask)

                    self.current_index = 0
                    self.last_value = 0
                    normalization = dt / .001  # convert to kcounts/sec

                    while True:
                        if self.current_index >= len(dist_array):  # if the maximum time is hit
                            # self._abort = True  # tell the script to abort
                            break

                        if self._abort:
                            break

                        self.progress = self.current_index * 100. / len(dist_array)
                        self.updateProgress.emit(int(self.progress))

                        # print('read_ai')
                        # daq_in_AI.read needs to be before daq_in_DI.read so the counter will not read with zero gaps
                        raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                        # print('read_ctr')
                        raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                        # store analog data
                        if self.current_index == 0:
                            # throw the first data point
                            raw_data_analog = raw_data_analog[1:]
                            # print('raw_data_analog',raw_data_analog)
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog'][self.current_round - 1][
                            N - self.current_index - len(raw_data_analog): N - self.current_index] = np.flip(
                                raw_data_analog)
                        else:
                            self.data['data_analog'][self.current_round - 1][0:N - self.current_index] = np.flip(
                                raw_data_analog[0:N - self.current_index])

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr'][self.current_round - 1][N - self.current_index - 1] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                        # print('self.current_index:', self.current_index)

                    # self.daq_out.waitToFinish(aotask)
                    self.daq_out.stop(ctrtask)
                    self.daq_out.stop(aitask)
                    self.daq_out.stop(aotask)

            if self._abort == True:
                break
            else:
                self.current_round += 1
                if self.current_round <= self.settings['num_of_rounds'] or self.settings['num_of_rounds'] == -1:
                    self.data['data_ctr'] = np.concatenate((self.data['data_ctr'], [np.zeros(N)]), axis=0)
                    self.data['data_analog'] = np.concatenate((self.data['data_analog'], [np.zeros(N)]), axis=0)
                else:
                    self._abort = True
                    break

        # Return the scanner to certain positions
        if self.settings['ending_behavior'] == 'return_to_initial':
            self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
            self.scripts['SetScanner'].run()
            print('Scanner returned to the initial position.')
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.scripts['SetScanner'].settings['point']['x'] = 0.0
            self.scripts['SetScanner'].settings['point']['y'] = 0.0
            self.scripts['SetScanner'].run()
            print('Scanner returned to the origin.')
        else:
            print('Scanner is left at the last point.')

        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))
        # self.log('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))

        # turn off laser and apd_switch
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['PB']['instance'].update({'apd_switch': {'status': False}})
        print('Laser and APD readout are off.')

    def _get_scan_params(self, verbose=True):
        '''
        Returns an array of points to go to in the 1D scan.
        '''
        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        T_tot = dist / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']
        N = int(np.ceil(dist * ptspervolt / 2) * 2)  # number of sample
        dt = T_tot / N  # time to stay at each point
        scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
        scan_pos_1d_flipped = np.transpose(np.linspace(Vend, Vstart, N, endpoint=True))
        dist_array = np.linspace(0, dist, N, endpoint=True)

        if verbose:
            print('dt={:.3f}s, N={:d}.'.format(dt, N))

        return scan_pos_1d, scan_pos_1d_flipped, dist_array, dt, N, T_tot

    def is_at_point_a(self):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(self.settings['point_a']['x'] - current_position[0]) < 0.01 and np.abs(
                self.settings['point_a']['y'] - current_position[1]) < 0.01:
            return True
        else:
            return False

    def is_at_point_b(self):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(self.settings['point_b']['x'] - current_position[0]) < 0.01 and np.abs(
                self.settings['point_b']['y'] - current_position[1]) < 0.01:
            return True
        else:
            return False

    def _find_refresh_N(self, dt, N, verbose=True):
        refresh_N = 2
        while True:
            if refresh_N >= N:
                refresh_N = int(np.min([N + 1, int(4 / dt)]))
                break
            if ((N + 1 + refresh_N) % refresh_N == 0 or (
                    N + 1 + refresh_N) % refresh_N >= 0.66 * refresh_N) and 2 <= dt * refresh_N <= 5:
                break
            else:
                refresh_N += 1
        if verbose:
            # print('type of refresh N', type(refresh_N))
            print('Plot refresh per {:d} points.'.format(refresh_N))
        return refresh_N

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data
        plot_counts_vs_pos_multilines(axes_list[0], data['data_ctr'], data['dist_array'], x_label='Position [V]',
                                      title='NV-AFM 1D Scan')
        plot_counts_vs_pos_multilines(axes_list[1], data['data_analog'], data['dist_array'], x_label='Position [V]',
                                      y_label='Z_out [V]')

    def _update_plot(self, axes_list):
        plot_counts_vs_pos_multilines(axes_list[0], self.data['data_ctr'], self.data['dist_array'],
                                      x_label='Position [V]',
                                      title='NV-AFM 1D Scan')
        plot_counts_vs_pos_multilines(axes_list[1], self.data['data_analog'], self.data['dist_array'],
                                      x_label='Position [V]',
                                      y_label='Z_out [V]')

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])

        return axes_list


class AFM1D_qm(Script):
    """
        AFM 1D scan by specifying the start and end points. It allows multiple rounds.
        - Ziwei Qiu 7/30/2020

    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('point_a',
                  [Parameter('x', 0, float, 'x-coordinate [V]'),
                   Parameter('y', 0, float, 'y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0.2, float, 'x-coordinate [V]'),
                   Parameter('y', 0.2, float, 'y-coordinate [V]')
                   ]),
        Parameter('resolution', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        Parameter('height', 'relative', ['relative','absolute'], 'if relative: the first analog point will be reference.'),
        Parameter('num_of_rounds', -1, int,
                  'define number of rounds of scan. Odd (oven) number represents forward (backward) scan. -1 means infinite roudns, until user click stop.'),
        Parameter('monitor_AFM', False, bool,
                  'monitor the AFM Z_out voltage and retract the tip when the feedback loop is out of control'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for monitoring the z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending bahavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'NI6210': NI6210}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes AFM1D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_usbAI = self.instruments['NI6210']['instance']
        self.qm_connect()

    def qm_connect(self):
        if self.settings['IP_address'] == 'automatic':
            try:
                self.qmm = QuantumMachinesManager()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def turn_on_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as laser_on:
                with infinite_loop_():
                    play('trig', 'laser', duration=3000)

            self.qm.execute(laser_on)
            print('Laser is on.')

    def turn_off_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as job_stop:
                play('trig', 'laser', duration=10)

            self.qm.execute(job_stop)
            print('Laser is off.')

    def _setup_anc(self):
        self.anc_sample_connected = False
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_last = z_out[0]

        if self.settings['monitor_AFM']:
            try:
                self.anc_sample = Positioner()
                self.anc_sample_connected = self.anc_sample.is_connected
            except Exception as e:
                print('** ATTENTION in creating ANC_sample **')
                print(e)


    def _check_AFM(self):
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_now = z_out[0]
        if np.abs(self.Z_scanner_now - self.Z_scanner_last) > 0.35:
            try:
                self.anc_sample.dcInEnable(5, False)
                state = self.anc_sample.getDcInEnable(5)
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                print('Z scanner dcInEnable is ' + str(state))
                self.log('Z scanner dcInEnable is ' + str(state))
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed and the tip has been retracted. Be relaxed and try again!"""

                send_email(receiver_email, message)


            except Exception as e:
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                print('** But the tip CANNOT be Retracted!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                self.log('** But the tip CANNOT be Retracted!! **')
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed BUT the tip CANNOT be retracted. Please take action!"""

                send_email(receiver_email, message)

            self._abort = True

        else:
            self.Z_scanner_last = self.Z_scanner_now

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """
        self._setup_anc()
        if self.settings['monitor_AFM'] and not self.anc_sample_connected:
            print('** Attention ** ANC350 v2 (sample) is not connected. No scanning started.')
            self._abort = True

        self.scripts['SetScanner'].update({'to_do': 'set'})
        self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['SetScanner'].settings['step_size'] = self.settings['resolution']

        # turn on laser
        self.turn_on_laser()

        # Get initial positions
        self.varinitialpos = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
        # self.log('initial_position: Vx={:.4}V and Vy={:.4}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

        # move to point_a
        self.scripts['SetScanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['SetScanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['SetScanner'].run()

        scan_pos_1d, scan_pos_1d_flipped, dist_array, dt, N, T_tot = self._get_scan_params()
        refresh_N = self._find_refresh_N(dt, N)

        self.sample_rate = float(1) / dt
        # Set the corresponding sample rates for all the DAQ channels
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_DI.settings['digital_input'][
            self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
        self.daq_in_AI.settings['analog_input'][
            self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

        # refresh_N = self.settings['refresh_per_N_pt']

        self.data['scan_pos_1d'] = scan_pos_1d
        self.data['dist_array'] = dist_array
        self.data['data_ctr'] = np.array([np.zeros(N)])
        self.data['data_analog'] = np.array([np.zeros(N)])
        self.current_round = 1  # odd means forward scan, even means backward scan

        self.data['ref_analog'] = 0.0

        while True:
            if self._abort == True:
                break

            # do scan
            if self.current_round % 2 == 1:  # odd number: forward scan
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d,
                    ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)

                # start counter and scanning sequence
                if not self.is_at_point_a():
                    print('**ATTENTION** Scanner is NOT at point_a --> No scanning started.')
                elif self.is_at_point_a():
                    print('Scanner is at point_a --> Forward scan (ETA={:.2f}s).'.format(dt * (N + 1)))
                    self.daq_out.run(aotask)
                    self.daq_in_AI.run(aitask)
                    self.daq_in_DI.run(ctrtask)

                    self.current_index = 0
                    self.last_value = 0
                    normalization = dt / .001  # convert to kcounts/sec

                    while True:
                        if self.current_index >= len(dist_array):  # if the maximum time is hit
                            # self._abort = True  # tell the script to abort
                            break
                        if self._abort:
                            break

                        self.progress = self.current_index * 100. / len(dist_array)
                        self.updateProgress.emit(int(self.progress))

                        # print('read_ai')
                        # daq_in_AI needs to be before daq_in_DI
                        raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)

                        # print('read_ctr')
                        raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                        # store analog data
                        if self.current_index == 0:
                            if self.current_round == 1 and self.settings['height'] == 'relative':
                                self.data['ref_analog'] = raw_data_analog[1]
                            # throw the first data point
                            raw_data_analog = raw_data_analog[1:]

                        raw_data_analog = np.array(raw_data_analog) - self.data['ref_analog']
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog'][self.current_round - 1][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                        else:
                            self.data['data_analog'][self.current_round - 1][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                            0:N - self.current_index]

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr'][self.current_round - 1][self.current_index] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                        # print('self.current_index:', self.current_index)

                    # self.daq_out.waitToFinish(aotask)
                    self.daq_out.stop(ctrtask)
                    self.daq_out.stop(aitask)
                    self.daq_out.stop(aotask)

            else:  # even number: backward scan
                # print('scan_pos_1d_flipped',scan_pos_1d_flipped)
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d_flipped,
                    ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)

                # start counter and scanning sequence
                if not self.is_at_point_b():
                    print('**ATTENTION** Scanner is NOT at point_b --> No scanning started.')
                elif self.is_at_point_b():
                    print('Scanner is at point_b --> Backward scan (ETA={:.2f}s).'.format(dt * (N + 1)))
                    self.daq_out.run(aotask)
                    self.daq_in_AI.run(aitask)
                    self.daq_in_DI.run(ctrtask)

                    self.current_index = 0
                    self.last_value = 0
                    normalization = dt / .001  # convert to kcounts/sec

                    while True:
                        if self.current_index >= len(dist_array):  # if the maximum time is hit
                            # self._abort = True  # tell the script to abort
                            break

                        if self._abort:
                            break

                        self.progress = self.current_index * 100. / len(dist_array)
                        self.updateProgress.emit(int(self.progress))


                        # daq_in_AI.read needs to be before daq_in_DI.read so the counter will not read with zero gaps
                        raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                        raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                        # store analog data
                        if self.current_index == 0:
                            # throw the first data point
                            raw_data_analog = raw_data_analog[1:]
                        raw_data_analog = np.array(raw_data_analog) - self.data['ref_analog']
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog'][self.current_round - 1][
                            N - self.current_index - len(raw_data_analog): N - self.current_index] = np.flip(
                                raw_data_analog)
                        else:
                            self.data['data_analog'][self.current_round - 1][0:N - self.current_index] = np.flip(
                                raw_data_analog[0:N - self.current_index])

                        # store counter data
                        for value in raw_data_ctr:

                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr'][self.current_round - 1][N - self.current_index - 1] = new_val
                                self.current_index += 1

                            self.last_value = value


                    # self.daq_out.waitToFinish(aotask)
                    self.daq_out.stop(ctrtask)
                    self.daq_out.stop(aitask)
                    self.daq_out.stop(aotask)

            if self._abort == True:
                break
            else:
                self.current_round += 1
                if self.current_round <= self.settings['num_of_rounds'] or self.settings['num_of_rounds'] == -1:
                    self.data['data_ctr'] = np.concatenate((self.data['data_ctr'], [np.zeros(N)]), axis=0)
                    self.data['data_analog'] = np.concatenate((self.data['data_analog'], [np.zeros(N)]), axis=0)
                else:
                    self._abort = True
                    break

        # Return the scanner to certain positions
        if self.settings['ending_behavior'] == 'return_to_initial':
            self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
            self.scripts['SetScanner'].run()
            print('Scanner returned to the initial position.')
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.scripts['SetScanner'].settings['point']['x'] = 0.0
            self.scripts['SetScanner'].settings['point']['y'] = 0.0
            self.scripts['SetScanner'].run()
            print('Scanner returned to the origin.')
        else:
            print('Scanner is left at the last point.')

        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        print('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))
        # self.log('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))

        # turn off laser
        self.turn_off_laser()

        # close the ANC350
        if self.anc_sample_connected:
            try:
                self.anc_sample.close()
                print('ANC350 v2 (sample) is closed.')
                self.log('ANC350 v2 (sample) is closed.')
            except Exception as e:
                print('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')
                self.log('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')

    def _get_scan_params(self, verbose=True):
        '''
        Returns an array of points to go to in the 1D scan.
        '''
        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        T_tot = dist / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']
        N = int(np.ceil(dist * ptspervolt / 2) * 2)  # number of sample
        if N == 0:
            N =1
        dt = T_tot / N  # time to stay at each point
        scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
        scan_pos_1d_flipped = np.transpose(np.linspace(Vend, Vstart, N, endpoint=True))
        dist_array = np.linspace(0, dist, N, endpoint=True)

        if verbose:
            print('dt={:.3f}s, N={:d}.'.format(dt, N))

        return scan_pos_1d, scan_pos_1d_flipped, dist_array, dt, N, T_tot

    def is_at_point_a(self):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(self.settings['point_a']['x'] - current_position[0]) < 0.01 and np.abs(
                self.settings['point_a']['y'] - current_position[1]) < 0.01:
            return True
        else:
            return False

    def is_at_point_b(self):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(self.settings['point_b']['x'] - current_position[0]) < 0.01 and np.abs(
                self.settings['point_b']['y'] - current_position[1]) < 0.01:
            return True
        else:
            return False

    def _find_refresh_N(self, dt, N, verbose=True):
        refresh_N = 2
        while True:
            if refresh_N >= N:
                refresh_N = int(np.min([N + 1, int(4 / dt)]))
                break
            if ((N + 1 + refresh_N) % refresh_N == 0 or (
                    N + 1 + refresh_N) % refresh_N >= 0.66 * refresh_N) and 2 <= dt * refresh_N <= 5:
                break
            else:
                refresh_N += 1
        if verbose:
            # print('type of refresh N', type(refresh_N))
            print('Plot refresh per {:d} points.'.format(refresh_N))
        return refresh_N

    def _plot(self, axes_list, data=None, title=True):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data
        if title:
            title_name = 'NV-AFM 1D Scan\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                                          self.settings['point_a']['x'], self.settings['point_a']['y'],
                                          self.settings['point_b']['x'], self.settings['point_b']['y'])
        else:
            title_name = None

        if 'data_ctr' in data.keys() and 'dist_array' in data.keys():
            plot_counts_vs_pos_multilines(axes_list[0], data['data_ctr'], data['dist_array'], x_label='Position [V]',
                                          title=title_name)
        if 'data_analog' in data.keys() and 'dist_array' in data.keys():
            plot_counts_vs_pos_multilines(axes_list[1], data['data_analog'], data['dist_array'], x_label='Position [V]',
                                          y_label='Z_out [V]')

    def _update_plot(self, axes_list, title=True, monitor_AFM=True):
        if monitor_AFM and self.anc_sample_connected:
            self._check_AFM()

        if title:
            title_name = 'NV-AFM 1D Scan\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                                          self.settings['point_a']['x'], self.settings['point_a']['y'],
                                          self.settings['point_b']['x'], self.settings['point_b']['y'])
        else:
            title_name = None

        if 'data_ctr' in self.data.keys() and 'dist_array' in self.data.keys():
            plot_counts_vs_pos_multilines(axes_list[0], self.data['data_ctr'], self.data['dist_array'],
                                          x_label='Position [V]', title=title_name)
        if 'data_analog' in self.data.keys() and 'dist_array' in self.data.keys():
            plot_counts_vs_pos_multilines(axes_list[1], self.data['data_analog'], self.data['dist_array'],
                                          x_label='Position [V]', y_label='Z_out [V]')

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])

        return axes_list


class AFM2D(Script):
    """
        AFM 2D scan. Each line will be scanned back and forth.
        - Ziwei Qiu 8/4/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('scan_center',
                  [Parameter('x', 0.5, float, 'x-coordinate [V]'),
                   Parameter('y', 0.5, float, 'y-coordinate [V]')
                   ]),
        Parameter('scan_direction',
                  [Parameter('pt1',
                             [Parameter('x', 0.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('pt2',
                             [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size', 1.0, float, '[V]'),
        Parameter('resolution', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        # Parameter('refresh_per_N_pt', 11, [5, 6, 8, 10, 11, 12, 15,16,20,24,30], 'refresh the data plot per N samples'),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending bahavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'PB': G22BPulseBlaster}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes AFM2D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """
        self._get_scan_extent()
        T_tot = self.settings['scan_size'] / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']
        N = int(np.ceil(self.settings['scan_size'] * ptspervolt / 2) * 2)  # number of samples
        N -= 1
        dt = T_tot / N
        refresh_N = self._find_refresh_N(dt, N)

        if np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:
            self.scripts['SetScanner'].update({'to_do': 'set'})
            self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['SetScanner'].settings['step_size'] = self.settings['resolution']

            # turn on laser and apd_switch
            print('turn on laser and APD readout channel.')
            self.instruments['PB']['instance'].update({'laser': {'status': True}})
            self.instruments['PB']['instance'].update({'apd_switch': {'status': True}})

            # Get initial positions
            self.varinitialpos = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
            # self.log('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

            # Set proper sample rates for all the DAQ channels
            self.sample_rate = N / T_tot
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_DI.settings['digital_input'][
                self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

            # refresh_N = self.settings['refresh_per_N_pt']
            # scanner move to self.pta
            self.scripts['SetScanner'].settings['point']['x'] = self.pta[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.pta[1]
            self.scripts['SetScanner'].run()

            self.line_index = -1
            self.current_round = 0  # odd means forward scan, even means backward scan

            # self.data['scan_pos'] = np.zeros([N,2,N])
            self.data = {'data_ctr_for': np.zeros([N, N]), 'data_analog_for': np.zeros([N, N]),
                         'data_ctr_back': np.zeros([N, N]),
                         'data_analog_back': np.zeros([N, N]),
                         'rotated_data_ctr': ndimage.rotate(np.zeros([N, N]), self.rotation_angle),
                         'rotated_data_analog': ndimage.rotate(np.zeros([N, N]), self.rotation_angle)}
            # self.data['data_ctr_for'] = np.zeros([N,N])
            # self.data['data_analog_for'] = np.zeros([N,N])
            # self.data['data_ctr_back'] = np.zeros([N, N])
            # self.data['data_analog_back'] = np.zeros([N, N])
            # self.data['rotated_data_ctr'] = ndimage.rotate(np.zeros([N,N]), self.rotation_angle)
            # self.data['rotated_data_analog'] = ndimage.rotate(np.zeros([N, N]), self.rotation_angle)
            to_actual_extent = self.settings['scan_size'] / N
            self.data['extent'] = [
                self.scan_center[0] - np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
                self.scan_center[0] + np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
                self.scan_center[1] + np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.,
                self.scan_center[1] - np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.]
            self.data['scan_center'] = self.scan_center
            self.data['scan_size'] = self.settings['scan_size']
            self.data['vector_x'] = self.vector_x

            ETA = 2. * T_tot * N
            print('Start AFM scan (ETA = {:.1f}s):'.format(ETA))
            self.current_index = 0

            tik = time.time()
            while True:
                time.sleep(0.1)
                if self.current_round % 2 == 0:
                    self.line_index += 1
                # print('current line index', self.line_index)

                if self.line_index >= N:  # if the maximum time is hit
                    break
                if self._abort:
                    break

                self.current_round += 1

                if self.current_round % 2 == 1:  # odd, forward scan
                    print('--> Line index: {} / {}. Forward scan. ETA={:.1f}s.'.format(self.line_index, N, T_tot))
                    Vstart = self.pta + self.settings['resolution'] * self.vector_y * self.line_index
                    Vend = self.ptb + self.settings['resolution'] * self.vector_y * self.line_index
                else:  # even, backward scan
                    print('--> Line index: {} / {}. Backward scan. ETA={:.1f}s.'.format(self.line_index, N, T_tot))
                    Vstart = self.ptb + self.settings['resolution'] * self.vector_y * self.line_index
                    Vend = self.pta + self.settings['resolution'] * self.vector_y * self.line_index

                # self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                # self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                # self.scripts['SetScanner'].run()

                scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
                # if self.current_round%2 == 1:
                # self.data['scan_pos'][self.line_index] = scan_pos_1d

                # while not self.is_at_point(Vstart):
                #     if self._abort:
                #         break
                #     print('**ATTENTION** Sample scanner is NOT at Vstart --> Now moving there.')
                #     print('Vstart is', Vstart)
                #     self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                #     self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                #     self.scripts['SetScanner'].run()

                # Setup DAQ
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d, ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)
                self.daq_out.run(aotask)
                self.daq_in_AI.run(aitask)
                self.daq_in_DI.run(ctrtask)

                # Start 1D scan
                self.current_index = 0
                self.last_value = 0
                normalization = dt / .001  # convert to kcounts/sec

                while True:

                    self.progress = (self.current_round * N + self.current_index) * 100. / (2. * N * N)
                    self.updateProgress.emit(int(self.progress))

                    # print('current_index', self.current_index)
                    pt_ETA = refresh_N * self.settings['resolution'] / self.settings['scan_speed']
                    print('     Point index: {} / {}. ETA = {:.1f}s'.format(self.current_index, N, pt_ETA))
                    if self.current_index >= N:  # if the maximum time is hit
                        # self._abort = True  # tell the script to abort
                        break
                    if self._abort:
                        break

                    raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                    raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                    # store analog data
                    if self.current_index == 0:
                        # throw the first data point
                        raw_data_analog = raw_data_analog[1:]

                    if self.current_round % 2 == 1:  # forward scan
                        if self.current_index + len(raw_data_analog) < N:
                            # print(type(self.line_index))
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                        else:
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                            0:N - self.current_index]

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_for'][self.line_index][self.current_index] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                        # print('self.current_index:', self.current_index)
                    else:  # backward scan
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog_back'][self.line_index][
                            N - self.current_index - len(raw_data_analog): N - self.current_index] = np.flip(
                                raw_data_analog)
                        else:
                            self.data['data_analog_back'][self.line_index][0:N - self.current_index] = np.flip(
                                raw_data_analog[0:N - self.current_index])

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_back'][self.line_index][N - self.current_index - 1] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                self.daq_out.stop(ctrtask)
                self.daq_out.stop(aitask)
                self.daq_out.stop(aotask)
            tok = time.time()
            print('Actual scanning time: {:.1f}s.'.format(tok - tik))

            # Return the scanner to certain positions
            if self.settings['ending_behavior'] == 'return_to_initial':
                self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
                self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the initial position.')
            elif self.settings['ending_behavior'] == 'return_to_origin':
                self.scripts['SetScanner'].settings['point']['x'] = 0.0
                self.scripts['SetScanner'].settings['point']['y'] = 0.0
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the origin.')
            else:
                print('Sample scanner is left at the last point.')

            current_position = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))
            self.log('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))

            # turn off laser and apd_switch
            self.instruments['PB']['instance'].update({'laser': {'status': False}})
            self.instruments['PB']['instance'].update({'apd_switch': {'status': False}})
            print('Laser and APD readout are off.')

            self.data['rotated_data_ctr'] = ndimage.rotate(
                0.5 * (self.data['data_ctr_for'] + self.data['data_ctr_back']), self.rotation_angle)
            self.data['rotated_data_analog'] = ndimage.rotate(
                0.5 * (self.data['data_analog_for'] + self.data['data_analog_back']), self.rotation_angle)

        else:
            print('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')
            self.log('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')

    def is_at_point(self, pt, daq_read_error=0.01):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(current_position[0] - pt[0]) < daq_read_error and np.abs(
                current_position[1] - pt[1]) < daq_read_error:
            return True
        else:
            return False

    def _get_scan_extent(self, verbose=True):
        """
        Define 4 points and two unit vectors.
        self.pta - first point to scan, self.ptb- last point of first line,
        self.ptc - first point of last line, self.ptd - last point of last line
        self.vector_x - scanning dirction vector, self.vector_y - orthorgonl direction
        """
        pt1 = np.array([self.settings['scan_direction']['pt1']['x'], self.settings['scan_direction']['pt1']['y']])
        pt2 = np.array([self.settings['scan_direction']['pt2']['x'], self.settings['scan_direction']['pt2']['y']])
        if (pt1 == pt2)[0] == True and (pt1 == pt2)[1] == True:
            print('**ATTENTION** pt1 and pt2 are the same. Please define a valid scan direction. No action.')
            self._abort = True
        vector_1to2 = self._to_unit_vector(pt1, pt2)

        if self.settings['scan_direction']['type'] == 'parallel':
            self.vector_x = vector_1to2
            self.vector_y = self._get_ortho_vector(self.vector_x)
        else:
            self.vector_y = vector_1to2
            self.vector_x = self._get_ortho_vector(-self.vector_y)

        self.rotation_angle = math.acos(np.dot(self.vector_x, np.array([1, 0]))) / np.pi * 180
        if self.vector_x[1] > 0:
            self.rotation_angle = -self.rotation_angle

        if verbose:
            print('Scanning details:')
            print('     vector_x (inner loop):', self.vector_x)
            print('     vector_y (outer loop):', self.vector_y)
            print('     rotation_angle:', self.rotation_angle)

        self.scan_center = np.array([self.settings['scan_center']['x'], self.settings['scan_center']['y']])
        scan_size = self.settings['scan_size']

        # define the 4 points
        self.pta = self.scan_center - self.vector_x * scan_size / 2. - self.vector_y * scan_size / 2.
        self.ptb = self.scan_center + self.vector_x * scan_size / 2. - self.vector_y * scan_size / 2.
        self.ptc = self.scan_center - self.vector_x * scan_size / 2. + self.vector_y * scan_size / 2.
        self.ptd = self.scan_center + self.vector_x * scan_size / 2. + self.vector_y * scan_size / 2.

        if verbose:
            print('     self.pta (first point of first line):', self.pta)
            print('     self.ptb (last point of first line):', self.ptb)
            print('     self.ptc (first point of last line):', self.ptc)
            print('     self.ptd (last point of last line):', self.ptd)

    def _to_unit_vector(self, pt1, pt2):
        unit_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
        return unit_vector

    def _get_ortho_vector(self, vector):
        ortho_vector = np.array([-vector[1], vector[0]])
        return ortho_vector

    def _find_refresh_N(self, dt, N, verbose=True):
        refresh_N = 4
        while True:
            if refresh_N >= N:
                refresh_N = int(np.min([N + 1, int(8 / dt)]))
                break
            if ((N + 1 + refresh_N) % refresh_N == 0 or (
                    N + 1 + refresh_N) % refresh_N >= 0.66 * refresh_N) and 4 <= dt * refresh_N <= 15:
                break
            else:
                refresh_N += 1
        if verbose:
            # print('     type of refresh N', type(refresh_N))
            print('     dt={:.3f}s, N={:d}. Plot refresh per {:d} points.'.format(dt, N, refresh_N))
        return refresh_N

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        # print('_plot')
        if data is None:
            data = self.data
        # print('     _plot')
        plot_fluorescence_new(data['data_ctr_for'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[0], max_counts=self.settings['max_counts_plot'], aspect='equal',
                              min_counts=self.settings['min_counts_plot'],
                              axes_labels=['1', '2'],
                              title='Counts (forward)')
        plot_fluorescence_new(data['data_ctr_back'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[1], max_counts=self.settings['max_counts_plot'], aspect='equal',
                              min_counts=self.settings['min_counts_plot'],
                              axes_labels=['1', '2'],
                              title='Counts (backward)')
        plot_fluorescence_new(data['data_analog_for'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[2], axes_labels=['1', '2'], aspect='equal', title='Height (forward)',
                              colorbar_name='Z [V]')
        plot_fluorescence_new(data['data_analog_back'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[3], axes_labels=['1', '2'], aspect='equal', title='Height (backward)',
                              colorbar_name='Z [V]')
        plot_fluorescence_new(data['rotated_data_ctr'], self.data['extent'],
                              axes_list[4], aspect='equal',
                              max_counts=np.max(data['data_ctr_for']),
                              min_counts=np.min(data['data_ctr_for']),
                              axes_labels=['x', 'y'],
                              title=None)
        plot_fluorescence_new(data['rotated_data_analog'], self.data['extent'],
                              axes_list[5], aspect='equal',
                              max_counts=np.max(data['data_analog_for']),
                              min_counts=np.min(data['data_analog_for']),
                              axes_labels=['x', 'y'], title=None,
                              colorbar_name='Z [V]')

    def _update_plot(self, axes_list):

        update_fluorescence(self.data['data_ctr_for'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                            min_counts=self.settings['min_counts_plot'])
        update_fluorescence(self.data['data_ctr_back'], axes_list[1], max_counts=self.settings['max_counts_plot'],
                            min_counts=self.settings['min_counts_plot'])
        update_fluorescence(self.data['data_analog_for'], axes_list[2])
        update_fluorescence(self.data['data_analog_back'], axes_list[3])

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(221))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(222))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])

        return axes_list


class AFM2D_qm(Script):
    """
        AFM 2D scan. Each line will be scanned back and forth.
        - Ziwei Qiu 8/4/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'print_info', ['print_info', 'execution'],
                  'choose to print information of the scanning settings or do real scanning'),
        Parameter('scan_center',
                  [Parameter('x', 0.5, float, 'x-coordinate [V]'),
                   Parameter('y', 0.5, float, 'y-coordinate [V]')
                   ]),
        Parameter('scan_direction',
                  [Parameter('pt1',
                             [Parameter('x', 0.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('pt2',
                             [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size', 1.0, float, '[V]'),
        Parameter('resolution', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        # Parameter('refresh_per_N_pt', 11, [5, 6, 8, 10, 11, 12, 15,16,20,24,30], 'refresh the data plot per N samples'),
        Parameter('laser_on', True, bool, 'turn on laser during scanning'),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending behavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes AFM2D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.qm_connect()

    def qm_connect(self):
        if self.settings['IP_address'] == 'automatic':
            try:
                self.qmm = QuantumMachinesManager()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def turn_on_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as laser_on:
                with infinite_loop_():
                    play('trig', 'laser', duration=3000)

            self.qm.execute(laser_on)
            print('Laser is on.')

    def turn_off_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as job_stop:
                play('trig', 'laser', duration=10)

            self.qm.execute(job_stop)
            print('Laser is off.')

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """
        self._get_scan_extent()
        T_tot = self.settings['scan_size'] / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']
        N = int(np.ceil(self.settings['scan_size'] * ptspervolt / 2) * 2)  # number of samples
        N -= 1
        dt = T_tot / N
        refresh_N = self._find_refresh_N(dt, N)

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:
            self.scripts['SetScanner'].update({'to_do': 'set'})
            self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['SetScanner'].settings['step_size'] = self.settings['resolution']

            # turn on laser
            if self.settings['laser_on']:
                self.turn_on_laser()

            # Get initial positions
            self.varinitialpos = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
            # self.log('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

            # Set proper sample rates for all the DAQ channels
            self.sample_rate = N / T_tot
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_DI.settings['digital_input'][
                self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

            # refresh_N = self.settings['refresh_per_N_pt']
            # scanner move to self.pta
            self.scripts['SetScanner'].settings['point']['x'] = self.pta[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.pta[1]
            self.scripts['SetScanner'].run()

            self.line_index = -1
            self.current_round = 0  # odd means forward scan, even means backward scan

            self.data = {'data_ctr_for': np.zeros([N, N]), 'data_analog_for': np.zeros([N, N]),
                         'data_ctr_back': np.zeros([N, N]),
                         'data_analog_back': np.zeros([N, N]),
                         'rotated_data_ctr': ndimage.rotate(np.zeros([N, N]), self.rotation_angle),
                         'rotated_data_analog': ndimage.rotate(np.zeros([N, N]), self.rotation_angle)}

            to_actual_extent = self.settings['scan_size'] / N
            self.data['extent'] = [
                self.scan_center[0] - np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
                self.scan_center[0] + np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
                self.scan_center[1] + np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.,
                self.scan_center[1] - np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.]
            self.data['scan_center'] = self.scan_center
            self.data['scan_size'] = self.settings['scan_size']
            self.data['vector_x'] = self.vector_x

            ETA = 2. * T_tot * N
            print('Start AFM scan (ETA = {:.1f}s):'.format(ETA))
            self.current_index = 0

            tik = time.time()
            while True:
                time.sleep(0.1)
                if self.current_round % 2 == 0:
                    self.line_index += 1
                # print('current line index', self.line_index)

                if self.line_index >= N:  # if the maximum time is hit
                    break
                if self._abort:
                    break

                self.current_round += 1

                if self.current_round % 2 == 1:  # odd, forward scan
                    print('--> Line index: {} / {}. Forward scan. ETA={:.1f}s.'.format(self.line_index, N, T_tot))
                    Vstart = self.pta + self.settings['resolution'] * self.vector_y * self.line_index
                    Vend = self.ptb + self.settings['resolution'] * self.vector_y * self.line_index
                else:  # even, backward scan
                    print('--> Line index: {} / {}. Backward scan. ETA={:.1f}s.'.format(self.line_index, N, T_tot))
                    Vstart = self.ptb + self.settings['resolution'] * self.vector_y * self.line_index
                    Vend = self.pta + self.settings['resolution'] * self.vector_y * self.line_index

                # self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                # self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                # self.scripts['SetScanner'].run()

                scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
                # if self.current_round%2 == 1:
                # self.data['scan_pos'][self.line_index] = scan_pos_1d

                # while not self.is_at_point(Vstart):
                #     if self._abort:
                #         break
                #     print('**ATTENTION** Sample scanner is NOT at Vstart --> Now moving there.')
                #     print('Vstart is', Vstart)
                #     self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                #     self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                #     self.scripts['SetScanner'].run()

                # Setup DAQ
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d, ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)
                self.daq_out.run(aotask)
                self.daq_in_AI.run(aitask)
                self.daq_in_DI.run(ctrtask)

                # Start 1D scan
                self.current_index = 0
                self.last_value = 0
                normalization = dt / .001  # convert to kcounts/sec

                while True:

                    self.progress = (self.current_round * N + self.current_index) * 100. / (2. * N * N)
                    self.updateProgress.emit(int(self.progress))

                    # print('current_index', self.current_index)
                    pt_ETA = refresh_N * self.settings['resolution'] / self.settings['scan_speed']
                    print('     Point index: {} / {}. ETA = {:.1f}s'.format(self.current_index, N, pt_ETA))
                    if self.current_index >= N:  # if the maximum time is hit
                        # self._abort = True  # tell the script to abort
                        break
                    if self._abort:
                        break

                    raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                    raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)

                    # store analog data
                    if self.current_index == 0:
                        # throw the first data point
                        raw_data_analog = raw_data_analog[1:]

                    if self.current_round % 2 == 1:  # forward scan
                        if self.current_index + len(raw_data_analog) < N:
                            # print(type(self.line_index))
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                        else:
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                            0:N - self.current_index]

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_for'][self.line_index][self.current_index] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                        # print('self.current_index:', self.current_index)
                    else:  # backward scan
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog_back'][self.line_index][
                            N - self.current_index - len(raw_data_analog): N - self.current_index] = np.flip(
                                raw_data_analog)
                        else:
                            self.data['data_analog_back'][self.line_index][0:N - self.current_index] = np.flip(
                                raw_data_analog[0:N - self.current_index])

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_back'][self.line_index][N - self.current_index - 1] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                self.daq_out.stop(ctrtask)
                self.daq_out.stop(aitask)
                self.daq_out.stop(aotask)
            tok = time.time()
            print('Actual scanning time: {:.1f}s.'.format(tok - tik))

            # Return the scanner to certain positions
            if self.settings['ending_behavior'] == 'return_to_initial':
                self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
                self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the initial position.')
            elif self.settings['ending_behavior'] == 'return_to_origin':
                self.scripts['SetScanner'].settings['point']['x'] = 0.0
                self.scripts['SetScanner'].settings['point']['y'] = 0.0
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the origin.')
            else:
                print('Sample scanner is left at the last point.')

            current_position = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))
            self.log('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))

            # turn off laser
            self.turn_off_laser()

            self.data['rotated_data_ctr'] = ndimage.rotate(
                0.5 * (self.data['data_ctr_for'] + self.data['data_ctr_back']), self.rotation_angle)
            self.data['rotated_data_analog'] = ndimage.rotate(
                0.5 * (self.data['data_analog_for'] + self.data['data_analog_back']), self.rotation_angle)

        else:
            print('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')
            self.log('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')

    def is_at_point(self, pt, daq_read_error=0.01):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(current_position[0] - pt[0]) < daq_read_error and np.abs(
                current_position[1] - pt[1]) < daq_read_error:
            return True
        else:
            return False

    def _get_scan_extent(self, verbose=True):
        """
        Define 4 points and two unit vectors.
        self.pta - first point to scan, self.ptb- last point of first line,
        self.ptc - first point of last line, self.ptd - last point of last line
        self.vector_x - scanning dirction vector, self.vector_y - orthorgonl direction
        """
        pt1 = np.array([self.settings['scan_direction']['pt1']['x'], self.settings['scan_direction']['pt1']['y']])
        pt2 = np.array([self.settings['scan_direction']['pt2']['x'], self.settings['scan_direction']['pt2']['y']])
        if (pt1 == pt2)[0] == True and (pt1 == pt2)[1] == True:
            print('**ATTENTION** pt1 and pt2 are the same. Please define a valid scan direction. No action.')
            self._abort = True
        vector_1to2 = self._to_unit_vector(pt1, pt2)

        if self.settings['scan_direction']['type'] == 'parallel':
            self.vector_x = vector_1to2
            self.vector_y = self._get_ortho_vector(self.vector_x)
        else:
            self.vector_y = vector_1to2
            self.vector_x = self._get_ortho_vector(-self.vector_y)

        self.rotation_angle = math.acos(np.dot(self.vector_x, np.array([1, 0]))) / np.pi * 180
        if self.vector_x[1] > 0:
            self.rotation_angle = -self.rotation_angle

        if verbose:
            print('Scanning details:')
            print('     vector_x (inner loop):', self.vector_x)
            print('     vector_y (outer loop):', self.vector_y)
            print('     rotation_angle:', self.rotation_angle)

        self.scan_center = np.array([self.settings['scan_center']['x'], self.settings['scan_center']['y']])
        scan_size = self.settings['scan_size']

        # define the 4 points
        self.pta = self.scan_center - self.vector_x * scan_size / 2. - self.vector_y * scan_size / 2.
        self.ptb = self.scan_center + self.vector_x * scan_size / 2. - self.vector_y * scan_size / 2.
        self.ptc = self.scan_center - self.vector_x * scan_size / 2. + self.vector_y * scan_size / 2.
        self.ptd = self.scan_center + self.vector_x * scan_size / 2. + self.vector_y * scan_size / 2.

        if verbose:
            print('     self.pta (first point of first line):', self.pta)
            print('     self.ptb (last point of first line):', self.ptb)
            print('     self.ptc (first point of last line):', self.ptc)
            print('     self.ptd (last point of last line):', self.ptd)

    def _to_unit_vector(self, pt1, pt2):
        unit_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
        return unit_vector

    def _get_ortho_vector(self, vector):
        ortho_vector = np.array([-vector[1], vector[0]])
        return ortho_vector

    def _find_refresh_N(self, dt, N, verbose=True):
        refresh_N = 4
        while True:
            if refresh_N >= N:
                refresh_N = int(np.min([N + 1, int(8 / dt)]))
                break
            if ((N + 1 + refresh_N) % refresh_N == 0 or (
                    N + 1 + refresh_N) % refresh_N >= 0.66 * refresh_N) and 4 <= dt * refresh_N <= 15:
                break
            else:
                refresh_N += 1
        if verbose:
            # print('     type of refresh N', type(refresh_N))
            print('     dt={:.3f}s, N={:d}. Plot refresh per {:d} points.'.format(dt, N, refresh_N))
        return refresh_N

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        # print('_plot')
        if data is None:
            data = self.data
        # print('     _plot')
        plot_fluorescence_new(data['data_ctr_for'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[0], max_counts=self.settings['max_counts_plot'], aspect='equal',
                              min_counts=self.settings['min_counts_plot'],
                              axes_labels=['1', '2'],
                              title='Counts (forward)')
        plot_fluorescence_new(data['data_ctr_back'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[1], max_counts=self.settings['max_counts_plot'], aspect='equal',
                              min_counts=self.settings['min_counts_plot'],
                              axes_labels=['1', '2'],
                              title='Counts (backward)')
        plot_fluorescence_new(data['data_analog_for'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[2], axes_labels=['1', '2'], aspect='equal', title='Height (forward)',
                              colorbar_name='Z [V]')
        plot_fluorescence_new(data['data_analog_back'], [0, self.settings['scan_size'], self.settings['scan_size'], 0],
                              axes_list[3], axes_labels=['1', '2'], aspect='equal', title='Height (backward)',
                              colorbar_name='Z [V]')
        plot_fluorescence_new(data['rotated_data_ctr'], self.data['extent'],
                              axes_list[4], aspect='equal',
                              max_counts=np.max(data['data_ctr_for']),
                              min_counts=np.min(data['data_ctr_for']),
                              axes_labels=['x', 'y'],
                              title=None)
        plot_fluorescence_new(data['rotated_data_analog'], self.data['extent'],
                              axes_list[5], aspect='equal',
                              max_counts=np.max(data['data_analog_for']),
                              min_counts=np.min(data['data_analog_for']),
                              axes_labels=['x', 'y'], title=None,
                              colorbar_name='Z [V]')

    def _update_plot(self, axes_list):
        update_fluorescence(self.data['data_ctr_for'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                            min_counts=self.settings['min_counts_plot'])
        update_fluorescence(self.data['data_ctr_back'], axes_list[1], max_counts=self.settings['max_counts_plot'],
                            min_counts=self.settings['min_counts_plot'])
        update_fluorescence(self.data['data_analog_for'], axes_list[2])
        update_fluorescence(self.data['data_analog_back'], axes_list[3])

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(221))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(222))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])

        return axes_list


class AFM2D_qm_v2(Script):
    """
        AFM 2D scan. Each line will be scanned back and forth.
        - Ziwei Qiu 8/4/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'print_info', ['print_info', 'execution', 'qm_reconnection'],
                  'choose to print information of the scanning settings or do real scanning'),
        Parameter('scan_center',
                  [Parameter('x', 0.5, float, 'x-coordinate [V]'),
                   Parameter('y', 0.5, float, 'y-coordinate [V]')
                   ]),
        Parameter('scan_direction',
                  [Parameter('pt1',
                             [Parameter('x', 0.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('pt2',
                             [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size',
                  [Parameter('axis1', 1.0, float, 'inner loop [V]'),
                   Parameter('axis2', 1.0, float, 'outer loop [V]')
                   ]),
        Parameter('resolution',
                  [Parameter('axis1', 0.0001, [0.0001],
                             '[V] inner loop, step size between scanning points. 0.0001V is roughly 0.5nm.'),
                   Parameter('axis2', 0.1, float, '[V] outer loop, step size between lines. 0.1V is roughly 500nm.')
                   ]),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s. Note that the jump between neighboring points is instantaneous. Scan_speed determines the time at each point.'),
        Parameter('height', 'relative', ['relative', 'absolute'],
                  'if relative: the first analog point will be reference.'),
        Parameter('monitor_AFM', False, bool,
                  'monitor the AFM Z_out voltage and retract the tip when the feedback loop is out of control'),
        # Parameter('refresh_per_N_pt', 11, [5, 6, 8, 10, 11, 12, 15,16,20,24,30], 'refresh the data plot per N samples'),
        Parameter('laser_on', True, bool, 'turn on laser during scanning'),
        Parameter('max_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('min_counts_plot', -1, int, 'Rescales colorbar with this as the maximum counts on replotting'),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('x_ai_channel', 'ai3',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring x-scanner voltage'),
                   Parameter('y_ai_channel', 'ai4',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring y-scanner voltage'),
                   Parameter('z_ai_channel', 'ai2',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai14'],
                             'Daq channel used for measuring z-scanner voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for monitoring the z-scanner voltage'),
                   Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1', 'ctr2', 'ctr3'],
                             'Daq channel used for counter')
                   ]),
        Parameter('ending_behavior', 'leave_at_last', ['return_to_initial', 'return_to_origin', 'leave_at_last'],
                  'select the ending behavior'),
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6602': NI6602, 'NI6220': NI6220, 'NI6210': NI6210}
    _SCRIPTS = {'SetScanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        '''
        Initializes AFM2D script for use in gui
        Args:
            instruments: list of instrument objects
            name: name to give to instantiated script object
            settings: dictionary of new settings to pass in to override defaults
            log_function: log function passed from the gui to direct log calls to the gui log
            data_path: path to save data
        '''
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_in_DI = self.instruments['NI6602']['instance']
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_usbAI = self.instruments['NI6210']['instance']
        self.qm_connect()

    def qm_connect(self):
        if self.settings['IP_address'] == 'automatic':
            try:
                self.qmm = QuantumMachinesManager()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def turn_on_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as laser_on:
                with infinite_loop_():
                    play('trig', 'laser', duration=3000)

            self.qm.execute(laser_on)
            print('Laser is on.')

    def turn_off_laser(self):
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            with program() as job_stop:
                play('trig', 'laser', duration=10)

            self.qm.execute(job_stop)
            print('Laser is off.')

    def _setup_anc(self):
        self.anc_sample_connected = False
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_last = z_out[0]

        if self.settings['monitor_AFM']:
            try:
                self.anc_sample = Positioner()
                self.anc_sample_connected = self.anc_sample.is_connected
            except Exception as e:
                print('** ATTENTION in creating ANC_sample **')
                print(e)


    def _check_AFM(self):
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_now = z_out[0]
        if np.abs(self.Z_scanner_now - self.Z_scanner_last) > 0.35:
            try:
                self.anc_sample.dcInEnable(5, False)
                state = self.anc_sample.getDcInEnable(5)

                print('** ATTENTION: AFM Fails!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                print('Z scanner dcInEnable is ' + str(state))
                self.log('Z scanner dcInEnable is ' + str(state))

            except Exception as e:
                print('** ATTENTION: AFM Fails!! **')
                print('** But the tip CANNOT be Retracted!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                self.log('** But the tip CANNOT be Retracted!! **')
            self._abort = True

        else:
            self.Z_scanner_last = self.Z_scanner_now

    def _function(self):
        """
            Executes threaded 1D sample scanning
        """
        # to prevent errors
        self._setup_anc()

        self.data = {}
        self.line_index = 0

        self._get_scan_extent()
        T_tot = self.settings['scan_size']['axis1'] / self.settings['scan_speed']
        ptspervolt = float(1) / self.settings['resolution']['axis1']
        N = int(np.ceil(self.settings['scan_size']['axis1'] * ptspervolt / 2) * 2)  # number of samples per line
        N -= 1
        dt = T_tot / N
        refresh_N = self._find_refresh_N(dt, N)

        # for the outer loop
        ptspervolt2 = float(1) / self.settings['resolution']['axis2']
        N2 = int(np.ceil(self.settings['scan_size']['axis2'] * ptspervolt2 / 2) * 2)  # number of lines
        # N2 -= 1
        self.N2 = N2
        print('     Number of lines: {}. Line resolution: {:.3f}V ({:.1f}nm).'.format(N2, self.settings['resolution'][
            'axis2'], self.settings['resolution']['axis2'] * 5000))

        Total_ETA = 2. * T_tot * N2 / 3600.
        print('The AFM scan will take ETA = {:.3f} hour.'.format(Total_ETA))

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif self.settings['monitor_AFM'] and not self.anc_sample_connected:
            print('** Attention ** ANC350 v2 (sample) is not connected. No scanning started.')
            self._abort = True

        elif self.settings['to_do'] == 'qm_reconnection':
            self.qm_connect()

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:
            self.scripts['SetScanner'].update({'to_do': 'set'})
            self.scripts['SetScanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['SetScanner'].update({'step_size': self.settings['resolution']['axis1']})

            # turn on laser
            if self.settings['laser_on']:
                self.turn_on_laser()

            # Get initial positions
            self.varinitialpos = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))
            # self.log('Initial position: Vx={:.3}V, Vy={:.3}V'.format(self.varinitialpos[0], self.varinitialpos[1]))

            # Set proper sample rates for all the DAQ channels
            self.sample_rate = N / T_tot
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_out.settings['analog_output'][
                self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_DI.settings['digital_input'][
                self.settings['DAQ_channels']['counter_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['x_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['y_ai_channel']]['sample_rate'] = self.sample_rate
            self.daq_in_AI.settings['analog_input'][
                self.settings['DAQ_channels']['z_ai_channel']]['sample_rate'] = self.sample_rate

            # refresh_N = self.settings['refresh_per_N_pt']
            # scanner move to self.pta
            self.scripts['SetScanner'].settings['point']['x'] = self.pta[0]
            self.scripts['SetScanner'].settings['point']['y'] = self.pta[1]
            self.scripts['SetScanner'].run()

            self.line_index = -1
            self.current_round = 0  # odd means forward scan, even means backward scan

            self.data = {'data_ctr_for': np.zeros([N2, N]), 'data_analog_for': np.zeros([N2, N]),
                         'data_ctr_back': np.zeros([N2, N]),
                         'data_analog_back': np.zeros([N2, N])}
            # 'rotated_data_ctr': ndimage.rotate(np.zeros([N2, N]), self.rotation_angle),
            # 'rotated_data_analog': ndimage.rotate(np.zeros([N2, N]), self.rotation_angle)}

            # If the inner and outer loops have different resolutions, the following will be difficult
            # to_actual_extent = self.settings['scan_size'] / N
            # self.data['extent'] = [
            #     self.scan_center[0] - np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
            #     self.scan_center[0] + np.shape(self.data['rotated_data_analog'])[0] * to_actual_extent / 2.,
            #     self.scan_center[1] + np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.,
            #     self.scan_center[1] - np.shape(self.data['rotated_data_analog'])[1] * to_actual_extent / 2.]
            self.data['scan_center'] = self.scan_center
            self.data['scan_size_1'] = self.settings['scan_size']['axis1']
            self.data['scan_size_2'] = self.settings['scan_size']['axis2']
            self.data['vector_x'] = self.vector_x

            print('***************************')
            print('***** AFM Scan Starts *****')
            print('***************************')
            self.current_index = 0

            self.data['ref_analog'] = 0

            tik = time.time()
            while True:
                time.sleep(0.1)
                if self.current_round % 2 == 0:
                    self.line_index += 1

                if self.line_index >= N2:  # if the maximum time is hit
                    break
                if self._abort:
                    break

                self.current_round += 1

                if self.current_round % 2 == 1:  # odd, forward scan
                    print('--> Line index: {} / {}. Forward scan. ETA={:.1f}s.'.format(self.line_index, N2, T_tot))
                    Vstart = self.pta + self.settings['resolution']['axis2'] * self.vector_y * self.line_index
                    Vend = self.ptb + self.settings['resolution']['axis2'] * self.vector_y * self.line_index
                else:  # even, backward scan
                    print('--> Line index: {} / {}. Backward scan. ETA={:.1f}s.'.format(self.line_index, N2, T_tot))
                    Vstart = self.ptb + self.settings['resolution']['axis2'] * self.vector_y * self.line_index
                    Vend = self.pta + self.settings['resolution']['axis2'] * self.vector_y * self.line_index

                self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                self.scripts['SetScanner'].run()

                scan_pos_1d = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))
                # if self.current_round%2 == 1:
                # self.data['scan_pos'][self.line_index] = scan_pos_1d

                # while not self.is_at_point(Vstart):
                #     if self._abort:
                #         break
                #     print('**ATTENTION** Sample scanner is NOT at Vstart --> Now moving there.')
                #     print('Vstart is', Vstart)
                #     self.scripts['SetScanner'].settings['point']['x'] = Vstart[0]
                #     self.scripts['SetScanner'].settings['point']['y'] = Vstart[1]
                #     self.scripts['SetScanner'].run()

                # Setup DAQ
                ctrtask = self.daq_in_DI.setup_counter(self.settings['DAQ_channels']['counter_channel'], refresh_N,
                                                       continuous_acquisition=True)
                aotask = self.daq_out.setup_AO(
                    [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']],
                    scan_pos_1d, ctrtask)
                aitask = self.daq_in_AI.setup_AI(self.settings['DAQ_channels']['z_ai_channel'],
                                                 refresh_N, continuous=True, clk_source=ctrtask)
                self.daq_out.run(aotask)
                self.daq_in_AI.run(aitask)
                self.daq_in_DI.run(ctrtask)

                # Start 1D scan
                self.current_index = 0
                self.last_value = 0
                normalization = dt / .001  # convert to kcounts/sec

                while True:

                    self.progress = (self.current_round * N + self.current_index) * 100. / (2. * N * N2)
                    self.updateProgress.emit(int(self.progress))

                    pt_ETA = refresh_N * self.settings['resolution']['axis1'] / self.settings['scan_speed']
                    print('     Point index: {} / {}. ETA = {:.1f}s'.format(self.current_index, N, pt_ETA))
                    if self.current_index >= N:  # if the maximum time is hit
                        # self._abort = True  # tell the script to abort
                        break
                    if self._abort:
                        break

                    raw_data_analog, num_read_analog = self.daq_in_AI.read(aitask)
                    raw_data_ctr, num_read_ctr = self.daq_in_DI.read(ctrtask)


                    if self.current_index == 0:
                        if self.current_round == 1 and self.settings['height'] == 'relative':
                            self.data['ref_analog'] = raw_data_analog[1]
                        # throw the first data point
                        raw_data_analog = raw_data_analog[1:]

                    raw_data_analog = np.array(raw_data_analog) - self.data['ref_analog']

                    if self.current_round % 2 == 1:  # forward scan
                        # store analog data
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog
                        else:
                            self.data['data_analog_for'][self.line_index][
                            self.current_index:self.current_index + len(raw_data_analog)] = raw_data_analog[
                                                                                            0:N - self.current_index]
                        # store counter data
                        for value in raw_data_ctr:

                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_for'][self.line_index][self.current_index] = new_val
                                self.current_index += 1

                            self.last_value = value

                    else:  # backward scan
                        # store analog data
                        if self.current_index + len(raw_data_analog) < N:
                            self.data['data_analog_back'][self.line_index][
                            N - self.current_index - len(raw_data_analog): N - self.current_index] = np.flip(
                                raw_data_analog)
                        else:
                            self.data['data_analog_back'][self.line_index][0:N - self.current_index] = np.flip(
                                raw_data_analog[0:N - self.current_index])

                        # store counter data
                        for value in raw_data_ctr:
                            # print('self.last_value', self.last_value)
                            if self.current_index >= N:
                                break
                            new_val = ((float(value) - self.last_value) / normalization)
                            if self.last_value != 0:
                                self.data['data_ctr_back'][self.line_index][N - self.current_index - 1] = new_val
                                self.current_index += 1
                            # print('new_val', new_val)
                            self.last_value = value
                self.daq_out.stop(ctrtask)
                self.daq_out.stop(aitask)
                self.daq_out.stop(aotask)
            tok = time.time()
            print('Actual scanning time: {:.1f}s.'.format(tok - tik))

            # Return the scanner to certain positions
            if self.settings['ending_behavior'] == 'return_to_initial':
                self.scripts['SetScanner'].settings['point']['x'] = self.varinitialpos[0]
                self.scripts['SetScanner'].settings['point']['y'] = self.varinitialpos[0]
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the initial position.')
            elif self.settings['ending_behavior'] == 'return_to_origin':
                self.scripts['SetScanner'].settings['point']['x'] = 0.0
                self.scripts['SetScanner'].settings['point']['y'] = 0.0
                self.scripts['SetScanner'].run()
                print('Sample scanner returned to the origin.')
            else:
                print('Sample scanner is left at the last point.')

            current_position = self.daq_in_AI.get_analog_voltages([
                self.settings['DAQ_channels']['x_ai_channel'],
                self.settings['DAQ_channels']['y_ai_channel']]
            )
            print('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))
            self.log('Scanner: Vx={:.4}V, Vy={:.4}V'.format(current_position[0], current_position[1]))

            # turn off laser
            self.turn_off_laser()

            # self.data['rotated_data_ctr'] = ndimage.rotate(self.data['data_ctr_for'], self.rotation_angle)
            # self.data['rotated_data_analog'] = ndimage.rotate(self.data['data_analog_for'], self.rotation_angle)

        else:
            print('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')
            self.log('**ATTENTION**: Scanning voltage exceeds limit [0 8]. No action.')

        if self.anc_sample_connected:
            try:
                self.anc_sample.close()
                print('ANC350 v2 (sample) is closed.')
                self.log('ANC350 v2 (sample) is closed.')
            except Exception as e:
                print('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')
                self.log('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')

    def is_at_point(self, pt, daq_read_error=0.01):
        current_position = self.daq_in_AI.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        if np.abs(current_position[0] - pt[0]) < daq_read_error and np.abs(
                current_position[1] - pt[1]) < daq_read_error:
            return True
        else:
            return False

    def _get_scan_extent(self, verbose=True):
        """
        Define 4 points and two unit vectors.
        self.pta - first point to scan, self.ptb- last point of first line,
        self.ptc - first point of last line, self.ptd - last point of last line
        self.vector_x - scanning direction vector, self.vector_y - orthorgonl direction
        """
        pt1 = np.array([self.settings['scan_direction']['pt1']['x'], self.settings['scan_direction']['pt1']['y']])
        pt2 = np.array([self.settings['scan_direction']['pt2']['x'], self.settings['scan_direction']['pt2']['y']])
        if (pt1 == pt2)[0] == True and (pt1 == pt2)[1] == True:
            print('**ATTENTION** pt1 and pt2 are the same. Please define a valid scan direction. No action.')
            self._abort = True
        vector_1to2 = self._to_unit_vector(pt1, pt2)

        if self.settings['scan_direction']['type'] == 'parallel':
            self.vector_x = vector_1to2
            self.vector_y = self._get_ortho_vector(self.vector_x)
        else:
            self.vector_y = vector_1to2
            self.vector_x = self._get_ortho_vector(-self.vector_y)

        self.rotation_angle = math.acos(np.dot(self.vector_x, np.array([1, 0]))) / np.pi * 180
        if self.vector_x[1] > 0:
            self.rotation_angle = -self.rotation_angle

        if verbose:
            print('Scanning details:')
            print('     vector_x (inner loop):', self.vector_x)
            print('     vector_y (outer loop):', self.vector_y)
            print('     rotation_angle:', self.rotation_angle)

        self.scan_center = np.array([self.settings['scan_center']['x'], self.settings['scan_center']['y']])
        scan_size_1 = self.settings['scan_size']['axis1']  # inner loop
        scan_size_2 = self.settings['scan_size']['axis2']  # outer loop

        # define the 4 points
        self.pta = self.scan_center - self.vector_x * scan_size_1 / 2. - self.vector_y * scan_size_2 / 2.
        self.ptb = self.scan_center + self.vector_x * scan_size_1 / 2. - self.vector_y * scan_size_2 / 2.
        self.ptc = self.scan_center - self.vector_x * scan_size_1 / 2. + self.vector_y * scan_size_2 / 2.
        self.ptd = self.scan_center + self.vector_x * scan_size_1 / 2. + self.vector_y * scan_size_2 / 2.

        if verbose:
            print('     self.pta (first point of first line):', self.pta)
            print('     self.ptb (last point of first line):', self.ptb)
            print('     self.ptc (first point of last line):', self.ptc)
            print('     self.ptd (last point of last line):', self.ptd)

    def _to_unit_vector(self, pt1, pt2):
        unit_vector = (pt2 - pt1) / np.linalg.norm(pt2 - pt1)
        return unit_vector

    def _get_ortho_vector(self, vector):
        ortho_vector = np.array([-vector[1], vector[0]])
        return ortho_vector

    def _find_refresh_N(self, dt, N, verbose=True):
        refresh_N = 4
        while True:
            if refresh_N >= N:
                refresh_N = int(np.min([N + 1, int(8 / dt)]))
                break
            if ((N + 1 + refresh_N) % refresh_N == 0 or (
                    N + 1 + refresh_N) % refresh_N >= 0.66 * refresh_N) and 4 <= dt * refresh_N <= 15:
                break
            else:
                refresh_N += 1
        if verbose:
            # print('     type of refresh N', type(refresh_N))
            print(
                '     In each line, dt={:.3f}s, N={:d}, resolution={:.5f}V ({:.1f}nm). Plot refresh per {:d} points.'.format(
                    dt, N,
                    self.settings[
                        'resolution'][
                        'axis1'], self.settings[
                                      'resolution'][
                                      'axis1'] * 5000,
                    refresh_N))
        return refresh_N

    def plot(self, figure_list):
        super(AFM2D_qm_v2, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data

        if 'data_ctr_for' in data.keys():
            plot_fluorescence_new(data['data_ctr_for'],
                                  [0, self.settings['scan_size']['axis1'], self.settings['scan_size']['axis2'], 0],
                                  axes_list[0], max_counts=self.settings['max_counts_plot'], aspect='equal',
                                  min_counts=self.settings['min_counts_plot'],
                                  axes_labels=['1', '2'],
                                  title='Counts (forward)')
        if 'data_ctr_back' in data.keys():
            plot_fluorescence_new(data['data_ctr_back'],
                                  [0, self.settings['scan_size']['axis1'], self.settings['scan_size']['axis2'], 0],
                                  axes_list[1], max_counts=self.settings['max_counts_plot'], aspect='equal',
                                  min_counts=self.settings['min_counts_plot'],
                                  axes_labels=['1', '2'],
                                  title='Counts (backward)')
        if 'data_analog_for' in data.keys():
            plot_fluorescence_new(data['data_analog_for'],
                                  [0, self.settings['scan_size']['axis1'], self.settings['scan_size']['axis2'], 0],
                                  axes_list[2],
                                  axes_labels=['1', '2'], aspect='equal', title='Height (forward)',
                                  colorbar_name='Z [V]')

        if 'data_analog_back' in data.keys():
            plot_fluorescence_new(data['data_analog_back'],
                                  [0, self.settings['scan_size']['axis1'], self.settings['scan_size']['axis2'], 0],
                                  axes_list[3], axes_labels=['1', '2'], aspect='equal', title='Height (backward)',
                                  colorbar_name='Z [V]')

        try:
            num_of_pts = len(data['data_ctr_for'][-1])
            axes_list[4].plot(np.linspace(0, data['scan_size_1'], num_of_pts), data['data_ctr_for'][-1],
                              label='forward')
            axes_list[4].plot(np.linspace(0, data['scan_size_1'], num_of_pts), data['data_ctr_back'][-1], '--',
                              label='backward')
            axes_list[4].set_xlabel('Scanning position')
            axes_list[4].set_ylabel('counts [kcps]')
            axes_list[4].legend(fontsize=9, loc = 'upper right')

            axes_list[5].plot(np.linspace(0, data['scan_size_1'], num_of_pts), data['data_analog_for'][-1],
                              label='forward')
            axes_list[5].plot(np.linspace(0, data['scan_size_1'], num_of_pts), data['data_analog_back'][-1], '--',
                              label='backward')
            axes_list[5].set_xlabel('Scanning position')
            axes_list[5].set_ylabel('Z_out [V]')
            axes_list[5].legend(fontsize=9, loc = 'upper right')
        except Exception as e:
            print('** ATTENTION **')
            print(e)

        # if 'rotated_data_ctr' in data.keys() and 'rotated_data_analog' in data.keys():
        #     plot_fluorescence_new(data['rotated_data_ctr'], None,
        #                           axes_list[4],
        #                           max_counts=np.max(data['data_ctr_for']),
        #                           min_counts=np.min(data['data_ctr_for']),
        #                           axes_labels=['x', 'y'],
        #                           title=None, axis_off = True)
        #     plot_fluorescence_new(data['rotated_data_analog'], None,
        #                           axes_list[5],
        #                           max_counts=np.max(data['data_analog_for']),
        #                           min_counts=np.min(data['data_analog_for']),
        #                           axes_labels=['x', 'y'], title=None,
        #                           colorbar_name='Z [V]', axis_off = True)

    def _update_plot(self, axes_list, monitor_AFM=True):
        if monitor_AFM and self.anc_sample_connected:
            self._check_AFM()

        try:
            update_fluorescence(self.data['data_ctr_for'], axes_list[0], max_counts=self.settings['max_counts_plot'],
                                min_counts=self.settings['min_counts_plot'])
            update_fluorescence(self.data['data_ctr_back'], axes_list[1], max_counts=self.settings['max_counts_plot'],
                                min_counts=self.settings['min_counts_plot'])
            update_fluorescence(self.data['data_analog_for'], axes_list[2])
            update_fluorescence(self.data['data_analog_back'], axes_list[3])

            axes_list[4].lines[0].set_ydata(self.data['data_ctr_for'][self.line_index])
            axes_list[4].lines[1].set_ydata(self.data['data_ctr_back'][self.line_index])
            axes_list[4].relim()
            axes_list[4].autoscale_view()
            axes_list[4].set_title('Current Line: {} / {}'.format(self.line_index, self.N2), fontsize=9.5)

            axes_list[5].lines[0].set_ydata(self.data['data_analog_for'][self.line_index])
            axes_list[5].lines[1].set_ydata(self.data['data_analog_back'][self.line_index])
            axes_list[5].relim()
            axes_list[5].autoscale_view()
            axes_list[5].set_title('Current Line: {} / {}'.format(self.line_index, self.N2), fontsize=9.5)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
            self._plot(axes_list)

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            axes_list.append(figure_list[0].add_subplot(221))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(222))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


if __name__ == '__main__':
    script, failed, instruments = Script.load_and_append(script_dict={'ObjectiveScan': 'ObjectiveScan'})

    print(script)
    print(failed)
    # print(instruments)
