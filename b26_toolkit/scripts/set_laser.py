"""
    This file is part of b26_toolkit, a pylabcontrol add-on for experiments in Harvard LISE B26.
    Copyright (C) <2016>  Arthur Safira, Jan Gieseler, Aaron Kabcenell

    b26_toolkit is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    b26_toolkit is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with b26_toolkit.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from matplotlib import patches
import matplotlib.collections
import scipy.spatial
import time

# from b26_toolkit.instruments import NI6259, NI9263
# from b26_toolkit.instruments import NI6353
from b26_toolkit.instruments import NI6733, NI6220, NI6210
from pylabcontrol.core import Script, Parameter

class SetObjectiveXY(Script):
    """
    This script sets the objective XY position
    updated by Ziwei Qiu 7/13/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('x', 0.0, float, '[V] x-coordinate (from -10V to 10V)'),
                   Parameter('y', 0.0, float, '[V] y-coordinate (from -10V to 10V)')
                   ]),
        Parameter('patch_size', 0.005, [0.0005, 0.005, 0.05, 0.5], 'size of the red circle'),
        Parameter('DAQ_channels',
            [Parameter('x_ao_channel', 'ao5', ['ao5', 'ao6', 'ao7'], 'Daq channel used for x voltage analog output'),
            Parameter('y_ao_channel', 'ao6', ['ao5', 'ao6', 'ao7'], 'Daq channel used for y voltage analog output')
            ]),
        Parameter('daq_type', 'PCI', ['PCI'], 'Type of daq to use for scan')
    ]

    _INSTRUMENTS = {'NI6733': NI6733}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        if -10<=self.settings['point']['x']<=10 and -10<=self.settings['point']['y']<=10:
            self._setup_daq()
            self.daq_out.set_analog_voltages(
                {self.settings['DAQ_channels']['x_ao_channel']: self.settings['point']['x']})
            self.daq_out.set_analog_voltages(
                {self.settings['DAQ_channels']['y_ao_channel']: self.settings['point']['y']})

            # pt = (self.settings['point']['x'], self.settings['point']['y'])
            #
            # # daq API only accepts either one point and one channel or multiple points and multiple channels
            # pt = np.transpose(np.column_stack((pt[0],pt[1])))
            # pt = (np.repeat(pt, 2, axis=1))
            # # print(pt)
            #
            # self._setup_daq()
            #
            # task = self.daq_out.setup_AO([self.settings['DAQ_channels']['x_ao_channel'],
            #                               self.settings['DAQ_channels']['y_ao_channel']],
            #                              pt)
            #
            # self.daq_out.run(task)
            # self.daq_out.waitToFinish(task)
            # self.daq_out.stop(task)

            self.log('objective set to Vx={:.4}, Vy={:.4}'.format(self.settings['point']['x'], self.settings['point']['y']))


        else:
            self.log('ATTENTION: Voltage exceeds limit [-10 10]. No action.')
            self._abort = True

    def _setup_daq(self):
        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_out = self.instruments['NI6733']['instance']

    # def get_galvo_position(self):
    #     """
    #     reads the current position from the x and y channels and returns it
    #     Returns:
    #
    #     """
    #     if self.settings['daq_type'] == 'PCI':
    #         galvo_position = self.daq_out.get_analog_voltages([
    #             self.settings['DAQ_channels']['x_ao_channel'],
    #             self.settings['DAQ_channels']['y_ao_channel']]
    #         )
    #     elif self.settings['daq_type'] == 'cDAQ':
    #         print("WARNING cDAQ doesn't allow to read values")
    #         galvo_position = []
    #
    #     return galvo_position
    #must be passed figure with galvo plot on first axis

    def plot(self, figure_list):
        try:
            axes_Image = figure_list[0].axes[0]

            # removes patches
            [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]

            patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='r')
            axes_Image.add_patch(patch)
        except Exception as e:
            print('** ATTENTION **')
            print(e)


class SetObjectiveZ(Script):
    """
    This script points the laser to a point
    updated by Ziwei Qiu 7/13/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('z', 0.0, float, '[V] z-coordinate (from -10V to 10V)')]),
        Parameter('patch_size', 0.005, [0.0005, 0.005, 0.05, 0.5], 'size of the red circle'),
        Parameter('DAQ_channels',
            [Parameter('z_ao_channel', 'ao7', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                       'Daq channel used for x voltage analog output')]),
        Parameter('daq_type', 'PCI', ['PCI'], 'Type of daq to use for scan')
    ]

    _INSTRUMENTS = {'NI6733': NI6733}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        # pt = (self.settings['point']['z'], self.settings['point']['dummy'])
        #
        # # daq API only accepts either one point and one channel or multiple points and multiple channels
        # pt = np.transpose(np.column_stack((pt[0],pt[1])))
        # pt = (np.repeat(pt, 2, axis=1))
        # print(pt)
        # task = self.daq_out.setup_AO([self.settings['DAQ_channels']['z_ao_channel'],
        #                               self.settings['DAQ_channels']['dummy_channel']],
        #                              pt)
        #
        # self.daq_out.run(task)
        # self.daq_out.waitToFinish(task)
        # self.daq_out.stop(task)

        if -10 <= self.settings['point']['z'] <= 10:

            self._setup_daq()
            self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['z_ao_channel']: self.settings['point']['z']})


            self.log('Objective set to Vz={:.4}V'.format(self.settings['point']['z']))
        else:
            self.log('ATTENTION: Voltage exceeds limit [-10 10]. No action.')
            self._abort = True

    def _setup_daq(self):
        # defines which daqs contain the input and output based on user selection of daq interface

        self.daq_out = self.instruments['NI6733']['instance']

    # def get_galvo_position(self):
    #     """
    #     reads the current position from the x and y channels and returns it
    #     Returns:
    #
    #     """
    #     if self.settings['daq_type'] == 'PCI':
    #         galvo_position = self.daq_out.get_analog_voltages([
    #             self.settings['DAQ_channels']['z_ao_channel'],
    #             self.settings['DAQ_channels']['dummy_channel']]
    #         )
    #     elif self.settings['daq_type'] == 'cDAQ':
    #         print("WARNING cDAQ doesn't allow to read values")
    #         galvo_position = []
    #
    #     return galvo_position
    #must be passed figure with galvo plot on first axis

    # def plot(self, figure_list):
    #     axes_Image = figure_list[0].axes[0]
    #
    #     # removes patches
    #     [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]
    #
    #     patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='r')
    #     axes_Image.add_patch(patch)


class SetScannerXY(Script):
    """
    This script sets the attocube scanner X and Y voltages
    updated by Ziwei Qiu 7/13/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('x', 0.0, float, 'x-coordinate (from 0V to 8V)'),
                   Parameter('y', 0.0, float, 'y-coordinate (from 0V to 8V)')
                   ]),
        Parameter('DAQ_channels',
            [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                       'Daq channel used for x voltage analog output'),
            Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                      'Daq channel used for y voltage analog output')
            ]),
        Parameter('daq_type', 'PCI', ['PCI'], 'Type of daq to use for scan')
    ]

    _INSTRUMENTS = {'NI6733': NI6733}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)


    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        if 0<=self.settings['point']['x']<=8 and 0<=self.settings['point']['y']<=8:
            self._setup_daq()
            self.daq_out.set_analog_voltages(
                {self.settings['DAQ_channels']['x_ao_channel']: self.settings['point']['x']})
            self.daq_out.set_analog_voltages(
                {self.settings['DAQ_channels']['y_ao_channel']: self.settings['point']['y']})


            self.log('Scanner set to Vx={:.4}, Vy={:.4}'.format(self.settings['point']['x'], self.settings['point']['y']))


        else:
            self.log('ATTENTION: Voltage exceeds limit [0 8]. No action.')
            self._abort = True

    def _setup_daq(self):
        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_out = self.instruments['NI6733']['instance']

    # def get_scanner_position(self):
    #     """
    #     reads the current position from the x and y channels and returns it
    #     Returns:
    #
    #     """
    #     if self.settings['daq_type'] == 'PCI':
    #         scanner_position = self.daq_out.get_analog_voltages([
    #             self.settings['DAQ_channels']['x_ao_channel'],
    #             self.settings['DAQ_channels']['y_ao_channel']]
    #         )
    #     elif self.settings['daq_type'] == 'cDAQ':
    #         print("WARNING cDAQ doesn't allow to read values")
    #         scanner_position = []
    #
    #     return scanner_position
    #must be passed figure with galvo plot on first axis

    # def plot(self, figure_list):
    #     axes_Image = figure_list[0].axes[0]
    #
    #     # removes patches
    #     [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]
    #
    #     patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='r')
    #     axes_Image.add_patch(patch)


class SetScannerXY_gentle(Script):
    """
    This script sets the attocube scanner X and Y voltages
    User can define the step size.
    - Ziwei Qiu 9/9/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('to_do', 'read', ['read', 'set'],
                  'read - read current scanner positions, set - set scanner positions'),
        Parameter('verbose', True, bool, 'print the scanner current status'),
        Parameter('point',
                  [Parameter('x', 0.0, float, 'x-coordinate (from 0V to 8V)'),
                   Parameter('y', 0.0, float, 'y-coordinate (from 0V to 8V)')
                   ]),
        Parameter('DAQ_channels',
            [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                       'Daq channel used for x voltage analog output'),
             Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
                      'Daq channel used for y voltage analog output'),
             Parameter('x_ai_channel', 'ai3',
                       ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11', 'ai12',
                        'ai13', 'ai14', 'ai14'],
                       'Daq channel used for measuring x voltage analog input'),
             Parameter('y_ai_channel', 'ai4',
                       ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11', 'ai12',
                        'ai13', 'ai14', 'ai14'],
                       'Daq channel used for measuring y voltage analog input'),
            ]),
        Parameter('step_size', 0.0001, [0.0001], '[V] step size between scanning points. 0.0001V is roughly 0.5nm.'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1], 'moving speed in [V/s], suggest <0.04'),
        Parameter('daq_type', 'PCI', ['PCI'], 'Type of daq to use for scan')
    ]

    _INSTRUMENTS = {'NI6733': NI6733, 'NI6220': NI6220}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        self._setup_daq()
        current_position = self.get_scanner_position()
        if self.settings['verbose']:
            print('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))
            self.log('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))

        if self.settings['to_do'] == 'set':
            if 0<=self.settings['point']['x']<=8 and 0<=self.settings['point']['y']<=8:
                self.go_to_pos(current_position, [self.settings['point']['x'], self.settings['point']['y']],
                               self.settings['scan_speed'])
                current_position = self.get_scanner_position()
                if self.settings['verbose']:
                    print('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))
                    self.log('Scanner: Vx={:.3}V, Vy={:.3}V'.format(current_position[0], current_position[1]))
            else:
                if self.settings['verbose']:
                    print('**ATTENTION**: Voltage exceeds limit [0 8]. No action.')
                    self.log('**ATTENTION**: Voltage exceeds limit [0 8]. No action.')
                self._abort = True

    def go_to_pos(self, Vstart, Vend, scan_speed):
        Vstart = np.array(Vstart)
        Vend = np.array(Vend)
        dist = np.linalg.norm(Vend-Vstart)
        T_tot = dist / scan_speed
        ptspervolt = float(1) / self.settings['step_size'] # points per volt
        N = int(np.ceil(dist * ptspervolt / 2) * 2)
        dt = T_tot / N # time to count at each point
        # print('dt = ',dt)
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['x_ao_channel']]['sample_rate'] = float(1) / dt
        self.daq_out.settings['analog_output'][
            self.settings['DAQ_channels']['y_ao_channel']]['sample_rate'] = float(1) / dt
        # print(float(1) / dt)

        buffer_move = np.transpose(np.linspace(Vstart, Vend, N, endpoint=True))

        task = self.daq_out.setup_AO(
            [self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], buffer_move)
        if self.settings['verbose']:
            print('**ATTENTION** Scanner is MOVING to ' + str(Vend) + '. Speed={:.3}V/s. ETA={:.1f}s.'.format(scan_speed,T_tot))
        self.daq_out.run(task)
        start_time = time.time()
        while True:
            if time.time() - start_time > T_tot + 1: # if the maximum time is hit
                self._abort = True # tell the script to abort

            if self._abort:
                break
            time.sleep(1)

            self.progress = (time.time() - start_time) * 100. / T_tot
            self.updateProgress.emit(int(self.progress))

        # self.daq_out.waitToFinish(task)

        self.daq_out.stop(task)

    def _setup_daq(self):
        # defines which daqs contain the input and output based on user selection of daq interface
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in = self.instruments['NI6220']['instance']

    def get_scanner_position(self):
        """
        reads the current position from the x and y channels and returns it
        Returns:

        """
        scanner_position = self.daq_in.get_analog_voltages([
            self.settings['DAQ_channels']['x_ai_channel'],
            self.settings['DAQ_channels']['y_ai_channel']]
        )
        # if self.settings['daq_type'] == 'PCI':
        #     scanner_position = self.daq_in.get_analog_voltages([
        #         self.settings['DAQ_channels']['x_ai_channel'],
        #         self.settings['DAQ_channels']['y_ai_channel']]
        #     )
        # elif self.settings['daq_type'] == 'cDAQ':
        #     print("WARNING cDAQ doesn't allow to read values")
        #     scanner_position = []

        return scanner_position
    #must be passed figure with galvo plot on first axis

    # def plot(self, figure_list):
    #     axes_Image = figure_list[0].axes[0]
    #
    #     # removes patches
    #     [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]
    #
    #     patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='r')
    #     axes_Image.add_patch(patch)


class ReadScannerZ(Script):
    """
        This script reads the attocube scanner X and Y voltages using NI6210
        - Ziwei Qiu 12/7/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('DAQ_channels',
                  [Parameter('z_ai_channel', 'ai1',
                             ['ai0', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7', 'ai8', 'ai9', 'ai10', 'ai11',
                              'ai12', 'ai13', 'ai14', 'ai15'],
                             'Daq channel used for measuring z voltage analog input')
                   ])
    ]
    _INSTRUMENTS = {'NI6210': NI6210}
    _SCRIPTS = {}

    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None,
                 data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts,
                        log_function= log_function, data_path = data_path)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        self.daq = self.instruments['NI6210']['instance']
        scanner_position = self.daq.get_analog_voltages([self.settings['DAQ_channels']['z_ai_channel']])
        print('Scanner: Vz={:.5}V'.format(scanner_position[0]))
        self.log('Scanner: Vz={:.5}V'.format(scanner_position[0]))


# The following script causes error because the NI DAQ does not accept an odd number of samples...
# class SetObjectiveZ(Script):
#     """
#         This script sets the objective Z position.
#         updated by Ziwei Qiu 7/13/2020
#     """
#
#     _DEFAULT_SETTINGS = [
#         Parameter('point',
#                   [Parameter('z', 0.0, float, 'z-coordinate')
#                    ]),
#         Parameter('patch_size', 0.005, [0.0005, 0.005, 0.05, 0.5], 'size of the red circle'),
#         Parameter('DAQ_channels',
#             [Parameter('z_ao_channel', 'ao7', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
#                        'Daq channel used for z voltage analog output'),
#              Parameter('dummy_ao_channel', 'ao4', ['ao0', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'ao6', 'ao7'],
#                        'The dummy channel used to make the number of samples be an even number'),
#              Parameter('sample_rates',5000,[1000,2000,5000],'set same sample rates for all three ao channels [Hz]')
#             ]),
#         Parameter('daq_type', 'PCI', ['PCI'], 'Type of daq to use for scan')
#     ]
#
#     _INSTRUMENTS = {'NI6733': NI6733}
#     _SCRIPTS = {}
#
#
#     def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)
#
#
#     def _function(self):
#         """
#         This is the actual function that will be executed. It uses only information that is provided in the settings property
#         will be overwritten in the __init__
#         """
#         pt = (self.settings['point']['z'])
#
#         # daq API only accepts either one point and one channel or multiple points and multiple channels
#
#         pt = np.transpose(np.column_stack((pt, 0.0)))
#         pt = (np.repeat(pt,2, axis=1))
#
#         print(pt)
#
#         self._setup_daq()
#
#         # # force all channels to have the same sample rates (ZQ 1/7/2019 7:37pm)
#         # self.daq_out.settings['analog_output']['ao0']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao1']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao2']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao3']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao4']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao5']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao6']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#         # self.daq_out.settings['analog_output']['ao7']['sample_rate'] = self.settings['DAQ_channels']['sample_rates']
#
#         task = self.daq_out.setup_AO(
#             [self.settings['DAQ_channels']['z_ao_channel'],
#             self.settings['DAQ_channels']['dummy_ao_channel']],
#             pt)
#         # Here is why we need this dummy channel...
#         # https://forums.ni.com/t5/LabVIEW/Why-odd-number-of-samples-in-DAQ-is-not-allowed/td-p/3160652?profile.language=en
#
#
#         self.daq_out.run(task)
#         self.daq_out.waitToFinish(task)
#         self.daq_out.stop(task)
#
#         confocal_position = self.daq_out.get_analog_voltages(
#             [self.settings['DAQ_channels']['z_ao_channel']])
#
#
#         print('laser is set to Vz={:.4}'.format(confocal_position[0]))
#         self.log('laser is set to Vz={:.4}'.format(confocal_position[0]))
#
#         # if self.settings['daq_type'] == 'PCI':
#         #     self.daq_out = self.instruments['NI6259']['instance']
#         # elif self.settings['daq_type'] == 'cDAQ':
#         #     self.daq_out = self.instruments['NI9263']['instance']
#         self.daq_out = self.instruments['NI6733']['instance']
#
#     def _setup_daq(self):
#         # defines which daqs contain the input and output based on user selection of daq interface
#         # if self.settings['daq_type'] == 'PCI':
#         #     self.daq_out = self.instruments['NI6259']['instance']
#         # elif self.settings['daq_type'] == 'cDAQ':
#         #     self.daq_out = self.instruments['NI9263']['instance']
#         self.daq_out = self.instruments['NI6733']['instance']
#
#     def get_galvo_position(self):
#         """
#         reads the current position from the x and y channels and returns it
#         Returns:
#
#         """
#         if self.settings['daq_type'] == 'PCI':
#             galvo_position = self.daq_out.get_analog_voltages([
#
#                 self.settings['DAQ_channels']['z_ao_channel']]
#             )
#         elif self.settings['daq_type'] == 'cDAQ':
#             print("WARNING cDAQ doesn't allow to read values")
#             galvo_position = []
#
#         return galvo_position
#     #must be passed figure with galvo plot on first axis
#
#     # def plot(self, figure_list):
#     #     axes_Image = figure_list[0].axes[0]
#     #
#     #
#     #
#     #     # removes patches
#     #     [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]
#     #
#     #     patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='r')
#     #     axes_Image.add_patch(patch)



if __name__ == '__main__':
    from pylabcontrol.core import Instrument

    # instruments, instruments_failed = Instrument.load_and_append({'daq':  'NI6259'})
    script, failed, instruments = Script.load_and_append(script_dict={'SetObjectiveXY': 'SetObjectiveXY'})
    # script, failed, instruments = Script.load_and_append(script_dict={'SetObjectiveZ': 'SetObjectiveZ'})
    # script, failed, instruments = Script.load_and_append(script_dict={'SetScannerXY': 'SetScannerXY'})

    print(script)
    print(failed)
    # print(instruments)