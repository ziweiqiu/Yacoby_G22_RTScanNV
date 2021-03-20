import numpy as np
import time
from pylabcontrol.core import Script, Parameter
from b26_toolkit.instruments import MagnetX, MagnetY, MagnetZ

from b26_toolkit.scripts.find_nv import FindNV
from b26_toolkit.scripts.set_laser import SetObjectiveXY
from b26_toolkit.scripts.qm_scripts.counter_time_trace import CounterTimeTrace
from b26_toolkit.scripts.qm_scripts.basic import ESRQM_FitGuaranteed, RabiQM
from b26_toolkit.scripts.qm_scripts.echo import PDDQM, PDDSingleTau

from b26_toolkit.scripts.optimize import optimize
from b26_toolkit.plotting.plots_1d import plot_counts_vs_pos, update_counts_vs_pos, plot_magnet_sweep1D_ESR, \
    plot_magnet_sweep1D_Fluor
# from b26_toolkit.plotting.plots_2d import plot_magnet_sweep2D_Fluor, update_magnet_sweep2D_Fluor
from b26_toolkit.plotting.plots_2d import plot_fluorescence_new, update_fluorescence
from collections import deque
import scipy as sp
from b26_toolkit.plotting.plots_2d import plot_fluorescence_pos, update_fluorescence


class MagnetSweep1D(Script):
    """
        MagnetSweep1D sweeps the position of the automatic translation stages, in 1D or 2D scans, and does the following experiments:
        (1) NV fluorescence
        (2) ESR
        (3) Rabi (optional)
        (4) Periodic Dynamical Decoupling - PDD (optional)
        (5) Ramsey (not implemented yet)
        For the optional scripts, set the right parameters in the subscripts.
        Note that only 1D scan is allowed, and only 1 ESR can be done.
        --> Ziwei Qiu 9/18/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('to-do', 'read', ['initialize', 'move', 'sweep', 'read'],
                  'Choose to move to a point, do a magnet sweep or just read the magnet positions'),
        Parameter('servo_initial',
                  [Parameter('initialize', True, bool,
                             'whether or not to intialize the servo position before sweeping? (highly recommended)'),
                   Parameter('Xservo', 0, float, 'initial position of Xservo'),
                   Parameter('Yservo', 0, float, 'initial position of Yservo'),
                   Parameter('Zservo', -5.0, float, 'initial position of Zservo'),
                   Parameter('Xservo_min', -12.5, float, 'minimum allowed position of Xservo'),
                   Parameter('Xservo_max', 12.5, float, 'maximum allowed position of Xservo'),
                   Parameter('Yservo_min', -12.5, float, 'minimum allowed position of Yservo'),
                   Parameter('Yservo_max', 12.5, float, 'maximum allowed position of Yservo'),
                   Parameter('Zservo_min', -11, float, 'minimum allowed position of Zservo'),
                   Parameter('Zservo_max', 4.5, float, 'maximum allowed position of Zservo'),
                   ]),
        Parameter('scan_axis', 'x', ['x', 'y', 'z'],
                  'Choose which axis to perform 1D magnet sweep'),
        Parameter('move_to',
                  [Parameter('x', -6.6, float, 'move to x-coordinate [mm]'),
                   Parameter('y', 0, float, 'move to y-coordinate [mm]'),
                   Parameter('z', -5, float, 'move to z-coordinate [mm]')
                   ]),
        Parameter('sweep_center',
                  [Parameter('x', 0.0, float, 'x-coordinate [mm] of the sweep center'),
                   Parameter('y', 0.0, float, 'y-coordinate [mm] of the sweep center'),
                   Parameter('z', 0.0, float, 'z-coordinate [mm] of the sweep center')
                   ]),
        Parameter('sweep_span',
                  [Parameter('x', 2.0, float, 'x-coordinate [mm]'),
                   Parameter('y', 0.0, float, 'y-coordinate [mm]'),
                   Parameter('z', 0.0, float, 'z-coordinate [mm]')
                   ]),
        Parameter('num_points',
                  [Parameter('x', 15, int, 'number of x points to scan'),
                   Parameter('y', 0, int, 'number of y points to scan'),
                   Parameter('z', 0, int, 'number of z points to scan')
                   ]),
        Parameter('exp_to_do', [Parameter('backward_sweep', False, bool, 'whether to do a backward sweep (not supported now'),
                                Parameter('fluorescence', True, bool, 'measure the NV fluorescence'),
                                Parameter('esr', True, bool, 'measure the ESR of NV'),
                                Parameter('Rabi', True, bool, 'measure Rabi at the ESR resonance frequency'),
                                Parameter('PDD', True, bool, 'measure T2 coherence times using dynamical decoupling'),
                                Parameter('Ramsey', False, bool, 'measure Ramsey (not implemented yet)')]),
        Parameter('exp_settings', [
            Parameter('intensity_wheel_esr', 14.5, float, 'microwave power for ESR scan'),
            Parameter('intensity_wheel_pulse', 16.5, float, 'minimum number of esr averages'),
            Parameter('fluorescence_time_per_pt', 0.4, float, 'time for fluorescence measurement at each point (s)'),
            Parameter('esr_mw_pwr', -35, float, 'microwave power for ESR scan'),
            Parameter('esr_LO_freq', 2.87e9, float, 'LO frequency on the RF generator'),
            Parameter('esr_cntr_freq', 0, float, 'center IF frequency for ESR scan'),
            Parameter('esr_freq_range', 8.5e7, float, 'frequency range for ESR scan (suggest 6e7 - 9e7)'),
            Parameter('esr_avg_min', 2000, int, 'minimum number of esr averages'),
            Parameter('esr_avg_max', 20000, int, 'maximum number of esr averages'),
            Parameter('esr_num_of_pts', 200, int, 'number of frequency points for ESR scan'),

            Parameter('to_plot', 'pdd', ['fwhm', 'contrast', 'pdd'], 'choose to plot fwhm or contrast or pdd in 1D sweep')
        ]),
        Parameter('tracking_settings', [Parameter('track_focus', False, bool,
                                                  'check to use find_nv to track to the NV'),
                                        Parameter('track_focus_every_N', 1, int, 'track every N points'),
                                        Parameter('track_frequency', False, bool,
                                                  'keep track of the frequency and set it to the central frequency of the next ESR scan (recommended)'),
                                        Parameter('track_frequency_every_N', 1, int, 'track every N points')]),
    ]

    _INSTRUMENTS = {'XServo': MagnetX, 'YServo': MagnetY, 'ZServo': MagnetZ}

    _SCRIPTS = {'read_counter': CounterTimeTrace, 'optimize': optimize,
                'esr': ESRQM_FitGuaranteed, 'rabi': RabiQM, 'pdd': PDDQM}

    def __init__(self, scripts, name=None, settings=None, instruments=None, log_function=None, timeout=1000000000,
                 data_path=None):
        """
        Example of a script that makes use of an instrument
        Args:
            instruments: instruments the script will make use of
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """

        # call init of superclass
        Script.__init__(self, name, scripts=scripts, settings=settings, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _get_instr(self):
        """
        Assigns an instrument relevant to the 1D scan axis.
        """
        if self.settings['scan_axis'] == 'x':
            return self.instruments['XServo']['instance']
        elif self.settings['scan_axis'] == 'y':
            return self.instruments['YServo']['instance']
        elif self.settings['scan_axis'] == 'z':
            return self.instruments['ZServo']['instance']

    def _get_instr_2D(self):
        """
        Assigns an instrument relevant to the 2D scan axis.
        """
        if self.settings['scan_axis'] == 'xy':
            return self.instruments['XServo']['instance'], self.instruments['YServo']['instance']
        elif self.settings['scan_axis'] == 'yx':
            return self.instruments['YServo']['instance'], self.instruments['XServo']['instance']
        elif self.settings['scan_axis'] == 'xz':
            return self.instruments['XServo']['instance'], self.instruments['ZServo']['instance']
        elif self.settings['scan_axis'] == 'zx':
            return self.instruments['ZServo']['instance'], self.instruments['XServo']['instance']
        elif self.settings['scan_axis'] == 'yz':
            return self.instruments['YServo']['instance'], self.instruments['ZServo']['instance']
        elif self.settings['scan_axis'] == 'zy':
            return self.instruments['ZServo']['instance'], self.instruments['YServo']['instance']

    def _get_scan_positions(self, verbose=True):
        '''
        Returns an array of points to go to in the 1D scan.
        '''
        if self.settings['scan_axis'] in ['x', 'y', 'z']:
            min_pos = self.settings['sweep_center'][self.settings['scan_axis']] - 0.5 * self.settings['sweep_span'][
                self.settings['scan_axis']]

            max_pos = self.settings['sweep_center'][self.settings['scan_axis']] + 0.5 * self.settings['sweep_span'][
                self.settings['scan_axis']]
            num_points = self.settings['num_points'][self.settings['scan_axis']]
            scan_pos = [np.linspace(min_pos, max_pos, num_points)]
            if verbose:
                print('-------------Scan Settings---------------')
                print('Scan axis:' + self.settings['scan_axis'])
                print('Values for the primary scan are (in mm):' + self.settings['scan_axis'] + ' = ', scan_pos)

            return scan_pos
        else:
            print('NotImplementedError: multiple dimensional scans not yet implemented')
            NotImplementedError('multiple dimensional scans not yet implemented')

    def _get_scan_positions_2D(self, verbose=True):
        if self.settings['scan_axis'] in ['xy', 'yx', 'xz', 'zx', 'yz', 'zy']:

            primary_min_pos = self.settings['sweep_center'][self.settings['scan_axis'][0]] - \
                              self.settings['sweep_span'][
                                  self.settings['scan_axis'][0]] / 2.0

            primary_max_pos = self.settings['sweep_center'][self.settings['scan_axis'][0]] + \
                              self.settings['sweep_span'][
                                  self.settings['scan_axis'][0]] / 2.0
            primary_num_points = self.settings['num_points'][self.settings['scan_axis'][0]]
            secondary_min_pos = self.settings['sweep_center'][self.settings['scan_axis'][1]] - \
                                self.settings['sweep_span'][self.settings['scan_axis'][1]] / 2.0
            secondary_max_pos = self.settings['sweep_center'][self.settings['scan_axis'][1]] + \
                                self.settings['sweep_span'][self.settings['scan_axis'][1]] / 2.0
            secondary_num_points = self.settings['num_points'][self.settings['scan_axis'][1]]

            primary_scan_pos = np.linspace(primary_min_pos, primary_max_pos, num=primary_num_points)
            secondary_scan_pos = np.linspace(secondary_min_pos, secondary_max_pos, num=secondary_num_points)

            if verbose:
                print('-------------Scan Settings---------------')
                print('Scan axis:' + self.settings['scan_axis'])
                print('Values for the primary scan are (in mm): ' + self.settings['scan_axis'][0] + ' = ',
                      primary_scan_pos)
                print('Values for the secondary scan are (in mm): ' + self.settings['scan_axis'][1] + ' = ',
                      secondary_scan_pos)
            return primary_scan_pos, secondary_scan_pos

        else:
            print('NotImplementedError: dimension is not right.')
            # NotImplementedError('Dimension is not right')

    @staticmethod
    def pts_to_extent(pta, ptb):
        """
        Args:
            pta: point a
            ptb: point b
            roi_mode:   mode how to calculate region of interest
                        corner: pta and ptb are diagonal corners of rectangle.
                        center: pta is center and ptb is extend or rectangle

        Returns: extend of region of interest [xVmin, xVmax, yVmax, yVmin]
        """
        xVmin = pta['x'] - float(ptb['x']) / 2.
        xVmax = pta['x'] + float(ptb['x']) / 2.
        yVmin = pta['y'] - float(ptb['y']) / 2.
        yVmax = pta['y'] + float(ptb['y']) / 2.
        zVmin = pta['z'] - float(ptb['z']) / 2.
        zVmax = pta['z'] + float(ptb['z']) / 2.

        return [xVmin, xVmax, yVmin, yVmax, zVmin, zVmax]

    def meas_fluorescence(self, index=-1):

        # update the tag of the read_counter script
        if index >= 0:
            self.scripts['read_counter'].settings['tag'] = 'read_counter_ind' + str(index)

        self.scripts['read_counter'].settings['total_int_time'] = self.settings['exp_settings'][
            'fluorescence_time_per_pt']

        # # set the intensity wheel to the right position (not implemented yet)
        # self.scripts['wheel'].settings['to-do'] = 'move'
        # self.scripts['wheel'].settings['move_to'] = self.settings['exp_settings']['intensity_wheel_esr']
        # self.scripts['wheel'].run()

        # run read_counter or the relevant script to get fluorescence
        print('==> Start measuring FLUORESCENCE...')
        self.scripts['read_counter'].run()
        # time.sleep(self.settings['exp_settings']['fluorescence_time_per_pt'])
        # self.scripts['read_counter'].stop()
        data = np.array(self.scripts['read_counter'].data['counts1']) + np.array(
            self.scripts['read_counter'].data['counts2'])

        print('--> Mean counts = {:0.2f} kcps'.format(np.mean(data)))
        return data

    def do_esr(self, esr_cntr_freq, esr_freq_range, label=None, index=-1, verbose=False):

        # update the tag of the esr script
        if label is not None and index >= 0:
            self.scripts['esr'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['esr'].settings['tag'] = label
        elif index >= 0:
            self.scripts['esr'].settings['tag'] = 'esr_ind' + str(index)

        # # set the intensity wheel to the right position (not implemented yet)
        # self.scripts['wheel'].settings['to-do'] = 'move'
        # self.scripts['wheel'].settings['move_to'] = self.settings['exp_settings']['intensity_wheel_esr']
        # self.scripts['wheel'].run()

        # set the right parameters for the ESR scan
        self.scripts['esr'].settings['to_do'] = 'execution'
        self.scripts['esr'].settings['power_out'] = self.settings['exp_settings']['esr_mw_pwr']
        self.scripts['esr'].settings['esr_avg_min'] = self.settings['exp_settings']['esr_avg_min']
        self.scripts['esr'].settings['esr_avg_max'] = self.settings['exp_settings']['esr_avg_max']
        self.scripts['esr'].settings['mw_frequency'] = self.settings['exp_settings']['esr_LO_freq']
        self.scripts['esr'].settings['freq_points'] = self.settings['exp_settings']['esr_num_of_pts']

        self.scripts['esr'].settings['IF_center'] = float(esr_cntr_freq)
        self.scripts['esr'].settings['IF_range'] = float(esr_freq_range)

        if not self._abort:
            print('==> Start measuring ESR...')
            self.scripts['esr'].run()
            esr_fit_data = self.scripts['esr'].data['fit_params']
            if verbose:
                print('len(esr_fit_data) =  ', esr_fit_data)

            return esr_fit_data

    def do_rabi(self, IF_freq, label=None, index=-1, verbose=False):
        # update the tag of rabi script
        if label is not None and index >= 0:
            self.scripts['rabi'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['rabi'].settings['tag'] = label
        elif index >= 0:
            self.scripts['rabi'].settings['tag'] = 'rabi_ind' + str(index)

        # # set the intensity wheel to the right position (not implemented yet)
        # self.scripts['wheel'].settings['to-do'] = 'move'
        # self.scripts['wheel'].settings['move_to'] = self.settings['exp_settings']['intensity_wheel_pulse']
        # self.scripts['wheel'].run()

        # set the right parameters for the Rabi
        self.scripts['rabi'].settings['to_do'] = 'execution'
        self.scripts['rabi'].settings['mw_pulses']['mw_frequency'] = self.settings['exp_settings']['esr_LO_freq']
        self.scripts['rabi'].settings['mw_pulses']['IF_frequency'] = float(IF_freq)
        if not self._abort:
            print('==> Start measuring Rabi...')
            self.scripts['rabi'].run()

            if 'fits' in self.scripts['rabi'].data.keys() and self.scripts['rabi'].data['fits'] is not None:
                Rabi_Success = True
                pi_time = self.scripts['rabi'].data['pi_time']
                pi_half_time = self.scripts['rabi'].data['pi_half_time']
                three_pi_half_time = self.scripts['rabi'].data['three_pi_half_time']
                mw_power = self.scripts['rabi'].settings['mw_pulses']['mw_power']
            else:
                Rabi_Success = False
                pi_time = -1.0
                pi_half_time = -1.0
                three_pi_half_time = -1.0
                mw_power = self.scripts['rabi'].settings['mw_pulses']['mw_power']

            return Rabi_Success, mw_power, IF_freq, pi_half_time, pi_time, three_pi_half_time

    # def do_ramsey(self, mw_power, mw_freq, pi_half_time, three_pi_half_time, label=None, index=-1, verbose=False):
    #     # update the tag of rabi script
    #     if label is not None and index >= 0:
    #         self.scripts['ramsey'].settings['tag'] = label + '_ind' + str(index)
    #     elif label is not None:
    #         self.scripts['ramsey'].settings['tag'] = label
    #     elif index >= 0:
    #         self.scripts['ramsey'].settings['tag'] = 'rabi_ind' + str(index)
    #
    #     # set the intensity wheel to the right position
    #     self.scripts['wheel'].settings['to-do'] = 'move'
    #     self.scripts['wheel'].settings['move_to'] = self.settings['exp_settings']['intensity_wheel_pulse']
    #     self.scripts['wheel'].run()
    #
    #     # set the right parameters for the Ramsey
    #     self.scripts['ramsey'].settings['mw_pulses']['mw_power'] = float(mw_power)
    #     self.scripts['ramsey'].settings['mw_pulses']['resonant_freq'] = float(mw_freq)
    #     self.scripts['ramsey'].settings['mw_pulses']['pi_half_pulse_time'] = float(pi_half_time)
    #     self.scripts['ramsey'].settings['mw_pulses']['three_pi_half_pulse_time'] = float(three_pi_half_time)
    #
    #     print('==> Start measuring Ramsey...')
    #     self.scripts['ramsey'].run()

    def do_pdd(self, mw_power, IF_freq, pi_half_time, pi_time, three_pi_half_time, label=None, index=-1, verbose=False):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['pdd'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['pdd'].settings['tag'] = label
        elif index >= 0:
            self.scripts['pdd'].settings['tag'] = 'pdd_ind' + str(index)

        # # set the intensity wheel to the right position (not implemented yet)
        # self.scripts['wheel'].settings['to-do'] = 'move'
        # self.scripts['wheel'].settings['move_to'] = self.settings['exp_settings']['intensity_wheel_pulse']
        # self.scripts['wheel'].run()

        # set the right parameters for PDD
        self.scripts['pdd'].settings['to_do'] = 'execution'
        self.scripts['pdd'].settings['mw_pulses']['mw_power'] = float(mw_power)
        self.scripts['pdd'].settings['mw_pulses']['mw_frequency'] = self.settings['exp_settings']['esr_LO_freq']
        self.scripts['pdd'].settings['mw_pulses']['IF_frequency'] = float(IF_freq)
        self.scripts['pdd'].settings['mw_pulses']['pi_pulse_time'] = float(pi_time)
        self.scripts['pdd'].settings['mw_pulses']['pi_half_pulse_time'] = float(pi_half_time)
        self.scripts['pdd'].settings['mw_pulses']['3pi_half_pulse_time'] = float(three_pi_half_time)

        if not self._abort:
            print('==> Start measuring PDD...')
            self.scripts['pdd'].run()

            if 'pdd' not in self.data.keys():
                self.data['pdd'] = np.zeros([len(self.data['positions']), len(self.scripts['pdd'].data['t_vec'])])
                self.data['pdd_tau'] = self.scripts['pdd'].data['t_vec']

            if index >= 0:
                try:
                    self.data['pdd'][index] = self.scripts['pdd'].data['signal_norm']
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)

                    length = np.min([len(self.data['pdd'][index]), len(self.scripts['pdd'].data['signal_norm'])])
                    self.data['pdd'][index][0:length] = self.scripts['pdd'].data['signal_norm'][0:length]

    def _function(self):
        self.settings['exp_to_do']['backward_sweep'] = False # Not spported for now ZQ1/3/2021

        self.flag_image0_update_plot = True
        if self.settings['exp_to_do']['Rabi']:  # if Rabi is selected, set peak # to be 1.
            self.scripts['esr'].settings['fit_constants']['num_of_peaks'] = 1

        def do_tracking(index):

            # update the tag of scripts
            if index >= 0:
                self.scripts['optimize'].settings['tag'] = 'optimize_ind' + str(index)

            # track to the NV if it's time to
            if self.settings['tracking_settings']['track_focus']:
                if index > 0 and index % self.settings['tracking_settings']['track_focus_every_N'] == 0:
                    print('==> Do track_focus now:')
                    print('    ==> optimize starts')
                    self.flag_optimize_plot = True
                    self.scripts['optimize'].run()
            if self.settings['tracking_settings']['track_frequency']:
                if index > 0 and index % self.settings['tracking_settings']['track_frequency_every_N'] == 0:
                    print('==> track_frequency')
                    self.esr_cntr_freq_todo = self.current_esr_cntr_freq

        if self.settings['to-do'] == 'read':
            Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
            self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
            Yservo_position = self.instruments['YServo']['instance'].get_current_position()
            self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
            Xservo_position = self.instruments['XServo']['instance'].get_current_position()
            self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))

        else:
            # initialize the servo positions
            if self.settings['servo_initial']['initialize'] or self.settings['to-do'] == 'initialize':
                print('----------- Servo Initialization -----------')
                print('Xservo:')
                self.instruments['XServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Xservo_min']})
                self.instruments['XServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Xservo_max']})

                if self.settings['servo_initial']['Xservo'] <= self.settings['servo_initial']['Xservo_max'] and \
                        self.settings['servo_initial']['Xservo'] >= self.settings['servo_initial']['Xservo_min']:
                    self.instruments['XServo']['instance'].enable_motion()
                    error_code_x = self.instruments['XServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Xservo'])
                    self.instruments['XServo']['instance'].disable_motion()
                    if error_code_x != 0:
                        print('Xservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Xservo exceeds limit. No action.')

                print('Yservo:')
                self.instruments['YServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Yservo_min']})
                self.instruments['YServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Yservo_max']})
                if self.settings['servo_initial']['Yservo'] <= self.settings['servo_initial']['Yservo_max'] and \
                        self.settings['servo_initial']['Yservo'] >= self.settings['servo_initial']['Yservo_min']:
                    self.instruments['YServo']['instance'].enable_motion()
                    error_code_y = self.instruments['YServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Yservo'])
                    self.instruments['YServo']['instance'].disable_motion()
                    if error_code_y != 0:
                        print('Yservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Yservo exceeds limit. No action.')

                print('Zservo:')
                self.instruments['ZServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Zservo_min']})
                self.instruments['ZServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Zservo_max']})
                if self.settings['servo_initial']['Zservo'] <= self.settings['servo_initial']['Zservo_max'] and \
                        self.settings['servo_initial']['Zservo'] >= self.settings['servo_initial']['Zservo_min']:
                    self.instruments['ZServo']['instance'].enable_motion()
                    error_code_z = self.instruments['ZServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Zservo'])
                    self.instruments['ZServo']['instance'].disable_motion()

                    if error_code_z != 0:
                        print('Zservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Zservo exceeds limit. No action.')

                Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
                self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
                Yservo_position = self.instruments['YServo']['instance'].get_current_position()
                self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
                Xservo_position = self.instruments['XServo']['instance'].get_current_position()
                self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))

                print('>>>> Servo initialization done')

            if self.settings['to-do'] == 'move':
                print('----------- Servo moving along the scanning axis -----------')
                scan_instr = self._get_instr()
                print('     ' + self.settings['scan_axis'][0] + ' Servo is moving to ' + self.settings['scan_axis'][
                    0] + ' = ' + str(self.settings['move_to'][self.settings['scan_axis'][0]]) + 'mm')
                scan_instr.enable_motion()
                servo_move = scan_instr.absolute_move(self.settings['move_to'][self.settings['scan_axis'][0]])
                if servo_move != 0:
                    print(
                        self.settings['scan_axis'] + 'servo fails to move. Experiment stopped. self._abort = True')
                    self._abort = True
                scan_instr.disable_motion()
                Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
                self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
                Yservo_position = self.instruments['YServo']['instance'].get_current_position()
                self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
                Xservo_position = self.instruments['XServo']['instance'].get_current_position()
                self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))

                print('>>>> Servo Moving done')

            elif self.settings['to-do'] == 'sweep':
                # ESR frequency initial settings (for both 1D and 2D)
                self.current_esr_cntr_freq = self.settings['exp_settings']['esr_cntr_freq']
                self.esr_cntr_freq_todo = self.current_esr_cntr_freq

                # 1D scan (forward and backward) (note that 2D scan is disabled for now)
                # get the relevant instrument (servo) for controlling the magnet.
                scan_instr = self._get_instr()
                scan_instr.enable_motion()
                # get positions for the scan.
                scan_pos = self._get_scan_positions()

                # forward and backward sweeps
                self.data = {'counts': deque(), 'counts_r': deque(), 'esr_fo': deque(), 'esr_fo_r': deque(),
                             'esr_wo': deque(), 'esr_wo_r': deque(), 'esr_ctrst': deque(), 'esr_ctrst_r': deque()}

                self.data['positions'] = scan_pos[0]
                self.data['positions_r'] = scan_pos[0][::-1]
                self.positions = {'positions': deque()}

                # loop over scan positions and call the scripts
                index = 0
                # 1D Forward Sweep
                for pos_index in range(0, len(self.data['positions'])):
                    print('len(self.data[]', len(self.data['positions']))
                    if self._abort:
                        break

                    # Set the magnet to be at the initial position
                    new_pos = float(self.data['positions'][pos_index])
                    print('============= Start (index = ' + str(index) + ', Forward) =================')
                    print('----------- Magnet Position: {:0.2f} mm -----------'.format(new_pos))
                    # scan_instr.update({'position': new_pos})  # actually move the instrument to that location.
                    servo_move = scan_instr.absolute_move(new_pos)
                    if servo_move != 0:
                        print(self.settings['scan_axis'][0] + 'servo fails to move. Experiment stopped.')
                        self._abort = True
                        break

                    # If this is not within the safety limits of the instruments, it will not actually move and say so in the log

                    # Do the tracking if it's time to
                    print('---------------- Tracking ----------------')
                    do_tracking(index)

                    # Do the actual measurements
                    print('---------------- Experiment ----------------')

                    ESR_Success = False
                    Rabi_Success = False

                    if self.settings['exp_to_do']['fluorescence']:
                        fluor_data = self.meas_fluorescence(index=index)
                        # add to output structures which will be plotted
                        self.data['counts'].append(np.mean(fluor_data))

                    if self.settings['exp_to_do']['esr']:
                        esr_fit_data = self.do_esr(self.esr_cntr_freq_todo,
                                                   self.settings['exp_settings']['esr_freq_range'], label='esr1',
                                                   index=index)

                        if esr_fit_data is None:
                            print('--> No ESR fitting')
                            # add to output structures which will be plotted
                            self.data['esr_fo'].append(0.0)
                            self.data['esr_wo'].append(0.0)
                            self.data['esr_ctrst'].append(0.0)

                        elif len(esr_fit_data) == 4:
                            if esr_fit_data[3] < 0.5e6:
                                self.data['esr_fo'].append(0.0)
                                self.data['esr_wo'].append(0.0)
                                self.data['esr_ctrst'].append(0.0)
                                print(
                                    '--> Find one ESR peak, but it is not good fit because the width is < 0.5 MHz which is impossible')
                            else:
                                print(
                                    '--> Good, find one ESR peak :). fo = ' + str(
                                        esr_fit_data[2]) + ' Hz, wo = ' + str(
                                        esr_fit_data[3]) + 'Hz')
                                # add to output structures which will be plotted
                                self.data['esr_fo'].append(esr_fit_data[2])
                                self.data['esr_wo'].append(esr_fit_data[3])
                                self.data['esr_ctrst'].append(esr_fit_data[1])
                                # update the ESR center frequency
                                self.current_esr_cntr_freq = esr_fit_data[2] - self.settings['exp_settings'][
                                    'esr_LO_freq']
                                ESR_Success = True

                        elif len(esr_fit_data) == 6:
                            if esr_fit_data[1] < 0.5e6:
                                self.data['esr_fo'].append(0.0)
                                self.data['esr_wo'].append(0.0)
                                self.data['esr_ctrst'].append(0.0)
                                print(
                                    '--> Find two ESRs peak, but it is not good fit because the width is < 0.5 MHz which is impossible')
                            else:
                                print('--> Find two ESR peaks, only record the first peak info')
                                # add to output structures which will be plotted
                                self.data['esr_fo'].append(esr_fit_data[4])
                                self.data['esr_wo'].append(esr_fit_data[1])
                                self.data['esr_ctrst'].append(esr_fit_data[2])
                                # update the ESR center frequency
                                self.current_esr_cntr_freq = esr_fit_data[4] - self.settings['exp_settings'][
                                    'esr_LO_freq']

                    if self.settings['exp_to_do']['Rabi']:

                        if ESR_Success:
                            self.flag_rabi_plot = True
                            print('==> rabi starts')
                            Rabi_Success, mw_power, mw_freq, pi_half_time, pi_time, three_pi_half_time = self.do_rabi(
                                self.current_esr_cntr_freq, label='rabi', index=index)

                        else:
                            Rabi_Success = False
                            print('--> No ESR resonance found. Abort doing Rabi.')

                    if self.settings['exp_to_do']['PDD']:
                        if Rabi_Success:
                            self.flag_pdd_plot = True
                            print('==> pdd starts')
                            self.do_pdd(mw_power, mw_freq, pi_half_time, pi_time, three_pi_half_time, label='pdd',
                                        index=index)

                        else:
                            print('--> No Rabi information found. Abort doing PDD.')

                    # if self.settings['exp_to_do']['Ramsey']:
                    #     if Rabi_Success:
                    #         self.flag_ramsey_plot = True
                    #         print('==> ramsey starts')
                    #         self.do_ramsey(mw_power, mw_freq, pi_half_time, three_pi_half_time,
                    #                        label='ramsey', index=index)
                    #     else:
                    #         print('--> No Rabi information found. Abort doing Ramsey.')

                    print('==================== Finished (Forward) =======================')
                    # record the position
                    self.positions['positions'].append(new_pos)
                    self.progress = index * 100. / (len(self.data['positions']) + len(self.data['positions_r']))
                    self.updateProgress.emit(int(self.progress))

                    index = index + 1
                # the end of the for loop for the forward 1D sweep

                if self.settings['exp_to_do']['backward_sweep']:
                    # 1D Backward Sweep
                    for pos_r_index in range(0, len(self.data['positions_r'])):
                        if self._abort:
                            break

                        # Set the sweeping TDC001 to be at the initial position for the backward sweep
                        new_pos = float(self.data['positions_r'][pos_r_index])
                        print('============ Start (index = ' + str(index) + ', Backward) =================')
                        print('----------- Magnet Position: {:0.2f} mm -----------'.format(new_pos))
                        servo_move = scan_instr.absolute_move(new_pos)
                        if servo_move != 0:
                            print(self.settings['scan_axis'][0] + 'servo fails to move. Experiment stopped.')
                            # If this is not within the safety limits of the instruments, it will not actually move and say so in the log
                            self._abort = True
                            break

                        # Do the tracking if it's time to
                        print('---------------- Tracking ----------------')
                        do_tracking(index)

                        # Do the actual measurements
                        print('---------------- Experiment ----------------')

                        ESR_Success = False
                        Rabi_Success = False

                        if self.settings['exp_to_do']['fluorescence']:
                            fluor_data = self.meas_fluorescence(index=index)
                            # add to output structures which will be plotted
                            self.data['counts_r'].append(np.mean(fluor_data))

                        if self.settings['exp_to_do']['esr']:
                            esr_fit_data = self.do_esr(self.esr_cntr_freq_todo,
                                                       self.settings['exp_settings']['esr_freq_range'],
                                                       label='esr1',
                                                       index=index)

                            if esr_fit_data is None:
                                print('--> No ESR fitting')
                                # add to output structures which will be plotted
                                self.data['esr_fo_r'].append(0.0)
                                self.data['esr_wo_r'].append(0.0)
                                self.data['esr_ctrst_r'].append(0.0)

                            elif len(esr_fit_data) == 4:
                                if esr_fit_data[3] < 0.5e6:
                                    self.data['esr_fo_r'].append(0.0)
                                    self.data['esr_wo_r'].append(0.0)
                                    self.data['esr_ctrst_r'].append(0.0)
                                    print(
                                        '--> Find one ESR peak, but it is not good fit because the width is < 0.5 MHz which is impossible')
                                else:
                                    print(
                                        '--> Good, find one ESR peak :). fo = ' + str(
                                            esr_fit_data[2]) + ' Hz, wo = ' + str(
                                            esr_fit_data[3]) + 'Hz')
                                    # add to output structures which will be plotted
                                    self.data['esr_fo_r'].append(esr_fit_data[2])
                                    self.data['esr_wo_r'].append(esr_fit_data[3])
                                    self.data['esr_ctrst_r'].append(esr_fit_data[1])
                                    # update the ESR center frequency
                                    self.current_esr_cntr_freq = esr_fit_data[2] - self.settings['exp_settings'][
                                        'esr_LO_freq']
                                    ESR_Success = False

                            elif len(esr_fit_data) == 6:
                                if esr_fit_data[1] < 0.5e6:
                                    self.data['esr_fo_r'].append(0.0)
                                    self.data['esr_wo_r'].append(0.0)
                                    self.data['esr_ctrst_r'].append(0.0)
                                    print(
                                        '--> Find two ESRs peak, but it is not good fit because the width is < 0.5 MHz which is impossible')
                                else:
                                    print('--> Find two ESR peaks, only record the first peak info')
                                    # add to output structures which will be plotted
                                    self.data['esr_fo_r'].append(esr_fit_data[4])
                                    self.data['esr_wo_r'].append(esr_fit_data[1])
                                    self.data['esr_ctrst_r'].append(esr_fit_data[2])
                                    # update the ESR center frequency
                                    self.current_esr_cntr_freq = esr_fit_data[4] - self.settings['exp_settings'][
                                        'esr_LO_freq']

                        if self.settings['exp_to_do']['Rabi']:

                            if ESR_Success:
                                self.flag_rabi_plot = True
                                print('==> rabi starts')
                                Rabi_Success, mw_power, mw_freq, pi_half_time, pi_time, three_pi_half_time = self.do_rabi(
                                    self.current_esr_cntr_freq, label='rabi', index=index)

                            else:
                                Rabi_Success = False
                                print('--> No ESR resonance found. Abort doing Rabi.')

                        if self.settings['exp_to_do']['PDD']:
                            if Rabi_Success:
                                self.flag_pdd_plot = True
                                print('==> pdd starts')
                                self.do_pdd(mw_power, mw_freq, pi_half_time, pi_time, three_pi_half_time,
                                            label='pdd', index=index)
                            else:
                                print('--> No Rabi information found. Abort doing PDD.')

                        # if self.settings['exp_to_do']['Ramsey']:
                        #     if Rabi_Success:
                        #         self.flag_ramsey_plot = True
                        #         print('==> ramsey starts')
                        #         self.do_ramsey(mw_power, mw_freq, pi_half_time, three_pi_half_time,
                        #                        label='ramsey', index=index)
                        #     else:
                        #         print('--> No Rabi information found. Abort doing Ramsey.')

                        print('=================== Finished (Backward) =======================')
                        # record the position
                        self.positions['positions'].append(new_pos)
                        self.progress = index * 100. / (len(self.data['positions']) + len(self.data['positions_r']))
                        self.updateProgress.emit(int(self.progress))

                        index = index + 1

                    # the end of the for loop for the backward 1D sweep

                # convert deque object to numpy array or list
                if 'counts' in self.data.keys() is not None:
                    self.data['counts'] = np.asarray(self.data['counts'])
                if 'esr_fo' in self.data.keys() is not None:
                    self.data['esr_fo'] = np.asarray(self.data['esr_fo'])
                if 'esr_wo' in self.data.keys() is not None:
                    self.data['esr_wo'] = np.asarray(self.data['esr_wo'])
                if 'esr_ctrst' in self.data.keys() is not None:
                    self.data['esr_ctrst'] = np.asarray(self.data['esr_ctrst'])
                if 'counts_r' in self.data.keys() is not None:
                    self.data['counts_r'] = np.asarray(self.data['counts_r'])
                if 'esr_fo_r' in self.data.keys() is not None:
                    self.data['esr_fo_r'] = np.asarray(self.data['esr_fo_r'])
                if 'esr_wo_r' in self.data.keys() is not None:
                    self.data['esr_wo_r'] = np.asarray(self.data['esr_wo_r'])
                if 'esr_ctrst_r' in self.data.keys() is not None:
                    self.data['esr_ctrst_r'] = np.asarray(self.data['esr_ctrst_r'])

    def _plot(self, axes_list, data=None):
        # COMMENT_ME

        if data is None:
            data = self.data

        if self.settings['to-do'] == 'sweep':
            print('(1D plot)')
            if data['counts'] is not None:
                lbls1 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]',
                         'counts [kcps]', 'Fluorescence']
                plot_magnet_sweep1D_Fluor([axes_list[0]], data['positions'], np.array(data['counts']),
                                          lbls1, x_r=data['positions_r'], y1_r=np.array(data['counts_r']))

            if self.settings['exp_settings']['to_plot'] == 'contrast':  # to plot contrast
                if data['esr_fo'] is not None and data['esr_ctrst'] is not None:
                    lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]', 'f0 [Hz]',
                             'contrast', 'ESR']
                    plot_magnet_sweep1D_ESR([axes_list[2], axes_list[3]], data['positions'],
                                            np.array(data['esr_fo']),
                                            np.array(data['esr_ctrst']), lbls2, x_r=data['positions_r'],
                                            y1_r=np.array(data['esr_fo_r']),
                                            y2_r=np.array(data['esr_ctrst_r']))
            elif self.settings['exp_settings']['to_plot'] == 'pdd':  # to plot pdd signal
                if self.data['esr_fo'] is not None:
                    lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]',
                             'f0 [Hz]', 'ESR']
                    plot_magnet_sweep1D_Fluor([axes_list[2]], self.data['positions'], np.array(self.data['esr_fo']),
                                              lbls2)
                if 'pdd' in self.data.keys() and 'pdd_tau' in self.data.keys():
                    plot_fluorescence_new(self.data['pdd'],
                                          [self.data['pdd_tau'][0]/1000, self.data['pdd_tau'][-1]/1000, self.data['positions'][-1],
                                           self.data['positions'][0]],
                                          axes_list[3], axes_labels=['Tau (us)', 'magnet position ' + self.settings[
                            'scan_axis'] + ' [mm]'], axes_not_voltage=True, title='PDD', colorbar_name='Contrast')
            else:
                if data['esr_fo'] is not None and data['esr_wo'] is not None:
                    lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]', 'f0 [Hz]',
                             'wo[Hz]', 'ESR']
                    plot_magnet_sweep1D_ESR([axes_list[2], axes_list[3]], data['positions'],
                                            np.array(data['esr_fo']),
                                            np.array(data['esr_wo']), lbls2, x_r=data['positions_r'],
                                            y1_r=np.array(data['esr_fo_r']), y2_r=np.array(data['esr_wo_r']))

    def _update_plot(self, axes_list):

        if self._current_subscript_stage['current_subscript'] is self.scripts['read_counter'] and self.scripts[
            'read_counter'].is_running:
            self.scripts['read_counter']._plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] is self.scripts['esr'] and self.scripts[
            'esr'].is_running:
            self.scripts['esr']._update_plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] is self.scripts['rabi'] and self.scripts[
            'rabi'].is_running:
            if self.flag_rabi_plot:
                self.scripts['rabi']._plot(axes_list)
                self.flag_rabi_plot = False
            else:
                self.scripts['rabi']._update_plot(axes_list)

        elif self._current_subscript_stage['current_subscript'] is self.scripts['pdd'] and self.scripts[
            'pdd'].is_running:
            if self.flag_pdd_plot:
                self.scripts['pdd']._plot(axes_list)
                self.flag_pdd_plot = False
            else:
                self.scripts['pdd']._update_plot(axes_list)

        # elif self._current_subscript_stage['current_subscript'] is self.scripts['ramsey'] and self.scripts[
        #     'ramsey'].is_running:
        #     # print('ramsey is running, update plot')
        #     if self.flag_ramsey_plot:
        #         self.scripts['ramsey']._plot(axes_list)
        #         self.flag_ramsey_plot = False
        #     else:
        #         self.scripts['ramsey']._update_plot(axes_list)

        else:
            if self.settings['to-do'] == 'sweep':
                print('(updating 1D plot)')

                if self.data['counts'] is not None:
                    lbls1 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]',
                             'counts [kcps]', 'Fluorescence']
                    plot_magnet_sweep1D_Fluor([axes_list[0]], self.data['positions'], np.array(self.data['counts']),
                                              lbls1, x_r=self.data['positions_r'], y1_r=np.array(self.data['counts_r']))

                if self.settings['exp_settings']['to_plot'] == 'contrast':  # to plot contrast
                    if self.data['esr_fo'] is not None and self.data['esr_ctrst'] is not None:
                        lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]', 'f0 [Hz]',
                                 'contrast', 'ESR']
                        plot_magnet_sweep1D_ESR([axes_list[2], axes_list[3]], self.data['positions'],
                                                np.array(self.data['esr_fo']),
                                                np.array(self.data['esr_ctrst']), lbls2, x_r=self.data['positions_r'],
                                                y1_r=np.array(self.data['esr_fo_r']),
                                                y2_r=np.array(self.data['esr_ctrst_r']))

                elif self.settings['exp_settings']['to_plot'] == 'pdd': # to plot pdd signal
                    if self.data['esr_fo'] is not None:
                        lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]',
                                 'f0 [Hz]', 'ESR']
                        plot_magnet_sweep1D_Fluor([axes_list[2]], self.data['positions'], np.array(self.data['esr_fo']),
                                                  lbls2)
                    if 'pdd' in self.data.keys():
                        if len(axes_list[3].images) > 0 :
                            update_fluorescence(self.data['pdd'], axes_list[3])
                        else:
                            if 'pdd_tau' in self.data.keys():
                                plot_fluorescence_new(self.data['pdd'],
                                                      [self.data['pdd_tau'][0]/1000, self.data['pdd_tau'][-1]/1000,
                                                       self.data['positions'][-1],
                                                       self.data['positions'][0]],
                                                      axes_list[3],
                                                      axes_labels=['Tau (us)', 'magnet position ' + self.settings[
                                                          'scan_axis'] + ' [mm]'], axes_not_voltage=True, title='PDD',
                                                      colorbar_name='Contrast')
                else:  # to plot width
                    if self.data['esr_fo'] is not None and self.data['esr_wo'] is not None:
                        lbls2 = ['magnet position ' + self.settings['scan_axis'] + ' [mm]', 'f0 [Hz]',
                                 'wo[Hz]', 'ESR']
                        plot_magnet_sweep1D_ESR([axes_list[2], axes_list[3]], self.data['positions'],
                                                np.array(self.data['esr_fo']),
                                                np.array(self.data['esr_wo']), lbls2, x_r=self.data['positions_r'],
                                                y1_r=np.array(self.data['esr_fo_r']),
                                                y2_r=np.array(self.data['esr_wo_r']))

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
            # 3 subplots in total (since ESR2 is not going to be done)
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[3]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])

        return axes_list


class FineTuneMagAngle(Script):
    """
        FineTuneAngle is based on the angle dependence of the echo signal near 90 deg.
        Magnet is swept in 1D and at each position we measure the PDD signal at a fixed tau.
        For PDD script settings, edit it in the subscript.

        -- Ziwei Qiu 10/12/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('to-do', 'read', ['initialize', 'move', 'sweep', 'read'],
                  'Choose to move to a point, do a magnet sweep or just read the magnet positions'),
        Parameter('servo_initial',
                  [Parameter('initialize', True, bool,
                             'whether or not to intialize the servo position before sweeping? (highly recommended)'),
                   Parameter('Xservo', 0, float, 'initial position of Xservo'),
                   Parameter('Yservo', 0, float, 'initial position of Yservo'),
                   Parameter('Zservo', -5.0, float, 'initial position of Zservo'),
                   Parameter('Xservo_min', -12.5, float, 'minimum allowed position of Xservo'),
                   Parameter('Xservo_max', 12.5, float, 'maximum allowed position of Xservo'),
                   Parameter('Yservo_min', -12.5, float, 'minimum allowed position of Yservo'),
                   Parameter('Yservo_max', 12.5, float, 'maximum allowed position of Yservo'),
                   Parameter('Zservo_min', -11, float, 'minimum allowed position of Zservo'),
                   Parameter('Zservo_max', 4.5, float, 'maximum allowed position of Zservo'),
                   ]),
        Parameter('scan_axis', 'x', ['x', 'y', 'z'], 'Choose which axis to perform 1D magnet sweep'),
        Parameter('move_to',
                  [Parameter('x', -6.6, float, 'move to x-coordinate [mm]'),
                   Parameter('y', 0, float, 'move to y-coordinate [mm]'),
                   Parameter('z', -5, float, 'move to z-coordinate [mm]')
                   ]),
        Parameter('sweep_center',
                  [Parameter('x', 0.0, float, 'x-coordinate [mm] of the sweep center'),
                   Parameter('y', 0.0, float, 'y-coordinate [mm] of the sweep center'),
                   Parameter('z', 0.0, float, 'z-coordinate [mm] of the sweep center')
                   ]),
        Parameter('sweep_span', 2.0, float, 'x-coordinate [mm]'),
        Parameter('num_points', 15, int, 'number of x points to scan'),
        Parameter('backward_sweep', False, bool, 'choose whether to do a backward magnet sweep'),

    ]

    _INSTRUMENTS = {'XServo': MagnetX, 'YServo': MagnetY, 'ZServo': MagnetZ}

    _SCRIPTS = {'pdd': PDDSingleTau}

    def __init__(self, scripts, name=None, settings=None, instruments=None, log_function=None, timeout=1000000000,
                 data_path=None):
        """
        Example of a script that makes use of an instrument
        Args:
            instruments: instruments the script will make use of
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """

        # call init of superclass
        Script.__init__(self, name, scripts=scripts, settings=settings, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _get_instr(self):
        """
        Assigns an instrument relevant to the 1D scan axis.
        """
        if self.settings['scan_axis'] == 'x':
            return self.instruments['XServo']['instance']
        elif self.settings['scan_axis'] == 'y':
            return self.instruments['YServo']['instance']
        elif self.settings['scan_axis'] == 'z':
            return self.instruments['ZServo']['instance']

    def _get_scan_positions(self, verbose=True):
        '''
        Returns an array of points to go to in the 1D scan.
        '''
        if self.settings['scan_axis'] in ['x', 'y', 'z']:
            min_pos = self.settings['sweep_center'][self.settings['scan_axis']] - 0.5 * self.settings['sweep_span']
            max_pos = self.settings['sweep_center'][self.settings['scan_axis']] + 0.5 * self.settings['sweep_span']
            num_points = self.settings['num_points']
            scan_pos = [np.linspace(min_pos, max_pos, num_points)]

            if verbose:
                print('-------------Scan Settings---------------')
                print('Scan axis:' + self.settings['scan_axis'])
                print('Values for the primary scan are (in mm):' + self.settings['scan_axis'] + ' = ', scan_pos)

            return scan_pos

        else:
            print('NotImplementedError: multiple dimensional scans not yet implemented')
            NotImplementedError('multiple dimensional scans not yet implemented')

    def do_pdd(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['pdd'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['pdd'].settings['tag'] = label
        elif index >= 0:
            self.scripts['pdd'].settings['tag'] = 'pdd_ind' + str(index)
        else:
            self.scripts['pdd'].settings['tag'] = 'pdd'

        if verbose:
            print('==> Start measuring PDD...')

        self.scripts['pdd'].settings['to_do'] = 'execution'
        self.scripts['pdd'].run()

        pdd_sig = self.scripts['pdd'].data['signal_avg_vec']

        return pdd_sig

    def _function(self):
        if self.settings['to-do'] == 'read':
            Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
            self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
            Yservo_position = self.instruments['YServo']['instance'].get_current_position()
            self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
            Xservo_position = self.instruments['XServo']['instance'].get_current_position()
            self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))
        else:
            # initialize the servo positions
            if self.settings['servo_initial']['initialize'] or self.settings['to-do'] == 'initialize':
                print('----------- Servo Initialization -----------')
                print('Xservo:')
                self.instruments['XServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Xservo_min']})
                self.instruments['XServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Xservo_max']})

                if self.settings['servo_initial']['Xservo'] <= self.settings['servo_initial']['Xservo_max'] and \
                        self.settings['servo_initial']['Xservo'] >= self.settings['servo_initial']['Xservo_min']:
                    self.instruments['XServo']['instance'].enable_motion()
                    error_code_x = self.instruments['XServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Xservo'])
                    self.instruments['XServo']['instance'].disable_motion()
                    if error_code_x != 0:
                        print('Xservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Xservo exceeds limit. No action.')

                print('Yservo:')
                self.instruments['YServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Yservo_min']})
                self.instruments['YServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Yservo_max']})
                if self.settings['servo_initial']['Yservo'] <= self.settings['servo_initial']['Yservo_max'] and \
                        self.settings['servo_initial']['Yservo'] >= self.settings['servo_initial']['Yservo_min']:
                    self.instruments['YServo']['instance'].enable_motion()
                    error_code_y = self.instruments['YServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Yservo'])
                    self.instruments['YServo']['instance'].disable_motion()
                    if error_code_y != 0:
                        print('Yservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Yservo exceeds limit. No action.')

                print('Zservo:')
                self.instruments['ZServo']['instance'].update(
                    {'lower_limit': self.settings['servo_initial']['Zservo_min']})
                self.instruments['ZServo']['instance'].update(
                    {'upper_limit': self.settings['servo_initial']['Zservo_max']})
                if self.settings['servo_initial']['Zservo'] <= self.settings['servo_initial']['Zservo_max'] and \
                        self.settings['servo_initial']['Zservo'] >= self.settings['servo_initial']['Zservo_min']:
                    self.instruments['ZServo']['instance'].enable_motion()
                    error_code_z = self.instruments['ZServo']['instance'].absolute_move(
                        target=self.settings['servo_initial']['Zservo'])
                    self.instruments['ZServo']['instance'].disable_motion()

                    if error_code_z != 0:
                        print('Zservo fails to initialize. Experiment stopped. self._abort = True')
                        self._abort = True
                else:
                    print('Zservo exceeds limit. No action.')

                Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
                self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
                Yservo_position = self.instruments['YServo']['instance'].get_current_position()
                self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
                Xservo_position = self.instruments['XServo']['instance'].get_current_position()
                self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))

                print('>>>> Servo initialization done')

            if self.settings['to-do'] == 'move':
                print('----------- Servo moving along the scanning axis -----------')
                scan_instr = self._get_instr()
                print('     ' + self.settings['scan_axis'][0] + ' Servo is moving to ' + self.settings['scan_axis'][
                    0] + ' = ' + str(self.settings['move_to'][self.settings['scan_axis'][0]]) + 'mm')
                scan_instr.enable_motion()
                servo_move = scan_instr.absolute_move(self.settings['move_to'][self.settings['scan_axis'][0]])
                if servo_move != 0:
                    print(
                        self.settings['scan_axis'] + 'servo fails to move. Experiment stopped. self._abort = True')
                    self._abort = True
                scan_instr.disable_motion()
                Zservo_position = self.instruments['ZServo']['instance'].get_current_position()
                self.log('MagnetZ position: {:.3f} mm'.format(Zservo_position))
                Yservo_position = self.instruments['YServo']['instance'].get_current_position()
                self.log('MagnetY position: {:.3f} mm'.format(Yservo_position))
                Xservo_position = self.instruments['XServo']['instance'].get_current_position()
                self.log('MagnetX position: {:.3f} mm'.format(Xservo_position))

                print('>>>> Servo Moving done')

            elif self.settings['to-do'] == 'sweep':
                # 1D scan (forward and backward) (note that 2D scan is disabled for now)
                # get the relevant instrument (servo) for controlling the magnet.
                scan_instr = self._get_instr()
                scan_instr.enable_motion()
                # get positions for the scan.
                scan_pos = self._get_scan_positions()

                # forward and backward sweeps
                self.data = {'pdd_sig1': deque(), 'pdd_sig2': deque(), 'pdd_norm': deque(), 'pdd_sig1_r': deque(),
                             'pdd_sig2_r': deque(), 'pdd_norm_r': deque()}
                self.data['positions'] = scan_pos[0]
                self.data['positions_r'] = scan_pos[0][::-1]
                self.positions = {'positions': deque()}

                if self.settings['backward_sweep']:
                    tot_pos_pts = len(self.data['positions']) + len(self.data['positions_r'])
                else:
                    tot_pos_pts = len(self.data['positions'])

                # loop over scan positions and call the scripts
                index = 0
                # 1D Forward Sweep
                for pos_index in range(0, len(self.data['positions'])):
                    # print('len(self.data[]', len(self.data['positions']))
                    if self._abort:
                        break

                    # Set the magnet to be at the initial position
                    new_pos = float(self.data['positions'][pos_index])
                    print('============= Start (index = ' + str(index) + ', Forward) =================')
                    print('----------- Magnet Position: {:0.2f} mm -----------'.format(new_pos))
                    # scan_instr.update({'position': new_pos})  # actually move the instrument to that location.
                    servo_move = scan_instr.absolute_move(new_pos)
                    if servo_move != 0:
                        print(self.settings['scan_axis'][0] + 'servo fails to move. Experiment stopped.')
                        self._abort = True
                        break

                    # Do the actual measurements
                    print('---------------- Experiment ----------------')

                    # do PDD
                    self.flag_pdd_plot = True
                    print('==> pdd starts')
                    pdd_sig = self.do_pdd(label='pdd', index=index)
                    self.data['pdd_sig1'].append(pdd_sig[0])
                    self.data['pdd_sig2'].append(pdd_sig[1])
                    self.data['pdd_norm'].append(2 * (pdd_sig[1] - pdd_sig[0]) / (pdd_sig[0] + pdd_sig[1]))

                    print('==================== Finished (Forward) =======================')

                    # record the position
                    self.positions['positions'].append(new_pos)
                    self.progress = index * 100. / tot_pos_pts
                    self.updateProgress.emit(int(self.progress))
                    time.sleep(0.2)
                    index = index + 1
                    # the end of the for loop for the forward 1D sweep

                if self.settings['backward_sweep']:

                    # 1D Backward Sweep
                    for pos_r_index in range(0, len(self.data['positions_r'])):
                        if self._abort:
                            break

                        # Set the sweeping TDC001 to be at the initial position for the backward sweep
                        new_pos = float(self.data['positions_r'][pos_r_index])
                        print('============ Start (index = ' + str(index) + ', Backward) =================')
                        print('----------- Magnet Position: {:0.2f} mm -----------'.format(new_pos))
                        servo_move = scan_instr.absolute_move(new_pos)
                        if servo_move != 0:
                            print(self.settings['scan_axis'][0] + 'servo fails to move. Experiment stopped.')
                            # If this is not within the safety limits of the instruments, it will not actually move and say so in the log
                            self._abort = True
                            break

                        pass
                        # Do the actual measurements
                        print('---------------- Experiment ----------------')

                        # do PDD
                        self.flag_pdd_plot = True
                        print('==> pdd starts')
                        pdd_sig = self.do_pdd(label='pdd', index=index)
                        self.data['pdd_sig1_r'].append(pdd_sig[0])
                        self.data['pdd_sig2_r'].append(pdd_sig[1])
                        self.data['pdd_norm_r'].append(2 * (pdd_sig[1] - pdd_sig[0]) / (pdd_sig[0] + pdd_sig[1]))

                        print('=================== Finished (Backward) =======================')
                        # record the position
                        self.positions['positions'].append(new_pos)
                        self.progress = index * 100. / tot_pos_pts
                        self.updateProgress.emit(int(self.progress))
                        time.sleep(0.2)
                        index = index + 1

                        # the end of the for loop for the backward 1D sweep

            # convert deque object to numpy array or list
            if 'pdd_sig1' in self.data.keys() is not None:
                self.data['pdd_sig1'] = np.asarray(self.data['pdd_sig1'])
            if 'pdd_sig2' in self.data.keys() is not None:
                self.data['pdd_sig2'] = np.asarray(self.data['pdd_sig2'])
            if 'pdd_norm' in self.data.keys() is not None:
                self.data['pdd_norm'] = np.asarray(self.data['pdd_norm'])
            if 'pdd_sig1_r' in self.data.keys() is not None:
                self.data['pdd_sig1_r'] = np.asarray(self.data['pdd_sig1_r'])
            if 'pdd_sig2_r' in self.data.keys() is not None:
                self.data['pdd_sig2_r'] = np.asarray(self.data['pdd_sig2_r'])
            if 'pdd_norm_r' in self.data.keys() is not None:
                self.data['pdd_norm_r'] = np.asarray(self.data['pdd_norm_r'])

    def _plot(self, axes_list, data=None):
        # COMMENT_ME

        if data is None:
            data = self.data

        if self._current_subscript_stage['current_subscript'] == self.scripts['pdd'] and self.scripts[
            'pdd'].is_running:
            self.scripts['pdd']._plot([axes_list[2]], title=False)

        if self.settings['to-do'] == 'sweep':
            # print('(1D plot)')
            if len(data['pdd_sig1']) > 0:
                axes_list[0].clear()
                axes_list[1].clear()
                axes_list[0].plot(data['positions'][0:len(data['pdd_sig1'])], data['pdd_sig1'],
                                  label="+x (forward)")
                axes_list[0].plot(data['positions'][0:len(data['pdd_sig2'])], data['pdd_sig2'],
                                  label="-x (forward)")
                axes_list[0].axhline(y=1.0, color='r', ls='--', lw=1.3)
                axes_list[1].plot(data['positions'][0:len(data['pdd_norm'])], data['pdd_norm'],
                                  label="norm sig (forward)")
                axes_list[1].axhline(y=0.0, color='r', ls='--', lw=1.3)

            if len(data['pdd_sig1_r']) > 0:

                axes_list[0].plot(data['positions_r'][0:len(data['pdd_sig1_r'])], data['pdd_sig1_r'],
                                  label="+x (backward)")
                axes_list[0].plot(data['positions_r'][0:len(data['pdd_sig2_r'])], data['pdd_sig2_r'],
                                  label="-x (backward)")
                axes_list[1].plot(data['positions_r'][0:len(data['pdd_norm_r'])], data['pdd_norm_r'],
                                  label="norm sig (backward)")

            axes_list[0].legend(loc='upper right')
            axes_list[1].legend(loc='upper right')
            axes_list[1].set_xlabel('magnet position [mm]')
            axes_list[0].set_ylabel('Contrast')
            axes_list[1].set_ylabel('Normalized contrast')
            axes_list[0].set_title(
                'Periodic Dynamical Decoupling\n{:s} {:d} block(s), tau_total = {:2.2f}us\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                    self.scripts['pdd'].settings['decoupling_seq']['type'],
                    self.scripts['pdd'].settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.scripts['pdd'].settings['tau'], self.scripts['pdd'].settings['rep_num'],
                    self.scripts['pdd'].settings['mw_pulses']['pi_half_pulse_time'],
                    self.scripts['pdd'].settings['mw_pulses']['pi_pulse_time'],
                    self.scripts['pdd'].settings['mw_pulses']['3pi_half_pulse_time'],
                    self.scripts['pdd'].settings['mw_pulses']['mw_power'],
                    self.scripts['pdd'].settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.scripts['pdd'].settings['mw_pulses']['IF_amp'],
                    self.scripts['pdd'].settings['mw_pulses']['IF_frequency'] * 1e-6))

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['pdd'] and self.scripts[
            'pdd'].is_running:
            self.scripts['pdd']._update_plot([axes_list[2]], title=False)
        else:
            self._plot(axes_list)
        # if self.settings['to-do'] == 'sweep':
        #     print('(updating 1D plot)')
        #     if len(self.data['pdd_sig1']) > 0:
        #         axes_list[0].lines[0].set_xdata(self.data['positions'][0:len(self.data['pdd_sig1'])])
        #         axes_list[0].lines[0].set_ydata(self.data['pdd_sig1'])
        #         axes_list[0].lines[1].set_xdata(self.data['positions'][0:len(self.data['pdd_sig2'])])
        #         axes_list[0].lines[1].set_ydata(self.data['pdd_sig2'])
        #         axes_list[1].lines[0].set_xdata(self.data['positions'][0:len(self.data['pdd_norm'])])
        #         axes_list[1].lines[0].set_ydata(self.data['pdd_norm'])
        #     if len(self.data['pdd_sig1_r']) > 0:
        #         axes_list[0].lines[2].set_xdata(self.data['positions_r'][0:len(self.data['pdd_sig1_r'])])
        #         axes_list[0].lines[2].set_ydata(self.data['pdd_sig1_r'])
        #         axes_list[0].lines[3].set_xdata(self.data['positions_r'][0:len(self.data['pdd_sig2_r'])])
        #         axes_list[0].lines[3].set_ydata(self.data['pdd_sig2_r'])
        #         axes_list[1].lines[1].set_xdata(self.data['positions_r'][0:len(self.data['pdd_norm_r'])])
        #         axes_list[1].lines[1].set_ydata(self.data['pdd_norm_r'])
        #
        #     axes_list[0].relim()
        #     axes_list[0].autoscale_view()
        #     axes_list[1].relim()
        #     axes_list[1].autoscale_view()

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
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[2]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])

        return axes_list