import numpy as np
import time
import math

from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.set_laser import SetScannerXY_gentle
from b26_toolkit.scripts.galvo_scan.confocal_scan_G22B import AFM1D_qm
from b26_toolkit.scripts.qm_scripts.basic import RabiQM
from b26_toolkit.scripts.qm_scripts.afm_sync_sensing import PDDSyncAFM
from b26_toolkit.scripts.optimize import optimize
from b26_toolkit.instruments import NI6733, NI6220, NI6210, SGS100ARFSource, Agilent33120A, YokogawaGS200


class ScanningDCSensing(Script):
    """
        Perform DC sensing measurements (based on PDDSyncAFM) and scan along a 1D line.
        One can also choose to do Rabi and fluorescence optimize every several points.
        - Ziwei Qiu 2/19/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'initial x-coordinate [V]'),
                   Parameter('y', 0, float, 'initial y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 4, float, 'last x-coordinate [V]'),
                   Parameter('y', 0, float, 'last y-coordinate [V]')
                   ]),
        Parameter('num_points', 100, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.02, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('dc_voltage', [
            Parameter('level', 0.0, float, 'define the DC voltage [V] to infinity load'),
            Parameter('source', 'afg', ['afg', 'yokogawa', 'keithley', 'None'],
                      'choose the voltage source. afg limit: +/-10V, yokogawa: +/-30V, keithley: not implemented. If None, then no voltage source is connected.')
        ]),
        Parameter('do_afm1d',
                  [Parameter('before', False, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int,
                             'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool,
                             'whether to scan back to the original point after the experiment is done'),
                   ]),
        Parameter('do_rabi',
                  [Parameter('on', False, bool, 'choose whether to do Rabi calibration'),
                   Parameter('frequency', 5, int, 'do the Rabi once per N points'),
                   Parameter('start_power', -10, float, 'choose the MW power at the first point'),
                   Parameter('end_power', -20, float, 'choose the MW power at the last point'),
                   ]),
        Parameter('do_optimize',
                  [Parameter('on', False, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 50, int, 'do the optimization once per N points'),
                   Parameter('range_xy', 0.25, float, 'the range [V] in which the xy focus can fluctuate reasonably'),
                   Parameter('range_z', 0.5, float, 'the range [V] in which the z focus can fluctuate reasonably')
                   ]),
        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao5', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao6', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('z_ao_channel', 'ao7', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for z voltage analog output'),
                   Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'],
                             'Daq channel used for measuring obj x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'],
                             'Daq channel used for measuring obj y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'],
                             'Daq channel used for measuring obj z voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for measuring scanner z voltage')
                   ])
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6220': NI6220, 'NI6210': NI6210, 'afg': Agilent33120A,
                    'yokogawa': YokogawaGS200}
    _SCRIPTS = {'trig_pdd': PDDSyncAFM, 'rabi': RabiQM, 'optimize': optimize, 'set_scanner': SetScannerXY_gentle,
                'afm1d': AFM1D_qm, 'afm1d_before_after': AFM1D_qm}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_usbAI = self.instruments['NI6210']['instance']

    def _get_scan_array(self):
        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        N = self.settings['num_points']
        scan_pos_1d = np.linspace(Vstart, Vend, N, endpoint=True)
        dist_array = np.linspace(0, dist, N, endpoint=True)

        return scan_pos_1d, dist_array

    def setup_afm(self, afm_start, afm_end):
        self.scripts['afm1d'].settings['scan_speed'] = self.settings['scan_speed']
        self.scripts['afm1d'].settings['resolution'] = 0.0001
        self.scripts['afm1d'].settings['num_of_rounds'] = 1
        self.scripts['afm1d'].settings['ending_behavior'] = 'leave_at_last'
        self.scripts['afm1d'].settings['height'] = 'absolute'
        self.scripts['afm1d'].settings['point_a']['x'] = afm_start[0]
        self.scripts['afm1d'].settings['point_a']['y'] = afm_start[1]
        self.scripts['afm1d'].settings['point_b']['x'] = afm_end[0]
        self.scripts['afm1d'].settings['point_b']['y'] = afm_end[1]

    def do_dc_sensing(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['trig_pdd'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['trig_pdd'].settings['tag'] = label
        elif index >= 0:
            self.scripts['trig_pdd'].settings['tag'] = 'trig_pdd' + '_ind' + str(index)
        else:
            self.scripts['trig_pdd'].settings['tag'] = 'trig_pdd'

        if self.settings['do_optimize']['on'] and index >= 0 and index % self.settings['do_optimize'][
            'frequency'] == 0 and not self._abort:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_ind' + str(index)

            pos_before = self.daq_in_AI.get_analog_voltages(
                [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
                 self.settings['DAQ_channels']['z_ai_channel']])

            if verbose:
                print('==> Executing optimize ...')
            self.scripts['optimize'].run()

            pos_after = self.daq_in_AI.get_analog_voltages(
                [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
                 self.settings['DAQ_channels']['z_ai_channel']])

            if np.abs(pos_after[0] - pos_before[0]) > self.settings['do_optimize']['range_xy'] or np.abs(
                    pos_after[1] - pos_before[1]) > self.settings['do_optimize']['range_xy'] or np.abs(
                pos_after[2] - pos_before[2]) > self.settings['do_optimize']['range_z']:
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['x_ao_channel']: pos_before[0]})
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['y_ao_channel']: pos_before[1]})
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['z_ao_channel']: pos_before[2]})
                print('---> Optimize does not make sense. Set laser back.')

        if self.settings['do_rabi']['on'] and index >= 0 and index % self.settings['do_rabi'][
            'frequency'] == 0 and not self._abort:
            self.scripts['rabi'].settings['tag'] = 'rabi' + '_ind' + str(index)
            self.scripts['rabi'].settings['mw_pulses']['phase'] = 0
            mw_power = self.settings['do_rabi']['start_power'] + (
                    self.settings['do_rabi']['end_power'] - self.settings['do_rabi']['start_power']) * index / \
                       self.settings['num_points']
            self.scripts['rabi'].settings['mw_pulses']['mw_power'] = mw_power

            try:
                if verbose:
                    print('==> Executing Rabi ...')
                self.scripts['rabi'].run()
                pi_time = self.scripts['rabi'].data['pi_time']
                pi_half_time = self.scripts['rabi'].data['pi_half_time']
                three_pi_half_time = self.scripts['rabi'].data['three_pi_half_time']
                rabi_freq = np.array([self.scripts['rabi'].data['rabi_freq']])
                rabi_power = np.array([self.scripts['rabi'].settings['mw_pulses']['mw_power']])

            except Exception as e:
                print('** ATTENTION in Rabi **')
                print(e)
            else:
                if pi_time >= 16 and pi_half_time >= 16 and three_pi_half_time >= 16:
                    self.scripts['trig_pdd'].settings['mw_pulses']['mw_frequency'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['mw_frequency'])
                    self.scripts['trig_pdd'].settings['mw_pulses']['mw_power'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['mw_power'])
                    self.scripts['trig_pdd'].settings['mw_pulses']['IF_frequency'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['IF_frequency'])
                    self.scripts['trig_pdd'].settings['mw_pulses']['IF_amp'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['IF_amp'])
                    self.scripts['trig_pdd'].settings['mw_pulses']['pi_pulse_time'] = float(pi_time)
                    self.scripts['trig_pdd'].settings['mw_pulses']['pi_half_pulse_time'] = float(pi_half_time)
                    self.scripts['trig_pdd'].settings['mw_pulses']['3pi_half_pulse_time'] = float(
                        three_pi_half_time)

                # save the rabi data
                if 'rabi_freq' in self.data.keys():
                    self.data['rabi_freq'] = np.concatenate((self.data['rabi_freq'], rabi_freq))
                else:
                    self.data['rabi_freq'] = rabi_freq

                if 'rabi_power' in self.data.keys():
                    self.data['rabi_power'] = np.concatenate((self.data['rabi_power'], rabi_power))
                else:
                    self.data['rabi_power'] = rabi_power

                if 'rabi_dist_array' in self.data.keys():
                    self.data['rabi_dist_array'] = np.concatenate(
                        (self.data['rabi_dist_array'], np.array([self.dist_array[index]])))
                else:
                    self.data['rabi_dist_array'] = np.array([self.dist_array[index]])

        if not self._abort:
            if verbose:
                print('==> Executing DC sensing based on AFM-synced PDD...')

            try:
                self.scripts['trig_pdd'].run()
            except Exception as e:
                print('** ATTENTION in Running trig_pdd **')
                print(e)
            else:
                dc_sig = self.scripts['trig_pdd'].data['signal_avg_vec']
                norm_sig = np.array([[2 * (dc_sig[1] - dc_sig[0]) / (dc_sig[1] + dc_sig[0]),
                                      2 * (dc_sig[3] - dc_sig[2]) / (dc_sig[3] + dc_sig[2])]])
                dc_sig = np.array([[dc_sig[0], dc_sig[1], dc_sig[2], dc_sig[3]]])
                exact_tau = np.array([self.scripts['trig_pdd'].data['tau']])
                coherence = np.sqrt(norm_sig[0, 0] ** 2 + norm_sig[0, 1] ** 2)
                phase = np.arccos(norm_sig[0, 0] / coherence) * np.sign(norm_sig[0, 1])
                coherence = np.array([coherence])
                phase = np.array([phase])

                return dc_sig, norm_sig, exact_tau, coherence, phase

        else:
            return None, None, None, None, None

    def do_afm1d_before(self, verbose=True):
        if verbose:
            print('===> AFM1D starts before the sensing experiments.')
        self.scripts['afm1d_before_after'].settings['tag'] = 'afm1d_before'
        self.scripts['afm1d_before_after'].settings['scan_speed'] = self.settings['scan_speed']
        self.scripts['afm1d_before_after'].settings['resolution'] = 0.0001
        self.scripts['afm1d_before_after'].settings['num_of_rounds'] = int(
            self.settings['do_afm1d']['num_of_cycles'] * 2)
        self.scripts['afm1d_before_after'].settings['ending_behavior'] = 'leave_at_last'
        self.scripts['afm1d_before_after'].settings['height'] = 'relative'
        self.scripts['afm1d_before_after'].settings['point_a']['x'] = self.settings['point_a']['x']
        self.scripts['afm1d_before_after'].settings['point_a']['y'] = self.settings['point_a']['y']
        self.scripts['afm1d_before_after'].settings['point_b']['x'] = self.settings['point_b']['x']
        self.scripts['afm1d_before_after'].settings['point_b']['y'] = self.settings['point_b']['y']
        self.scripts['afm1d_before_after'].run()

    def do_afm1d_after(self, verbose=True):
        if verbose:
            print('===> AFM1D starts after the sensing experiments.')
        self.scripts['afm1d_before_after'].settings['tag'] = 'afm1d_after'
        self.scripts['afm1d_before_after'].settings['scan_speed'] = self.settings['scan_speed']
        self.scripts['afm1d_before_after'].settings['resolution'] = 0.0001
        self.scripts['afm1d_before_after'].settings['num_of_rounds'] = 1
        self.scripts['afm1d_before_after'].settings['ending_behavior'] = 'leave_at_last'
        self.scripts['afm1d_before_after'].settings['height'] = 'relative'
        self.scripts['afm1d_before_after'].settings['point_a']['x'] = self.settings['point_b']['x']
        self.scripts['afm1d_before_after'].settings['point_a']['y'] = self.settings['point_b']['y']
        self.scripts['afm1d_before_after'].settings['point_b']['x'] = self.settings['point_a']['x']
        self.scripts['afm1d_before_after'].settings['point_b']['y'] = self.settings['point_a']['y']
        self.scripts['afm1d_before_after'].run()

    def set_up_voltage_source(self):
        if self.settings['dc_voltage']['source'] == 'afg':
            self.vol_source = self.instruments['afg']['instance']
            self.vol_source.update({'output_load': 'INFinity'})
            self.vol_source.update({'wave_shape': 'DC'})
            self.vol_source.update({'burst_mod': False})
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            self.vol_source = self.instruments['yokogawa']['instance']
            self.vol_source.update({'source': 'VOLT'})
            self.vol_source.update({'level': 0.0})
            self.vol_source.update({'enable_output': True})
        elif self.settings['dc_voltage']['source'] == 'keithley':
            self.vol_source = None
        else:
            self.vol_source = None

    def set_voltage(self, vol):
        steps = np.linspace(0, 1, 21)
        if self.settings['dc_voltage']['source'] == 'afg':
            print('Ramping up the voltage on AFG...')
            for step in steps:
                self.vol_source.update({'offset': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'offset': vol})
            print('Voltage is set on AFG.')
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            print('Ramping up the voltage on Yokogawa...')
            for step in steps:
                self.vol_source.update({'level': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'level': vol})
            print('Voltage is set on Yokogawa.')
        # elif self.settings['dc_voltage']['source'] == 'keithley':
        #     pass

    def close_voltage_source(self, vol):
        steps = np.linspace(1, 0, 21)
        if self.settings['dc_voltage']['source'] == 'afg':
            print('Ramping down the voltage on AFG...')
            for step in steps:
                self.vol_source.update({'offset': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'offset': 0.0})
            print('Voltage is 0 on AFG.')
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            print('Ramping down the voltage on Yokogawa...')
            for step in steps:
                self.vol_source.update({'level': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'level': 0.0})
            print('Voltage is 0 on Yokogawa.')
            self.vol_source.update({'enable_output': False})
        # elif self.settings['dc_voltage']['source'] == 'keithley':
        #     pass

    def _function(self):
        self.set_up_voltage_source()
        self.set_voltage(float(self.settings['dc_voltage']['level']))

        scan_pos_1d, dist_array = self._get_scan_array()
        self.dist_array = dist_array

        # Move to the initial scanning position
        self.scripts['set_scanner'].update({'to_do': 'set'})
        self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['set_scanner'].update({'step_size': 0.0001})

        self.scripts['afm1d'].settings['scan_speed'] = self.settings['scan_speed']
        self.scripts['afm1d'].settings['resolution'] = 0.0001
        self.scripts['afm1d'].settings['num_of_rounds'] = 1
        self.scripts['afm1d'].settings['ending_behavior'] = 'leave_at_last'
        self.scripts['afm1d'].settings['height'] = 'absolute'

        self.scripts['set_scanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['set_scanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['set_scanner'].run()

        # Initialize the data structure
        self.data = {'scan_pos_1d': scan_pos_1d, 'dc_dist_array': np.array([]),
                     'afm_dist_array': np.array([]), 'afm_ctr': np.array([]), 'afm_analog': np.array([])}

        if self.settings['do_afm1d']['before']:
            self.do_afm1d_before()

        self.scripts['trig_pdd'].settings['to_do'] = 'execution'
        if not self._abort:
            # The first point
            try:
                dc_sig, norm_sig, exact_tau, coherence, phase = self.do_dc_sensing(index=0)
            except Exception as e:
                print('** ATTENTION do_dc_sensing (index=0) **')
                print(e)
            else:
                self.data['dc_data'] = dc_sig
                self.data['norm_data'] = norm_sig
                self.data['exact_tau'] = exact_tau
                self.data['coherence'] = coherence
                self.data['phase'] = phase
                self.data['dc_dist_array'] = np.concatenate((self.data['dc_dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                break

            afm_start = scan_pos_1d[i]
            afm_end = scan_pos_1d[i + 1]
            self.setup_afm(afm_start, afm_end)

            try:
                self.scripts['afm1d'].run()
                afm_dist_array = self.scripts['afm1d'].data['dist_array'] + dist_array[i]
                self.data['afm_dist_array'] = np.concatenate((self.data['afm_dist_array'], afm_dist_array))
                afm_ctr = self.scripts['afm1d'].data['data_ctr'][0]
                self.data['afm_ctr'] = np.concatenate((self.data['afm_ctr'], afm_ctr))
                afm_analog = self.scripts['afm1d'].data['data_analog'][0]
                self.data['afm_analog'] = np.concatenate((self.data['afm_analog'], afm_analog))
            except Exception as e:
                print('** ATTENTION **')
                print(e)

            try:
                dc_sig, norm_sig, exact_tau, coherence, phase = self.do_dc_sensing(index=i + 1)
                self.data['dc_data'] = np.concatenate((self.data['dc_data'], dc_sig), axis=0)
                self.data['norm_data'] = np.concatenate((self.data['norm_data'], norm_sig), axis=0)
                self.data['exact_tau'] = np.concatenate((self.data['exact_tau'], exact_tau))
                self.data['coherence'] = np.concatenate((self.data['coherence'], coherence))
                self.data['phase'] = np.concatenate((self.data['phase'], phase))
                self.data['dc_dist_array'] = np.concatenate((self.data['dc_dist_array'], np.array([dist_array[i + 1]])))
            except Exception as e:
                print('** ATTENTION in do_dc_sensing **')
                print(e)

        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

        self.close_voltage_source(float(self.settings['dc_voltage']['level']))

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()
        axes_list[4].clear()
        axes_list[5].clear()
        axes_list[6].clear()
        axes_list[7].clear()

        try:
            if 'afm_dist_array' in data.keys() and len(data['afm_dist_array']) > 0:
                axes_list[0].plot(data['afm_dist_array'], data['afm_ctr'])
                axes_list[1].plot(data['afm_dist_array'], data['afm_analog'])
            else:
                axes_list[0].plot(np.zeros([10]), np.zeros([10]))
                axes_list[1].plot(np.zeros([10]), np.zeros([10]))

            if 'dc_data' in data.keys() and len(data['dc_dist_array']) > 0:
                axes_list[2].plot(data['dc_dist_array'], data['norm_data'][:, 0], label="norm sig1")
                axes_list[2].plot(data['dc_dist_array'], data['norm_data'][:, 1], label="norm sig2")
                axes_list[2].plot(data['dc_dist_array'], data['coherence'], '--', label="coherence")
                axes_list[2].axhline(y=0.0, color='r', ls='--', lw=1.3)
                axes_list[7].plot(data['dc_dist_array'], data['phase'])
                # axes_list[7].axhline(y=3.14159, color='r', ls='--', lw=1.1)
                # axes_list[7].axhline(y=-3.14159, color='r', ls='--', lw=1.1)
            else:
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig1")
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig2")
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), '--', label="coherence")
                axes_list[2].axhline(y=0.0, color='r', ls='--', lw=1.3)
                axes_list[7].plot(np.zeros([10]), np.zeros([10]))
                # axes_list[7].axhline(y=3.14159, color='r', ls='--', lw=1.1)
                # axes_list[7].axhline(y=-3.14159, color='r', ls='--', lw=1.1)

            if 'rabi_freq' in data.keys() and 'rabi_power' in data.keys() and 'rabi_dist_array' in data.keys():
                axes_list[4].plot(data['rabi_dist_array'], data['rabi_freq'], label="freq")
                axes_list[5].plot(data['rabi_dist_array'], data['rabi_power'], '--', label="pwr")
            else:
                axes_list[4].plot(np.zeros([10]), np.zeros([10]), label="freq")
                axes_list[5].plot(np.zeros([10]), np.zeros([10]), '--', label="pwr")

            axes_list[0].set_ylabel('Counts [kcps]')
            axes_list[1].set_ylabel('Z_out [V]')
            axes_list[2].set_ylabel('Contrast')
            axes_list[4].set_ylabel('Rabi freq. [MHz]')
            axes_list[5].set_ylabel('RF power [dB]')
            axes_list[7].set_ylabel('Phase [rad]')

            axes_list[0].xaxis.set_ticklabels([])
            axes_list[1].xaxis.set_ticklabels([])
            axes_list[2].xaxis.set_ticklabels([])
            axes_list[4].xaxis.set_ticklabels([])
            axes_list[5].xaxis.set_ticklabels([])

            axes_list[7].set_xlabel('Position [V]')

            # axes_list[2].legend(loc='upper left')
            axes_list[4].legend(loc='upper left')
            axes_list[5].legend(loc='upper right')

            dc_title = 'AFM freq={:0.2f}kHz, {:s} {:d} block(s), π/2={:2.1f}ns, π={:2.1f}ns, Repetition:{:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                self.scripts['trig_pdd'].settings['f_exc'],
                self.scripts['trig_pdd'].settings['decoupling_seq']['type'],
                self.scripts['trig_pdd'].settings['decoupling_seq']['num_of_pulse_blocks'],
                self.scripts['trig_pdd'].settings['mw_pulses']['pi_half_pulse_time'],
                self.scripts['trig_pdd'].settings['mw_pulses']['pi_pulse_time'],
                self.scripts['trig_pdd'].settings['rep_num'],
                self.scripts['trig_pdd'].settings['mw_pulses']['mw_power'],
                self.scripts['trig_pdd'].settings['mw_pulses']['mw_frequency'] * 1e-9,
                self.scripts['trig_pdd'].settings['mw_pulses']['IF_amp'],
                self.scripts['trig_pdd'].settings['mw_pulses']['IF_frequency'] * 1e-6)

            axes_list[0].set_title(
                '1D DC Sensing: pta=({:0.3f}V, {:0.3f}V), ptb=({:0.3f}V, {:0.3f}V), {:0.2}Vdc\n'.format(
                    self.settings['point_a']['x'], self.settings['point_a']['y'],
                    self.settings['point_b']['x'], self.settings['point_b']['y'],
                    self.settings['dc_voltage']['level']) + dc_title)

            # axes_list[0].set_title('1D DC Sensing: pta=({:0.3f}V, {:0.3f}V), ptb=({:0.3f}V, {:0.3f}V)\n'.format(
            #     self.settings['point_a']['x'], self.settings['point_a']['y'],
            #     self.settings['point_b']['x'], self.settings['point_b']['y']))
        except Exception as e:
            print('** ATTENTION in _plot 1d **')
            print(e)

        if self._current_subscript_stage['current_subscript'] == self.scripts['trig_pdd']:
            self.scripts['trig_pdd']._plot([axes_list[3]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
            self.scripts['rabi']._plot([axes_list[3]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[3]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[3], axes_list[6]], title=False)

    def _update_plot(self, axes_list):
        try:
            if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._update_plot([axes_list[3]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['trig_pdd']:
                self.scripts['trig_pdd']._update_plot([axes_list[3]], title=False)

            elif self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
                self.scripts['rabi']._update_plot([axes_list[3]], title=False)

            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[3], axes_list[6]], title=False,
                                                                monitor_AFM=False)
            else:
                if 'afm_dist_array' in self.data.keys() and len(self.data['afm_dist_array']) > 0:
                    axes_list[0].lines[0].set_xdata(self.data['afm_dist_array'])
                    axes_list[0].lines[0].set_ydata(self.data['afm_ctr'])
                    axes_list[0].relim()
                    axes_list[0].autoscale_view()

                    axes_list[1].lines[0].set_xdata(self.data['afm_dist_array'])
                    axes_list[1].lines[0].set_ydata(self.data['afm_analog'])
                    axes_list[1].relim()
                    axes_list[1].autoscale_view()
                if 'dc_data' in self.data.keys() and len(self.data['dc_dist_array']) > 0:
                    axes_list[2].lines[0].set_xdata(self.data['dc_dist_array'])
                    axes_list[2].lines[0].set_ydata(self.data['norm_data'][:, 0])
                    axes_list[2].lines[1].set_xdata(self.data['dc_dist_array'])
                    axes_list[2].lines[1].set_ydata(self.data['norm_data'][:, 1])
                    axes_list[2].lines[2].set_xdata(self.data['dc_dist_array'])
                    axes_list[2].lines[2].set_ydata(self.data['coherence'])
                    axes_list[2].relim()
                    axes_list[2].autoscale_view()

                    axes_list[7].lines[0].set_xdata(self.data['dc_dist_array'])
                    axes_list[7].lines[0].set_ydata(self.data['phase'])
                    axes_list[7].relim()
                    axes_list[7].autoscale_view()

                if 'rabi_freq' in self.data.keys() and 'rabi_power' in self.data.keys() and 'rabi_dist_array' in self.data.keys():
                    axes_list[4].lines[0].set_xdata(self.data['rabi_dist_array'])
                    axes_list[4].lines[0].set_ydata(self.data['rabi_freq'])
                    axes_list[5].lines[0].set_xdata(self.data['rabi_dist_array'])
                    axes_list[5].lines[0].set_ydata(self.data['rabi_power'])
                    axes_list[4].relim()
                    axes_list[4].autoscale_view()
                    axes_list[5].relim()
                    axes_list[5].autoscale_view()

        except Exception as e:
            print('** ATTENTION in _update_plot **')
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
            axes_list.append(figure_list[0].add_subplot(511))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(512))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(513))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[3]
            axes_list.append(figure_list[0].add_subplot(514))  # axes_list[4]
            axes_list.append(axes_list[4].twinx())  # axes_list[5]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[6]
            axes_list.append(figure_list[0].add_subplot(515))  # axes_list[7]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[0].axes[4])
            axes_list.append(figure_list[1].axes[1])
            axes_list.append(figure_list[0].axes[5])

        return axes_list


class ScanningDCSensing2D(Script):
    """
        Perform DC sensing measurements (based on PDDSyncAFM) and perform AFM scan in 2D.
        One can also choose to do Rabi and fluorescence optimize every several points.
        One needs to go to child subscripts to define the settings.
        - Ziwei Qiu 2/22/2021
    """
    _DEFAULT_SETTINGS = [
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
                             [Parameter('x', 0.0, float, 'x-coordinate [V]'),
                              Parameter('y', 1.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size',
                  [Parameter('axis1', 1.5, float, 'inner loop [V]'),
                   Parameter('axis2', 1.5, float, 'outer loop [V]')
                   ]),
        Parameter('num_points',
                  [Parameter('axis1', 60, int, 'number of points to scan in each line (inner loop)'),
                   Parameter('axis2', 20, int, 'number of lines to scan (outer loop)'),
                   ]),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('dc_voltage', [
            Parameter('level', 0.0, float, 'define the DC voltage [V] to infinity load'),
            Parameter('source', 'None', ['afg', 'yokogawa', 'keithley', 'None'],
                      'choose the voltage source. afg limit: +/-10V, yokogawa: +/-30V, keithley: not implemented. If None, then no voltage source is connected.')
        ]),
        Parameter('do_rabi',
                  [Parameter('pta_power', -15, float, 'choose the MW power at pta, first point of first line'),
                   Parameter('ptb_power', -15, float, 'choose the MW power at ptb, last point of first line'),
                   Parameter('ptc_power', -15, float, 'choose the MW power at ptc, first point of last line'),
                   Parameter('ptd_power', -15, float, 'choose the MW power at ptd, last point of last line'),
                   ]),
        # Parameter('monitor_AFM', False, bool,
        #           'monitor the AFM Z_out voltage and retract the tip when the feedback loop is out of control'),
        Parameter('DAQ_channels',
                  [Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring z voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for measuring scanner z voltage')
                   ])

    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6220': NI6220, 'NI6210': NI6210, 'afg': Agilent33120A,
                    'yokogawa': YokogawaGS200}
    _SCRIPTS = {'scandc1d': ScanningDCSensing, 'set_scanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.daq_in_usbAI = self.instruments['NI6210']['instance']

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

    def _get_scan_array(self):
        self._get_scan_extent()
        self.scan_pos_1d_ac = np.linspace(self.pta, self.ptc, self.settings['num_points']['axis2'], endpoint=True)
        self.scan_pos_1d_bd = np.linspace(self.ptb, self.ptd, self.settings['num_points']['axis2'], endpoint=True)
        self.start_power_ac = np.linspace(self.settings['do_rabi']['pta_power'], self.settings['do_rabi']['ptc_power'],
                                          self.settings['num_points']['axis2'], endpoint=True)
        self.end_power_bd = np.linspace(self.settings['do_rabi']['ptb_power'], self.settings['do_rabi']['ptd_power'],
                                        self.settings['num_points']['axis2'], endpoint=True)

    def set_up_voltage_source(self):
        if self.settings['dc_voltage']['source'] == 'afg':
            self.vol_source = self.instruments['afg']['instance']
            self.vol_source.update({'output_load': 'INFinity'})
            self.vol_source.update({'wave_shape': 'DC'})
            self.vol_source.update({'burst_mod': False})
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            self.vol_source = self.instruments['yokogawa']['instance']
            self.vol_source.update({'source': 'VOLT'})
            self.vol_source.update({'level': 0.0})
            self.vol_source.update({'enable_output': True})
        elif self.settings['dc_voltage']['source'] == 'keithley':
            self.vol_source = None
        else:
            self.vol_source = None

    def set_voltage(self, vol):
        steps = np.linspace(0, 1, 21)
        if self.settings['dc_voltage']['source'] == 'afg':
            print('Ramping up the voltage on AFG...')
            for step in steps:
                self.vol_source.update({'offset': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'offset': vol})
            print('Voltage is set on AFG.')
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            print('Ramping up the voltage on Yokogawa...')
            for step in steps:
                self.vol_source.update({'level': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'level': vol})
            print('Voltage is set on Yokogawa.')
        # elif self.settings['dc_voltage']['source'] == 'keithley':
        #     pass

    def close_voltage_source(self, vol):
        steps = np.linspace(1, 0, 21)
        if self.settings['dc_voltage']['source'] == 'afg':
            print('Ramping down the voltage on AFG...')
            for step in steps:
                self.vol_source.update({'offset': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'offset': 0.0})
            print('Voltage is 0 on AFG.')
        elif self.settings['dc_voltage']['source'] == 'yokogawa':
            print('Ramping down the voltage on Yokogawa...')
            for step in steps:
                self.vol_source.update({'level': float(vol * step)})
                time.sleep(0.5)
            self.vol_source.update({'level': 0.0})
            print('Voltage is 0 on Yokogawa.')
            self.vol_source.update({'enable_output': False})
        # elif self.settings['dc_voltage']['source'] == 'keithley':
        #     pass


    def _function(self):

        self.scripts['scandc1d'].settings['dc_voltage']['source'] = 'None'
        self._get_scan_array()

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:
            self.set_up_voltage_source()
            self.set_voltage(float(self.settings['dc_voltage']['level']))

            # Move to the initial scanning position
            self.scripts['set_scanner'].update({'to_do': 'set'})
            self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['set_scanner'].update({'step_size': 0.0001})

            self.data = {'scan_center': self.scan_center, 'scan_size_1': self.settings['scan_size']['axis1'],
                         'scan_size_2': self.settings['scan_size']['axis2'], 'vector_x': self.vector_x,
                         'extent': np.array([self.pta, self.ptb, self.ptc, self.ptd])}

            print('*********************************')
            print('********* 2D Scan Starts ********')
            print('*********************************')

            for i in range(self.settings['num_points']['axis2']):
                if self._abort:
                    break
                print('---------------------------------')
                print('----- Line index {:d} out of {:d} ----'.format(i, self.settings['num_points']['axis2']))
                print('---------------------------------')

                self.scripts['set_scanner'].settings['point']['x'] = self.scan_pos_1d_ac[i][0]
                self.scripts['set_scanner'].settings['point']['y'] = self.scan_pos_1d_ac[i][1]
                self.scripts['set_scanner'].run()

                try:
                    self.flag_1d_plot = True
                    self.scripts['scandc1d'].settings['tag'] = 'sensing1d_ind' + str(i)
                    # self.scripts['scandc1d'].settings['monitor_AFM'] = False
                    self.scripts['scandc1d'].settings['scan_speed'] = self.settings['scan_speed']
                    self.scripts['scandc1d'].settings['point_a']['x'] = self.scan_pos_1d_ac[i][0]
                    self.scripts['scandc1d'].settings['point_a']['y'] = self.scan_pos_1d_ac[i][1]
                    self.scripts['scandc1d'].settings['point_b']['x'] = self.scan_pos_1d_bd[i][0]
                    self.scripts['scandc1d'].settings['point_b']['y'] = self.scan_pos_1d_bd[i][1]
                    self.scripts['scandc1d'].settings['num_points'] = self.settings['num_points']['axis1']
                    # we always need to scan back in each line
                    self.scripts['scandc1d'].settings['do_afm1d']['after'] = True
                    self.scripts['scandc1d'].settings['do_rabi']['start_power'] = float(
                        self.start_power_ac[i])
                    self.scripts['scandc1d'].settings['do_rabi']['end_power'] = float(self.end_power_bd[i])
                    self.scripts['scandc1d'].run()

                    norm_sig1 = self.scripts['scandc1d'].data['norm_data'][:, 0]
                    norm_sig2 = self.scripts['scandc1d'].data['norm_data'][:, 1]

                    if 'norm_sig1_data' in self.data.keys():
                        self.data['norm_sig1_data'] = np.concatenate((self.data['norm_sig1_data'], norm_sig1), axis=0)
                    else:
                        self.data['norm_sig1_data'] = norm_sig1
                    if 'norm_sig2_data' in self.data.keys():
                        self.data['norm_sig2_data'] = np.concatenate((self.data['norm_sig2_data'], norm_sig2), axis=0)
                    else:
                        self.data['norm_sig2_data'] = norm_sig2

                    exact_tau = self.scripts['scandc1d'].data['exact_tau']
                    if 'exact_tau' in self.data.keys():
                        self.data['exact_tau'] = np.concatenate((self.data['exact_tau'], exact_tau), axis=0)
                    else:
                        self.data['exact_tau'] = exact_tau

                    phase = self.scripts['scandc1d'].data['phase']
                    if 'phase' in self.data.keys():
                        self.data['phase'] = np.concatenate((self.data['phase'], phase), axis=0)
                    else:
                        self.data['phase'] = phase

                except Exception as e:
                    print('** ATTENTION in scanning_dcsensing_1d **')
                    print(e)

            self.close_voltage_source(float(self.settings['dc_voltage']['level']))

    def _plot(self, axes_list, data=None):

        if self._current_subscript_stage['current_subscript'] == self.scripts['scandc1d']:
            self.scripts['scandc1d']._plot(axes_list)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['scandc1d']:

            if self.flag_1d_plot:
                self.scripts['scandc1d']._plot(axes_list)
                self.flag_1d_plot = False
            else:
                self.scripts['scandc1d']._update_plot(axes_list)

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
            axes_list.append(figure_list[0].add_subplot(511))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(512))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(513))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[3]
            axes_list.append(figure_list[0].add_subplot(514))  # axes_list[4]
            axes_list.append(axes_list[4].twinx())  # axes_list[5]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[6]
            axes_list.append(figure_list[0].add_subplot(515))  # axes_list[7]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[0].axes[4])
            axes_list.append(figure_list[1].axes[1])
            axes_list.append(figure_list[0].axes[5])

        return axes_list
