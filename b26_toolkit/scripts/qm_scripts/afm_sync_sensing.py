import time
import numpy as np
from collections import deque

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from pylabcontrol.core import Script, Parameter
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from b26_toolkit.instruments import SGS100ARFSource, Agilent33120A, YokogawaGS200
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser, optimize

class EchoSyncAFM(Script):
    """
        This scripts implements AFM motion-enabled DC sensing.
        An echo sequence is used for sensing and synchronized with AFM motion. Tau is fixed.
        AFM frequency is typically of 32.65kHz.
        Be sure to check on the scope that the echo sequence is indeed synced with the AFM motion.
            - Ziwei Qiu 1/22/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -10.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
        ]),
        Parameter('tau', 10000, int, 'time between the two pi/2 pulses (in ns), max allowed tau = 25us'),
        Parameter('f_exc', 32.65, float, 'tuning fork excitation frequency (in kHz)'),
        Parameter('initial_delay_offset', -2250, float, 'initial delay manual offset in ns'),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in ns'),
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

        self._connect()

    def _connect(self):

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

    def _function(self):
        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                # unit: cycle of 4ns
                pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

                # unit: ns
                config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
                config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

                self.qm = self.qmm.open_qm(config)

            except Exception as e:
                print('** ATTENTION in open_qm **')
                print(e)

            else:
                rep_num = self.settings['rep_num']
                self.meas_len = round(self.settings['read_out']['meas_len'])
                nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
                laser_off = round(self.settings['read_out']['laser_off'] / 4)
                delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
                delay_readout = round(self.settings['read_out']['delay_readout'] / 4)

                IF_amp = self.settings['mw_pulses']['IF_amp']
                if IF_amp > 1.0:
                    IF_amp = 1.0
                elif IF_amp < 0.0:
                    IF_amp = 0.0

                num_of_evolution_blocks = 1
                # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
                t = round(self.settings['tau'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                # delay between the trigger and the pi/2 pulse edge (in cycles of 4ns)
                initial_delay = round(
                    1000000 / self.settings['f_exc'] / 2 / 4 - t - pi2_time - pi_time / 2 + self.settings[
                        'initial_delay_offset'] / 4)
                initial_delay = int(np.max([initial_delay, 4]))
                print('initial_delay in ns:', initial_delay*4)

                # define the qua program
                with program() as triggered_echo:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    # t = declare(int) this needs to be commented out!!!
                    n = declare(int)
                    k = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_(k, 0, k < 4, k + 1):
                            wait_for_trigger('qubit')
                            reset_frame('qubit')
                            wait(initial_delay, 'qubit')
                            with if_(k == 0):  # +x readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('pi' * amp(IF_amp), 'qubit')
                                wait(t, 'qubit')
                                frame_rotation(-np.pi * 0.5, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 1):  # -x readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('pi' * amp(IF_amp), 'qubit')
                                wait(t, 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 2):  # +y readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('pi' * amp(IF_amp), 'qubit')
                                wait(t, 'qubit')
                                frame_rotation(np.pi * 0.0, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 3):  # -y readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('pi' * amp(IF_amp), 'qubit')
                                wait(t, 'qubit')
                                frame_rotation(np.pi * 1.0, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')

                            align('qubit', 'laser', 'readout1', 'readout2')
                            wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
                            play('trig', 'laser', duration=nv_reset_time)
                            wait(delay_readout, 'readout1', 'readout2')
                            measure('readout', 'readout1', None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure('readout', 'readout2', None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))

                            align('qubit', 'laser', 'readout1', 'readout2')
                            wait(laser_off, 'qubit')

                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            save(n, rep_num_st)
                            save(total_counts, "total_counts")

                    with stream_processing():
                        total_counts_st.buffer(4).average().save("live_data")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(triggered_echo)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(triggered_echo, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):
        try:
            start = time.time()
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
                                       flags=['skip-add-implicit-align'])
            end = time.time()
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            print('QM simulation took {:.1f}s.'.format(end - start))
            self.log('QM simulation took {:.1f}s.'.format(end - start))
            samples = job_sim.get_simulated_samples().con1
            self.data = {'analog': samples.analog,
                         'digital': samples.digital}

    def _qm_execution(self, qua_program, job_stop):
        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, trigger on).')

        try:
            job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
        except Exception as e:
            print('** ATTENTION in QM execution **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)

            self.data = {'tau': float(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
                         'signal_norm': None, 'signal2_norm': None, 'squared_sum_root': None, 'phase': None,
                         'rep_num': None}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in vec_handle **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0] + echo_avg[1]),
                                      'ref_cnts': (echo_avg[0] + echo_avg[1]) / 2})
                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][1] - self.data['signal_avg_vec'][0]) / \
                                               (self.data['signal_avg_vec'][0] + self.data['signal_avg_vec'][1])
                    self.data['signal2_norm'] = 2 * (
                            self.data['signal_avg_vec'][3] - self.data['signal_avg_vec'][2]) / \
                                                (self.data['signal_avg_vec'][2] + self.data['signal_avg_vec'][3])
                    self.data['squared_sum_root'] = np.sqrt(self.data['signal_norm'] ** 2 + self.data['signal2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['signal_norm'] / self.data['squared_sum_root']) * np.sign(
                        self.data['signal2_norm'])

                try:
                    current_rep_num = progress_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle / tracking_handle **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(EchoSyncAFM, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None, title=True):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data

        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 'tau' in data.keys() and 'signal_avg_vec' in data.keys():
            axes_list[0].clear()
            axes_list[0].scatter(data['tau'], data['signal_avg_vec'][0], label="+x")
            axes_list[0].scatter(data['tau'], data['signal_avg_vec'][1], label="-x")
            axes_list[0].scatter(data['tau'], data['signal_avg_vec'][2], label="+y")
            axes_list[0].scatter(data['tau'], data['signal_avg_vec'][3], label="-y")
            axes_list[0].set_xlabel('Tau [ns]')
            axes_list[0].set_ylabel('Normalized Counts')
            axes_list[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

            if title:
                axes_list[0].set_title(
                    'Echo Synced with AFM\nRef fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, π time: {:2.1f}ns, 3π/2 time: {:2.1f}ns\ntau: {:2.1f}ns, Echo signal: {:0.2f}, phase: {:0.2f} rad'.format(
                        data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                        self.settings['mw_pulses']['pi_half_pulse_time'],
                        self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                        data['tau'], data['squared_sum_root'], data['phase'])
                )

    def _update_plot(self, axes_list, title=True):
        self._plot(axes_list, title=title)

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
            axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class DCSensingSyncAFM(Script):
    """
        This script performs DC sensing calibration based on AFM-synced Echo measurement, assuming there is spacially inhomogeneous electric field.
        The DC voltage is provided by an arbitrary function generator or yokogawa.
        Readout both the cosine and sine components (i.e. along x and y axis).
            - Ziwei Qiu 1/22/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('sweep', [
            Parameter('min_vol', -5.0, float, 'define the minimum DC voltage [V]'),
            Parameter('max_vol', 5.0, float, 'define the maximum DC voltage [V]'),
            Parameter('vol_step', 0.5, float, 'define the DC voltage step [V]')
        ]),
        Parameter('voltage_source', 'afg', ['afg', 'yokogawa', 'keithley'],
                  'choose the voltage source. afg limit: +/-10V, yokogawa: +/-30V'),
        Parameter('tracking_settings', [Parameter('track_focus', False, bool,
                                                  'check to use optimize to track to the NV'),
                                        Parameter('track_focus_every_N', 5, int, 'track every N points')]
                  )
    ]
    _INSTRUMENTS = {'afg': Agilent33120A, 'yokogawa':YokogawaGS200}
    _SCRIPTS = {'trig_echo': EchoSyncAFM, 'optimize': optimize}

    def __init__(self, scripts, name=None, settings=None, instruments=None, log_function=None, timeout=1000000000,
                 data_path=None):

        Script.__init__(self, name, scripts=scripts, settings=settings, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _get_voltage_array(self):
        min_vol = self.settings['sweep']['min_vol']
        max_vol = self.settings['sweep']['max_vol']
        vol_step = self.settings['sweep']['vol_step']

        if self.settings['voltage_source'] == 'afg':
            if min_vol < -10:
                min_vol = -10.0
            if max_vol > 10:
                max_vol = 10.0
        elif self.settings['voltage_source'] == 'yokogawa':
            if min_vol < -30:
                min_vol = -30.0
            if max_vol > 30:
                max_vol = 30.0

        self.sweep_array = np.arange(min_vol, max_vol, vol_step)

    def do_echo(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['trig_echo'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['trig_echo'].settings['tag'] = label
        elif index >= 0:
            self.scripts['trig_echo'].settings['tag'] = 'trig_echo_ind' + str(index)
        else:
            self.scripts['trig_echo'].settings['tag'] = 'trig_echo'

        if verbose:
            print('==> Start measuring Triggered Echo ...')

        self.scripts['trig_echo'].settings['to_do'] = 'execution'

        try:
            self.scripts['trig_echo'].run()
        except Exception as e:
            print('** ATTENTION in Running trig_echo **')
            print(e)
        else:
            raw_sig = self.scripts['trig_echo'].data['signal_avg_vec']
            self.data['plus_x'].append(raw_sig[0])
            self.data['minus_x'].append(raw_sig[1])
            self.data['plus_y'].append(raw_sig[2])
            self.data['minus_y'].append(raw_sig[3])
            self.data['tau'].append(self.scripts['trig_echo'].data['tau'])
            self.data['norm1'].append(self.scripts['trig_echo'].data['signal_norm'])
            self.data['norm2'].append(self.scripts['trig_echo'].data['signal2_norm'])
            self.data['squared_sum_root'].append(self.scripts['trig_echo'].data['squared_sum_root'])
            self.data['phase'].append(self.scripts['trig_echo'].data['phase'])

    def set_up_voltage_source(self):
        if self.settings['voltage_source'] == 'afg':
            self.vol_source = self.instruments['afg']['instance']
            self.vol_source.update({'output_load': 'INFinity'})
            self.vol_source.update({'wave_shape': 'DC'})
            self.vol_source.update({'burst_mod': False})
        elif self.settings['voltage_source'] == 'yokogawa':
            self.vol_source = self.instruments['yokogawa']['instance']
            self.vol_source.update({'source': 'VOLT'})
            self.vol_source.update({'level': 0.0})
            self.vol_source.update({'enable_output': True})
        elif self.settings['voltage_source'] == 'keithley':
            self.vol_source = None

    def set_voltage(self, vol):
        if self.settings['voltage_source'] == 'afg':
            self.vol_source.update({'offset': vol})
        elif self.settings['voltage_source'] == 'yokogawa':
            self.vol_source.update({'level': vol})
        elif self.settings['voltage_source'] == 'keithley':
            pass

    def close_voltage_source(self):
        if self.settings['voltage_source'] == 'afg':
            self.vol_source.update({'offset': 0.0})
        elif self.settings['voltage_source'] == 'yokogawa':
            self.vol_source.update({'level': 0.0})
            self.vol_source.update({'enable_output': False})
        elif self.settings['voltage_source'] == 'keithley':
            pass

    def _function(self):
        self._get_voltage_array()
        self.set_up_voltage_source()

        self.data = {'vol': deque(), 'tau': deque(), 'plus_x': deque(), 'minus_x': deque(), 'plus_y': deque(),
                     'minus_y': deque(), 'norm1': deque(), 'norm2': deque(), 'squared_sum_root': deque(),
                     'phase': deque()}

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

        for index in range(0, len(self.sweep_array)):
            if self._abort:
                break
            vol = self.sweep_array[index]
            self.set_voltage(float(vol))
            try:
                do_tracking(index)
                self.do_echo(label='trig_echo', index=index)
            except Exception as e:
                print('** ATTENTION in self.do_echo **')
                print(e)
            else:
                self.data['vol'].append(vol)

            self.progress = index * 100. / len(self.sweep_array)
            self.updateProgress.emit(int(self.progress))
            time.sleep(0.2)


        self.close_voltage_source()

        # convert deque object to numpy array or list for saving
        if 'vol' in self.data.keys() is not None:
            self.data['vol'] = np.asarray(self.data['vol'])
        if 'tau' in self.data.keys() is not None:
            self.data['tau'] = np.asarray(self.data['tau'])
        if 'plus_x' in self.data.keys() is not None:
            self.data['plus_x'] = np.asarray(self.data['plus_x'])
        if 'minus_x' in self.data.keys() is not None:
            self.data['minus_x'] = np.asarray(self.data['minus_x'])
        if 'plus_y' in self.data.keys() is not None:
            self.data['plus_y'] = np.asarray(self.data['plus_y'])
        if 'minus_y' in self.data.keys() is not None:
            self.data['minus_y'] = np.asarray(self.data['minus_y'])
        if 'norm1' in self.data.keys() is not None:
            self.data['norm1'] = np.asarray(self.data['norm1'])
        if 'norm2' in self.data.keys() is not None:
            self.data['norm2'] = np.asarray(self.data['norm2'])
        if 'squared_sum_root' in self.data.keys() is not None:
            self.data['squared_sum_root'] = np.asarray(self.data['squared_sum_root'])
        if 'phase' in self.data.keys() is not None:
            self.data['phase'] = np.asarray(self.data['phase'])

    def _plot(self, axes_list, data=None):
        # COMMENT_ME

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()

        if len(data['vol']) > 0:
            axes_list[0].plot(data['vol'], data['norm1'], label="cosine")
            axes_list[0].plot(data['vol'], data['norm2'], label="sine")
            axes_list[0].plot(data['vol'], data['squared_sum_root'], label="squared_sum_root")
            axes_list[1].plot(data['vol'], data['phase'], label="phase")

        axes_list[0].set_ylabel('Contrast')
        axes_list[1].set_ylabel('Phase [rad]')
        axes_list[1].set_xlabel('DC Voltage [V]')
        axes_list[0].legend(loc='upper right')
        axes_list[1].legend(loc='upper right')
        axes_list[0].set_title(
            'AFM-synced Echo-based DC Sensing\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, π time: {:2.1f}ns, 3π/2 time: {:2.1f}ns\nRepetition number: {:d}, tau: {:2.1f}ns '.format(
                self.scripts['trig_echo'].settings['mw_pulses']['mw_power'],
                self.scripts['trig_echo'].settings['mw_pulses']['mw_frequency'] * 1e-9,
                self.scripts['trig_echo'].settings['mw_pulses']['IF_amp'],
                self.scripts['trig_echo'].settings['mw_pulses']['IF_frequency'] * 1e-6,
                self.scripts['trig_echo'].settings['mw_pulses']['pi_half_pulse_time'],
                self.scripts['trig_echo'].settings['mw_pulses']['pi_pulse_time'],
                self.scripts['trig_echo'].settings['mw_pulses']['3pi_half_pulse_time'],
                self.scripts['trig_echo'].settings['rep_num'], self.scripts['trig_echo'].settings['tau'])
        )

        if self._current_subscript_stage['current_subscript'] == self.scripts['trig_echo'] and self.scripts[
            'trig_echo'].is_running:
            self.scripts['trig_echo']._plot([axes_list[2]], title=False)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[2]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[2]])

        elif self._current_subscript_stage['current_subscript'] == self.scripts['trig_echo'] and self.scripts[
            'trig_echo'].is_running:
            self.scripts['trig_echo']._update_plot([axes_list[2]], title=False)
        else:
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
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[2]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])

        return axes_list


    pass