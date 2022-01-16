import time
import numpy as np
from collections import deque

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from pylabcontrol.core import Script, Parameter
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from b26_toolkit.instruments import SGS100ARFSource
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser, optimize
from b26_toolkit.data_processing.fit_functions import fit_sine_amplitude, sine

gate_wait = 55 # the delay between the RF pulses and electric fields in ns

class RamseyLockInOPX(Script):
    """
        This script measures a time series of Ramsey signals synchronized with a external slowly oscillating electric field.
        This is effectively a classical lock-in measurement based on NV Ramsey measurement.
        The external electric field is applied from the OPX port 5.

        - Ziwei Qiu 2/12/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('signal', [
            Parameter('freq_kHz', 10, [2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50], 'oscillating frequency [kHz]'),
            Parameter('amplitude', 0.05, float, 'amplitude of the oscillating voltage [V] from the OPX, i.e. 0.5Vpp'),
            Parameter('offset', 0.0, float, 'offset of the oscillating voltage [V]')
        ]),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'resonant IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('detuning', 1e6, float, 'detuning for Ramsey experiment in Hz]'),
            Parameter('pi_half_pulse_time', 50, float, 'time duration of a pi/2 pulse (in ns)'),
        ]),
        Parameter('tau', 800, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('ramsey_pts', 30, int,
                  'number of measurement points. if -1, then ramsey_pts will be automatically determined by the signal period. Each point is approximately 4us'),
        Parameter('fit', False, bool, 'fit the data in real time'),
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

    def _get_signal_duration(self):
        ramsey_time = self.settings['tau'] + self.settings['read_out']['nv_reset_time'] \
                      + self.settings['read_out']['laser_off'] + self.settings['read_out']['delay_mw_readout'] + 125
        # time of each ramsey point in ns
        min_length_in_cycle = round(self.settings['ramsey_pts']*ramsey_time / 4)
        signal_period_cycle = int(1e6 / self.settings['signal']['freq_kHz'] / 4) # in ns
        self.signal_length_ns = int(np.ceil(min_length_in_cycle / signal_period_cycle) * signal_period_cycle * 4)

    def _function(self):

        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                # unit: cycle of 4ns
                # pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                # pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

                # unit: ns
                # config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
                # config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

                self._get_signal_duration()
                config['elements']['gate']['intermediate_frequency'] = int(self.settings['signal']['freq_kHz'] * 1000)
                config['pulses']['gate_pulse1']['length'] = self.signal_length_ns
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['signal']['offset']
                config['waveforms']['const_gate1']['sample'] = self.settings['signal']['amplitude']

                self.qm = self.qmm.open_qm(config)

            except Exception as e:
                print('** ATTENTION **')
                print(e)

            else:
                rep_num = self.settings['rep_num']
                # tracking_num = self.settings['NV_tracking']['tracking_num']
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

                self.t = round(self.settings['tau'] / 4)  # unit: in 4ns cycle
                self.t = int(np.max([self.t, 4]))  # unit: in 4ns cycle
                self.tau = self.t * 4  # in ns
                print('tau [ns]: ', self.tau)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
                IF_freq = self.settings['mw_pulses']['IF_frequency'] - self.settings['mw_pulses']['detuning']

                self.ramsey_pts = self.settings['ramsey_pts']

                def readout_save():
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

                # define the qua program
                with program() as ramsey_lockin:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    n = declare(int)
                    m = declare(int)
                    # k = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()
                    reset_phase('gate')
                    reset_frame('gate', 'qubit')

                    with for_(n, 0, n < rep_num, n + 1):
                        align('gate', 'qubit')
                        # duration (already in the configuration) needs to be integer multiples of the signal period
                        play('gate1', 'gate')
                        # for m in np.arange(0, self.ramsey_pts, 1):
                        with for_(m, 0, m < self.ramsey_pts, m + 1):
                            play('pi2' * amp(IF_amp), 'qubit')
                            wait(self.t, 'qubit')
                            frame_rotation(np.pi * 0.0, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            frame_rotation(np.pi * 0.0, 'qubit')
                            readout_save()

                        align('gate', 'qubit')
                        play('gate1', 'gate')
                        # for m in np.arange(0, self.ramsey_pts, 1):
                        with for_(m, 0, m < self.ramsey_pts, m + 1):
                            play('pi2' * amp(IF_amp), 'qubit')
                            wait(self.t, 'qubit')
                            frame_rotation(np.pi, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            frame_rotation(-np.pi, 'qubit')
                            readout_save()

                        align('gate', 'qubit')
                        play('gate1', 'gate')
                        # for m in np.arange(0, self.ramsey_pts, 1):
                        with for_(m, 0, m < self.ramsey_pts, m + 1):
                            play('pi2' * amp(IF_amp), 'qubit')
                            wait(self.t, 'qubit')
                            frame_rotation(0.5*np.pi, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            frame_rotation(-0.5*np.pi, 'qubit')
                            readout_save()

                        align('gate', 'qubit')
                        play('gate1',
                             'gate')
                        # for k in np.arange(0, self.ramsey_pts, 1):
                        with for_(m, 0, m < self.ramsey_pts, m + 1):
                            play('pi2' * amp(IF_amp), 'qubit')
                            wait(self.t, 'qubit')
                            frame_rotation(1.5*np.pi, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            frame_rotation(-1.5*np.pi, 'qubit')
                            readout_save()

                        save(n, rep_num_st)

                    with stream_processing():
                        total_counts_st.buffer(4, self.ramsey_pts).average().save("live_data")
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(ramsey_lockin)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(ramsey_lockin, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):

        try:
            start = time.time()
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
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
            job = self.qm.execute(qua_program)
            # counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION in QM execution **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")
            # tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            # tracking_handle.wait_for_values(1)
            self.data = {'tau': float(self.tau), 'signal_avg_vec': None, 'ref_cnts': None,
                         'signal_norm': None, 'signal2_norm': None, 'signal_ramsey': None, 'phase': None,
                         'rep_num': None, 'ramsey_pts': self.ramsey_pts,
                         'signal_freq_kHz': self.settings['signal']['freq_kHz'],
                         'signal_amp': self.settings['signal']['amplitude']}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                    echo_avg = vec * 1e6 / self.meas_len
                    echo_avg = echo_avg.reshape((4, self.ramsey_pts))
                    ref_counts = (echo_avg[0, :].mean() + echo_avg[1, :].mean()) / 2

                    self.data.update({'signal_avg_vec': echo_avg / ref_counts,
                                      'ref_cnts': ref_counts})

                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][1, :] - self.data['signal_avg_vec'][0, :]) / \
                                               (self.data['signal_avg_vec'][0, :] + self.data['signal_avg_vec'][1, :])
                    self.data['signal2_norm'] = 2 * (
                            self.data['signal_avg_vec'][3, :] - self.data['signal_avg_vec'][2, :]) / \
                                                (self.data['signal_avg_vec'][2, :] + self.data['signal_avg_vec'][3, :])

                    self.data['signal_ramsey'] = np.sqrt(self.data['signal_norm'] ** 2 + self.data['signal2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['signal_norm'] / self.data['signal_ramsey']) * np.sign(
                        self.data['signal2_norm'])

                except Exception as e:
                    print('** ATTENTION in vec_handle **')
                    print(e)

                # do fitting
                if self.settings['fit']:
                    try:
                        us_per_pt = 4
                        first_pt_us = 4
                        A_fit, A_fit_err, phi_fit, phi_fit_err, freq_fit, offset_fit = \
                            fit_sine_amplitude(np.arange(self.ramsey_pts)*us_per_pt + first_pt_us, self.data['phase'])
                    except Exception as e:
                        print('** ATTENTION in fitting **')
                        print(e)
                    else:
                        self.data['fits'] = np.array([A_fit, A_fit_err, phi_fit, phi_fit_err, freq_fit, offset_fit])


                try:
                    current_rep_num = progress_handle.fetch_all()
                    # current_counts_vec = tracking_handle.fetch_all()
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

                time.sleep(1.0)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

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
            plot_qmsimulation_samples(axes_list[2], data)

        if 'tau' in data.keys() and 'signal_avg_vec' in data.keys():
            x_array = np.arange(self.ramsey_pts)

            axes_list[2].clear()
            if data['signal_avg_vec'] is not None:
                axes_list[2].plot(x_array, data['signal_avg_vec'][0, :], label="+x")
                axes_list[2].plot(x_array, data['signal_avg_vec'][1, :], label="-x")
                axes_list[2].plot(x_array, data['signal_avg_vec'][2, :], label="+y")
                axes_list[2].plot(x_array, data['signal_avg_vec'][3, :], label="-y")
            axes_list[2].set_xlabel('Time [a.u.]')
            axes_list[2].set_ylabel('Normalized Counts')
            axes_list[2].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

            axes_list[0].clear()
            if data['signal_norm'] is not None:
                axes_list[0].plot(x_array, data['signal_norm'], label="cosine")
            if data['signal2_norm'] is not None:
                axes_list[0].plot(x_array, data['signal2_norm'], label="sine")
            if data['signal_ramsey'] is not None:
                axes_list[0].plot(x_array, data['signal_ramsey'], label="squared sum root")
            axes_list[0].set_ylabel('Contrast')
            axes_list[0].legend(loc='upper right')

            axes_list[1].clear()
            if data['phase'] is not None:
                axes_list[1].plot(x_array, data['phase'])

            if self.settings['fit'] and 'fits' in data.keys():
                us_per_pt = 4
                first_pt_us = 4
                axes_list[1].plot(x_array, sine(x_array * us_per_pt + first_pt_us, data['fits'][0],
                                                data['fits'][2] / 180 * np.pi, data['fits'][4] / 1000 * 2 * np.pi,
                                                data['fits'][5]),
                                  label='sinusoidal fit\n A={:.3f}+/-{:0.2f} rad\n phi={:0.1f}+/-{:0.1f} deg\n freq={:0.4f}kHz'.format(
                                      data['fits'][0],
                                      data['fits'][1],
                                      data['fits'][2],
                                      data['fits'][3],
                                      data['fits'][4]))


            axes_list[1].set_xlabel('Time [a.u.]')
            axes_list[1].set_ylabel('Phase [rad]')

            axes_list[0].set_title(
                'Ramsey Lock-In OPX\nRef fluor: {:0.1f}kcps, Repetition number: {:d}, tau: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz\nsignal: freq: {:0.2f}kHz, amp: {:0.2f}Vpp, offset: {:0.2f}V'.format(
                    data['ref_cnts'], int(data['rep_num']), self.settings['tau'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                    self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['detuning'] * 1e-6,
                    self.settings['signal']['freq_kHz'], self.settings['signal']['amplitude'],
                    self.settings['signal']['offset']
                )
            )

    def _update_plot(self, axes_list):
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


class PDDLockInSweepGate(Script):
    """
        This script implements AC sensing based on PDD (Periodic Dynamical Decoupling) sequence.
        The voltage is sinusoidal and phase is varied, so NV can measure both the amplitude and the phase of the signal.
        This is effectively a quantum lock-in measurement.
        The external electric field is applied from the OPX port 5. No NV tracking option for now.

        - Ziwei Qiu 2/11/2021
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
        Parameter('tau', 10000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('phase', 0, float, 'define the initial phase of the oscillating voltage')
        ]),
        Parameter('sweep', [
            Parameter('min_vol', 0, float, 'define the minimum amplitude in V (i.e. 0.5 Vpp)'),
            Parameter('max_vol', 0.48, float, 'define the maximum amplitude in V (i.e. 0.5 Vpp), max 0.5V'),
            Parameter('vol_step', 0.05, float, 'define the voltage step in V')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'XY8', ['XY4', 'XY8'],
                      'type of dynamical decoupling sequences, here only xy4 and xy8 are allowed'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),

        Parameter('rep_num', 200000, int, 'define the repetition number, suggest at least 100000'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):

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

            except Exception as e:
                print('** ATTENTION **')
                print(e)

            else:
                rep_num = self.settings['rep_num']
                # tracking_num = self.settings['NV_tracking']['tracking_num']
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

                num_of_evolution_blocks = 0
                number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
                if self.settings['decoupling_seq']['type'] == 'XY4':
                    num_of_evolution_blocks = 4 * number_of_pulse_blocks
                elif self.settings['decoupling_seq']['type'] == 'XY8':
                    num_of_evolution_blocks = 8 * number_of_pulse_blocks

                # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
                t = round(self.settings['tau'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
                # signal_period = 2 * self.tau_total / num_of_evolution_blocks # in ns
                signal_period = (2 * t + pi2_time + pi_time) * 2 * 4 # in ns
                signal_freq = 1e9 / signal_period
                self.signal_freq_kHz = signal_freq / 1000 # in kHz
                config['elements']['gate']['intermediate_frequency'] = signal_freq
                config['pulses']['gate_pulse1']['length'] = int(self.tau_total)

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)
                print('Signal frequency = {:0.2f}kHz, period = {:0.3}us'.format(self.signal_freq_kHz, signal_period/1000))

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                # Amplitude array
                self.amp_vec = np.arange(self.settings['sweep']['min_vol'], self.settings['sweep']['max_vol'],
                                           self.settings['sweep']['vol_step'])
                gate1_config = np.max(np.abs(self.amp_vec))
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = gate1_config
                self.qm = self.qmm.open_qm(config)

                amp_list = (self.amp_vec / gate1_config).tolist()

                print('self.amp_vec in V (i.e. 0.5Vpp):', self.amp_vec)

                self.amp_num = len(self.amp_vec)

                def xy4_block(is_last_block, IF_amp=IF_amp):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def xy8_block(is_last_block, IF_amp=IF_amp):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False)
                        xy4_block(is_last_block=True)

                    else: # XY8
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False)
                        elif number_of_pulse_blocks > 1:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False)
                        xy8_block(is_last_block=True)

                p = self.settings['gate_voltages']['phase'] / 180 # phase in units of pi
                # define the qua program
                with program() as pdd_lockin:
                    update_frequency('qubit', round(IF_freq))
                    update_frequency('gate', round(signal_freq))
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    n = declare(int)
                    k = declare(int)
                    i = declare(int)
                    g1 = declare(fixed)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(g1, amp_list):
                            with for_(k, 0, k < 4, k + 1):

                                with if_(k == 0):  # +x readout
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    align('qubit', 'gate')

                                    wait(round(pi2_time/2 + gate_wait/4), 'gate')  # gate off
                                    frame_rotation(p*np.pi, 'gate') # it seems using qua variable phi will cause incorrect phase...
                                    play('gate1' * amp(g1), 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(0*np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 1):  # -x readout
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    align('qubit', 'gate')

                                    wait(round(pi2_time / 2+ gate_wait/4), 'gate')  # gate off
                                    frame_rotation(p*np.pi, 'gate')
                                    play('gate1' * amp(g1), 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 2):  # -y readout

                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    align('qubit', 'gate')

                                    wait(round(pi2_time / 2+ gate_wait/4), 'gate')  # gate off
                                    frame_rotation(p*np.pi, 'gate')
                                    play('gate1' * amp(g1), 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 3):  # -y readout

                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    align('qubit', 'gate')

                                    wait(round(pi2_time / 2+ gate_wait/4), 'gate')  # gate off
                                    frame_rotation(p*np.pi, 'gate')
                                    play('gate1' * amp(g1), 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation( 1.5 * np.pi, 'qubit')
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
                        total_counts_st.buffer(self.amp_num, 4).average().save("live_data")
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(pdd_lockin)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(pdd_lockin, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):

        try:
            start = time.time()
            # No need for flags=['skip-add-implicit-align'] here!!! It took three hours to figure this out...
            # job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
            #                            flags=['skip-add-implicit-align'])
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
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
            # job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
            job = self.qm.execute(qua_program)
            # counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)

            self.data = {'amp_vec': self.amp_vec, 'signal_freq_kHz': self.signal_freq_kHz, 'tau': self.tau_total,
                         'initial_phase_deg': self.settings['gate_voltages']['phase'],
                         'signal_avg_vec': None, 'ref_cnts': None, 'sig1_norm': None, 'sig2_norm': None,
                         'rep_num': None, 'squared_sum_root': None, 'phase': None}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
                                      'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})

                    self.data['sig1_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                             (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])

                    self.data['sig2_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 3] - self.data['signal_avg_vec'][:, 2]) / \
                                             (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
                                                                                  3])

                    self.data['squared_sum_root'] = np.sqrt(self.data['sig1_norm'] ** 2 + self.data['sig2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['sig1_norm'] / self.data['squared_sum_root']) * np.sign(
                        self.data['sig2_norm'])

                try:
                    current_rep_num = progress_handle.fetch_all()
                    # current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(1.0)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(PDDLockInSweepGate, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 'amp_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[0].clear()
                axes_list[1].clear()
                axes_list[2].clear()

                axes_list[1].plot(data['amp_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['amp_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['amp_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['amp_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")

                axes_list[1].set_xlabel('Signal amplitude [V]')
                axes_list[1].set_ylabel('Normalized Counts')
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].plot(data['amp_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['amp_vec'], data['sig2_norm'], label="sig2_norm")
                axes_list[0].plot(data['amp_vec'], data['squared_sum_root'], label="squared_sum_root")

                axes_list[2].plot(data['amp_vec'], data['phase'], label="phase")
                axes_list[2].grid(b=True, which='major', color='#666666', linestyle='--')

                axes_list[2].set_xlabel('Signal amplitude [V]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[2].set_ylabel('Phase [rad]')

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'PDD Lock-In Amplitude Sweep\nInput signal offset: {:0.2f}V, frequency: {:0.2f}kHz, initial phase: {:0.2f}deg\n{:s} {:d} block(s), tau = {:0.3}us, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}, pi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['gate_voltages']['offset'],
                        data['signal_freq_kHz'], self.settings['gate_voltages']['phase'],
                        self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                        self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))

            except Exception as e:
                print('** ATTENTION in _plot **')
                print(e)

    def _update_plot(self, axes_list):
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
            # axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
            # axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[2]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[1])

        return axes_list


class PDDLockInFixedGate(Script):
    """
        This script implements AC sensing based on PDD (Periodic Dynamical Decoupling) sequence.
        The voltage is sinusoidal and phase is varied, so NV can measure both the amplitude and the phase of the signal.
        This is effectively a quantum lock-in measurement.
        The external electric field is applied from the OPX port 5. No NV tracking option for now.

        - Ziwei Qiu 2/12/2021
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
        Parameter('tau', 10000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('amplitude', 0.48, float, 'define the oscillating voltage amplitude in V, i.e. 0.5Vpp'),
            Parameter('phase', 0, float, 'define the initial phase of the oscillating voltage'),
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'XY8', ['XY4', 'XY8'],
                      'type of dynamical decoupling sequences, here only xy4 and xy8 are allowed'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),

        Parameter('rep_num', 200000, int, 'define the repetition number, suggest at least 100000'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):

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

            except Exception as e:
                print('** ATTENTION **')
                print(e)

            else:
                rep_num = self.settings['rep_num']
                # tracking_num = self.settings['NV_tracking']['tracking_num']
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

                num_of_evolution_blocks = 0
                number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
                if self.settings['decoupling_seq']['type'] == 'XY4':
                    num_of_evolution_blocks = 4 * number_of_pulse_blocks
                elif self.settings['decoupling_seq']['type'] == 'XY8':
                    num_of_evolution_blocks = 8 * number_of_pulse_blocks

                # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
                t = round(self.settings['tau'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
                # signal_period = 2 * self.tau_total / num_of_evolution_blocks # in ns
                signal_period = (2 * t + pi2_time + pi_time) * 2 * 4 # in ns
                signal_freq = 1e9 / signal_period
                self.signal_freq_kHz = signal_freq / 1000 # in kHz

                config['elements']['gate']['intermediate_frequency'] = signal_freq
                config['pulses']['gate_pulse1']['length'] = int(self.tau_total)
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = self.settings['gate_voltages']['amplitude']
                self.qm = self.qmm.open_qm(config)

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)
                print('Signal frequency = {:0.2f}kHz, period = {:0.3}us'.format(self.signal_freq_kHz, signal_period/1000))

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def xy8_block(is_last_block, IF_amp=IF_amp):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False)
                        xy4_block(is_last_block=True)

                    else: # XY8
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False)
                        elif number_of_pulse_blocks > 1:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False)
                        xy8_block(is_last_block=True)

                p = self.settings['gate_voltages']['phase'] / 180  # phase in units of pi
                # define the qua program
                with program() as pdd_lockin:
                    update_frequency('qubit', round(IF_freq))
                    update_frequency('gate', round(signal_freq))
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    n = declare(int)
                    k = declare(int)
                    i = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_(k, 0, k < 4, k + 1):
                            with if_(k == 0):  # +x readout
                                # align('qubit', 'gate')
                                reset_phase('gate')
                                reset_frame('gate', 'qubit')

                                wait(round(pi2_time / 2 + gate_wait/4), 'gate')  # gate off
                                frame_rotation(p * np.pi,
                                               'gate')  # it seems using qua variable phi will cause incorrect phase...
                                play('gate1', 'gate')

                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()
                                frame_rotation(0 * np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')

                            with if_(k == 1):  # -x readout
                                # align('qubit', 'gate')
                                reset_phase('gate')
                                reset_frame('gate', 'qubit')

                                wait(round(pi2_time / 2 + gate_wait/4), 'gate')  # gate off
                                frame_rotation(p * np.pi, 'gate')
                                play('gate1', 'gate')

                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()
                                frame_rotation(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')

                            with if_(k == 2):  # -y readout

                                # align('qubit', 'gate')
                                reset_phase('gate')
                                reset_frame('gate', 'qubit')

                                wait(round(pi2_time / 2 + gate_wait/4), 'gate')  # gate off
                                frame_rotation(p * np.pi, 'gate')
                                play('gate1', 'gate')

                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()
                                frame_rotation(np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')

                            with if_(k == 3):  # -y readout

                                # align('qubit', 'gate')
                                reset_phase('gate')
                                reset_frame('gate', 'qubit')

                                wait(round(pi2_time / 2 + gate_wait/4), 'gate')  # gate off
                                frame_rotation(p * np.pi, 'gate')
                                play('gate1', 'gate')

                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()
                                frame_rotation(1.5 * np.pi, 'qubit')
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
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(pdd_lockin)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(pdd_lockin, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):

        try:
            start = time.time()
            # No need for flags=['skip-add-implicit-align'] here!!! It took three hours to figure this out...
            # job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
            #                            flags=['skip-add-implicit-align'])
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
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
            # job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
            job = self.qm.execute(qua_program)
            # counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)

            self.data = {'amp': self.settings['gate_voltages']['amplitude'], 'signal_freq_kHz': self.signal_freq_kHz,
                         'tau': self.tau_total, 'initial_phase_deg': self.settings['gate_voltages']['phase'],
                         'signal_avg_vec': None, 'ref_cnts': None, 'sig1_norm': None, 'sig2_norm': None,
                         'rep_num': None, 'squared_sum_root': None, 'phase': None}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0] + echo_avg[1]),
                                      'ref_cnts': (echo_avg[0] + echo_avg[1]) / 2})

                    self.data['sig1_norm'] = 2 * (
                            self.data['signal_avg_vec'][1] - self.data['signal_avg_vec'][0]) / \
                                             (self.data['signal_avg_vec'][0] + self.data['signal_avg_vec'][1])

                    self.data['sig2_norm'] = 2 * (
                            self.data['signal_avg_vec'][3] - self.data['signal_avg_vec'][2]) / \
                                             (self.data['signal_avg_vec'][2] + self.data['signal_avg_vec'][3])

                    self.data['squared_sum_root'] = np.sqrt(self.data['sig1_norm'] ** 2 + self.data['sig2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['sig1_norm'] / self.data['squared_sum_root']) * np.sign(
                        self.data['sig2_norm'])

                try:
                    current_rep_num = progress_handle.fetch_all()
                    # current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(1.0)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(PDDLockInFixedGate, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None, title=True):
        if data is None:
            data = self.data

        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 'amp' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[0].clear()
                axes_list[0].scatter(data['amp'], data['signal_avg_vec'][0], label="sig1 +")
                axes_list[0].scatter(data['amp'], data['signal_avg_vec'][1], label="sig1 -")
                axes_list[0].scatter(data['amp'], data['signal_avg_vec'][2], label="sig2 +")
                axes_list[0].scatter(data['amp'], data['signal_avg_vec'][3], label="sig2 -")
                axes_list[0].axhline(y=1.0, color='r', ls='--', lw=1.3)
                axes_list[0].set_xlabel('Amplitude [V]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[0].legend(loc='upper right')
                if title:
                    axes_list[0].set_title(
                        'PDD Lock-In\nInput signal offset: {:0.2f}V, amp = {:0.2}V, freq: {:0.2f}kHz, initial phase: {:0.2f}deg\n{:s} {:d} block(s), tau = {:0.3}us, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}, pi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                            self.settings['gate_voltages']['offset'], self.settings['gate_voltages']['amplitude'],
                            data['signal_freq_kHz'], self.settings['gate_voltages']['phase'],
                            self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                            data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                            self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                            self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                            self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                            self.settings['mw_pulses']['IF_frequency'] * 1e-6))

            except Exception as e:
                print('** ATTENTION in _plot **')
                print(e)

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


class PDDLockInDelayMeas(Script):
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('delay_times', [
            Parameter('min_time', 0, int, 'minimum gate delay time (in ns)'),
            Parameter('max_time', 500, int, 'maximum gate delay time (in ns)'),
            Parameter('time_step', 28, int,
                      'time step increment(in ns), using multiples of 4ns'),
            Parameter('qubit_wait', 136, int, 'qubit wait time (in ns), >=16ns, to account for the OPX timing artifect'),
        ]),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -10.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
        ]),
        Parameter('tau', 10000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('amplitude', 0.48, float, 'define the oscillating voltage amplitude in V, i.e. 0.5Vpp'),
            Parameter('phase', 0, float, 'define the initial phase of the oscillating voltage'),
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'XY8', ['XY4', 'XY8'],
                      'type of dynamical decoupling sequences, here only xy4 and xy8 are allowed'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),

        Parameter('rep_num', 200000, int, 'define the repetition number, suggest at least 100000'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {}
    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):

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

            except Exception as e:
                print('** ATTENTION **')
                print(e)

            else:
                rep_num = self.settings['rep_num']
                # tracking_num = self.settings['NV_tracking']['tracking_num']
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

                num_of_evolution_blocks = 0
                number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
                if self.settings['decoupling_seq']['type'] == 'XY4':
                    num_of_evolution_blocks = 4 * number_of_pulse_blocks
                elif self.settings['decoupling_seq']['type'] == 'XY8':
                    num_of_evolution_blocks = 8 * number_of_pulse_blocks

                # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
                t = round(self.settings['tau'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
                # signal_period = 2 * self.tau_total / num_of_evolution_blocks # in ns
                signal_period = (2 * t + pi2_time + pi_time) * 2 * 4 # in ns
                signal_freq = 1e9 / signal_period
                self.signal_freq_kHz = signal_freq / 1000 # in kHz

                config['elements']['gate']['intermediate_frequency'] = signal_freq
                # config['elements']['gate']['intermediate_frequency'] = 0.0
                config['pulses']['gate_pulse1']['length'] = int(self.tau_total)
                # config['pulses']['gate_pulse1']['length'] = int(t * 4)
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = self.settings['gate_voltages']['amplitude']
                self.qm = self.qmm.open_qm(config)

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)
                print('Signal frequency = {:0.2f}kHz, period = {:0.3}us'.format(self.signal_freq_kHz, signal_period/1000))

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def xy8_block(is_last_block, IF_amp=IF_amp):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    frame_rotation(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False)
                        xy4_block(is_last_block=True)

                    else: # XY8
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False)
                        elif number_of_pulse_blocks > 1:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False)
                        xy8_block(is_last_block=True)

                p = self.settings['gate_voltages']['phase'] / 180  # phase in units of pi
                #
                # self.t_vec = [int(a_) for a_ in
                #               np.arange(round(np.ceil(self.settings['delay_times']['min_time'] / 4 + pi2_time / 2)),
                #                         round(np.ceil(self.settings['delay_times']['max_time'] / 4 + pi2_time / 2)),
                #                         round(np.ceil(self.settings['delay_times']['time_step'] / 4)))]
                self.t_vec = [int(a_) for a_ in
                              np.arange(round(np.ceil(self.settings['delay_times']['min_time'] / 4)),
                                        round(np.ceil(self.settings['delay_times']['max_time'] / 4)),
                                        round(np.ceil(self.settings['delay_times']['time_step'] / 4)))]
                t_vec = self.t_vec
                print('Gate delay times [ns]: ', np.array(t_vec)*4)
                t_num = len(self.t_vec)
                qubit_wait = round(self.settings['delay_times']['qubit_wait'] / 4)

                # define the qua program
                with program() as pdd_lockin:
                    update_frequency('qubit', round(IF_freq))
                    update_frequency('gate', round(signal_freq))
                    # update_frequency('gate', 0)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    n = declare(int)
                    k = declare(int)
                    i = declare(int)
                    delay = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(delay, t_vec):
                            with for_(k, 0, k < 4, k + 1):
                                with if_(k == 0):  # +x readout
                                    # align('qubit', 'gate')
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')
                                    wait(qubit_wait, 'qubit')
                                    wait(delay + 16, 'gate')  # gate off
                                    frame_rotation(p * np.pi,
                                                   'gate')  # it seems using qua variable phi will cause incorrect phase...
                                    play('gate1', 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(0 * np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 1):  # -x readout
                                    # align('qubit', 'gate')
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')
                                    wait(qubit_wait, 'qubit')
                                    wait(delay + 16, 'gate')  # gate off
                                    frame_rotation(p * np.pi, 'gate')
                                    play('gate1', 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 2):  # -y readout

                                    # align('qubit', 'gate')
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    wait(qubit_wait, 'qubit')
                                    wait(delay + 16, 'gate')  # gate off
                                    frame_rotation(p * np.pi, 'gate')
                                    play('gate1', 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 3):  # -y readout

                                    # align('qubit', 'gate')
                                    reset_phase('gate')
                                    reset_frame('gate', 'qubit')

                                    wait(qubit_wait, 'qubit')
                                    wait(delay + 16, 'gate')  # gate off
                                    frame_rotation(p * np.pi, 'gate')
                                    play('gate1', 'gate')

                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    pi_pulse_train()
                                    frame_rotation(1.5 * np.pi, 'qubit')
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
                        total_counts_st.buffer(t_num, 4).average().save("live_data")
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(pdd_lockin)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(pdd_lockin, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):

        try:
            start = time.time()
            # No need for flags=['skip-add-implicit-align'] here!!! It took three hours to figure this out...
            # job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
            #                            flags=['skip-add-implicit-align'])
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
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
            # job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
            job = self.qm.execute(qua_program)
            # counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)

            self.data = {'amp': self.settings['gate_voltages']['amplitude'], 'signal_freq_kHz': self.signal_freq_kHz,
                         'tau': self.tau_total, 'initial_phase_deg': self.settings['gate_voltages']['phase'],
                         'signal_avg_vec': None, 'ref_cnts': None, 'sig1_norm': None, 'sig2_norm': None,
                         'rep_num': None, 'squared_sum_root': None, 'phase': None,
                         'gate_delay': np.array(self.t_vec) * 4}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
                                      'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})

                    self.data['sig1_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                             (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])

                    self.data['sig2_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 3] - self.data['signal_avg_vec'][:, 2]) / \
                                             (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:, 3])

                    self.data['squared_sum_root'] = np.sqrt(self.data['sig1_norm'] ** 2 + self.data['sig2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['sig1_norm'] / self.data['squared_sum_root']) * np.sign(
                        self.data['sig2_norm'])

                try:
                    current_rep_num = progress_handle.fetch_all()
                    # current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(1.0)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(PDDLockInDelayMeas, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None, title=True):
        if data is None:
            data = self.data

        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 'gate_delay' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[0].clear()
                axes_list[1].clear()
                axes_list[2].clear()

                axes_list[1].plot(data['gate_delay'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['gate_delay'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['gate_delay'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['gate_delay'], data['signal_avg_vec'][:, 3], label="sig2 -")
                axes_list[1].axhline(y=1.0, color='r', ls='--', lw=1.3)
                axes_list[1].set_xlabel('Gate delay time [ns]')
                axes_list[1].set_ylabel('Normalized Counts')
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].plot(data['gate_delay'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['gate_delay'], data['sig2_norm'], label="sig2_norm")
                axes_list[0].plot(data['gate_delay'], data['squared_sum_root'], label="squared_sum_root")
                axes_list[2].plot(data['gate_delay'], data['phase'])
                axes_list[2].grid(b=True, which='major', color='#666666', linestyle='--')

                axes_list[2].set_xlabel('Gate delay time [ns]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[2].set_ylabel('Phase [rad]')
                axes_list[0].legend(loc='upper right')
                if title:
                    axes_list[0].set_title(
                        'PDD Lock-In Delay Calibration\nInput signal offset: {:0.2f}V, amp = {:0.2}V, freq: {:0.2f}kHz, initial phase: {:0.2f}deg\n{:s} {:d} block(s), tau = {:0.3}us, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}, pi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                            self.settings['gate_voltages']['offset'], self.settings['gate_voltages']['amplitude'],
                            data['signal_freq_kHz'], self.settings['gate_voltages']['phase'],
                            self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                            data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                            self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                            self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                            self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                            self.settings['mw_pulses']['IF_frequency'] * 1e-6))

            except Exception as e:
                print('** ATTENTION in _plot **')
                print(e)

    def _update_plot(self, axes_list, title=True):
        self._plot(axes_list, title=title)

    def get_axes_layout(self, figure_list):

        axes_list = []
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()
            # axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
            # axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]

            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[2]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[1])

        return axes_list


class PDDLockInSweepPhase(Script):
    """
        This script measures the PDD signal locked in with an oscillating reference signal as a function of phase.
        Note phase sweep is the outer loop, which means all repetitions are done before going to the next phase.
        The voltage is applied from the OPX analog port 5.

        - Ziwei Qiu 1/20/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('sweep', [
            Parameter('min_phase', 0, float, 'define the minimum phase in degree'),
            Parameter('max_phase', 180, float, 'define the maximum phase in degree'),
            Parameter('phase_step', 10, float, 'define the phase step in degree')
        ])
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'pdd_lockin': PDDLockInFixedGate}

    def __init__(self, scripts, name=None, settings=None, instruments=None, log_function=None, timeout=1000000000,
                 data_path=None):

        Script.__init__(self, name, scripts=scripts, settings=settings, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def do_pdd_lockin(self, ini_phase, label=None, index=-1, verbose=True):
        self.scripts['pdd_lockin'].settings['gate_voltages']['phase'] = ini_phase
        print('Current script initial phase [deg]: ', ini_phase)

        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['pdd_lockin'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['pdd_lockin'].settings['tag'] = label
        elif index >= 0:
            self.scripts['pdd_lockin'].settings['tag'] = 'pdd_lockin_ind' + str(index)
        else:
            self.scripts['pdd_lockin'].settings['tag'] = 'pdd_lockin'

        if verbose:
            print('==> Start measuring PDD Lock-In Fixed Gate...')

        self.scripts['pdd_lockin'].settings['to_do'] = 'execution'

        try:
            self.scripts['pdd_lockin'].run()
        except Exception as e:
            print('** ATTENTION in pdd_lockin **')
            print(e)
        else:
            pdd_lockin_sig = self.scripts['pdd_lockin'].data['signal_avg_vec']
            self.data['plus_x'].append(pdd_lockin_sig[0])
            self.data['minus_x'].append(pdd_lockin_sig[1])
            self.data['plus_y'].append(pdd_lockin_sig[2])
            self.data['minus_y'].append(pdd_lockin_sig[3])
            self.data['tau'].append(self.scripts['pdd_lockin'].data['tau'])
            self.data['sig1_norm'].append(self.scripts['pdd_lockin'].data['sig1_norm'])
            self.data['sig2_norm'].append(self.scripts['pdd_lockin'].data['sig2_norm'])
            self.data['squared_sum_root'].append(self.scripts['pdd_lockin'].data['squared_sum_root'])
            self.data['phase'].append(self.scripts['pdd_lockin'].data['phase'])

            self.data['amp'] = self.scripts['pdd_lockin'].data['amp']
            self.data['signal_freq_kHz'] = self.scripts['pdd_lockin'].data['signal_freq_kHz']

    def _function(self):
        self.phase_array = np.arange(self.settings['sweep']['min_phase'],
                                     self.settings['sweep']['max_phase'] + self.settings['sweep']['phase_step'],
                                     self.settings['sweep']['phase_step']) # this is to include the end point

        self.data = {'initial_phase_deg': deque(), 'tau': deque(), 'plus_x': deque(), 'minus_x': deque(), 'plus_y': deque(),
                     'minus_y': deque(), 'sig1_norm': deque(), 'sig2_norm': deque(), 'squared_sum_root': deque(),
                     'phase': deque(), 'amp': None, 'signal_freq_kHz': None}

        for index in range(0, len(self.phase_array)):

            if self._abort:
                break
            try:
                ini_phase = self.phase_array[index]
                self.do_pdd_lockin(ini_phase = ini_phase, label='pdd_lockin', index=index)
            except Exception as e:
                print('** ATTENTION in self.do_pdd_lockin **')
                print(e)
            else:
                self.data['initial_phase_deg'].append(ini_phase)

            self.progress = index * 100. / len(self.phase_array)
            self.updateProgress.emit(int(self.progress))
            time.sleep(0.2)

        # convert deque object to numpy array or list
        if 'initial_phase_deg' in self.data.keys() is not None:
            self.data['initial_phase_deg'] = np.asarray(self.data['initial_phase_deg'])
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
        if 'sig1_norm' in self.data.keys() is not None:
            self.data['sig1_norm'] = np.asarray(self.data['sig1_norm'])
        if 'sig2_norm' in self.data.keys() is not None:
            self.data['sig2_norm'] = np.asarray(self.data['sig2_norm'])
        if 'squared_sum_root' in self.data.keys() is not None:
            self.data['squared_sum_root'] = np.asarray(self.data['squared_sum_root'])
        if 'phase' in self.data.keys() is not None:
            self.data['phase'] = np.asarray(self.data['phase'])

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        # print('data['initial_phase_deg']:',data['initial_phase_deg'])
        try:
            if len(data['initial_phase_deg']) > 0:
                axes_list[0].plot(data['initial_phase_deg'], data['sig1_norm'], label="cosine")
                axes_list[0].plot(data['initial_phase_deg'], data['sig2_norm'], label="sine")
                axes_list[0].plot(data['initial_phase_deg'], data['squared_sum_root'], label="squared_sum_root")
                axes_list[1].plot(data['initial_phase_deg'], data['phase'], label="phase")

            axes_list[0].set_ylabel('Contrast')
            axes_list[1].set_ylabel('Phase [rad]')
            axes_list[1].set_xlabel('Input signal initial phase [deg]')
            axes_list[0].legend(loc='upper right')
            axes_list[1].legend(loc='upper right')

            axes_list[0].set_title(
                'PDD Lock-In Phase Sweep\nInput signal offset: {:0.2f}V, amp = {:0.2f}V, freq: {:0.2f}kHz\n{:s} {:d} block(s), tau = {:0.3}us, Repetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                    self.scripts['pdd_lockin'].settings['gate_voltages']['offset'],
                    self.scripts['pdd_lockin'].settings['gate_voltages']['amplitude'],
                    data['signal_freq_kHz'],
                    self.scripts['pdd_lockin'].settings['decoupling_seq']['type'],
                    self.scripts['pdd_lockin'].settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.scripts['pdd_lockin'].settings['tau'] / 1000, self.scripts['pdd_lockin'].settings['rep_num'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['pi_half_pulse_time'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['pi_pulse_time'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['3pi_half_pulse_time'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['mw_power'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.scripts['pdd_lockin'].settings['mw_pulses']['IF_amp'],
                    self.scripts['pdd_lockin'].settings['mw_pulses']['IF_frequency'] * 1e-6))

            if self._current_subscript_stage['current_subscript'] == self.scripts['pdd_lockin'] and self.scripts[
                'pdd_lockin'].is_running:
                self.scripts['pdd_lockin']._plot([axes_list[2]], title=False)
        except Exception as e:
            print('** ATTENTION in _plot **')
            print(e)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['pdd_lockin'] and self.scripts[
            'pdd_lockin'].is_running:
            self.scripts['pdd_lockin']._update_plot([axes_list[2]], title=False)
        else:
            self._plot(axes_list)

    def get_axes_layout(self, figure_list):

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
