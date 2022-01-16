import time
import numpy as np
from collections import deque

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from pylabcontrol.core import Script, Parameter
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from b26_toolkit.instruments import SGS100ARFSource, Agilent33120A
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser, optimize
from b26_toolkit.data_processing.fit_functions import fit_sine_amplitude, sine


class RamseyQM(Script):
    """
        This script runs a Ramsey measurement on an NV center.
        Only readout the cosine components (i.e. along x axis).
        - Ziwei Qiu 1/16/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'resonant IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('detuning', 1e6, float, 'detuning for Ramsey experiment in Hz]'),
            Parameter('pi_half_pulse_time', 50, float, 'time duration of a pi/2 pulse (in ns)'),
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 100, int, 'minimum time between the two pi pulses'),
            Parameter('max_time', 3000, int, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in ns'),
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {'optimize': OptimizeNoLaser}

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
        #####################################
        # Open communication with the server:
        #####################################
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
                # pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                # pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

                # unit: ns
                # config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
                # config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rep_num = self.settings['rep_num']
                tracking_num = self.settings['NV_tracking']['tracking_num']
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

                tau_start = np.max([round(self.settings['tau_times']['min_time']), 16])

                tau_end = round(self.settings['tau_times']['max_time'])
                tau_step = round(self.settings['tau_times']['time_step'])
                self.t_vec = [int(a_) for a_ in
                              np.arange(round(np.ceil(tau_start / 4)), round(np.ceil(tau_end / 4)),
                                        round(np.ceil(tau_step / 4)))]
                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('t_vec [ns]: ', np.array(self.t_vec) * 4)

                if len(self.t_vec) > 1:
                    res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
                    IF_freq = self.settings['mw_pulses']['IF_frequency'] - self.settings['mw_pulses']['detuning']

                    # define the qua program
                    with program() as ramsey:
                        update_frequency('qubit', IF_freq)
                        result1 = declare(int, size=res_len)
                        counts1 = declare(int, value=0)
                        result2 = declare(int, size=res_len)
                        counts2 = declare(int, value=0)
                        total_counts = declare(int, value=0)

                        t = declare(int)
                        n = declare(int)
                        k = declare(int)

                        # the following two variable are used to flag tracking
                        assign(IO1, False)

                        total_counts_st = declare_stream()
                        rep_num_st = declare_stream()

                        with for_(n, 0, n < rep_num, n + 1):
                            # Check if tracking is called
                            with while_(IO1):
                                play('trig', 'laser', duration=10000)

                            with for_each_(t, t_vec):
                                with for_(k, 0, k < 2, k + 1):
                                    reset_frame('qubit')
                                    with if_(k == 0):  # +x readout
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        # frame_rotation(np.pi*0.5, 'qubit')
                                        play('pi2' * amp(IF_amp), 'qubit')
                                    with if_(k == 1):
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        frame_rotation(np.pi, 'qubit')
                                        # frame_rotation(np.pi*0.5, 'qubit')
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
                            total_counts_st.buffer(t_num, 2).average().save("live_data")
                            total_counts_st.buffer(tracking_num).save("current_counts")
                            rep_num_st.save("live_rep_num")

                    with program() as job_stop:
                        play('trig', 'laser', duration=10)

                    if self.settings['to_do'] == 'simulation':
                        self._qm_simulation(ramsey)
                    elif self.settings['to_do'] == 'execution':
                        self._qm_execution(ramsey, job_stop)
                    self._abort = True
                else:
                    print('t_vec length needs to be >= 2! No action.')

    def _qm_simulation(self, qua_program):
        try:
            start = time.time()
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
                                       flags=['skip-add-implicit-align'])
            end = time.time()
        except Exception as e:
            print('** ATTENTION in QM simulation **')
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
            counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION in QM execution **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            self.data = {'t_vec': np.array(self.t_vec) * 4, 'signal_avg_vec': None, 'ref_cnts': None,
                         'signal_norm': None, 'rep_num': None}

            ref_counts = -1
            tolerance = self.settings['NV_tracking']['tolerance']

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in vec_handle **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
                                      'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})

                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                               (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
                                                                                    1])

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle / tracking_handle **')
                    print(e)
                else:
                    # print('current_counts_vec',current_counts_vec)
                    current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
                    # print('current_counts', current_counts_kcps)
                    # print('ref_counts', ref_counts)
                    if ref_counts < 0:
                        ref_counts = current_counts_kcps

                    if self.settings['NV_tracking']['on']:
                        if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
                                1 - tolerance):
                            counts_out_num += 1

                            print(
                                '--> No.{:d}: Current counts {:0.2f}kcps is out of range [{:0.2f}kcps, {:0.2f}kcps].'.format(
                                    counts_out_num, current_counts_kcps, ref_counts * (1 - tolerance),
                                                                         ref_counts * (1 + tolerance)))

                            if counts_out_num > 5:
                                print('** Start tracking **')
                                self.qm.set_io1_value(True)
                                self.NV_tracking()
                                try:
                                    self.qm.set_io1_value(False)
                                except Exception as e:
                                    print('** ATTENTION in set_io1_value **')
                                    print(e)
                                else:
                                    counts_out_num = 0
                                    ref_counts = self.settings['NV_tracking']['ref_counts']

                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(RamseyQM, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
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

        if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            axes_list[1].clear()
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="+x")
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="-x")
            axes_list[1].set_xlabel('Total tau [ns]')
            axes_list[1].set_ylabel('Normalized Counts')
            axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

            axes_list[0].clear()
            axes_list[0].plot(data['t_vec'], data['signal_norm'], label="signal")
            axes_list[0].set_xlabel('Total tau [ns]')
            axes_list[0].set_ylabel('Contrast')

            axes_list[0].legend(loc='upper right')
            axes_list[0].set_title(
                'Ramsey\nRef fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz'.format(
                    data['ref_cnts'], int(data['rep_num']),
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                    self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['detuning'] * 1e-6)
            )


    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
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
            axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class RamseyQM_v2(Script):
    """
        This script runs a Ramsey measurement on an NV center.
        Readout both the cosine and sine components (i.e. along x and y axis).
        - Ziwei Qiu 1/19/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'resonant IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('detuning', 1e6, float, 'detuning for Ramsey experiment in Hz]'),
            Parameter('pi_half_pulse_time', 50, float, 'time duration of a pi/2 pulse (in ns)'),
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 100, int, 'minimum time between the two pi pulses'),
            Parameter('max_time', 3000, int, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in ns'),
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {'optimize': OptimizeNoLaser}

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
        #####################################
        # Open communication with the server:
        #####################################
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
                # pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                # pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

                # unit: ns
                # config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
                # config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rep_num = self.settings['rep_num']
                tracking_num = self.settings['NV_tracking']['tracking_num']
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

                tau_start = np.max([round(self.settings['tau_times']['min_time']), 16])

                tau_end = round(self.settings['tau_times']['max_time'])
                tau_step = round(self.settings['tau_times']['time_step'])
                self.t_vec = [int(a_) for a_ in
                              np.arange(round(np.ceil(tau_start / 4)), round(np.ceil(tau_end / 4)),
                                        round(np.ceil(tau_step / 4)))]
                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('t_vec [ns]: ', np.array(self.t_vec) * 4)

                if len(self.t_vec) > 1:
                    res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
                    IF_freq = self.settings['mw_pulses']['IF_frequency'] - self.settings['mw_pulses']['detuning']

                    # define the qua program
                    with program() as ramsey:
                        update_frequency('qubit', IF_freq)
                        result1 = declare(int, size=res_len)
                        counts1 = declare(int, value=0)
                        result2 = declare(int, size=res_len)
                        counts2 = declare(int, value=0)
                        total_counts = declare(int, value=0)

                        t = declare(int)
                        n = declare(int)
                        k = declare(int)

                        # the following two variable are used to flag tracking
                        assign(IO1, False)

                        total_counts_st = declare_stream()
                        rep_num_st = declare_stream()

                        with for_(n, 0, n < rep_num, n + 1):
                            # Check if tracking is called
                            with while_(IO1):
                                play('trig', 'laser', duration=10000)

                            with for_each_(t, t_vec):
                                with for_(k, 0, k < 4, k + 1):
                                    reset_frame('qubit')
                                    with if_(k == 0):  # +x readout
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        frame_rotation(np.pi * 0.0, 'qubit')
                                        play('pi2' * amp(IF_amp), 'qubit')
                                    with if_(k == 1): # -x readout
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        frame_rotation(np.pi, 'qubit')
                                        play('pi2' * amp(IF_amp), 'qubit')
                                    with if_(k == 2):  # +y readout
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        frame_rotation(np.pi*0.5, 'qubit')
                                        play('pi2' * amp(IF_amp), 'qubit')
                                    with if_(k == 3): # -y readout
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        wait(t, 'qubit')
                                        frame_rotation(np.pi*1.5, 'qubit')
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
                            total_counts_st.buffer(tracking_num).save("current_counts")
                            rep_num_st.save("live_rep_num")

                    with program() as job_stop:
                        play('trig', 'laser', duration=10)

                    if self.settings['to_do'] == 'simulation':
                        self._qm_simulation(ramsey)
                    elif self.settings['to_do'] == 'execution':
                        self._qm_execution(ramsey, job_stop)
                    self._abort = True
                else:
                    print('t_vec length needs to be >= 2! No action.')

    def _qm_simulation(self, qua_program):
        try:
            start = time.time()
            job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
                                       flags=['skip-add-implicit-align'])
            end = time.time()
        except Exception as e:
            print('** ATTENTION in QM simulation **')
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
            counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
        except Exception as e:
            print('** ATTENTION in QM execution **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            self.data = {'t_vec': np.array(self.t_vec) * 4, 'signal_avg_vec': None, 'ref_cnts': None,
                         'signal_norm': None, 'signal2_norm': None, 'signal_ramsey': None, 'rep_num': None}

            ref_counts = -1
            tolerance = self.settings['NV_tracking']['tolerance']

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in vec_handle **')
                    print(e)
                else:
                    echo_avg = vec * 1e6 / self.meas_len
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
                                      'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})

                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                               (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
                                                                                    1])
                    self.data['signal2_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 3] - self.data['signal_avg_vec'][:, 2]) / \
                                             (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
                                                                                  3])
                    self.data['signal_ramsey'] = np.sqrt(self.data['signal_norm'] ** 2 + self.data['signal2_norm'] ** 2)

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle / tracking_handle **')
                    print(e)
                else:
                    # print('current_counts_vec',current_counts_vec)
                    current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
                    # print('current_counts', current_counts_kcps)
                    # print('ref_counts', ref_counts)
                    if ref_counts < 0:
                        ref_counts = current_counts_kcps

                    if self.settings['NV_tracking']['on']:
                        if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
                                1 - tolerance):
                            counts_out_num += 1

                            print(
                                '--> No.{:d}: Current counts {:0.2f}kcps is out of range [{:0.2f}kcps, {:0.2f}kcps].'.format(
                                    counts_out_num, current_counts_kcps, ref_counts * (1 - tolerance),
                                                                         ref_counts * (1 + tolerance)))

                            if counts_out_num > 5:
                                print('** Start tracking **')
                                self.qm.set_io1_value(True)
                                self.NV_tracking()
                                try:
                                    self.qm.set_io1_value(False)
                                except Exception as e:
                                    print('** ATTENTION in set_io1_value **')
                                    print(e)
                                else:
                                    counts_out_num = 0
                                    ref_counts = self.settings['NV_tracking']['ref_counts']

                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt() # Currently not implemented. Will be implemented in future releases.
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(RamseyQM_v2, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
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

        if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            axes_list[1].clear()
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="+x")
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="-x")
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="+y")
            axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="-y")
            axes_list[1].set_xlabel('Total tau [ns]')
            axes_list[1].set_ylabel('Normalized Counts')
            axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

            axes_list[0].clear()
            axes_list[0].plot(data['t_vec'], data['signal_norm'], label="cosine")
            axes_list[0].plot(data['t_vec'], data['signal2_norm'], label="sine")
            axes_list[0].plot(data['t_vec'], data['signal_ramsey'], label="squared sum root")
            axes_list[0].set_xlabel('Total tau [ns]')
            axes_list[0].set_ylabel('Contrast')

            axes_list[0].legend(loc='upper right')
            axes_list[0].set_title(
                'Ramsey\nRef fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz'.format(
                    data['ref_cnts'], int(data['rep_num']),
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                    self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['detuning'] * 1e-6)
            )


    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
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
            axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class Ramsey_SingleTau(Script):
    """
        This script runs a Ramsey measurement on an NV center at a fixed tau
        Readout both the cosine and sine components (i.e. along x and y axis).
        No NV tracking option.
        - Ziwei Qiu 1/19/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'resonant IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('detuning', 1e6, float, 'detuning for Ramsey experiment in Hz]'),
            Parameter('pi_half_pulse_time', 50, float, 'time duration of a pi/2 pulse (in ns)'),
        ]),
        Parameter('tau', 1150, int, 'time between the two pi/2 pulses (in ns)'),
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
                # pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                # pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

                # unit: ns
                # config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
                # config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

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

                self.t = round(self.settings['tau'] / 4) # unit: in 4ns cycle
                self.t = int(np.max([self.t, 4])) # unit: in 4ns cycle
                self.tau = self.t * 4 # in ns
                print('tau [ns]: ', self.tau)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
                IF_freq = self.settings['mw_pulses']['IF_frequency'] - self.settings['mw_pulses']['detuning']

                # define the qua program
                with program() as ramsey:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    n = declare(int)
                    k = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_(k, 0, k < 4, k + 1):
                            reset_frame('qubit')
                            with if_(k == 0):  # +x readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(self.t, 'qubit')
                                frame_rotation(np.pi * 0.0, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 1):  # -x readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(self.t, 'qubit')
                                frame_rotation(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 2):  # +y readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(self.t, 'qubit')
                                frame_rotation(np.pi * 0.5, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 3):  # -y readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(self.t, 'qubit')
                                frame_rotation(np.pi * 1.5, 'qubit')
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
                    self._qm_simulation(ramsey)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(ramsey, job_stop)
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
            counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
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


                    self.data['signal_ramsey'] = np.sqrt(self.data['signal_norm'] ** 2 + self.data['signal2_norm'] ** 2)
                    self.data['phase'] = np.arccos(self.data['signal_norm'] / self.data['signal_ramsey']) * np.sign(
                        self.data['signal2_norm'])

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

                time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(Ramsey_SingleTau, self).plot([figure_list[0], figure_list[1]])

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
            axes_list[0].set_xlabel('Ramsey tau [ns]')
            axes_list[0].set_ylabel('Normalized Counts')
            axes_list[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

            if title:
                axes_list[0].set_title(
                    'Ramsey\nRef fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz\ntau: {:2.1f}ns, Ramsey signal: {:0.2f}, phase: {:0.2f} rad'.format(
                        data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['detuning'] * 1e-6,
                        data['tau'], data['signal_ramsey'], data['phase'])
                )
            else:
                axes_list[0].set_title('Ramsey: {:0.1f}kcps'.format(data['ref_cnts']))

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


class DCSensing(Script):
    """
        This script performs DC sensing calibration based on a single tau Ramsey measurement.
        The DC voltage is provided by an arbitrary function generator.
        Readout both the cosine and sine components (i.e. along x and y axis).

        - Ziwei Qiu 1/20/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('sweep', [
            Parameter('min_vol', 0, float, 'define the minimum DC voltage [V]'),
            Parameter('max_vol', 10.0, float, 'define the maximum DC voltage [V]'),
            Parameter('vol_step', 0.5, float, 'define the DC voltage step [V]')
        ]),
        Parameter('tracking_settings', [Parameter('track_focus', False, bool,
                                                  'check to use optimize to track to the NV'),
                                        Parameter('track_focus_every_N', 5, int, 'track every N points')]
                  )
    ]
    _INSTRUMENTS = {'afg': Agilent33120A}
    _SCRIPTS = {'ramsey': Ramsey_SingleTau, 'optimize': optimize}

    def __init__(self, scripts, name=None, settings=None, instruments=None, log_function=None, timeout=1000000000,
                 data_path=None):

        Script.__init__(self, name, scripts=scripts, settings=settings, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _get_voltage_array(self):
        min_vol = self.settings['sweep']['min_vol']
        max_vol = self.settings['sweep']['max_vol']
        vol_step = self.settings['sweep']['vol_step']

        if min_vol < -10:
            min_vol = -10.0
        if max_vol > 10:
            max_vol = 10.0

        self.sweep_array = np.arange(min_vol, max_vol, vol_step)

    def do_ramsey(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts['ramsey'].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts['ramsey'].settings['tag'] = label
        elif index >= 0:
            self.scripts['ramsey'].settings['tag'] = 'ramsey_ind' + str(index)
        else:
            self.scripts['ramsey'].settings['tag'] = 'ramsey'

        if verbose:
            print('==> Start measuring Ramsey...')

        self.scripts['ramsey'].settings['to_do'] = 'execution'

        try:
            self.scripts['ramsey'].run()
        except Exception as e:
            print('** ATTENTION in Ramsey **')
            print(e)
        else:
            ramsey_sig = self.scripts['ramsey'].data['signal_avg_vec']
            self.data['plus_x'].append(ramsey_sig[0])
            self.data['minus_x'].append(ramsey_sig[1])
            self.data['plus_y'].append(ramsey_sig[2])
            self.data['minus_y'].append(ramsey_sig[3])
            self.data['tau'].append(self.scripts['ramsey'].data['tau'])
            self.data['norm1'].append(self.scripts['ramsey'].data['signal_norm'])
            self.data['norm2'].append(self.scripts['ramsey'].data['signal2_norm'])
            self.data['squared_sum_root'].append(self.scripts['ramsey'].data['signal_ramsey'])
            self.data['phase'].append(self.scripts['ramsey'].data['phase'])

    def _function(self):
        afg = self.instruments['afg']['instance']
        afg.update({'output_load': 'INFinity'})
        afg.update({'wave_shape': 'DC'})
        afg.update({'burst_mod': False})

        self._get_voltage_array()

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
            # print('update:', vol)
            afg.update({'offset': float(vol)})
            try:
                do_tracking(index)
                self.do_ramsey(label='ramsey', index=index)
            except Exception as e:
                print('** ATTENTION in self.do_ramsey **')
                print(e)
            else:
                self.data['vol'].append(vol)

            self.progress = index * 100. / len(self.sweep_array)
            self.updateProgress.emit(int(self.progress))
            time.sleep(0.2)

        # return the AFG offset back to 0V
        afg.update({'offset': 0.0})

        # convert deque object to numpy array or list
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
            'Ramsey DC Sensing\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz\nRepetition number: {:d}, tau: {:2.1f}ns '.format(
                self.scripts['ramsey'].settings['mw_pulses']['mw_power'],
                self.scripts['ramsey'].settings['mw_pulses']['mw_frequency'] * 1e-9,
                self.scripts['ramsey'].settings['mw_pulses']['IF_amp'],
                self.scripts['ramsey'].settings['mw_pulses']['IF_frequency'] * 1e-6,
                self.scripts['ramsey'].settings['mw_pulses']['pi_half_pulse_time'],
                self.scripts['ramsey'].settings['mw_pulses']['detuning'] * 1e-6,
                self.scripts['ramsey'].settings['rep_num'], self.scripts['ramsey'].settings['tau'])
        )

        if self._current_subscript_stage['current_subscript'] == self.scripts['ramsey'] and self.scripts[
            'ramsey'].is_running:
            self.scripts['ramsey']._plot([axes_list[2]], title=False)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[2]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[2]])

        elif self._current_subscript_stage['current_subscript'] == self.scripts['ramsey'] and self.scripts[
            'ramsey'].is_running:
            self.scripts['ramsey']._update_plot([axes_list[2]], title=False)
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


class RamseySyncReadout(Script):
    """
        This script measures a time series of Ramsey signals synchronized with a external slowly oscillating electric field.
        This is effectively a classical lock-in measurement.
        The external electric field is applied from the function generator.

        - Ziwei Qiu 2/5/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('signal', [
            Parameter('frequency', 1000.0, float, 'oscillating frequency [Hz]'),
            Parameter('amplitude', 2.0, float, 'peak-to-peak amplitude of the oscillating voltage [V]'),
            Parameter('offset', 0.0, float, 'offset of the oscillating voltage [V]'),
            Parameter('wave_shape', 'SINusoid', ['SINusoid', 'SQUare'], 'wave shape of the oscillating signal')
        ]),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'resonant IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('detuning', 1e6, float, 'detuning for Ramsey experiment in Hz]'),
            Parameter('pi_half_pulse_time', 50, float, 'time duration of a pi/2 pulse (in ns)'),
        ]),
        Parameter('tau', 1000, int, 'time between the two pi/2 pulses (in ns)'),
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
    _INSTRUMENTS = {'afg': Agilent33120A, 'mw_gen_iq': SGS100ARFSource}
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
                pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
                # unit: ns
                config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
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

                self.t = round(self.settings['tau'] / 4) # unit: in 4ns cycle
                self.t = int(np.max([self.t, 4])) # unit: in 4ns cycle
                self.tau = self.t * 4 # in ns
                print('tau [ns]: ', self.tau)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
                IF_freq = self.settings['mw_pulses']['IF_frequency'] - self.settings['mw_pulses']['detuning']

                if self.settings['ramsey_pts'] < 0:
                    ramsey_meas_time = (nv_reset_time + laser_off + delay_mw_readout + self.t + pi2_time * 2) * 4 + 250 # in ns
                    ramsey_pts = round(1e9/self.settings['signal']['frequency']/ramsey_meas_time)
                else:
                    ramsey_pts = self.settings['ramsey_pts']
                self.ramsey_pts = ramsey_pts

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
                    # save(n, rep_num_st)
                    # save(total_counts, "total_counts")

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
                    k = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_(k, 0, k < 4, k + 1):
                            reset_frame('qubit')
                            if self.settings['to_do'] == 'execution':
                                wait_for_trigger('qubit')
                            with if_(k == 0):  # +x readout
                                align('qubit', 'laser', 'readout1', 'readout2')
                                play('trig', 'laser', duration=nv_reset_time)
                                align('qubit', 'laser', 'readout1', 'readout2')
                                wait(laser_off, 'qubit')
                                with for_(m, 0, m < ramsey_pts, m + 1):
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(self.t, 'qubit')
                                    frame_rotation(np.pi * 0.0, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    readout_save()

                            with if_(k == 1):  # -x readout
                                align('qubit', 'laser', 'readout1', 'readout2')
                                play('trig', 'laser', duration=nv_reset_time)
                                align('qubit', 'laser', 'readout1', 'readout2')
                                wait(laser_off, 'qubit')
                                with for_(m, 0, m < ramsey_pts, m + 1):
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(self.t, 'qubit')
                                    frame_rotation(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    readout_save()

                            with if_(k == 2):  # +y readout
                                align('qubit', 'laser', 'readout1', 'readout2')
                                play('trig', 'laser', duration=nv_reset_time)
                                align('qubit', 'laser', 'readout1', 'readout2')
                                wait(laser_off, 'qubit')
                                with for_(m, 0, m < ramsey_pts, m + 1):
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(self.t, 'qubit')
                                    frame_rotation(np.pi * 0.5, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    readout_save()

                            with if_(k == 3):  # -y readout
                                align('qubit', 'laser', 'readout1', 'readout2')
                                play('trig', 'laser', duration=nv_reset_time)
                                align('qubit', 'laser', 'readout1', 'readout2')
                                wait(laser_off, 'qubit')
                                with for_(m, 0, m < ramsey_pts, m + 1):
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(self.t, 'qubit')
                                    frame_rotation(np.pi * 1.5, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    readout_save()

                            save(n, rep_num_st)


                    with stream_processing():
                        total_counts_st.buffer(4, ramsey_pts).average().save("live_data")
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

        self.instruments['afg']['instance'].update({'output_load': 'INFinity'})
        self.instruments['afg']['instance'].update({'frequency': self.settings['signal']['frequency']})
        self.instruments['afg']['instance'].update({'amplitude': self.settings['signal']['amplitude']})
        self.instruments['afg']['instance'].update({'offset': self.settings['signal']['offset']})
        self.instruments['afg']['instance'].update({'wave_shape': self.settings['signal']['wave_shape']})
        self.instruments['afg']['instance'].update({'burst_mod': False})
        print('Function generator is ready.')

        try:
            job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
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
                         'rep_num': None, 'ramsey_pts': self.ramsey_pts}

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
                            fit_sine_amplitude(np.arange(self.ramsey_pts) * us_per_pt + first_pt_us, self.data['phase'])
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

        self.instruments['afg']['instance'].update({'offset': 0.0})
        self.instruments['afg']['instance'].update({'wave_shape': 'DC'})
        print('Function generator is outputting 0V dc.')

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
            x_array = np.arange(len(data['signal_avg_vec'][0, :]))

            if title: # if title is False, then no need to plot on axes_list in a scanning measurmenet
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
            if title:
                axes_list[0].legend(loc='upper right')

            axes_list[1].clear()
            if data['phase'] is not None:
                axes_list[1].plot(x_array, data['phase'])
            if self.settings['fit'] and 'fits' in data.keys():
                us_per_pt = 4
                first_pt_us = 4
                if title:
                    axes_list[1].plot(x_array, sine(x_array * us_per_pt + first_pt_us, data['fits'][0],
                                                    data['fits'][2] / 180 * np.pi, data['fits'][4] / 1000 * 2 * np.pi,
                                                    data['fits'][5]), lw=2,
                                      label='sinusoidal fit\n A={:0.2f}+/-{:0.2f} rad\n phi={:0.2f}+/-{:0.2f} deg\n freq={:0.1f}kHz'.format(
                                          data['fits'][0],
                                          data['fits'][1],
                                          data['fits'][2],
                                          data['fits'][3],
                                          data['fits'][4]))
                else:
                    axes_list[1].plot(x_array, sine(x_array * us_per_pt + first_pt_us, data['fits'][0],
                                                    data['fits'][2] / 180 * np.pi, data['fits'][4] / 1000 * 2 * np.pi,
                                                    data['fits'][5]), lw=2,
                                      label='A={:0.2f}rad, phi={:0.2f}deg'.format(data['fits'][0], data['fits'][2]))

                axes_list[1].legend(loc='upper right')
            if not title:
                axes_list[1].set_xlabel('Time [4us]')
            axes_list[1].set_xlabel('Time [4us]')
            axes_list[1].set_ylabel('Phase [rad]')

            if title:
                axes_list[0].set_title(
                    'Ramsey Lock-In Measurement\nRef fluor: {:0.1f}kcps, Repetition number: {:d}, tau: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nπ/2 time: {:2.1f}ns, detuning: {:0.4f}MHz\nsignal: freq: {:0.2f}kHz, amp: {:0.2f}Vpp, offset: {:0.2f}V, wave: {:s}'.format(
                        data['ref_cnts'], int(data['rep_num']), self.settings['tau'],
                        self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['detuning'] * 1e-6,
                                                                self.settings['signal']['frequency'] * 1e-3,
                        self.settings['signal']['amplitude'],
                        self.settings['signal']['offset'], self.settings['signal']['wave_shape']
                    )
                )
            else:
                axes_list[0].set_title('{:0.1f}kcps'.format(data['ref_cnts']))

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
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[2]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])

        return axes_list




if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'RamseyQM': 'RamseyQM'}, script, instr)

    print(script)
    print(failed)
    print(instr)