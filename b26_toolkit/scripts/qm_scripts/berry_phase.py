from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import frame_rotation as z_rot
from qm import SimulationConfig
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config

import numpy as np
import time

from b26_toolkit.instruments import SGS100ARFSource
from b26_toolkit.scripts.optimize import OptimizeNoLaser


class BerryPhaseAttempt(Script):
    """
        This script attempts to measure the NV geometric phase under a perpendicular magnetic field.
        Only spin-echo sequence is used.
        Tau is fixed and the currents are swept.
        - Ziwei Qiu 12/21/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution'],
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('voltage_pulses', [
            Parameter('frequency', 1e6, float, 'frequency [Hz] of the current modulation'),
            Parameter('gap_to_RF', 200, int, 'the gap [ns] between voltage and RF pulses to avoid overlap'),
        ]),
        Parameter('sweep', [
            Parameter('num_of_pts', 15, int, 'number of points to be swept'),
            Parameter('voltage1_min', 0.01, float, 'define the minimum voltage1 [V], into 50ohm load, <0.5V'),
            Parameter('voltage1_max', 0.4, float, 'define the maximum voltage1 [V], into 50ohm load, <0.5V'),
            Parameter('voltage2_min', 0.01, float, 'define the minimum voltage2 [V], into 50ohm load, <0.5V'),
            Parameter('voltage2_max', 0.4, float, 'define the maximum voltage2 [V], into 50ohm load, <0.5V')
        ]),
        # Parameter('decoupling_seq', [
        #     Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
        #               'type of dynamical decoupling sequences'),
        #     Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        # ]),
        # Parameter('sensing_type', 'cosine', ['cosine', 'sine', 'both'], 'choose the sensing type'),
        Parameter('read_out', [
            Parameter('meas_len', 200, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 400, int,
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
        Parameter('rep_num', 200000, int, 'define the repetition number, suggest at least 100000'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')

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

    def _get_voltage_array(self):
        self.gate1_list = np.linspace(self.settings['sweep']['voltage1_min'],
                                         self.settings['sweep']['voltage1_max'],
                                         self.settings['sweep']['num_of_pts'])
        self.gate2_list = np.linspace(self.settings['sweep']['voltage2_min'],
                                         self.settings['sweep']['voltage2_max'],
                                         self.settings['sweep']['num_of_pts'])

    def _function(self):
        try:
            # unit: cycle of 4ns
            pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
            pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
            pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

            # unit: ns
            config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
            config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
            config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

            # Make qe1 and qe2 to be non-sticky elements
            config['elements']['qe1'].pop('hold_offset', None)
            config['elements']['qe2'].pop('hold_offset', None)

            config['elements']['qe1']['intermediate_frequency'] = self.settings['voltage_pulses']['frequency']
            config['elements']['qe2']['intermediate_frequency'] = self.settings['voltage_pulses']['frequency']

            self._get_voltage_array()
            gate1_config = np.max(np.abs(self.gate1_list))
            gate2_config = np.max(np.abs(self.gate2_list))

            config['waveforms']['const_gate1']['sample'] = gate1_config
            config['waveforms']['const_gate2']['sample'] = gate2_config

            self.gate_list = ((np.array(self.gate1_list) / gate1_config).tolist(),
                              (np.array(self.gate2_list) / gate2_config).tolist())
            print(self.gate_list)
            self.gate_num = len(self.gate1_list)

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

            gap_to_RF = round(self.settings['voltage_pulses']['gap_to_RF'] / 4)

            # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
            t = round(self.settings['tau']  / 2 / 4 - pi2_time / 2 - pi_time / 2)
            t = int(np.max([t, 4 + 2*gap_to_RF]))

            # total evolution time in ns
            self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8

            print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
            print('Total evolution times [ns]: ', self.tau_total)

            res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

            IF_freq = self.settings['mw_pulses']['IF_frequency']

            # define the qua program
            with program() as berry_phase_sensing:

                update_frequency('qubit', IF_freq)
                result1 = declare(int, size=res_len)
                counts1 = declare(int, value=0)
                result2 = declare(int, size=res_len)
                counts2 = declare(int, value=0)
                total_counts = declare(int, value=0)

                g1 = declare(fixed)
                g2 = declare(fixed)
                n = declare(int)
                k = declare(int)

                # the following variable is used to flag tracking
                assign(IO1, False)

                total_counts_st = declare_stream()
                rep_num_st = declare_stream()

                with for_(n, 0, n < rep_num, n + 1):
                    # Check if tracking is called
                    with while_(IO1):
                        play('trig', 'laser', duration=10000)

                    with for_each_((g1, g2), self.gate_list):
                        with for_(k, 0, k < 4, k + 1):

                            reset_frame('qubit')
                            with if_(k == 0):  # +x readout
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                z_rot(np.pi / 2, 'qubit')
                                play('pi' * amp(IF_amp), 'qubit')  # pi_y

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 1):  # +x readout
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                z_rot(np.pi / 2, 'qubit')
                                play('pi' * amp(IF_amp), 'qubit')  # pi_y

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                z_rot(-np.pi / 2, 'qubit')
                                z_rot(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                            with if_(k == 2):  # +x readout
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'qe1', 'qe2')  # gate off

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time


                                reset_phase('qe1')
                                reset_phase('qe2')
                                reset_frame('qe1', 'qe2')
                                z_rot(-0.5 * np.pi, 'qe1')
                                z_rot(np.pi, 'qe2')
                                wait(gap_to_RF, 'qe1', 'qe2')
                                play('gate1' * amp(g1), 'qe1', duration=t - 2*gap_to_RF)  # gate 1
                                play('gate2' * amp(g2), 'qe2', duration=t - 2*gap_to_RF)  # gate 2
                                wait(gap_to_RF, 'qe1', 'qe2')

                                z_rot(np.pi / 2, 'qubit')
                                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                                wait(pi_time, 'qe1', 'qe2')  # gate off

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                reset_phase('qe1')
                                reset_phase('qe2')
                                reset_frame('qe1', 'qe2')
                                z_rot(0.5 * np.pi, 'qe1')
                                z_rot(np.pi, 'qe2')
                                wait(gap_to_RF, 'qe1', 'qe2')
                                play('gate1' * amp(g1), 'qe1', duration=t - 2*gap_to_RF)  # gate 1
                                play('gate2' * amp(g2), 'qe2', duration=t - 2*gap_to_RF)  # gate 2
                                wait(gap_to_RF, 'qe1', 'qe2')

                                z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'qe1', 'qe2')  # gate off
                            with if_(k == 3):  # +x readout
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                reset_phase('qe1')
                                reset_phase('qe2')
                                reset_frame('qe1', 'qe2')
                                z_rot(-0.5 * np.pi, 'qe1')
                                z_rot(np.pi, 'qe2')
                                wait(gap_to_RF, 'qe1', 'qe2')
                                play('gate1' * amp(g1), 'qe1', duration=t-2*gap_to_RF)  # gate 1
                                play('gate2' * amp(g2), 'qe2', duration=t-2*gap_to_RF)  # gate 2
                                wait(gap_to_RF, 'qe1', 'qe2')

                                z_rot(np.pi / 2, 'qubit')
                                play('pi' * amp(IF_amp), 'qubit')  # pi_y

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                reset_phase('qe1')
                                reset_phase('qe2')
                                reset_frame('qe1', 'qe2')
                                z_rot(0.5 * np.pi, 'qe1')
                                z_rot(np.pi, 'qe2')
                                wait(gap_to_RF, 'qe1', 'qe2')
                                play('gate1' * amp(g1), 'qe1', duration=t-2*gap_to_RF)  # gate 1
                                play('gate2' * amp(g2), 'qe2', duration=t-2*gap_to_RF)  # gate 2
                                wait(gap_to_RF, 'qe1', 'qe2')

                                z_rot(-np.pi / 2, 'qubit')
                                z_rot(np.pi, 'qubit')
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
                    total_counts_st.buffer(self.settings['sweep']['num_of_pts'], 4).average().save("live_data")
                    total_counts_st.buffer(tracking_num).save("current_counts")
                    rep_num_st.save("live_rep_num")

            with program() as job_stop:
                play('trig', 'laser', duration=10)

            if self.settings['to_do'] == 'simulation':
                self._qm_simulation(berry_phase_sensing)
            elif self.settings['to_do'] == 'execution':
                self._qm_execution(berry_phase_sensing, job_stop)
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
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            self.data = {'t_vec': self.gate1_list, 't_vec2': self.gate2_list, 'tau': self.tau_total,
                         'signal_avg_vec': None, 'ref_cnts': None, 'sig1_norm': None, 'sig2_norm': None,
                         'rep_num': None}

            ref_counts = -1
            tolerance = self.settings['NV_tracking']['tolerance']

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

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
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
                                    print('** ATTENTION **')
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

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(BerryPhaseAttempt, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data
        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[1].clear()
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")

                axes_list[1].set_xlabel('Voltage [V]')
                axes_list[1].set_ylabel('Normalized Counts')
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].clear()
                axes_list[0].plot(data['t_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['t_vec'], data['sig2_norm'], label="sig2_norm")
                axes_list[0].set_xlabel('Voltage1 [V]')
                axes_list[0].set_ylabel('Contrast')

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'Berry Phase Attempt (type: echo)\nvol1: [{:0.3}, {:0.3}] V, vol2: [{:0.3}, {:0.3}] V, {:d} points, mod freq = {:0.2f}MHz \ntau = {:0.3}us, Ref fluor: {:0.1f}kcps, Repetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['sweep']['voltage1_min'], self.settings['sweep']['voltage1_max'],
                        self.settings['sweep']['voltage2_min'], self.settings['sweep']['voltage2_max'],
                        self.settings['sweep']['num_of_pts'], self.settings['voltage_pulses']['frequency'] * 1e-6,
                        data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                        self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
                print('here')
                print(e)

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
            # axes_list.append(axes_list[1].twinx())             # axes_list[2]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
            # axes_list.append(figure_list[1].axes[1])
        return axes_list


class BerryPhaseSweepCurrent(Script):
    """
    This script measures the NV geometric phase under a perpendicular magnetic field.
    Tau is fixed and the currents are swept.
    - Ziwei Qiu 12/9/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution'],
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('ramps', [
            Parameter('ramping_time', 800, float, 'ramping time in [ns]'),
            Parameter('hold_time', 1600, float, 'the duration the voltage will be held high in [ns]'),
            Parameter('delay', 1200, float,
                      'delay between the two channels in [ns]. delay should be > ramping_time to accumulate a solid angle')
        ]),
        Parameter('sweep', [
            Parameter('num_of_pts', 15, int, 'number of points to be swept'),
            Parameter('voltage1_min', 0.01, float, 'define the minimum voltage1 [V], into 50ohm load'),
            Parameter('voltage1_max', 0.4, float, 'define the maximum voltage1 [V], into 50ohm load'),
            Parameter('voltage2_min', 0.01, float, 'define the minimum voltage2 [V], into 50ohm load'),
            Parameter('voltage2_max', 0.4, float, 'define the maximum voltage2 [V], into 50ohm load')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('sensing_type', 'cosine', ['cosine', 'sine', 'both'], 'choose the sensing type'),
        Parameter('read_out', [
            Parameter('meas_len', 200, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 400, int,
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
        Parameter('rep_num', 200000, int, 'define the repetition number, suggest at least 100000'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')

    ]

    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {'optimize': OptimizeNoLaser}

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
        try:
            # unit: cycle of 4ns
            pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
            pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
            pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

            ramping_time = round(self.settings['ramps']['ramping_time'] / 4)
            hold_time = round(self.settings['ramps']['hold_time'] / 4)
            delay = round(self.settings['ramps']['delay'] / 4)
            solid_angle_accumulation_time = 2*(ramping_time * 2 + hold_time + delay)

            # unit: ns
            config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
            config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
            config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

            # Make qe1 and qe2 to be sticky elements
            config['elements']['qe1']['hold_offset'] = {'duration': 10}
            config['elements']['qe2']['hold_offset'] = {'duration': 10}

            self.voltage1_list = np.linspace(self.settings['sweep']['voltage1_min'],
                                             self.settings['sweep']['voltage1_max'],
                                             self.settings['sweep']['num_of_pts'])
            ramp1_rate_list = self.voltage1_list / ramping_time / 4 # in V/ns

            self.voltage2_list = np.linspace(self.settings['sweep']['voltage2_min'],
                                             self.settings['sweep']['voltage2_max'],
                                             self.settings['sweep']['num_of_pts'])
            ramp2_rate_list = self.voltage2_list / ramping_time / 4 # in V/ns

            self.ramp_rate_list = (ramp1_rate_list.tolist(), ramp2_rate_list.tolist())

            print(self.ramp_rate_list)

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

            num_of_evolution_blocks = 0
            number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
            if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
                'type'] == 'CPMG':
                num_of_evolution_blocks = 1 * number_of_pulse_blocks
            elif self.settings['decoupling_seq']['type'] == 'XY4':
                num_of_evolution_blocks = 4 * number_of_pulse_blocks
            elif self.settings['decoupling_seq']['type'] == 'XY8':
                num_of_evolution_blocks = 8 * number_of_pulse_blocks

            # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
            t = round(self.settings['tau'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
            t = int(np.max([t, 8 + solid_angle_accumulation_time]))

            # used in the first evolution block
            wait_after_rf_half = round((t - solid_angle_accumulation_time) / 2)
            wait_before_rf_half = t - solid_angle_accumulation_time - wait_after_rf_half

            # used in the pi pulse train
            wait_after_rf = round((2 * t + pi2_time - solid_angle_accumulation_time) / 2)
            wait_before_rf = 2 * t + pi2_time - solid_angle_accumulation_time - wait_after_rf

            # total evolution time in ns
            self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

            print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
            print('Total evolution times [ns]: ', self.tau_total)

            res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

            IF_freq = self.settings['mw_pulses']['IF_frequency']

            def berry_phase_accumulation(sign=1):
                if sign == 1:
                    play(ramp(r1), 'qe1', duration=ramping_time)
                    wait(hold_time, 'qe1')
                    ramp_to_zero('qe1', ramping_time)
                    wait(delay, 'qe1')
                    play(ramp(r1), 'qe1', duration=ramping_time)
                    wait(hold_time, 'qe1')
                    ramp_to_zero('qe1', ramping_time)
                    wait(delay, 'qe1')

                    wait(delay, 'qe2')
                    play(ramp(r2), 'qe2', duration=ramping_time)
                    wait(hold_time, 'qe2')
                    ramp_to_zero('qe2', ramping_time)
                    wait(delay, 'qe2')
                    play(ramp(-r2), 'qe2', duration=ramping_time)
                    wait(hold_time, 'qe2')
                    ramp_to_zero('qe2', ramping_time)
                else:
                    wait(delay, 'qe1')
                    play(ramp(r1), 'qe1', duration=ramping_time)
                    wait(hold_time, 'qe1')
                    ramp_to_zero('qe1', ramping_time)
                    wait(delay, 'qe1')
                    play(ramp(r1), 'qe1', duration=ramping_time)
                    wait(hold_time, 'qe1')
                    ramp_to_zero('qe1', ramping_time)

                    play(ramp(r2), 'qe2', duration=ramping_time)
                    wait(hold_time, 'qe2')
                    ramp_to_zero('qe2', ramping_time)
                    wait(delay, 'qe2')
                    play(ramp(-r2), 'qe2', duration=ramping_time)
                    wait(hold_time, 'qe2')
                    ramp_to_zero('qe2', ramping_time)
                    wait(delay, 'qe2')

            def xy4_block(is_last_block, IF_amp=IF_amp, current=False):
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation()
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(-np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                if is_last_block:
                    wait(t, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf_half, 'qe1')
                        wait(wait_after_rf_half, 'qe2')
                        berry_phase_accumulation()
                        wait(wait_before_rf_half, 'qe1')
                        wait(wait_before_rf_half, 'qe2')
                else:
                    wait(2 * t + pi2_time, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf, 'qe1')
                        wait(wait_after_rf, 'qe2')
                        berry_phase_accumulation()
                        wait(wait_before_rf, 'qe1')
                        wait(wait_before_rf, 'qe2')

            def xy8_block(is_last_block, IF_amp=IF_amp, current=False):
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation()
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(-np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation()
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(-np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation()
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_y
                wait(2 * t + pi2_time, 'qubit')
                if current:
                    wait(pi_time, 'qe1')  # current1 off
                    wait(pi_time, 'qe2')  # current2 off
                    wait(wait_after_rf, 'qe1')
                    wait(wait_after_rf, 'qe2')
                    berry_phase_accumulation(sign=-1)
                    wait(wait_before_rf, 'qe1')
                    wait(wait_before_rf, 'qe2')

                z_rot(-np.pi / 2, 'qubit')
                play('pi' * amp(IF_amp), 'qubit')  # pi_x
                if is_last_block:
                    wait(t, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf_half, 'qe1')
                        wait(wait_after_rf_half, 'qe2')
                        berry_phase_accumulation()
                        wait(wait_before_rf_half, 'qe1')
                        wait(wait_before_rf_half, 'qe2')
                else:
                    wait(2 * t + pi2_time, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf, 'qe1')
                        wait(wait_after_rf, 'qe2')
                        berry_phase_accumulation()
                        wait(wait_before_rf, 'qe1')
                        wait(wait_before_rf, 'qe2')

            def spin_echo_block(is_last_block, IF_amp=IF_amp, current=False, sign=1):
                play('pi' * amp(IF_amp), 'qubit')
                if is_last_block:
                    wait(t, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf_half, 'qe1')
                        wait(wait_after_rf_half, 'qe2')
                        berry_phase_accumulation(sign=sign)
                        wait(wait_before_rf_half, 'qe1')
                        wait(wait_before_rf_half, 'qe2')
                else:
                    wait(2 * t + pi2_time, 'qubit')
                    if current:
                        wait(pi_time, 'qe1')  # current1 off
                        wait(pi_time, 'qe2')  # current2 off
                        wait(wait_after_rf, 'qe1')
                        wait(wait_after_rf, 'qe2')
                        berry_phase_accumulation(sign=sign)
                        wait(wait_before_rf, 'qe1')
                        wait(wait_before_rf, 'qe2')

            def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                               number_of_pulse_blocks=number_of_pulse_blocks, current=False):
                if decoupling_seq_type == 'XY4':
                    if number_of_pulse_blocks == 2:
                        xy4_block(is_last_block=False, current=current)
                    elif number_of_pulse_blocks > 2:
                        with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                            xy4_block(is_last_block=False, current=current)
                    xy4_block(is_last_block=True, current=current)
                elif decoupling_seq_type == 'XY8':
                    if number_of_pulse_blocks == 2:
                        xy8_block(is_last_block=False, current=current)
                    elif number_of_pulse_blocks > 2:
                        with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                            xy8_block(is_last_block=False, current=current)
                    xy8_block(is_last_block=True, current=current)
                else:
                    if number_of_pulse_blocks == 2:
                        spin_echo_block(is_last_block=False, current=current, sign=-1)
                    elif number_of_pulse_blocks > 2:
                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                spin_echo_block(is_last_block=False, current=current, sign=-1)
                                spin_echo_block(is_last_block=False, current=current, sign=1)
                        else:
                            with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                spin_echo_block(is_last_block=False, current=current, sign=-1)
                                spin_echo_block(is_last_block=False, current=current, sign=1)
                            spin_echo_block(is_last_block=False, current=current, sign=-1)
                    if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                        spin_echo_block(is_last_block=True, current=current, sign=-1)
                    else:  # even number of pi pulses
                        spin_echo_block(is_last_block=True, current=current, sign=1)

            # define the qua program
            with program() as berry_phase_sensing:

                update_frequency('qubit', IF_freq)
                result1 = declare(int, size=res_len)
                counts1 = declare(int, value=0)
                result2 = declare(int, size=res_len)
                counts2 = declare(int, value=0)
                total_counts = declare(int, value=0)

                r1 = declare(fixed)
                r2 = declare(fixed)
                n = declare(int)
                k = declare(int)
                i = declare(int)

                # the following variable is used to flag tracking
                assign(IO1, False)

                total_counts_st = declare_stream()
                rep_num_st = declare_stream()

                with for_(n, 0, n < rep_num, n + 1):
                    # Check if tracking is called
                    with while_(IO1):
                        play('trig', 'laser', duration=10000)

                    with for_each_((r1, r2), self.ramp_rate_list):
                        with for_(k, 0, k < 4, k + 1):
                            reset_frame('qubit')

                            with if_(k == 0):  # +x readout, no current
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                if self.settings['sensing_type'] == 'both':
                                    wait(wait_after_rf_half, 'qe1')
                                    wait(wait_after_rf_half, 'qe2')
                                    berry_phase_accumulation()
                                    wait(wait_before_rf_half, 'qe1')
                                    wait(wait_before_rf_half, 'qe2')
                                    pi_pulse_train(current=True)
                                else:
                                    wait(t, 'qe1')  # current1 off
                                    wait(t, 'qe2')  # current2 off
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                            with if_(k == 1):  # -x readout, no E field
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                if self.settings['sensing_type'] == 'both':
                                    wait(wait_after_rf_half, 'qe1')
                                    wait(wait_after_rf_half, 'qe2')
                                    berry_phase_accumulation()
                                    wait(wait_before_rf_half, 'qe1')
                                    wait(wait_before_rf_half, 'qe2')
                                    pi_pulse_train(current=True)
                                else:
                                    wait(t, 'qe1')  # current1 off
                                    wait(t, 'qe2')  # current2 off
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')
                                z_rot(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                            with if_(k == 2):  # +x readout, with E field
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                wait(wait_after_rf_half, 'qe1')
                                wait(wait_after_rf_half, 'qe2')
                                berry_phase_accumulation()
                                wait(wait_before_rf_half, 'qe1')
                                wait(wait_before_rf_half, 'qe2')
                                pi_pulse_train(current=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')

                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                            with if_(k == 3):  # -x readout, with E field
                                align('qubit', 'qe1', 'qe2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                wait(wait_after_rf_half, 'qe1')
                                wait(wait_after_rf_half, 'qe2')
                                berry_phase_accumulation()
                                wait(wait_before_rf_half, 'qe1')
                                wait(wait_before_rf_half, 'qe2')
                                pi_pulse_train(current=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                z_rot(np.pi, 'qubit')
                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'qe1')  # current1 off
                                wait(pi2_time, 'qe2')  # current2 off

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
                    total_counts_st.buffer(self.settings['sweep']['num_of_pts'], 4).average().save("live_data")
                    total_counts_st.buffer(tracking_num).save("current_counts")
                    rep_num_st.save("live_rep_num")

            with program() as job_stop:
                play('trig', 'laser', duration=10)

            if self.settings['to_do'] == 'simulation':
                self._qm_simulation(berry_phase_sensing)
            elif self.settings['to_do'] == 'execution':
                self._qm_execution(berry_phase_sensing, job_stop)
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
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("live_data")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            self.data = {'t_vec': self.voltage1_list, 't_vec2': self.voltage2_list, 'tau': self.tau_total,
                         'signal_avg_vec': None, 'ref_cnts': None, 'sig1_norm': None, 'sig2_norm': None,
                         'rep_num': None}

            ref_counts = -1
            tolerance = self.settings['NV_tracking']['tolerance']

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

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
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
                                    print('** ATTENTION **')
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

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(BerryPhaseSweepCurrent, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data
        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[1].clear()
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")

                axes_list[1].set_xlabel('Voltage [V]')
                axes_list[1].set_ylabel('Normalized Counts')
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].clear()
                axes_list[0].plot(data['t_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['t_vec'], data['sig2_norm'], label="sig2_norm")
                axes_list[0].set_xlabel('Voltage1 [V]')
                axes_list[0].set_ylabel('Contrast')

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'Berry Phase Sensing (type: {:s})\nvol1: [{:0.3}, {:0.3}] V, vol2: [{:0.3}, {:0.3}] V, {:d} points\n{:s} {:d} block(s), tau = {:0.3}us, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['sensing_type'],
                        self.settings['sweep']['voltage1_min'], self.settings['sweep']['voltage1_max'],
                        self.settings['sweep']['voltage2_min'], self.settings['sweep']['voltage2_max'],
                        self.settings['sweep']['num_of_pts'],
                        self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                        self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
                print('here')
                print(e)

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
            # axes_list.append(axes_list[1].twinx())             # axes_list[2]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
            # axes_list.append(figure_list[1].axes[1])
        return axes_list


class BerryRabi(Script):
    """ This is a test script potentially to drive Rabi in the |+>-|-> subspace"""
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'execution', ['simulation', 'execution'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)')
        ]),
        Parameter('voltage_pulses', [
            Parameter('amplitude', 0.2, float, 'the bias voltage [V] for applying the current, <0.5V'),
            Parameter('gap_to_RF', 1000, int, 'the gap [ns] between votlage and RF pulses to avoid overlap'),
        ]),

        Parameter('tau_times', [
            Parameter('min_time', 16, int, 'minimum time for rabi oscillations (in ns), >=16ns'),
            Parameter('max_time', 500, int, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 12, int,
                      'time step increment of rabi pulse duration (in ns), using multiples of 4ns')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 200, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 440, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV if the counts out of the reference range'),
            Parameter('tracking_num', 20000, int, 'number of recent APD windows used for calculating current counts'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.28, float, 'define the reference range (1+/-tolerance)*ref')
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
        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
            gap_to_RF = round(self.settings['voltage_pulses']['gap_to_RF'] / 4)
            # unit: ns
            config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
            config['waveforms']['const_gate']['sample'] = self.settings['voltage_pulses']['amplitude']

            rep_num = self.settings['rep_num']
            tracking_num = self.settings['NV_tracking']['tracking_num']
            self.meas_len = round(self.settings['read_out']['meas_len'])

            delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
            nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
            laser_off = round(self.settings['read_out']['laser_off'] / 4)
            delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)

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

            res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

            # define the qua program
            with program() as berry_rabi:
                update_frequency('qubit', self.settings['mw_pulses']['IF_frequency'])
                result1 = declare(int, size=res_len)
                counts1 = declare(int, value=0)
                result2 = declare(int, size=res_len)
                counts2 = declare(int, value=0)
                total_counts = declare(int, value=0)
                t = declare(int)
                n = declare(int)

                # the following two variable are used to flag tracking
                assign(IO1, False)
                flag = declare(bool, value=False)

                total_counts_st = declare_stream()
                rep_num_st = declare_stream()

                with for_(n, 0, n < rep_num, n + 1):
                    # Check if tracking is called
                    assign(flag, IO1)
                    # with while_(flag):
                    #     play('trig', 'laser', duration=10000)
                    with if_(flag):
                        pause()
                    with for_each_(t, t_vec):
                        reset_frame('qubit')
                        align('qubit', 'bias_vol')
                        play('pi' * amp(IF_amp), 'qubit')
                        wait(pi_time, 'bias_vol')  # voltage off

                        wait(gap_to_RF, 'qubit','bias_vol')

                        play('const', 'bias_vol', duration= t)
                        wait(t, 'qubit')

                        wait(gap_to_RF, 'qubit', 'bias_vol')

                        play('pi' * amp(IF_amp), 'qubit')
                        wait(pi_time, 'bias_vol')  # voltage off

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
                    total_counts_st.buffer(t_num).average().save("live_rabi_data")
                    total_counts_st.buffer(tracking_num).save("current_counts")
                    rep_num_st.save("live_rep_num")
            with program() as job_stop:
                play('trig', 'laser', duration=10)

            if self.settings['to_do'] == 'simulation':
                self._qm_simulation(berry_rabi)
            elif self.settings['to_do'] == 'execution':
                self._qm_execution(berry_rabi, job_stop)
            self._abort = True

    def _qm_simulation(self, qua_program):
        start = time.time()
        job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
        end = time.time()
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

        job = self.qm.execute(qua_program)

        vec_handle = job.result_handles.get("live_rabi_data")
        progress_handle = job.result_handles.get("live_rep_num")
        tracking_handle = job.result_handles.get("current_counts")

        vec_handle.wait_for_values(1)
        progress_handle.wait_for_values(1)
        tracking_handle.wait_for_values(1)
        self.data = {'t_vec': np.array(self.t_vec) * 4, 'signal_avg_vec': None, 'ref_cnts': None, 'rep_num': None}

        current_rep_num = 0
        ref_counts = -1
        tolerance = self.settings['NV_tracking']['tolerance']

        while vec_handle.is_processing():
            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rabi_avg = vec * 1e6 / self.meas_len
                self.data.update({'signal_avg_vec': rabi_avg / rabi_avg[0], 'ref_cnts': rabi_avg[0]})

            try:
                current_rep_num = progress_handle.fetch_all()
                current_counts_vec = tracking_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # Check if tracking is called
                self.data['rep_num'] = float(current_rep_num)
                self.progress = current_rep_num * 100. / self.settings['rep_num']
                self.updateProgress.emit(int(self.progress))


            if self._abort:
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
        # need to put a find_NV script here
        time.sleep(10)

    def plot(self, figure_list):
        super(BerryRabi, self).plot([figure_list[0], figure_list[1]])

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

        if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            try:
                axes_list[0].clear()
                axes_list[0].plot(data['t_vec'], data['signal_avg_vec'])
                axes_list[0].set_xlabel('Rabi tau [ns]')
                axes_list[0].set_ylabel('Contrast')
                if title:
                    axes_list[0].set_title(
                        'Berry Rabi\nRef fluor: {:0.1f}kcps\nRepetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nVoltage: {:0.3f}V'.format(
                            data['ref_cnts'], int(data['rep_num']),
                            self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                            self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                            self.settings['voltage_pulses']['amplitude']))

            except Exception as e:
                print('** ATTENTION **')
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

