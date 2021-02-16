from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import time
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser
import numpy as np
from b26_toolkit.instruments import SGS100ARFSource, YokogawaGS200
from b26_toolkit.data_processing.fit_functions import fit_exp_decay, exp_offset
from qm.qua import frame_rotation as z_rot

wait_pulse_artifect = 68 / 4
# analog_digital_align_artifect = 0
analog_digital_align_artifect = int(140 / 4)


# pi_filename = 'C:\\Users\\NV\\b26_toolkit-master\\b26_toolkit\\scripts\\qm_scripts\\pi_time'
# pi2_filename = 'C:\\Users\\NV\\b26_toolkit-master\\b26_toolkit\\scripts\\qm_scripts\\pi2_time'
# pi32_filename = 'C:\\Users\\NV\\b26_toolkit-master\\b26_toolkit\\scripts\\qm_scripts\\pi32_time'


class EchoQM(Script):
    """
        This script runs a spin-echo measurement on an NV center.
        - Ziwei Qiu 9/5/2020

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
        Parameter('tau_times', [
            Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
            Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 200, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 440, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns'),
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
                config['pulses']['pi_pulse']['length'] = int(round(self.settings['mw_pulses']['pi_pulse_time'] / 4) * 4)
                config['pulses']['pi2_pulse']['length'] = int(
                    round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4) * 4)
                config['pulses']['pi32_pulse']['length'] = int(
                    round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4) * 4)

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

                # tau time between MW pulses
                tau_start = np.max([round(self.settings['tau_times']['min_time'] / 2), 16])
                tau_end = round(self.settings['tau_times']['max_time'] / 2)
                tau_step = round(self.settings['tau_times']['time_step'] / 2)

                self.t_vec = [int(a_) for a_ in
                              np.arange(int(np.ceil(tau_start / 4)), int(np.ceil(tau_end / 4)),
                                        int(np.ceil(tau_step / 4)))]
                self.tau_total = np.array(self.t_vec) * 8
                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('tau_total [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']
                # define the qua program
                with program() as spin_echo:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    t = declare(int)
                    n = declare(int)
                    k = declare(int)

                    # # the following two variable are used to flag tracking
                    # assign(IO1, False)
                    # flag = declare(bool, value=False)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(t, t_vec):
                            with for_(k, 0, k < 2, k + 1):
                                # # Check if tracking is called
                                # assign(flag, IO1)
                                # with if_(flag):
                                #     pause()
                                reset_frame('qubit')

                                with if_(k == 0):
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                with else_():
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
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
                        total_counts_st.buffer(t_num, 2).average().save("live_echo_data")
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(spin_echo)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(spin_echo, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):
        start = time.time()
        job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
        # job.get_simulated_samples().con1.plot()
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

        vec_handle = job.result_handles.get("live_echo_data")
        progress_handle = job.result_handles.get("live_rep_num")
        # tracking_handle = job.result_handles.get("current_counts")

        vec_handle.wait_for_values(1)
        progress_handle.wait_for_values(1)
        self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None, 'signal_norm': None,
                     'rep_num': None}

        # ref_counts = -1
        # tolerance = self.settings['NV_tracking']['tolerance']

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
                if echo_avg[0, 0] > echo_avg[0, 1]:
                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
                                               (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
                else:
                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                               (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])

            # do fitting
            try:
                echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True, verbose=False)
                self.data['fits'] = echo_fits
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.data['T2'] = self.data['fits'][1]

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

            time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(EchoQM, self).plot([figure_list[0], figure_list[1]])

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

            if 'fits' in data.keys():
                tau = data['t_vec']
                fits = data['fits']
                tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
                axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]),
                                  label="exp fit (T2={:2.1f} ns)".format(fits[1]))
            axes_list[0].legend(loc='upper center')
            axes_list[0].set_title(
                'Spin-echo\nRef fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                    data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                    self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                    self.settings['mw_pulses']['mw_power'],
                    self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'],
                    self.settings['mw_pulses']['IF_frequency'] * 1e-6))

    # def _update_plot(self, axes_list):
    #     if self.data is None:
    #         return
    #
    #     if 't_vec' in self.data.keys() and 'signal_avg_vec' in self.data.keys():
    #         try:
    #             axes_list[0].lines[0].set_ydata(self.data['signal_avg_vec'][:,0])
    #             axes_list[0].lines[1].set_ydata(self.data['signal_avg_vec'][:,1])
    #             axes_list[0].lines[0].set_xdata(self.data['t_vec'])
    #             axes_list[0].relim()
    #             axes_list[0].autoscale_view()
    #             axes_list[0].set_title('Spin-Echo live plot\n' + time.asctime())
    #         except Exception as e:
    #             print('** ATTENTION **')
    #             print(e)
    #             self._plot(axes_list)
    #
    #     else:
    #         self._plot(axes_list)

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


class PDDQM(Script):
    """
        This script runs a PDD (Periodic Dynamical Decoupling) sequence for different number of pi pulses.
        To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
        Rhode Schwarz SGS100A is used and it has IQ modulation.
        For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
        The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
        Tau/2 is the time between the center of the pulses!

        - Ziwei Qiu 9/6/2020
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
        Parameter('tau_times', [
            Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
            Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
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
                tau_start = round(self.settings['tau_times'][
                                      'min_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_start = int(np.max([tau_start, 4]))
                tau_end = round(self.settings['tau_times'][
                                    'max_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks / 2 / 4)
                tau_step = int(np.max([tau_step, 1]))

                self.t_vec = [int(a_) for a_ in np.arange(int(tau_start), int(tau_end), int(tau_step))]

                # total evolution time in ns
                self.tau_total = (np.array(self.t_vec) + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('Time between the first pi/2 and pi pulse edges [ns]: ', np.array(self.t_vec) * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                if len(self.t_vec) > 1:

                    res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                    IF_freq = self.settings['mw_pulses']['IF_frequency']

                    def xy4_block(is_last_block, IF_amp=IF_amp):
                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(-np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        if is_last_block:
                            wait(t, 'qubit')
                        else:
                            wait(2 * t + pi2_time, 'qubit')

                    def xy8_block(is_last_block, IF_amp=IF_amp):

                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(-np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(-np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_y
                        wait(2 * t + pi2_time, 'qubit')

                        z_rot(-np.pi / 2, 'qubit')
                        play('pi' * amp(IF_amp), 'qubit')  # pi_x
                        if is_last_block:
                            wait(t, 'qubit')
                        else:
                            wait(2 * t + pi2_time, 'qubit')

                    def spin_echo_block(is_last_block, IF_amp=IF_amp):
                        play('pi' * amp(IF_amp), 'qubit')
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

                        elif decoupling_seq_type == 'XY8':
                            if number_of_pulse_blocks == 2:
                                xy8_block(is_last_block=False)
                            elif number_of_pulse_blocks > 1:
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                    xy8_block(is_last_block=False)
                            xy8_block(is_last_block=True)
                        else:
                            if number_of_pulse_blocks == 2:
                                spin_echo_block(is_last_block=False)
                            elif number_of_pulse_blocks > 1:
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                    spin_echo_block(is_last_block=False)
                            spin_echo_block(is_last_block=True)

                    # define the qua program
                    with program() as pdd:
                        update_frequency('qubit', IF_freq)
                        result1 = declare(int, size=res_len)
                        counts1 = declare(int, value=0)
                        result2 = declare(int, size=res_len)
                        counts2 = declare(int, value=0)
                        total_counts = declare(int, value=0)

                        t = declare(int)
                        n = declare(int)
                        k = declare(int)
                        i = declare(int)

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
                                        if self.settings['decoupling_seq']['type'] == 'CPMG':
                                            z_rot(np.pi / 2, 'qubit')

                                        wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                        pi_pulse_train()

                                        if self.settings['decoupling_seq']['type'] == 'CPMG':
                                            z_rot(-np.pi / 2, 'qubit')
                                        play('pi2' * amp(IF_amp), 'qubit')

                                    # with else_():  # -x readout
                                    with if_(k == 1):
                                        play('pi2' * amp(IF_amp), 'qubit')
                                        if self.settings['decoupling_seq']['type'] == 'CPMG':
                                            z_rot(np.pi / 2, 'qubit')

                                        wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                        pi_pulse_train()

                                        if self.settings['decoupling_seq']['type'] == 'CPMG':
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
                            total_counts_st.buffer(t_num, 2).average().save("live_data")
                            total_counts_st.buffer(tracking_num).save("current_counts")
                            rep_num_st.save("live_rep_num")

                    with program() as job_stop:
                        play('trig', 'laser', duration=10)

                    if self.settings['to_do'] == 'simulation':
                        self._qm_simulation(pdd)
                    elif self.settings['to_do'] == 'execution':
                        self._qm_execution(pdd, job_stop)

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
            self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
                         'signal_norm': None, 'rep_num': None}

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

                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
                                               (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
                                                                                    1])

                # do fitting
                if self.settings['fit']:
                    try:
                        echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True,
                                                  verbose=False)
                        self.data['fits'] = echo_fits
                    except Exception as e:
                        print('** ATTENTION **')
                        print(e)
                    else:
                        self.data['T2'] = self.data['fits'][1]

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

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        # time.sleep(15)
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(PDDQM, self).plot([figure_list[0], figure_list[1]])

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

            if 'fits' in data.keys():
                tau = data['t_vec']
                fits = data['fits']
                tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
                axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]),
                                  label="exp fit (T2={:2.1f} ns)".format(fits[1]))
            axes_list[0].legend(loc='upper right')
            axes_list[0].set_title(
                'Periodic Dynamical Decoupling\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                    self.settings['decoupling_seq']['type'],
                    self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                    self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                    self.settings['mw_pulses']['mw_power'],
                    self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'],
                    self.settings['mw_pulses']['IF_frequency'] * 1e-6))

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


class PDDSingleTau(Script):
    """
        This script runs PDD at a single fixed tau.
        -Ziwei Qiu 10/12/2020
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
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
            Parameter('tracking_num', 20000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.28, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
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
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def xy8_block(is_last_block, IF_amp=IF_amp):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                    else:
                        wait(2 * t + pi2_time, 'qubit')

                def spin_echo_block(is_last_block, IF_amp=IF_amp):
                    play('pi' * amp(IF_amp), 'qubit')
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

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False)
                        elif number_of_pulse_blocks > 1:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False)
                        xy8_block(is_last_block=True)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False)
                        elif number_of_pulse_blocks > 1:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                spin_echo_block(is_last_block=False)
                        spin_echo_block(is_last_block=True)

                # define the qua program
                with program() as pdd:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    # t = declare(int) this needs to be commented out!!!
                    n = declare(int)
                    k = declare(int)
                    i = declare(int)

                    # the following two variable are used to flag tracking
                    assign(IO1, False)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        # Check if tracking is called

                        with while_(IO1):
                            play('trig', 'laser', duration=10000)

                        with for_(k, 0, k < 2, k + 1):

                            reset_frame('qubit')

                            with if_(k == 0):  # +x readout
                                play('pi2' * amp(IF_amp), 'qubit')
                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')

                            # with else_():  # -x readout
                            with if_(k == 1):
                                play('pi2' * amp(IF_amp), 'qubit')
                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
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
                        total_counts_st.buffer(2).average().save("live_data")
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(pdd)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(pdd, job_stop)

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

            self.data = {'tau': float(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
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
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0] + echo_avg[1]),
                                      'ref_cnts': (echo_avg[0] + echo_avg[1]) / 2})

                    self.data['signal_norm'] = 2 * (
                            self.data['signal_avg_vec'][1] - self.data['signal_avg_vec'][0]) / \
                                               (self.data['signal_avg_vec'][0] + self.data['signal_avg_vec'][1])

                try:
                    current_rep_num = progress_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                try:
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in tracking_handle **')
                    print(e)
                else:
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
        super(PDDSingleTau, self).plot([figure_list[0], figure_list[1]])

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
            axes_list[0].axhline(y=1.0, color='r', ls='--', lw=1.3)
            axes_list[0].set_xlabel('Total tau [ns]')
            axes_list[0].set_ylabel('Contrast')
            axes_list[0].legend(loc='upper right')
            if title:
                axes_list[0].set_title(
                    'Periodic Dynamical Decoupling\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['decoupling_seq']['type'],
                        self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                        self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                        self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))

    def _update_plot(self, axes_list, title=True):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
        else:
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


# The following script still has memory problem.
class ACSensingDigitalGate(Script):
    """
        This script implements AC sensing based on PDD (Periodic Dynamical Decoupling) sequence.
        Due to limited OPX memory, at most output XY8-2 sequences.
        Modulating field is controlled by DIGITAL pulses, named as 'e_field2' (digital 5).
        A MOSFET and Yokogawa DC votlage source are used for outputting large voltage modulation.
        Rhode Schwarz SGS100A is used and it has IQ modulation.
        Tau is the total evolution time between the center of the first and last pi/2 pulses.

        - Ziwei Qiu 9/28/2020
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
        Parameter('tau_times', [
            Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
            Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('gate_voltages', [
            Parameter('source', 'qm-opx', ['qm-opx', 'yokogawa'],
                      'choose the source of the voltage. if qm-opx, output is TTL level; if yokogawa, output level is user defined.'),
            Parameter('level', 5.0, float, 'choose the ac modulation voltage Vpp from Yokogawa, between 4.5V and 21V.')

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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource, 'yokogawa':YokogawaGS200}
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
                tau_start = round(self.settings['tau_times'][
                                      'min_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_start = int(np.max([tau_start, 4]))
                tau_end = round(self.settings['tau_times'][
                                    'max_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks / 2 / 4)
                tau_step = int(np.max([tau_step, 1]))

                self.t_vec = [int(a_) for a_ in np.arange(int(tau_start), int(tau_end), int(tau_step))]

                # total evolution time in ns
                self.tau_total = (np.array(self.t_vec) + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('Time between the first pi/2 and pi pulse edges [ns]: ', np.array(self.t_vec) * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'e_field2')  # e field 2 off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                            wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'e_field2')  # e field 2 off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                            wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='e_field2', off='e_field1'):
                    play('pi' * amp(IF_amp), 'qubit')
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            if on == 'e_field2':
                                play('trig', 'e_field2', duration=t)  # e field  on
                            else:
                                wait(t, 'e_field2')  # e field  off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            if on == 'e_field2':
                                play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field  on
                            else:
                                wait(2 * t + pi2_time, 'e_field2')  # e field  off

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False, efield=efield)
                        xy4_block(is_last_block=True, efield=efield)

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False, efield=efield)
                        xy8_block(is_last_block=True, efield=efield)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                        elif number_of_pulse_blocks > 2:
                            if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')

                            else:
                                with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')
                                spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')

                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='e_field2', off='e_field1')
                        else:  # even number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='e_field1', off='e_field2')

                # define the qua program
                with program() as acsensing:

                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    t = declare(int)
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

                        with for_each_(t, t_vec):
                            # note that by having only 4 k values, I can do at most XY8-2 seq.
                            # by having 6 k values, I can do at most XY8 seq.
                            # the plot needs to be changed accordingly.
                            with for_(k, 0, k < 4, k + 1):

                                reset_frame('qubit')

                                # align('qubit', 'e_field2')
                                # play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                #
                                # if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #     z_rot(np.pi / 2, 'qubit')
                                #
                                # wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                #
                                # with if_(k < 2):
                                #     pi_pulse_train(efield=False)
                                # with else_():
                                #     pi_pulse_train(efield=True)
                                #
                                # if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #     z_rot(-np.pi / 2, 'qubit')
                                #
                                # with if_(k == 1 | k == 3 | k == 5):
                                #     z_rot(np.pi, 'qubit')
                                # with if_(k > 3):
                                #     z_rot(np.pi/2, 'qubit')
                                # play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 0):  # +x readout, no E field
                                    align('qubit', 'e_field2')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                                    wait(t, 'e_field2')  # e field 2 off

                                    if self.settings['sensing_type'] == 'both':
                                        pi_pulse_train(efield=True)
                                    else:
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                with if_(k == 1):  # -x readout, no E field
                                    align('qubit', 'e_field2')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                                    wait(t, 'e_field2')  # e field 2 off

                                    if self.settings['sensing_type'] == 'both':
                                        pi_pulse_train(efield=True)
                                    else:
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                with if_(k == 2):  # +x readout, with E field
                                    align('qubit', 'e_field2')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                                    wait(t, 'e_field2')  # e field 2 off

                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                with if_(k == 3):  # -x readout, with E field
                                    align('qubit', 'e_field2')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                                    wait(t, 'e_field2')  # e field 2 off
                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    # wait(pi2_time, 'e_field1')  # e field 1 off
                                    wait(pi2_time, 'e_field2')  # e field 2 off

                                # with if_(k == 4):  # +y readout, with E field
                                #     align('qubit', 'e_field2')
                                #     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                #     # wait(pi2_time, 'e_field1')  # e field 1 off
                                #     wait(pi2_time, 'e_field2')  # e field 2 off
                                #
                                #     if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #         z_rot(np.pi / 2, 'qubit')
                                #
                                #     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                #     # play('trig', 'e_field1', duration=t)  # e field 1 on
                                #     wait(t, 'e_field2')  # e field 2 off
                                #     pi_pulse_train(efield=True)
                                #
                                #     if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #         z_rot(-np.pi / 2, 'qubit')
                                #     z_rot(np.pi / 2, 'qubit')
                                #     play('pi2' * amp(IF_amp), 'qubit')
                                #     # wait(pi2_time, 'e_field1')  # e field 1 off
                                #     wait(pi2_time, 'e_field2')  # e field 2 off
                                #
                                # with if_(k == 5):  # -y readout, with E field
                                #     align('qubit', 'e_field2')
                                #     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                #     # wait(pi2_time, 'e_field1')  # e field 1 off
                                #     wait(pi2_time, 'e_field2')  # e field 2 off
                                #
                                #     if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #         z_rot(np.pi / 2, 'qubit')
                                #
                                #     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                #     # play('trig', 'e_field1', duration=t)  # e field 1 on
                                #     wait(t, 'e_field2')  # e field 2 off
                                #     pi_pulse_train(efield=True)
                                #
                                #     if self.settings['decoupling_seq']['type'] == 'CPMG':
                                #         z_rot(-np.pi / 2, 'qubit')
                                #
                                #     z_rot(np.pi * 1.5, 'qubit')
                                #     play('pi2' * amp(IF_amp), 'qubit')
                                #     # wait(pi2_time, 'e_field1')  # e field 1 off
                                #     wait(pi2_time, 'e_field2')  # e field 2 off

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
                    self._qm_simulation(acsensing)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(acsensing, job_stop)
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

        if self.settings['gate_voltages']['source'] == 'yokogawa':
            self.instruments['yokogawa']['instance'].update({'source': 'VOLT'})
            self.instruments['yokogawa']['instance'].update({'level': self.settings['gate_voltages']['level']})
            self.instruments['yokogawa']['instance'].update({'current_limit': 200E-3}) # use the maximum current limit
            self.instruments['yokogawa']['instance'].update({'enable_output': True})
            print('Turned on Yokogawa DC voltage source.')

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
            self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
                         'sig1_norm': None, 'sig2_norm': None, 'rep_num': None}

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

                    # self.data['esig_y_norm'] = 2 * (
                    #         self.data['signal_avg_vec'][:, 5] - self.data['signal_avg_vec'][:, 4]) / \
                    #                            (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:,
                    #                                                                 5])


                # # do fitting
                # if self.settings['fit']:
                #     try:
                #         echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True,
                #                                   verbose=False)
                #         self.data['fits'] = echo_fits
                #     except Exception as e:
                #         print('** ATTENTION **')
                #         print(e)
                #     else:
                #         self.data['T2'] = self.data['fits'][1]

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
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

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(ACSensingDigitalGate, self).plot([figure_list[0], figure_list[1]])

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
            try:
                axes_list[1].clear()
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")
                # axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
                # axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")

                axes_list[1].set_xlabel('Total tau [ns]')
                axes_list[1].set_ylabel('Normalized Counts')
                # axes_list[1].legend()
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].clear()
                axes_list[0].plot(data['t_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['t_vec'], data['sig2_norm'], label="sig2_norm")
                # axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
                axes_list[0].set_xlabel('Total tau [ns]')
                axes_list[0].set_ylabel('Contrast')

                # if 'fits' in data.keys():
                #     tau = data['t_vec']
                #     fits = data['fits']
                #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
                #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'AC Sensing (type: {:s})\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['sensing_type'], self.settings['decoupling_seq']['type'],
                        self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                        self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                        self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
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
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class AC_DGate_SingleTau(Script):
    """
        This script runs the script ACSensingDigitalGate at a single tau.
        - Ziwei Qiu 10/18/2020
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('decoupling_seq', [
            Parameter('type', 'CPMG', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('gate_voltages', [
            Parameter('source', 'qm-opx', ['qm-opx', 'yokogawa'],
                      'choose the source of the voltage. if qm-opx, output is TTL level; if yokogawa, output level is user defined.'),
            Parameter('level', 5.0, float, 'choose the ac modulation voltage Vpp from Yokogawa, between 4.5V and 21V.')

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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 20000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.28, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource, 'yokogawa':YokogawaGS200}
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
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'e_field2')  # e field 2 off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                            wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                        wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'e_field2')  # e field 1, 2 off
                        play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
                        # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'e_field2')  # e field 2 off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                            wait(2 * t + pi2_time, 'e_field2')  # e field 2 off

                def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='e_field2', off='e_field1'):
                    play('pi' * amp(IF_amp), 'qubit')
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            if on == 'e_field2':
                                play('trig', 'e_field2', duration=t)  # e field  on
                            else:
                                wait(t, 'e_field2')  # e field  off
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'e_field2')  # e field 1, 2 off
                            if on == 'e_field2':
                                play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field  on
                            else:
                                wait(2 * t + pi2_time, 'e_field2')  # e field  off

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False, efield=efield)
                        xy4_block(is_last_block=True, efield=efield)

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False, efield=efield)
                        xy8_block(is_last_block=True, efield=efield)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                        elif number_of_pulse_blocks > 2:
                            if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')

                            else:
                                with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
                                    spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')
                                spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')

                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='e_field2', off='e_field1')
                        else:  # even number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='e_field1', off='e_field2')

                # define the qua program
                with program() as acsensing:

                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    # t = declare(int)
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

                        # note that by having only 4 k values, I can do at most XY8-2 seq.
                        # by having 6 k values, I can do at most XY8 seq.
                        # the plot needs to be changed accordingly.
                        with for_(k, 0, k < 4, k + 1):

                            reset_frame('qubit')

                            with if_(k == 0):  # +x readout, no E field
                                align('qubit', 'e_field2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                # play('trig', 'e_field1', duration=t)  # e field 1 on
                                wait(t, 'e_field2')  # e field 2 off

                                if self.settings['sensing_type'] == 'both':
                                    pi_pulse_train(efield=True)
                                else:
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                            with if_(k == 1):  # -x readout, no E field
                                align('qubit', 'e_field2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                # play('trig', 'e_field1', duration=t)  # e field 1 on
                                wait(t, 'e_field2')  # e field 2 off

                                if self.settings['sensing_type'] == 'both':
                                    pi_pulse_train(efield=True)
                                else:
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                z_rot(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                            with if_(k == 2):  # +x readout, with E field
                                align('qubit', 'e_field2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                # play('trig', 'e_field1', duration=t)  # e field 1 on
                                wait(t, 'e_field2')  # e field 2 off

                                pi_pulse_train(efield=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')

                                play('pi2' * amp(IF_amp), 'qubit')
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                            with if_(k == 3):  # -x readout, with E field
                                align('qubit', 'e_field2')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                # play('trig', 'e_field1', duration=t)  # e field 1 on
                                wait(t, 'e_field2')  # e field 2 off
                                pi_pulse_train(efield=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                z_rot(np.pi, 'qubit')

                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')

                                play('pi2' * amp(IF_amp), 'qubit')
                                # wait(pi2_time, 'e_field1')  # e field 1 off
                                wait(pi2_time, 'e_field2')  # e field 2 off

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
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(acsensing)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(acsensing, job_stop)
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

        if self.settings['gate_voltages']['source'] == 'yokogawa':
            self.instruments['yokogawa']['instance'].update({'source': 'VOLT'})
            self.instruments['yokogawa']['instance'].update({'level': self.settings['gate_voltages']['level']})
            self.instruments['yokogawa']['instance'].update({'current_limit': 200E-3}) # use the maximum current limit
            self.instruments['yokogawa']['instance'].update({'enable_output': True})
            print('Turned on Yokogawa DC voltage source.')

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
            self.data = {'tau': float(self.tau_total),  'signal_avg_vec': None, 'ref_cnts': None,
                         'sig1_norm': None, 'sig2_norm': None, 'rep_num': None}

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
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0] + echo_avg[1]),
                                      'ref_cnts': (echo_avg[0] + echo_avg[1]) / 2})

                    self.data['sig1_norm'] = 2 * (
                            self.data['signal_avg_vec'][1] - self.data['signal_avg_vec'][0]) / \
                                             (self.data['signal_avg_vec'][0] + self.data['signal_avg_vec'][1])

                    self.data['sig2_norm'] = 2 * (
                            self.data['signal_avg_vec'][3] - self.data['signal_avg_vec'][2]) / \
                                             (self.data['signal_avg_vec'][2] + self.data['signal_avg_vec'][3])

                try:
                    current_rep_num = progress_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                try:
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in tracking_handle **')
                    print(e)
                else:
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
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(AC_DGate_SingleTau, self).plot([figure_list[0], figure_list[1]])

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
            try:
                axes_list[0].clear()
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][0], label="sig1 +")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][1], label="sig1 -")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][2], label="sig2 +")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][3], label="sig2 -")
                axes_list[0].axhline(y=1.0, color='r', ls='--', lw=1.3)
                axes_list[0].set_xlabel('Total tau [ns]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[0].legend(loc='upper right')
                if title:
                    axes_list[0].set_title(
                        'AC Sensing Digital Gate (type: {:s})\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                            self.settings['sensing_type'], self.settings['decoupling_seq']['type'],
                            self.settings['decoupling_seq']['num_of_pulse_blocks'],
                            data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                            self.settings['mw_pulses']['pi_pulse_time'],
                            self.settings['mw_pulses']['3pi_half_pulse_time'],
                            self.settings['mw_pulses']['mw_power'],
                            self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                            self.settings['mw_pulses']['IF_amp'],
                            self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def _update_plot(self, axes_list, title=True):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
        else:
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


# The following script still has memory problem.
class ACSensingAnalogGate(Script):
    """
        This script implements AC sensing based on PDD (Periodic Dynamical Decoupling) sequence.
        The gate voltages are fixed while tau is swept.
        Due to limited OPX memory, at most output XY8-2 sequences.
        Modulating field is controlled by analog pulses, named as 'gate' (analog 5).
        Rhode Schwarz SGS100A is used and it has IQ modulation.
        Tau is the total evolution time between the center of the first and last pi/2 pulses.

        - Ziwei Qiu 9/28/2020
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
        Parameter('tau_times', [
            Parameter('min_time', 200, int, 'minimum time between the two pi/2 pulses (in ns)'),
            Parameter('max_time', 10000, int, 'maximum time between the two pi/2 pulses (in ns)'),
            Parameter('time_step', 100, int, 'time step increment of time between the two pi/2 pulses (in ns)')
        ]),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('gate1', -0.001, float, 'define the first gate voltage in AC sensing, on top of the offset'),
            Parameter('gate2', 0.001, float, 'define the second gate voltage in AC sensing, on top of the offset')
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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
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

                config['elements']['gate']['intermediate_frequency'] = 0
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = self.settings['gate_voltages']['gate1']
                config['waveforms']['const_gate2']['sample'] = self.settings['gate_voltages']['gate2']

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
                tau_start = round(self.settings['tau_times'][
                                      'min_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_start = int(np.max([tau_start, 4]))
                tau_end = round(self.settings['tau_times'][
                                    'max_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
                tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks / 2 / 4)
                tau_step = int(np.max([tau_step, 1]))

                self.t_vec = [int(a_) for a_ in np.arange(int(tau_start), int(tau_end), int(tau_step))]

                # total evolution time in ns
                self.tau_total = (np.array(self.t_vec) + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                t_vec = self.t_vec
                t_num = len(self.t_vec)
                print('Time between the first pi/2 and pi pulse edges [ns]: ', np.array(self.t_vec) * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=t)  # e field 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=t)  # gate 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='gate1'):
                    play('pi' * amp(IF_amp), 'qubit')
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play(on, 'gate', duration=t)  # gate  on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play(on, 'gate', duration=2 * t + pi2_time)  # gate  on

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False, efield=efield)
                        xy4_block(is_last_block=True, efield=efield)

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False, efield=efield)
                        xy8_block(is_last_block=True, efield=efield)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                        elif number_of_pulse_blocks > 2:
                            if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                            else:
                                with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                                spin_echo_block(is_last_block=False, efield=efield, on='gate2')

                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate2')
                        else:  # even number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate1')

                # define the qua program
                with program() as acsensing:

                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    t = declare(int)
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

                        with for_each_(t, t_vec):
                            # note that by having only 4 k values, I can do at most XY8-2 seq.
                            # (by having 6 k values, I can do at most XY8 seq.)
                            with for_(k, 0, k < 4, k + 1):

                                reset_frame('qubit')
                                with if_(k == 0):  # +x readout, no E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                    if self.settings['sensing_type'] == 'both':
                                        play('gate1', 'gate', duration=t)  # gate 1
                                        pi_pulse_train(efield=True)
                                    else:
                                        wait(t, 'gate')  # gate off
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 1):  # -x readout, no E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                    if self.settings['sensing_type'] == 'both':
                                        play('gate1', 'gate', duration=t)  # gate 1
                                        pi_pulse_train(efield=True)
                                    else:
                                        wait(t, 'gate')  # gate off
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 2):  # +x readout, with E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    play('gate1', 'gate', duration=t)  # gate 1

                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 3):  # -x readout, with E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    play('gate1', 'gate', duration=t)  # gate 1

                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

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
                    self._qm_simulation(acsensing)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(acsensing, job_stop)
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
            self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
                         'sig1_norm': None, 'sig2_norm': None, 'rep_num': None}

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

                # # do fitting
                # if self.settings['fit']:
                #     try:
                #         echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True,
                #                                   verbose=False)
                #         self.data['fits'] = echo_fits
                #     except Exception as e:
                #         print('** ATTENTION **')
                #         print(e)
                #     else:
                #         self.data['T2'] = self.data['fits'][1]

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
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

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(ACSensingAnalogGate, self).plot([figure_list[0], figure_list[1]])

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
            try:
                axes_list[1].clear()
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")
                # axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
                # axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")

                axes_list[1].set_xlabel('Total tau [ns]')
                axes_list[1].set_ylabel('Normalized Counts')
                # axes_list[1].legend()
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].clear()
                axes_list[0].plot(data['t_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['t_vec'], data['sig2_norm'], label="sig2_norm")
                # axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
                axes_list[0].set_xlabel('Total tau [ns]')
                axes_list[0].set_ylabel('Contrast')

                # if 'fits' in data.keys():
                #     tau = data['t_vec']
                #     fits = data['fits']
                #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
                #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'AC Sensing (type: {:s})\ngate1: {:0.3f}V, gate2: {:0.3f}V, offset: {:0.3f}V\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['sensing_type'], self.settings['gate_voltages']['gate1'],
                        self.settings['gate_voltages']['gate2'], self.settings['gate_voltages']['offset'],
                        self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                        self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                        self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                        self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
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
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class AC_AGate_SingleTau(Script):
    """
        This script runs the script ACSensingAnalogGate at a single tau.
        - Ziwei Qiu 10/18/2020
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('gate1', -0.001, float, 'define the first gate voltage in AC sensing, on top of the offset'),
            Parameter('gate2', 0.001, float, 'define the second gate voltage in AC sensing, on top of the offset')
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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 50000, int,
                      'number of recent APD windows used for calculating current counts, suggest 50000'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
        ]),
        Parameter('rep_num', 500000, int, 'define the repetition number'),
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

                config['elements']['gate']['intermediate_frequency'] = 0
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = self.settings['gate_voltages']['gate1']
                config['waveforms']['const_gate2']['sample'] = self.settings['gate_voltages']['gate2']

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
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=t)  # e field 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2', 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=t)  # gate 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1', 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='gate1'):
                    play('pi' * amp(IF_amp), 'qubit')
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play(on, 'gate', duration=t)  # gate  on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play(on, 'gate', duration=2 * t + pi2_time)  # gate  on

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False, efield=efield)
                        xy4_block(is_last_block=True, efield=efield)

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False, efield=efield)
                        xy8_block(is_last_block=True, efield=efield)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                        elif number_of_pulse_blocks > 2:
                            if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                            else:
                                with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                                spin_echo_block(is_last_block=False, efield=efield, on='gate2')

                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate2')
                        else:  # even number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate1')

                # define the qua program
                with program() as acsensing:

                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    # t = declare(int)
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

                        # note that by having only 4 k values, I can do at most XY8-2 seq.
                        # (by having 6 k values, I can do at most XY8 seq.)
                        with for_(k, 0, k < 4, k + 1):

                            reset_frame('qubit')
                            with if_(k == 0):  # +x readout, no E field
                                align('qubit', 'gate')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'gate')  # gate off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                if self.settings['sensing_type'] == 'both':
                                    play('gate1', 'gate', duration=t)  # gate 1
                                    pi_pulse_train(efield=True)
                                else:
                                    wait(t, 'gate')  # gate off
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'gate')  # gate off

                            with if_(k == 1):  # -x readout, no E field
                                align('qubit', 'gate')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'gate')  # gate off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                if self.settings['sensing_type'] == 'both':
                                    play('gate1', 'gate', duration=t)  # gate 1
                                    pi_pulse_train(efield=True)
                                else:
                                    wait(t, 'gate')  # gate off
                                    pi_pulse_train()

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                z_rot(np.pi, 'qubit')
                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'gate')  # gate off

                            with if_(k == 2):  # +x readout, with E field
                                align('qubit', 'gate')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'gate')  # gate off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('gate1', 'gate', duration=t)  # gate 1

                                pi_pulse_train(efield=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')

                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'gate')  # gate off

                            with if_(k == 3):  # -x readout, with E field
                                align('qubit', 'gate')
                                play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                wait(pi2_time, 'gate')  # gate off

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(np.pi / 2, 'qubit')

                                wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                play('gate1', 'gate', duration=t)  # gate 1

                                pi_pulse_train(efield=True)

                                if self.settings['decoupling_seq']['type'] == 'CPMG':
                                    z_rot(-np.pi / 2, 'qubit')

                                z_rot(np.pi, 'qubit')

                                if self.settings['sensing_type'] != 'cosine':
                                    z_rot(np.pi / 2, 'qubit')

                                play('pi2' * amp(IF_amp), 'qubit')
                                wait(pi2_time, 'gate')  # gate off

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
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(acsensing)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(acsensing, job_stop)
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
            self.data = {'tau': float(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
                         'sig1_norm': None, 'sig2_norm': None, 'rep_num': None}

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
                    self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0] + echo_avg[1]),
                                      'ref_cnts': (echo_avg[0] + echo_avg[1]) / 2})

                    self.data['sig1_norm'] = 2 * (
                            self.data['signal_avg_vec'][1] - self.data['signal_avg_vec'][0]) / \
                                             (self.data['signal_avg_vec'][0] + self.data['signal_avg_vec'][1])

                    self.data['sig2_norm'] = 2 * (
                            self.data['signal_avg_vec'][3] - self.data['signal_avg_vec'][2]) / \
                                             (self.data['signal_avg_vec'][2] + self.data['signal_avg_vec'][3])

                try:
                    current_rep_num = progress_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                try:
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in tracking_handle **')
                    print(e)
                else:
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
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(AC_AGate_SingleTau, self).plot([figure_list[0], figure_list[1]])

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
            try:
                axes_list[0].clear()
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][0], label="sig1 +")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][1], label="sig1 -")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][2], label="sig2 +")
                axes_list[0].scatter(data['tau'], data['signal_avg_vec'][3], label="sig2 -")
                axes_list[0].axhline(y=1.0, color='r', ls='--', lw=1.3)
                axes_list[0].set_xlabel('Total tau [ns]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[0].legend(loc='upper right')
                if title:
                    axes_list[0].set_title(
                        'AC Sensing Analog Gate (type: {:s})\ngate1: {:0.3f}V, gate2: {:0.3f}V, offset: {:0.3f}V\n{:s} {:d} block(s), Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                            self.settings['sensing_type'], self.settings['gate_voltages']['gate1'],
                            self.settings['gate_voltages']['gate2'], self.settings['gate_voltages']['offset'],
                            self.settings['decoupling_seq']['type'],
                            self.settings['decoupling_seq']['num_of_pulse_blocks'],
                            data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['pi_half_pulse_time'],
                            self.settings['mw_pulses']['pi_pulse_time'],
                            self.settings['mw_pulses']['3pi_half_pulse_time'],
                            self.settings['mw_pulses']['mw_power'],
                            self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                            self.settings['mw_pulses']['IF_amp'],
                            self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
                print(e)

    def _update_plot(self, axes_list, title=True):
        if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
            'optimize'].is_running:
            if self.flag_optimize_plot:
                self.scripts['optimize']._plot([axes_list[1]])
                self.flag_optimize_plot = False
            else:
                self.scripts['optimize']._update_plot([axes_list[1]])
        else:
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


# The following script still has memory problem.
class ACSensingSweepGate(Script):
    """
        This script implements AC sensing based on PDD (Periodic Dynamical Decoupling) sequence.
        The gate voltages are swept while tau is fixed.
        Due to limited OPX memory, at most output XY8-2 sequences.
        Modulating field is controlled by analog pulses, named as 'gate' (analog 5).
        Rhode Schwarz SGS100A is used and it has IQ modulation.
        Tau is the total evolution time between the center of the first and last pi/2 pulses.

        - Ziwei Qiu 9/28/2020
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
        Parameter('tau', 20000, int, 'time between the two pi/2 pulses (in ns)'),
        Parameter('gate_voltages', [
            Parameter('offset', 0, float, 'define the offset gate voltage'),
            Parameter('gate1', -0.001, float, 'define the first gate voltage in AC sensing, on top of the offset'),
            Parameter('gate2', 0.001, float, 'define the second gate voltage in AC sensing, on top of the offset')
        ]),
        Parameter('sweep', [
            Parameter('type', 'diff', ['gate1', 'gate2', 'diff', 'mean'], 'choose what to sweep'),
            Parameter('min_vol', 0, float, 'define the minimum voltage'),
            Parameter('max_vol', 0.4, float, 'define the maximum voltage'),
            Parameter('vol_step', 0.02, float, 'define the voltage step')
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
        Parameter('fit', False, bool, 'fit the data with exponential decay'),
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

    def _get_voltage_array(self):
        min_vol = self.settings['sweep']['min_vol']
        max_vol = self.settings['sweep']['max_vol']
        vol_step = self.settings['sweep']['vol_step']
        self.sweep_array = np.arange(min_vol, max_vol, vol_step)

        if self.settings['sweep']['type'] == 'gate1':
            self.gate1_list = [float(a_) for a_ in np.arange(min_vol, max_vol, vol_step)]
            self.gate2_list = [self.settings['gate_voltages']['gate2']] * len(self.gate1_list)

        elif self.settings['sweep']['type'] == 'gate2':
            self.gate2_list = [float(a_) for a_ in np.arange(min_vol, max_vol, vol_step)]
            self.gate1_list = [self.settings['gate_voltages']['gate1']] * len(self.gate2_list)

        elif self.settings['sweep']['type'] == 'diff':
            self.gate1_list = [float(a_) for a_ in np.arange(-min_vol / 2, -max_vol / 2, -vol_step / 2)]
            self.gate2_list = [float(a_) for a_ in np.arange(min_vol / 2, max_vol / 2, vol_step / 2)]

        else:
            diff = self.settings['gate_voltages']['gate2'] - self.settings['gate_voltages']['gate1']
            self.gate1_list = [float(a_) for a_ in np.arange(min_vol - diff / 2, max_vol - diff / 2, vol_step)]
            self.gate2_list = [float(a_) for a_ in np.arange(min_vol + diff / 2, max_vol + diff / 2, vol_step)]

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

                self._get_voltage_array()
                gate1_config = np.max(np.abs(self.gate1_list))
                gate2_config = np.max(np.abs(self.gate2_list))

                config['elements']['gate']['intermediate_frequency'] = 0
                config['controllers']['con1']['analog_outputs'][5]['offset'] = self.settings['gate_voltages']['offset']
                config['waveforms']['const_gate1']['sample'] = gate1_config
                config['waveforms']['const_gate2']['sample'] = gate2_config

                self.gate_list = ((np.array(self.gate1_list) / gate1_config).tolist(),
                                  (np.array(self.gate2_list) / gate2_config).tolist())
                # print(self.gate_list)
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
                t = int(np.max([t, 4]))

                # total evolution time in ns
                self.tau_total = (t + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks

                print('Time between the first pi/2 and pi pulse edges [ns]: ', t * 4)
                print('Total evolution times [ns]: ', self.tau_total)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                IF_freq = self.settings['mw_pulses']['IF_frequency']

                def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1' * amp(g1), 'gate', duration=t)  # e field 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')

                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                    z_rot(np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_y
                    wait(2 * t + pi2_time, 'qubit')
                    if efield:
                        wait(pi_time, 'gate')  # gate off
                        play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate 2 on

                    z_rot(-np.pi / 2, 'qubit')
                    play('pi' * amp(IF_amp), 'qubit')  # pi_x
                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1' * amp(g1), 'gate', duration=t)  # gate 1 on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate 1 on

                def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='gate1'):
                    play('pi' * amp(IF_amp), 'qubit')

                    if is_last_block:
                        wait(t, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            if on == 'gate1':
                                play('gate1' * amp(g1), 'gate', duration=t)  # gate1  on
                            else:
                                play('gate2' * amp(g2), 'gate', duration=t)  # gate2  on
                    else:
                        wait(2 * t + pi2_time, 'qubit')
                        if efield:
                            wait(pi_time, 'gate')  # gate off
                            if on == 'gate1':
                                play('gate1' * amp(g1), 'gate', duration=2 * t + pi2_time)  # gate1  on
                            else:
                                play('gate2' * amp(g2), 'gate', duration=2 * t + pi2_time)  # gate2  on

                def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
                                   number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
                    if decoupling_seq_type == 'XY4':
                        if number_of_pulse_blocks == 2:
                            xy4_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy4_block(is_last_block=False, efield=efield)
                        xy4_block(is_last_block=True, efield=efield)

                    elif decoupling_seq_type == 'XY8':
                        if number_of_pulse_blocks == 2:
                            xy8_block(is_last_block=False, efield=efield)
                        elif number_of_pulse_blocks > 2:
                            with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
                                xy8_block(is_last_block=False, efield=efield)
                        xy8_block(is_last_block=True, efield=efield)
                    else:
                        if number_of_pulse_blocks == 2:
                            spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                        elif number_of_pulse_blocks > 2:
                            if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                                with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                            else:
                                with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate2')
                                    spin_echo_block(is_last_block=False, efield=efield, on='gate1')
                                spin_echo_block(is_last_block=False, efield=efield, on='gate2')

                        if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate2')
                        else:  # even number of pi pulses
                            spin_echo_block(is_last_block=True, efield=efield, on='gate1')

                # define the qua program
                with program() as acsensing:

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
                    i = declare(int)

                    # the following variable is used to flag tracking
                    assign(IO1, False)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):

                        # Check if tracking is called
                        with while_(IO1):
                            play('trig', 'laser', duration=10000)

                        with for_each_((g1, g2), self.gate_list):

                            # note that by having only 4 k values, I can do at most XY8-2 seq.
                            # (by having 6 k values, I can do at most XY8 seq.)
                            with for_(k, 0, k < 4, k + 1):

                                reset_frame('qubit')
                                with if_(k == 0):  # +x readout, no E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                    if self.settings['sensing_type'] == 'both':
                                        play('gate1' * amp(g1), 'gate', duration=t)  # gate 1
                                        pi_pulse_train(efield=True)
                                    else:
                                        wait(t, 'gate')  # gate off
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 1):  # -x readout, no E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time

                                    if self.settings['sensing_type'] == 'both':
                                        play('gate1' * amp(g1), 'gate', duration=t)  # gate 1
                                        pi_pulse_train(efield=True)
                                    else:
                                        wait(t, 'gate')  # gate off
                                        pi_pulse_train()

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 2):  # +x readout, with E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    play('gate1' * amp(g1), 'gate', duration=t)  # gate 1

                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

                                with if_(k == 3):  # -x readout, with E field
                                    align('qubit', 'gate')
                                    play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                                    wait(pi2_time, 'gate')  # gate off

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(np.pi / 2, 'qubit')

                                    wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                                    play('gate1' * amp(g1), 'gate', duration=t)  # gate 1

                                    pi_pulse_train(efield=True)

                                    if self.settings['decoupling_seq']['type'] == 'CPMG':
                                        z_rot(-np.pi / 2, 'qubit')

                                    z_rot(np.pi, 'qubit')

                                    if self.settings['sensing_type'] != 'cosine':
                                        z_rot(np.pi / 2, 'qubit')

                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(pi2_time, 'gate')  # gate off

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
                        total_counts_st.buffer(self.gate_num, 4).average().save("live_data")
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(acsensing)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(acsensing, job_stop)
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
            self.data = {'t_vec': self.sweep_array, 'tau': self.tau_total,'signal_avg_vec': None, 'ref_cnts': None,
                         'sig1_norm': None, 'sig2_norm': None, 'rep_num': None, 'squared_sum_root': None, 'phase': None}

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

                    if self.settings['sensing_type'] == 'both':
                        self.data['squared_sum_root'] = np.sqrt(self.data['sig1_norm'] ** 2 + self.data['sig2_norm'] ** 2)
                        self.data['phase'] = np.arccos(self.data['sig1_norm'] / self.data['squared_sum_root']) * np.sign(
                            self.data['sig2_norm'])

                # # do fitting
                # if self.settings['fit']:
                #     try:
                #         echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True,
                #                                   verbose=False)
                #         self.data['fits'] = echo_fits
                #     except Exception as e:
                #         print('** ATTENTION **')
                #         print(e)
                #     else:
                #         self.data['T2'] = self.data['fits'][1]

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
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

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['signal_full_vec'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def NV_tracking(self):
        # need to put a find_NV script here
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(ACSensingSweepGate, self).plot([figure_list[0], figure_list[1]])

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
            try:
                axes_list[0].clear()
                axes_list[1].clear()
                axes_list[2].clear()

                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="sig1 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="sig1 -")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="sig2 +")
                axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="sig2 -")

                axes_list[1].set_xlabel('Voltage [V]')
                axes_list[1].set_ylabel('Normalized Counts')
                axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)

                axes_list[0].plot(data['t_vec'], data['sig1_norm'], label="sig1_norm")
                axes_list[0].plot(data['t_vec'], data['sig2_norm'], label="sig2_norm")
                if data['squared_sum_root'] is not None:
                    axes_list[0].plot(data['t_vec'], data['squared_sum_root'], label="squared_sum_root")

                if data['phase'] is not None:
                    axes_list[2].plot(data['t_vec'], data['phase'], label="phase")
                    axes_list[2].grid(b=True, which='major', color='#666666', linestyle='--')

                axes_list[2].set_xlabel('Voltage [V]')
                axes_list[0].set_ylabel('Contrast')
                axes_list[2].set_ylabel('Phase [rad]')

                # if 'fits' in data.keys():
                #     tau = data['t_vec']
                #     fits = data['fits']
                #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
                #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))

                axes_list[0].legend(loc='upper right')
                axes_list[0].set_title(
                    'AC Sensing (type: {:s})\ngate1: {:0.3f}V, gate2: {:0.3f}V, offset: {:0.3f}V, sweep {:s}\n{:s} {:d} block(s), tau = {:0.3}us, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                        self.settings['sensing_type'],
                        self.settings['gate_voltages']['gate1'], self.settings['gate_voltages']['gate2'],
                        self.settings['gate_voltages']['offset'], self.settings['sweep']['type'],
                        self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                        data['tau'] / 1000, data['ref_cnts'], int(data['rep_num']),
                        self.settings['mw_pulses']['pi_half_pulse_time'], self.settings['mw_pulses']['pi_pulse_time'],
                        self.settings['mw_pulses']['3pi_half_pulse_time'], self.settings['mw_pulses']['mw_power'],
                        self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                        self.settings['mw_pulses']['IF_frequency'] * 1e-6))
            except Exception as e:
                print('** ATTENTION **')
                # print('here')
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


# # The following script (using both efield1 and efield2) still has memory problem.
# class ACSensing(Script):
#     """
#         This script runs a PDD (Periodic Dynamical Decoupling) sequence for different number of pi pulses.
#         To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
#         Rhode Schwarz SGS100A is used and it has IQ modulation.
#         For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
#         The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
#         Tau/2 is the time between the center of the pulses!
#
#         - Ziwei Qiu 9/27/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -10.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         Parameter('read_out', [
#             Parameter('meas_len', 180, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 400, int,
#                       'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#         ]),
#         Parameter('fit', False, bool, 'fit the data with exponential decay'),
#         Parameter('NV_tracking', [
#             Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
#             Parameter('tracking_num', 50000, int,
#                       'number of recent APD windows used for calculating current counts, suggest 50000'),
#             Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#             Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {'optimize': OptimizeNoLaser}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#                 # unit: cycle of 4ns
#                 pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
#                 pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
#                 pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)
#
#                 # unit: ns
#                 config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
#                 config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
#                 config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)
#
#                 self.qm = self.qmm.open_qm(config)
#
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 tracking_num = self.settings['NV_tracking']['tracking_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau times between the first pi/2 and pi pulse edges in cycles of 4ns
#                 tau_start = round(self.settings['tau_times'][
#                                       'min_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
#                 tau_start = int(np.max([tau_start, 4]))
#                 tau_end = round(self.settings['tau_times'][
#                                     'max_time'] / num_of_evolution_blocks / 2 / 4 - pi2_time / 2 - pi_time / 2)
#                 tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks / 2 / 4)
#                 tau_step = int(np.max([tau_step, 1]))
#
#                 self.t_vec = [int(a_) for a_ in np.arange(int(tau_start), int(tau_end), int(tau_step))]
#
#                 # total evolution time in ns
#                 self.tau_total = (np.array(self.t_vec) + pi2_time / 2 + pi_time / 2) * 8 * num_of_evolution_blocks
#
#                 t_vec = self.t_vec
#                 t_num = len(self.t_vec)
#                 print('Time between the first pi/2 and pi pulse edges [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total evolution times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 IF_freq = self.settings['mw_pulses']['IF_frequency']
#
#                 def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                         wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', 'e_field1', duration=t)  # e field 1 on
#                             wait(t, 'e_field2')  # e field 2 off
#                     else:
#                         wait(2 * t + pi2_time, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                             wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                 def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                         wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                         wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                         wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t + pi2_time, 'qubit')
#                     if efield:
#                         wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                         play('trig', 'e_field2', duration=2 * t + pi2_time)  # e field 2 on
#                         wait(2 * t + pi2_time, 'e_field1')  # e field 1 off
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', 'e_field1', duration=t)  # e field 1 on
#                             wait(t, 'e_field2')  # e field 2 off
#                     else:
#                         wait(2 * t + pi2_time, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
#                             wait(2 * t + pi2_time, 'e_field2')  # e field 2 off
#
#                 def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='e_field1', off='e_field2'):
#                     play('pi' * amp(IF_amp), 'qubit')
#                     if is_last_block:
#                         wait(t, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', on, duration=t)  # e field  on
#                             wait(t, off)  # e field  off
#                     else:
#                         wait(2 * t + pi2_time, 'qubit')
#                         if efield:
#                             wait(pi_time, 'e_field1', 'e_field2')  # e field 1, 2 off
#                             play('trig', on, duration=2 * t + pi2_time)  # e field  on
#                             wait(2 * t + pi2_time, off)  # e field  off
#
#                 def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
#                                    number_of_pulse_blocks=number_of_pulse_blocks, efield=False):
#                     if decoupling_seq_type == 'XY4':
#                         if number_of_pulse_blocks == 2:
#                             xy4_block(is_last_block=False, efield=efield)
#                         elif number_of_pulse_blocks > 2:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy4_block(is_last_block=False, efield=efield)
#                         xy4_block(is_last_block=True, efield=efield)
#
#                     elif decoupling_seq_type == 'XY8':
#                         if number_of_pulse_blocks == 2:
#                             xy8_block(is_last_block=False, efield=efield)
#                         elif number_of_pulse_blocks > 2:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy8_block(is_last_block=False, efield=efield)
#                         xy8_block(is_last_block=True, efield=efield)
#                     else:
#                         if number_of_pulse_blocks == 2:
#                             spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
#                         elif number_of_pulse_blocks > 2:
#                             if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
#                                 with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
#                                     spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
#                                     spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')
#
#                             else:
#                                 with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
#                                     spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
#                                     spin_echo_block(is_last_block=False, efield=efield, on='e_field1', off='e_field2')
#                                 spin_echo_block(is_last_block=False, efield=efield, on='e_field2', off='e_field1')
#
#                         if number_of_pulse_blocks % 2 == 1:  # odd number of pi pulses
#                             spin_echo_block(is_last_block=True, efield=efield, on='e_field2', off='e_field1')
#                         else:  # even number of pi pulses
#                             spin_echo_block(is_last_block=True, efield=efield, on='e_field1', off='e_field2')
#
#                 # define the qua program
#                 with program() as acsensing:
#
#                     update_frequency('qubit', IF_freq)
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#                     t = declare(int)
#                     n = declare(int)
#                     k = declare(int)
#                     i = declare(int)
#
#                     # the following variable is used to flag tracking
#                     assign(IO1, False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#
#                         # Check if tracking is called
#                         with while_(IO1):
#                             play('trig', 'laser', duration=10000)
#
#                         with for_each_(t, t_vec):
#
#                             with for_(k, 0, k < 6, k + 1):
#                                 reset_frame('qubit')
#
#                                 with if_(k == 0):  # +x readout, no E field
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train()
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 1):  # -x readout, no E field
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train()
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 2):  # +x readout, with E field
#                                     align('qubit', 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     play('trig', 'e_field1', duration=t)  # e field 1 on
#                                     wait(t, 'e_field2')  # e field 2 off
#
#                                     pi_pulse_train(efield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                 with if_(k == 3):  # -x readout, with E field
#                                     align('qubit', 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     play('trig', 'e_field1', duration=t)  # e field 1 on
#                                     wait(t, 'e_field2')  # e field 2 off
#                                     pi_pulse_train(efield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                 with if_(k == 4):  # +y readout, with E field
#                                     align('qubit', 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     play('trig', 'e_field1', duration=t)  # e field 1 on
#                                     wait(t, 'e_field2')  # e field 2 off
#                                     pi_pulse_train(efield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     z_rot(np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                 with if_(k == 5):  # -y readout, with E field
#                                     align('qubit', 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     play('trig', 'e_field1', duration=t)  # e field 1 on
#                                     wait(t, 'e_field2')  # e field 2 off
#                                     pi_pulse_train(efield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi * 1.5, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     wait(pi2_time, 'e_field1')  # e field 1 off
#                                     wait(pi2_time, 'e_field2')  # e field 2 off
#
#                                 align('qubit', 'laser', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                                 align('qubit', 'laser', 'readout1', 'readout2')
#                                 wait(laser_off, 'qubit')
#
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(n, rep_num_st)
#                                 save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 6).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(acsensing)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(acsensing, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
#                                        flags=['skip-add-implicit-align'])
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         try:
#             job = self.qm.execute(qua_program, flags=['skip-add-implicit-align'])
#             counts_out_num = 0  # count the number of times the NV fluorescence is out of tolerance
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             vec_handle = job.result_handles.get("live_data")
#             progress_handle = job.result_handles.get("live_rep_num")
#             tracking_handle = job.result_handles.get("current_counts")
#
#             vec_handle.wait_for_values(1)
#             self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None, 'ref_cnts': None,
#                          'pdd_norm': None, 'esig_x_norm': None, 'esig_y_norm': None}
#
#             ref_counts = -1
#             tolerance = self.settings['NV_tracking']['tolerance']
#
#             while vec_handle.is_processing():
#                 try:
#                     vec = vec_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     echo_avg = vec * 1e6 / self.meas_len
#                     self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
#                                       'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})
#
#                     if self.data['signal_avg_vec'][:, 0].mean() > self.data['signal_avg_vec'][:, 1].mean():
#
#                         self.data['pdd_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                                                 (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                         self.data['esig_x_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 2] - self.data['signal_avg_vec'][:, 3]) / \
#                                                    (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
#                                                                                         3])
#
#                         self.data['esig_y_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 4] - self.data['signal_avg_vec'][:, 5]) / \
#                                                    (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:,
#                                                                                         5])
#                     else:
#                         self.data['pdd_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
#                                                 (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                         self.data['esig_x_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 3] - self.data['signal_avg_vec'][:, 2]) / \
#                                                    (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
#                                                                                         3])
#
#                         self.data['esig_y_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 5] - self.data['signal_avg_vec'][:, 4]) / \
#                                                    (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:,
#                                                                                         5])
#
#                 # # do fitting
#                 # if self.settings['fit']:
#                 #     try:
#                 #         echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True,
#                 #                                   verbose=False)
#                 #         self.data['fits'] = echo_fits
#                 #     except Exception as e:
#                 #         print('** ATTENTION **')
#                 #         print(e)
#                 #     else:
#                 #         self.data['T2'] = self.data['fits'][1]
#
#                 try:
#                     current_rep_num = progress_handle.fetch_all()
#                     current_counts_vec = tracking_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
#                     # print('current_counts', current_counts_kcps)
#                     # print('ref_counts', ref_counts)
#                     if ref_counts < 0:
#                         ref_counts = current_counts_kcps
#
#                     if self.settings['NV_tracking']['on']:
#                         if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
#                                 1 - tolerance):
#                             counts_out_num += 1
#
#                             print(
#                                 '--> No.{:d}: Current counts {:0.2f}kcps is out of range [{:0.2f}kcps, {:0.2f}kcps].'.format(
#                                     counts_out_num, current_counts_kcps, ref_counts * (1 - tolerance),
#                                                                          ref_counts * (1 + tolerance)))
#
#                             if counts_out_num > 5:
#                                 print('** Start tracking **')
#                                 self.qm.set_io1_value(True)
#                                 self.NV_tracking()
#                                 try:
#                                     self.qm.set_io1_value(False)
#                                 except Exception as e:
#                                     print('** ATTENTION **')
#                                     print(e)
#                                 else:
#                                     counts_out_num = 0
#                                     ref_counts = self.settings['NV_tracking']['ref_counts']
#
#                     self.progress = current_rep_num * 100. / self.settings['rep_num']
#                     self.updateProgress.emit(int(self.progress))
#
#                 if self._abort:
#                     # job.halt() # Currently not implemented. Will be implemented in future releases.
#                     self.qm.execute(job_stop)
#                     break
#
#                 time.sleep(0.8)
#
#         # full_res = job.result_handles.get("total_counts")
#         # full_res_vec = full_res.fetch_all()
#         # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def NV_tracking(self):
#         # need to put a find_NV script here
#         # time.sleep(15)
#         self.flag_optimize_plot = True
#         self.scripts['optimize'].run()
#
#     def plot(self, figure_list):
#         super(ACSensing, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#             axes_list[1].clear()
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="pdd +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="pdd -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="esig +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="esig -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")
#
#             axes_list[1].set_xlabel('Total tau [ns]')
#             axes_list[1].set_ylabel('Normalized Counts')
#             axes_list[1].legend()
#
#             axes_list[0].clear()
#             axes_list[0].plot(data['t_vec'], data['pdd_norm'], label="pdd")
#             axes_list[0].plot(data['t_vec'], data['esig_x_norm'], label="esig x")
#             axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
#             axes_list[0].set_xlabel('Total tau [ns]')
#             axes_list[0].set_ylabel('Contrast')
#
#             # if 'fits' in data.keys():
#             #     tau = data['t_vec']
#             #     fits = data['fits']
#             #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
#             #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))
#             axes_list[0].legend()
#             axes_list[0].set_title(
#                 'AC Sensing\n{:s} {:d} block(s)\nRef fluor: {:0.1f}kcps\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
#                     self.settings['decoupling_seq']['type'],
#                     self.settings['decoupling_seq']['num_of_pulse_blocks'],
#                     self.data['ref_cnts'], self.settings['mw_pulses']['pi_half_pulse_time'],
#                     self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
#                     self.settings['mw_pulses']['mw_power'],
#                     self.settings['mw_pulses']['mw_frequency'] * 1e-9,
#                     self.settings['mw_pulses']['IF_amp'],
#                     self.settings['mw_pulses']['IF_frequency'] * 1e-6))
#
#     def _update_plot(self, axes_list):
#         if self._current_subscript_stage['current_subscript'] is self.scripts['optimize'] and self.scripts[
#             'optimize'].is_running:
#             if self.flag_optimize_plot:
#                 self.scripts['optimize']._plot([axes_list[1]])
#                 self.flag_optimize_plot = False
#             else:
#                 self.scripts['optimize']._update_plot([axes_list[1]])
#         else:
#             self._plot(axes_list)
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list


# class PDDQMwork(Script):
#     """
#         This script runs a PDD (Periodic Dynamical Decoupling) sequence for different number of pi pulses.
#         To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
#         Rhode Schwarz SGS100A is used and it has IQ modulation.
#         For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
#         The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
#         Tau/2 is the time between the edge of the pulses!
#
#         - Ziwei Qiu 9/6/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -10.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         Parameter('read_out', [
#             Parameter('meas_len', 250, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 440, int,
#                       'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#         ]),
#         Parameter('fit', False, bool, 'fit the data with exponential decay'),
#         Parameter('NV_tracking', [
#             Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
#             Parameter('tracking_num', 10000, int, 'number of recent APD windows used for calculating current counts'),
#             Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#             Parameter('tolerance', 0.3, float, 'define the reference range (1+/-tolerance)*ref')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {'optimize': optimize}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#                 # unit: cycle of 4ns
#                 pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
#                 pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
#                 pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)
#
#                 # unit: ns
#                 config['pulses']['pi_pulse']['length'] = int(pi_time*4)
#                 config['pulses']['pi2_pulse']['length'] = int(pi2_time*4)
#                 config['pulses']['pi32_pulse']['length'] = int(pi32_time*4)
#
#                 self.qm = self.qmm.open_qm(config)
#
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 tracking_num = self.settings['NV_tracking']['tracking_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau time between MW pulses (i.e. half of the each evolution block time)
#                 tau_start = round(self.settings['tau_times']['min_time'] / num_of_evolution_blocks)
#                 tau_end = round(self.settings['tau_times']['max_time'] / num_of_evolution_blocks)
#                 tau_step = np.max([round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks), 1])
#                 print('tau_step:', tau_step)
#
#                 self.t_vec = [int(a_) for a_ in
#                               np.arange(np.max([int(np.ceil(tau_start / 8)), 4]), int(np.ceil(tau_end / 8)),
#                                         int(np.ceil(tau_step / 8)))]
#
#                 self.tau_total = np.array(self.t_vec) * 8 * num_of_evolution_blocks
#                 self.tau_total_cntr = self.tau_total - (num_of_evolution_blocks-1)*pi_time - 2*pi2_time
#
#                 t_vec = self.t_vec
#                 t_num = len(self.t_vec)
#                 print('Half evolution block [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total_tau_times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 IF_freq = self.settings['mw_pulses']['IF_frequency']
#
#                 def xy4_block(is_last_block, IF_amp=IF_amp):
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                     else:
#                         wait(2 * t, 'qubit')
#
#                 def xy8_block(is_last_block, IF_amp=IF_amp):
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                     else:
#                         wait(2 * t, 'qubit')
#
#                 def spin_echo_block(is_last_block, IF_amp=IF_amp):
#                     play('pi' * amp(IF_amp), 'qubit')
#                     if is_last_block:
#                         wait(t, 'qubit')
#                     else:
#                         wait(2 * t, 'qubit')
#
#                 def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
#                                    number_of_pulse_blocks=number_of_pulse_blocks):
#                     if decoupling_seq_type == 'XY4':
#                         if number_of_pulse_blocks > 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy4_block(is_last_block=False)
#                         xy4_block(is_last_block=True)
#
#                     elif decoupling_seq_type == 'XY8':
#                         if number_of_pulse_blocks > 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy8_block(is_last_block=False)
#                         xy8_block(is_last_block=True)
#                     else:
#                         if number_of_pulse_blocks > 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 spin_echo_block(is_last_block=False)
#                         spin_echo_block(is_last_block=True)
#
#                 # define the qua program
#                 with program() as pdd:
#                     update_frequency('qubit', IF_freq)
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#
#                     t = declare(int)
#                     n = declare(int)
#                     k = declare(int)
#                     i = declare(int)
#
#                     # the following two variable are used to flag tracking
#                     assign(IO1, False)
#                     flag = declare(bool, value=False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         # Check if tracking is called
#                         assign(flag, IO1)
#                         with if_(flag):
#                             pause()
#
#                         with for_each_(t, t_vec):
#                             with for_(k, 0, k < 2, k + 1):
#
#                                 reset_frame('qubit')
#
#                                 with if_(k == 0):  # +x readout
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train()
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with else_():  # -x readout
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train()
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 align('qubit', 'laser', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                                 align('qubit', 'laser', 'readout1', 'readout2')
#                                 wait(laser_off, 'qubit')
#
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(n, rep_num_st)
#                                 save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 2).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(pdd)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(pdd, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
#             # job.get_simulated_samples().con1.plot()
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         try:
#             job = self.qm.execute(qua_program)
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             vec_handle = job.result_handles.get("live_data")
#             progress_handle = job.result_handles.get("live_rep_num")
#             tracking_handle = job.result_handles.get("current_counts")
#
#             vec_handle.wait_for_values(1)
#             self.data = {'t_vec': np.array(self.tau_total_cntr), 'signal_avg_vec': None, 'ref_cnts': None,
#                          'signal_norm': None}
#
#             ref_counts = -1
#             tolerance = self.settings['NV_tracking']['tolerance']
#
#             while vec_handle.is_processing():
#                 try:
#                     vec = vec_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     echo_avg = vec * 1e6 / self.meas_len
#                     self.data.update({'signal_avg_vec': 2 * echo_avg / (echo_avg[0, 0] + echo_avg[0, 1]),
#                                       'ref_cnts': (echo_avg[0, 0] + echo_avg[0, 1]) / 2})
#                     # self.data['signal_norm'] = 2 * (
#                     #         self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                     #                            (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
#                     #                                                                 1])
#
#                     if self.data['signal_avg_vec'][:, 0].mean() > self.data['signal_avg_vec'][:, 1].mean():
#                         self.data['signal_norm'] = 2 * (
#                                     self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                                                    (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
#                                                                                         1])
#                     else:
#                         self.data['signal_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
#                                                    (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:,
#                                                                                         1])
#
#                 # do fitting
#                 if self.settings['fit']:
#                     try:
#                         echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset=True, verbose=False)
#                         self.data['fits'] = echo_fits
#                     except Exception as e:
#                         print('** ATTENTION **')
#                         print(e)
#                     else:
#                         self.data['T2'] = self.data['fits'][1]
#
#                 try:
#                     current_rep_num = progress_handle.fetch_all()
#                     current_counts_vec = tracking_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
#                     # print('current kcps', current_counts_kcps)
#                     # print('ref counts', ref_counts)
#                     if ref_counts < 0:
#                         ref_counts = current_counts_kcps
#                     if self.settings['NV_tracking']['on']:
#                         if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
#                                 1 - tolerance):
#                             self.qm.set_io1_value(True)
#                             self.NV_tracking()
#                             self.qm.set_io1_value(False)
#                             job.resume()
#                     self.progress = current_rep_num * 100. / self.settings['rep_num']
#                     self.updateProgress.emit(int(self.progress))
#
#                 if self._abort:
#                     # job.halt() # Currently not implemented. Will be implemented in future releases.
#                     self.qm.execute(job_stop)
#                     break
#
#                 time.sleep(0.8)
#
#         # full_res = job.result_handles.get("total_counts")
#         # full_res_vec = full_res.fetch_all()
#         # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def NV_tracking(self):
#         # need to put a find_NV script here
#         self.scripts['optimize'].run()
#
#     def plot(self, figure_list):
#         super(PDDQM, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#
#             axes_list[1].clear()
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="+x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="-x")
#             axes_list[1].set_xlabel('Total tau [ns]')
#             axes_list[1].set_ylabel('Normalized Counts')
#             axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)
#
#             axes_list[0].clear()
#             axes_list[0].plot(data['t_vec'], data['signal_norm'], label="signal")
#             axes_list[0].set_xlabel('Total tau [ns]')
#             axes_list[0].set_ylabel('Contrast')
#
#             if 'fits' in data.keys():
#                 tau = data['t_vec']
#                 fits = data['fits']
#                 tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
#                 axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]),
#                                   label="exp fit (T2={:2.1f} ns)".format(fits[1]))
#             axes_list[0].legend(loc='upper right')
#             axes_list[0].set_title(
#                 'Periodic Dynamical Decoupling\n{:s} {:d} block(s)\nRef fluor: {:0.1f}kcps\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
#                     self.settings['decoupling_seq']['type'],
#                     self.settings['decoupling_seq']['num_of_pulse_blocks'],
#                     self.data['ref_cnts'], self.settings['mw_pulses']['pi_half_pulse_time'],
#                     self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
#                     self.settings['mw_pulses']['mw_power'],
#                     self.settings['mw_pulses']['mw_frequency'] * 1e-9,
#                     self.settings['mw_pulses']['IF_amp'],
#                     self.settings['mw_pulses']['IF_frequency'] * 1e-6))
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list


pass
# AC sensing is more complicated than spin-echo. Because I want to measure both +/-x and +/-y. To save memory, I need to
# use a real-time qua variable, hwoever with if_(k=?) take additional time because I have many k values.
# class ACSensing(Script):
#     """
#         This script does AC magnetometry / electrometry using an NV center.
#         To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
#         Rhode Schwarz SGS100A is used and it has IQ modulation.
#         For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
#         The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
#         Tau/2 is the time between the edge of the pulses!
#
#         - Ziwei Qiu 9/7/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -10.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         Parameter('read_out', [
#             Parameter('meas_len', 250, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 440, int,
#                       'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#         ]),
#         Parameter('NV_tracking', [
#             Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
#             Parameter('tracking_num', 10000, int, 'number of recent APD windows used for calculating current counts'),
#             Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#             Parameter('tolerance', 0.3, float, 'define the reference range (1+/-tolerance)*ref')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#                 config['pulses']['pi_pulse']['length'] = int(round(self.settings['mw_pulses']['pi_pulse_time'] / 4) * 4)
#                 pi_time = int(config['pulses']['pi_pulse']['length'] / 4)
#
#                 config['pulses']['pi2_pulse']['length'] = int(
#                     round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4) * 4)
#                 pi_half_time = int(config['pulses']['pi2_pulse']['length'] / 4)
#
#                 config['pulses']['pi32_pulse']['length'] = int(
#                     round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4) * 4)
#                 three_pi_half_time = int(config['pulses']['pi32_pulse']['length'] / 4)
#
#                 self.qm = self.qmm.open_qm(config)
#
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 tracking_num = self.settings['NV_tracking']['tracking_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau time between MW pulses (half of the each evolution block time)
#                 tau_start = round(self.settings['tau_times']['min_time'] / num_of_evolution_blocks)
#                 tau_end = round(self.settings['tau_times']['max_time'] / num_of_evolution_blocks)
#                 tau_step = np.max([round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks), 1])
#
#                 self.t_vec = [int(a_) for a_ in
#                               np.arange(np.max([int(np.ceil(tau_start / 8)), 4]), int(np.ceil(tau_end / 8)),
#                                         int(np.ceil(tau_step / 8)))]
#
#                 self.tau_total = np.array(self.t_vec) * 8 * num_of_evolution_blocks
#
#                 t_vec = self.t_vec
#                 t_num = len(self.t_vec)
#                 print('Half evolution block [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total_tau_times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 IF_freq = self.settings['mw_pulses']['IF_frequency']
#
#                 def xy4_block(is_last_block, with_acfield=False, IF_amp=IF_amp):
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time + 2 * t, 'e_field1')
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                         if with_acfield:
#                             wait(pi_time + t, 'e_field1')
#                     else:
#                         wait(2 * t, 'qubit')
#                         if with_acfield:
#                             wait(pi_time + 2 * t, 'e_field1')
#
#                 def xy8_block(is_last_block, with_acfield=False, IF_amp=IF_amp):
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time + 2 * t, 'e_field1')
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time + 2 * t, 'e_field1')
#
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time + 2 * t, 'e_field1')
#
#                     z_rot(np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                     wait(2 * t, 'qubit')
#                     if with_acfield:
#                         wait(pi_time, 'e_field1')
#                         play('trig', 'e_field1', duration=2 * t)
#
#                     z_rot(-np.pi / 2, 'qubit')
#                     play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                     if is_last_block:
#                         wait(t, 'qubit')
#                         if with_acfield:
#                             wait(pi_time + t, 'e_field1')
#                     else:
#                         wait(2 * t, 'qubit')
#                         if with_acfield:
#                             wait(pi_time + 2 * t, 'e_field1')
#
#                 def spin_echo_block(is_last_block, with_acfield=False, acfield_on=True, IF_amp=IF_amp):
#                     play('pi' * amp(IF_amp), 'qubit')
#
#                     if is_last_block:
#
#                         wait(t, 'qubit')
#                         if with_acfield:
#                             if acfield_on:
#                                 wait(pi_time, 'e_field1')
#                                 play('trig', 'e_field1', duration=t)
#                             else:
#                                 wait(pi_time + t, 'e_field1')
#
#                     else:
#                         wait(2 * t, 'qubit')
#                         if with_acfield:
#                             if acfield_on:
#                                 wait(pi_time, 'e_field1')
#                                 play('trig', 'e_field1', duration=2 * t)
#                             else:
#                                 wait(pi_time + 2 * t, 'e_field1')
#
#                 def pi_pulse_train(decoupling_seq_type=self.settings['decoupling_seq']['type'],
#                                    number_of_pulse_blocks=number_of_pulse_blocks, with_acfield=False):
#                     if decoupling_seq_type == 'XY4':
#                         if number_of_pulse_blocks > 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy4_block(is_last_block=False, with_acfield=with_acfield)
#                         xy4_block(is_last_block=True, with_acfield=with_acfield)
#
#                     elif decoupling_seq_type == 'XY8':
#                         if number_of_pulse_blocks > 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 1):
#                                 xy8_block(is_last_block=False, with_acfield=with_acfield)
#                         xy8_block(is_last_block=True, with_acfield=with_acfield)
#                     else:
#                         if number_of_pulse_blocks == 1:
#                             spin_echo_block(is_last_block=True, with_acfield=with_acfield, acfield_on=True)
#                         elif number_of_pulse_blocks % 2 == 1:
#                             with for_(i, 0, i < number_of_pulse_blocks - 1, i + 2):
#                                 spin_echo_block(is_last_block=False, with_acfield=with_acfield, acfield_on=True)
#                                 spin_echo_block(is_last_block=False, with_acfield=with_acfield, acfield_on=False)
#                             spin_echo_block(is_last_block=True, with_acfield=with_acfield, acfield_on=True)
#                         else:
#                             with for_(i, 0, i < number_of_pulse_blocks - 2, i + 2):
#                                 spin_echo_block(is_last_block=False, with_acfield=with_acfield, acfield_on=True)
#                                 spin_echo_block(is_last_block=False, with_acfield=with_acfield, acfield_on=False)
#                             spin_echo_block(is_last_block=False, with_acfield=with_acfield, acfield_on=True)
#                             spin_echo_block(is_last_block=True, with_acfield=with_acfield, acfield_on=False)
#
#                 # define the qua program
#
#                 with program() as ac_sensing:
#                     update_frequency('qubit', IF_freq)
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#
#                     t = declare(int)
#                     n = declare(int)
#                     k = declare(int)
#                     i = declare(int)
#
#                     # the following two variables are used to flag tracking
#                     assign(IO1, False)
#                     flag = declare(bool, value=False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         # Check if tracking is called
#                         assign(flag, IO1)
#                         with if_(flag):
#                             pause()
#
#                         with for_each_(t, t_vec):
#                             with for_(k, 0, k < 6, k + 1):
#                                 # align('qubit', 'laser', 'e_field1', 'readout1', 'readout2')
#                                 # for k in [-1, 0, 1, 2, 3, 4, 5]:
#
#                                 # k=0, +x readout, k=1, -x readout, no electric field
#                                 # k=2, +x readout, k=3, -x readout, with electric field
#                                 # k=4, +y readout, k=5, -y readout, with electric field
#
#                                 reset_frame('qubit')
#                                 align('qubit', 'laser', 'e_field1', 'readout1', 'readout2')
#
#                                 # for many k, with if_ takes additional time to process
#                                 with if_(k == 0):  # +x readout, no electric field
#                                     # if k == 0:
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train(with_acfield=False)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 1):  # -x readout, no electric field
#                                     # elif k == 1:
#                                     # align('qubit', 'laser', 'e_field1', 'readout1', 'readout2')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train(with_acfield=False)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 2):  # +x readout, with electric field
#                                     # elif k == 2:
#                                     # play('trig', 'e_field1', duration=20)
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#
#                                     wait(analog_digital_align_artifect + pi_half_time + t, 'e_field1')
#                                     pi_pulse_train(with_acfield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 # elif k == 3:
#                                 with if_(k == 3):  # -x readout, with electric field
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     wait(analog_digital_align_artifect + pi_half_time + t, 'e_field1')
#                                     pi_pulse_train(with_acfield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 # elif k == 4:
#                                 with if_(k == 4):  # +y readout, with electric field
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     wait(analog_digital_align_artifect + pi_half_time + t, 'e_field1')
#                                     pi_pulse_train(with_acfield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 # elif k == 5:
#                                 with if_(k == 5):  # +x readout, with electric field
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     wait(analog_digital_align_artifect + pi_half_time + t, 'e_field1')
#                                     pi_pulse_train(with_acfield=True)
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(1.5 * np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 align('qubit', 'laser', 'e_field1', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                                 align('qubit', 'laser', 'e_field1', 'readout1', 'readout2')
#                                 wait(laser_off, 'qubit')
#
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(n, rep_num_st)
#                                 save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 6).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#                 #
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(ac_sensing)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(ac_sensing, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
#             # job.get_simulated_samples().con1.plot()
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         try:
#             job = self.qm.execute(qua_program)
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             vec_handle = job.result_handles.get("live_data")
#             progress_handle = job.result_handles.get("live_rep_num")
#             tracking_handle = job.result_handles.get("current_counts")
#
#             vec_handle.wait_for_values(1)
#             self.data = {'t_vec': np.array(self.tau_total), 'ref_cnts': None, 'signal_avg_vec': None, 'pdd_norm': None,
#                          'esig_x_norm': None, 'esig_y_norm': None}
#
#             ref_counts = -1
#             tolerance = self.settings['NV_tracking']['tolerance']
#
#             while vec_handle.is_processing():
#                 try:
#                     vec = vec_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     sig_avg = vec * 1e6 / self.meas_len
#                     self.data.update({'signal_avg_vec': sig_avg / sig_avg[0, 0], 'ref_cnts': sig_avg[0, 0]})
#
#                     self.data['pdd_norm'] = 2 * (
#                         self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                                             (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                     self.data['esig_x_norm'] = 2 * (
#                         self.data['signal_avg_vec'][:, 2] - self.data['signal_avg_vec'][:, 3]) / \
#                                                (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:, 3])
#
#                     self.data['esig_y_norm'] = 2 *(
#                         self.data['signal_avg_vec'][:, 4] - self.data['signal_avg_vec'][:, 5]) / \
#                                                (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:, 5])
#
#                 # do fitting
#                 # try:
#                 #     echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset = True, verbose = False)
#                 #     self.data['fits'] = echo_fits
#                 # except Exception as e:
#                 #     print('** ATTENTION **')
#                 #     print(e)
#                 # else:
#                 #     self.data['T2'] = self.data['fits'][1]
#
#                 try:
#                     current_rep_num = progress_handle.fetch_all()
#                     current_counts_vec = tracking_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
#                     # print('current kcps', current_counts_kcps)
#                     # print('ref counts', ref_counts)
#                     if ref_counts < 0:
#                         ref_counts = current_counts_kcps
#                     if self.settings['NV_tracking']['on']:
#                         if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
#                                 1 - tolerance):
#                             self.qm.set_io1_value(True)
#                             self.NV_tracking()
#                             self.qm.set_io1_value(False)
#                             job.resume()
#                     self.progress = current_rep_num * 100. / self.settings['rep_num']
#                     self.updateProgress.emit(int(self.progress))
#
#                 if self._abort:
#                     # job.halt() # Currently not implemented. Will be implemented in future releases.
#                     self.qm.execute(job_stop)
#                     break
#
#                 time.sleep(0.8)
#
#             # full_res = job.result_handles.get("total_counts")
#             # full_res_vec = full_res.fetch_all()
#             # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def NV_tracking(self):
#         # need to put a find_NV script here
#         time.sleep(5)
#
#     def plot(self, figure_list):
#         super(ACSensing, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#             axes_list[1].clear()
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="pdd +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="pdd -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="esig +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="esig -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")
#
#             axes_list[1].set_xlabel('Total tau [ns]')
#             axes_list[1].set_ylabel('Normalized Counts')
#             # axes_list[1].legend()
#             axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)
#
#             axes_list[0].clear()
#             axes_list[0].plot(data['t_vec'], data['pdd_norm'], label="pdd")
#             axes_list[0].plot(data['t_vec'], data['esig_x_norm'], label="esig x")
#             axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
#             axes_list[0].set_xlabel('Total tau [ns]')
#             axes_list[0].set_ylabel('Contrast')
#
#             # if 'fits' in data.keys():
#             #     tau = data['t_vec']
#             #     fits = data['fits']
#             #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
#             #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))
#             axes_list[0].legend(loc='upper right')
#             axes_list[0].set_title(
#                 'AC Sensing\n{:s} {:d} block(s)\nRef fluor: {:0.1f}kcps\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
#                     self.settings['decoupling_seq']['type'],
#                     self.settings['decoupling_seq']['num_of_pulse_blocks'],
#                     self.data['ref_cnts'], self.settings['mw_pulses']['pi_half_pulse_time'],
#                     self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
#                     self.settings['mw_pulses']['mw_power'],
#                     self.settings['mw_pulses']['mw_frequency'] * 1e-9,
#                     self.settings['mw_pulses']['IF_amp'],
#                     self.settings['mw_pulses']['IF_frequency'] * 1e-6))
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list


pass

### The following version is working. DO NOT Change!!
# class ACSensing(Script):
#     """
#         This script does AC magnetometry / electrometry using an NV center.
#         To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
#         Rhode Schwarz SGS100A is used and it has IQ modulation.
#         For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
#         The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
#         Tau/2 is the time between the edge of the pulses!
#
#         - Ziwei Qiu 9/11/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -10.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         Parameter('analog_digital_align_artifect', 140, int, 'artifect between analog and digital pulses'),
#         Parameter('read_out', [
#             Parameter('meas_len', 250, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 440, int,
#                       'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#         ]),
#         Parameter('NV_tracking', [
#             Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
#             Parameter('tracking_num', 10000, int, 'number of recent APD windows used for calculating current counts'),
#             Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#             Parameter('tolerance', 0.3, float, 'define the reference range (1+/-tolerance)*ref')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#
#                 pi_time_config = int(round(self.settings['mw_pulses']['pi_pulse_time'] / 4) * 4)
#                 pi2_time_config = int(round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4) * 4)
#                 pi32_time_config = int(round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4) * 4)
#
#                 config['pulses']['pi_pulse']['length'] = pi_time_config
#                 config['pulses']['pi2_pulse']['length'] = pi2_time_config
#                 config['pulses']['pi32_pulse']['length'] = pi32_time_config
#
#                 self.qm = self.qmm.open_qm(config)
#
#                 pi_time = int(config['pulses']['pi_pulse']['length'] / 4)
#                 pi_half_time = int(config['pulses']['pi2_pulse']['length'] / 4)
#                 three_pi_half_time = int(config['pulses']['pi32_pulse']['length'] / 4)
#
#
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 tracking_num = self.settings['NV_tracking']['tracking_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau time between MW pulses (half of the each evolution block time)
#                 tau_start = round(self.settings['tau_times']['min_time'] / num_of_evolution_blocks)
#                 tau_end = round(self.settings['tau_times']['max_time'] / num_of_evolution_blocks)
#                 tau_step = np.max([round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks), 1])
#
#                 self.t_vec = [int(a_) for a_ in
#                               np.arange(np.max([int(np.ceil(tau_start / 8)), 4]), int(np.ceil(tau_end / 8)),
#                                         int(np.ceil(tau_step / 8)))]
#
#                 self.tau_total = np.array(self.t_vec) * 8 * num_of_evolution_blocks
#                 self.tau_total_cntr = self.tau_total - (
#                         num_of_evolution_blocks - 1) * pi_time_config - 2 * pi2_time_config
#
#                 t_vec = self.t_vec
#                 t_num = len(self.t_vec)
#                 print('Half evolution block [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total_tau_times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 IF_freq = self.settings['mw_pulses']['IF_frequency']
#
#                 analog_digital_align_artifect = round(self.settings['analog_digital_align_artifect'] / 4)
#
#                 def pi_pulse_train_no_efield():
#                     if self.settings['decoupling_seq']['type'] == 'XY4':
#                         for i in range(0, number_of_pulse_blocks):
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                     elif self.settings['decoupling_seq']['type'] == 'XY8':
#                         for i in range(0, number_of_pulse_blocks):
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                     else:
#                         for i in range(0, number_of_pulse_blocks):  # for CPMG or spin-echo
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')
#
#                 def pi_pulse_train_with_efield():
#                     if self.settings['decoupling_seq']['type'] == 'XY4':
#                         for i in range(0, number_of_pulse_blocks):
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#                                 wait(pi_time, 'e_field1', 'e_field2')
#                                 play('trig', 'e_field1', duration=2 * t)
#                                 wait(2 * t, 'e_field2')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field1', duration=2 * t)
#                             wait(2 * t, 'e_field2')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                     elif self.settings['decoupling_seq']['type'] == 'XY8':
#                         for i in range(0, number_of_pulse_blocks):
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#                                 wait(pi_time, 'e_field1', 'e_field2')
#                                 play('trig', 'e_field1', duration=2 * t)
#                                 wait(2 * t, 'e_field2')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field1', duration=2 * t)
#                             wait(2 * t, 'e_field2')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field1', duration=2 * t)
#                             wait(2 * t, 'e_field2')
#
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field1', duration=2 * t)
#                             wait(2 * t, 'e_field2')
#
#                             z_rot(np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                             wait(2 * t, 'qubit')
#                             wait(pi_time, 'e_field1', 'e_field2')
#                             play('trig', 'e_field2', duration=2 * t)
#                             wait(2 * t, 'e_field1')
#
#                             z_rot(-np.pi / 2, 'qubit')
#                             play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                     else:
#                         for i in range(0, number_of_pulse_blocks):  # for CPMG or spin-echo
#                             if i > 0:
#                                 wait(2 * t, 'qubit')
#                                 wait(pi_time, 'e_field1', 'e_field2')
#                                 if i % 2 == 0:
#                                     play('trig', 'e_field1', duration=2 * t)
#                                     wait(2 * t, 'e_field2')
#                                 else:
#                                     play('trig', 'e_field2', duration=2 * t)
#                                     wait(2 * t, 'e_field1')
#
#                             play('pi' * amp(IF_amp), 'qubit')
#
#                 # define the qua program
#                 with program() as ac_sensing:
#                     update_frequency('qubit', IF_freq)
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#
#                     t = declare(int)
#                     n = declare(int)
#                     k = declare(int)
#
#                     # the following two variable are used to tracking NV fluorescence
#                     assign(IO1, False)
#                     flag = declare(bool, value=False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         # Check if tracking is called
#                         assign(flag, IO1)
#                         with if_(flag):
#                             pause()
#
#                         with for_each_(t, t_vec):
#                             with for_(k, 0, k < 6, k + 1):
#                                 # for k in [0,1,2,3,4,5]:
#                                 # k=0, +x readout, k=1, -x readout, no electric field
#                                 # k=2, +x readout, k=3, -x readout, with electric field
#                                 # k=4, +y readout, k=5, -y readout, with electric field
#
#                                 reset_frame('qubit')
#                                 align('qubit', 'laser', 'e_field1', 'e_field2', 'readout1', 'readout2')
#                                 with if_(k == 0):  # +x readout, no electric field
#                                     # align('qubit', 'e_field1', 'e_field2') # this is not allowed in a with if_ flow control
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train_no_efield()
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 1):  # -x readout, no electric field
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     pi_pulse_train_no_efield()
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 2):  # +x readout, with electric field
#
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     wait(pi_half_time, 'e_field1', 'e_field2')
#                                     play('trig', 'e_field1', duration=t)
#                                     wait(t, 'e_field2')
#
#                                     pi_pulse_train_with_efield()
#
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                     wait(pi_time, 'e_field1', 'e_field2')
#                                     if number_of_pulse_blocks % 2 == 0 or self.settings['decoupling_seq']['type'] in [
#                                         'XY4', 'XY8']:
#                                         play('trig', 'e_field1', duration=t)
#                                         wait(t, 'e_field2')
#                                     else:
#                                         play('trig', 'e_field2', duration=t)
#                                         wait(t, 'e_field1')
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 3):  # -x readout, with electric field
#
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     wait(pi_half_time, 'e_field1', 'e_field2')
#                                     play('trig', 'e_field1', duration=t)
#                                     wait(t, 'e_field2')
#
#                                     pi_pulse_train_with_efield()
#
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                     wait(pi_time, 'e_field1', 'e_field2')
#                                     if number_of_pulse_blocks % 2 == 0 or self.settings['decoupling_seq']['type'] in [
#                                         'XY4', 'XY8']:
#                                         play('trig', 'e_field1', duration=t)
#                                         wait(t, 'e_field2')
#                                     else:
#                                         play('trig', 'e_field2', duration=t)
#                                         wait(t, 'e_field1')
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 4):  # +y readout, with electric field
#
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     wait(pi_half_time, 'e_field1', 'e_field2')
#                                     play('trig', 'e_field1', duration=t)
#                                     wait(t, 'e_field2')
#
#                                     pi_pulse_train_with_efield()
#
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                     wait(pi_time, 'e_field1', 'e_field2')
#                                     if number_of_pulse_blocks % 2 == 0 or self.settings['decoupling_seq']['type'] in [
#                                         'XY4', 'XY8']:
#                                         play('trig', 'e_field1', duration=t)
#                                         wait(t, 'e_field2')
#                                     else:
#                                         play('trig', 'e_field2', duration=t)
#                                         wait(t, 'e_field1')
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(np.pi / 2, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 with if_(k == 5):  # +x readout, with electric field
#
#                                     play('pi2' * amp(IF_amp), 'qubit')
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(np.pi / 2, 'qubit')
#                                     wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                     # wait(analog_digital_align_artifect, 'e_field1', 'e_field2')
#                                     wait(pi_half_time, 'e_field1', 'e_field2')
#                                     play('trig', 'e_field1', duration=t)
#                                     wait(t, 'e_field2')
#
#                                     pi_pulse_train_with_efield()
#
#                                     wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                     wait(pi_time, 'e_field1', 'e_field2')
#                                     if number_of_pulse_blocks % 2 == 0 or self.settings['decoupling_seq']['type'] in [
#                                         'XY4', 'XY8']:
#                                         play('trig', 'e_field1', duration=t)
#                                         wait(t, 'e_field2')
#                                     else:
#                                         play('trig', 'e_field2', duration=t)
#                                         wait(t, 'e_field1')
#
#                                     if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                         z_rot(-np.pi / 2, 'qubit')
#
#                                     z_rot(1.5 * np.pi, 'qubit')
#                                     play('pi2' * amp(IF_amp), 'qubit')
#
#                                 align('qubit', 'laser', 'e_field1', 'e_field2', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                                 align('qubit', 'laser', 'e_field1', 'e_field2', 'readout1', 'readout2')
#                                 wait(laser_off, 'qubit')
#
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(n, rep_num_st)
#                                 save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 6).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#                 #
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(ac_sensing)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(ac_sensing, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
#             # job.get_simulated_samples().con1.plot()
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         try:
#             job = self.qm.execute(qua_program)
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             vec_handle = job.result_handles.get("live_data")
#             progress_handle = job.result_handles.get("live_rep_num")
#             tracking_handle = job.result_handles.get("current_counts")
#
#             vec_handle.wait_for_values(1)
#             self.data = {'t_vec': np.array(self.tau_total_cntr), 'ref_cnts': None, 'signal_avg_vec': None,
#                          'pdd_norm': None,
#                          'esig_x_norm': None, 'esig_y_norm': None}
#
#             ref_counts = -1
#             tolerance = self.settings['NV_tracking']['tolerance']
#
#             while vec_handle.is_processing():
#                 try:
#                     vec = vec_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     sig_avg = vec * 1e6 / self.meas_len
#                     self.data.update({'signal_avg_vec': 2 * sig_avg / (sig_avg[0, 0] + sig_avg[0, 1]),
#                                       'ref_cnts': (sig_avg[0, 0] + sig_avg[0, 1]) / 2})
#
#                     if sig_avg[0, 0] > sig_avg[0, 1]:
#
#                         self.data['pdd_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                                                 (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                         self.data['esig_x_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 2] - self.data['signal_avg_vec'][:, 3]) / \
#                                                    (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
#                                                                                         3])
#
#                         self.data['esig_y_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 4] - self.data['signal_avg_vec'][:, 5]) / \
#                                                    (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:,
#                                                                                         5])
#                     else:
#                         self.data['pdd_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 1] - self.data['signal_avg_vec'][:, 0]) / \
#                                                 (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                         self.data['esig_x_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 3] - self.data['signal_avg_vec'][:, 2]) / \
#                                                    (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:,
#                                                                                         3])
#
#                         self.data['esig_y_norm'] = 2 * (
#                                 self.data['signal_avg_vec'][:, 5] - self.data['signal_avg_vec'][:, 4]) / \
#                                                    (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:,
#                                                                                         5])
#
#                 # do fitting
#                 # try:
#                 #     echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset = True, verbose = False)
#                 #     self.data['fits'] = echo_fits
#                 # except Exception as e:
#                 #     print('** ATTENTION **')
#                 #     print(e)
#                 # else:
#                 #     self.data['T2'] = self.data['fits'][1]
#
#                 try:
#                     current_rep_num = progress_handle.fetch_all()
#                     current_counts_vec = tracking_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION **')
#                     print(e)
#                 else:
#                     current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
#                     # print('current kcps', current_counts_kcps)
#                     # print('ref counts', ref_counts)
#                     if ref_counts < 0:
#                         ref_counts = current_counts_kcps
#                     if self.settings['NV_tracking']['on']:
#                         if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
#                                 1 - tolerance):
#                             self.qm.set_io1_value(True)
#                             self.NV_tracking()
#                             self.qm.set_io1_value(False)
#                             job.resume()
#                     self.progress = current_rep_num * 100. / self.settings['rep_num']
#                     self.updateProgress.emit(int(self.progress))
#
#                 if self._abort:
#                     # job.halt() # Currently not implemented. Will be implemented in future releases.
#                     self.qm.execute(job_stop)
#                     break
#
#                 time.sleep(0.8)
#
#             # full_res = job.result_handles.get("total_counts")
#             # full_res_vec = full_res.fetch_all()
#             # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def NV_tracking(self):
#         # need to put a find_NV script here
#         time.sleep(5)
#
#     def plot(self, figure_list):
#         super(ACSensing, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#             axes_list[1].clear()
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="pdd +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="pdd -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="esig +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="esig -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")
#
#             axes_list[1].set_xlabel('Total tau [ns]')
#             axes_list[1].set_ylabel('Normalized Counts')
#             # axes_list[1].legend()
#             axes_list[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)
#
#             axes_list[0].clear()
#             axes_list[0].plot(data['t_vec'], data['pdd_norm'], label="pdd")
#             axes_list[0].plot(data['t_vec'], data['esig_x_norm'], label="esig x")
#             axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
#             axes_list[0].set_xlabel('Total tau [ns]')
#             axes_list[0].set_ylabel('Contrast')
#
#             # if 'fits' in data.keys():
#             #     tau = data['t_vec']
#             #     fits = data['fits']
#             #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
#             #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))
#             axes_list[0].legend(loc='upper right')
#             axes_list[0].set_title(
#                 'AC Sensing\n{:s} {:d} block(s)\nRef fluor: {:0.1f}kcps\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
#                     self.settings['decoupling_seq']['type'],
#                     self.settings['decoupling_seq']['num_of_pulse_blocks'],
#                     self.data['ref_cnts'], self.settings['mw_pulses']['pi_half_pulse_time'],
#                     self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
#                     self.settings['mw_pulses']['mw_power'],
#                     self.settings['mw_pulses']['mw_frequency'] * 1e-9,
#                     self.settings['mw_pulses']['IF_amp'],
#                     self.settings['mw_pulses']['IF_frequency'] * 1e-6))
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list


pass
# class ACSensing2(Script):
#     """
#         This script does AC magnetometry / electrometry using an NV center.
#         To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
#         Rhode Schwarz SGS100A is used and it has IQ modulation.
#         For a single pi-pulse this is a Hahn-echo sequence. For zero pulses this is a Ramsey sequence.
#         The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2
#         Tau/2 is the time between the edge of the pulses!
#
#         - Ziwei Qiu 9/7/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -10.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 96, float, 'time duration of a 3pi/2 pulse (in ns)'),
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         Parameter('read_out', [
#             Parameter('meas_len', 250, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 440, int,
#                       'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#         ]),
#         Parameter('NV_tracking', [
#             Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
#             Parameter('tracking_num', 10000, int, 'number of recent APD windows used for calculating current counts'),
#             Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#             Parameter('tolerance', 0.3, float, 'define the reference range (1+/-tolerance)*ref')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#                 config['pulses']['pi_pulse']['length'] = int(round(self.settings['mw_pulses']['pi_pulse_time'] / 4) * 4)
#                 pi_time = int(config['pulses']['pi_pulse']['length'] / 4)
#
#                 config['pulses']['pi2_pulse']['length'] = int(
#                     round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4) * 4)
#                 pi_half_time = int(config['pulses']['pi2_pulse']['length'] / 4)
#
#                 config['pulses']['pi32_pulse']['length'] = int(
#                     round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4) * 4)
#                 three_pi_half_time = int(config['pulses']['pi32_pulse']['length'] / 4)
#
#                 self.qm = self.qmm.open_qm(config)
#
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 tracking_num = self.settings['NV_tracking']['tracking_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau time between MW pulses (half of the each evolution block time)
#                 tau_start = round(self.settings['tau_times']['min_time'] / num_of_evolution_blocks)
#                 tau_end = round(self.settings['tau_times']['max_time'] / num_of_evolution_blocks)
#                 tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks)
#
#                 self.t_vec = [int(a_) for a_ in
#                               np.arange(np.max([int(np.ceil(tau_start / 8)), 4]), int(np.ceil(tau_end / 8)),
#                                         int(np.ceil(tau_step / 8)))]
#
#                 self.tau_total = np.array(self.t_vec) * 8 * num_of_evolution_blocks
#
#                 t_vec = self.t_vec
#                 t_num = len(self.t_vec)
#                 print('Half evolution block [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total_tau_times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 IF_freq = self.settings['mw_pulses']['IF_frequency']
#
#                 # define the qua program
#                 with program() as ac_sensing:
#                     update_frequency('qubit', IF_freq)
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#
#                     t = declare(int)
#                     n = declare(int)
#
#                     # the following two variable are used to flag tracking
#                     assign(IO1, False)
#                     flag = declare(bool, value=False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         # Check if tracking is called
#                         assign(flag, IO1)
#                         with if_(flag):
#                             pause()
#                         with for_each_(t, t_vec):
#                             for k in [0, 1, 2, 3, 4,
#                                       5]:  # k is not a qua variable, so the program is almost using up the fpga memory...
#                                 # k=0, +x readout, k=1, -x readout, no electric field
#                                 # k=2, +x readout, k=3, -x readout, with electric field
#                                 # k=4, +y readout, k=5, -y readout, with electric field
#
#                                 reset_frame('qubit')
#                                 align('qubit', 'e_field1', 'e_field2')  # align
#                                 play('pi2' * amp(IF_amp), 'qubit')
#                                 if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                     z_rot(np.pi / 2, 'qubit')
#                                 wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
#                                 if k > 1:  # AC field applied
#                                     wait(pi_half_time, 'e_field1', 'e_field2')
#                                     play('trig', 'e_field1', duration=t)
#                                     wait(t, 'e_field2')
#
#                                 if self.settings['decoupling_seq']['type'] == 'XY4':
#                                     for i in range(0, number_of_pulse_blocks):
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             if k > 1:
#                                                 wait(pi_time, 'e_field1', 'e_field2')
#                                                 play('trig', 'e_field1', duration=2 * t)
#                                                 wait(2 * t, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field2', duration=2 * t)
#                                             wait(2 * t, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t)
#                                             wait(2 * t, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field2', duration=2 * t)
#                                             wait(2 * t, 'e_field1')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                                     for i in range(0, number_of_pulse_blocks):
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             if k > 1:
#                                                 wait(pi_time, 'e_field1', 'e_field2')
#                                                 play('trig', 'e_field1', duration=2 * t)
#                                                 wait(2 * t, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field2', duration=2 * t)
#                                             wait(2 * t, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t)
#                                             wait(2 * t, 'e_field2')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                                         wait(2 * t, 'qubit')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field2', duration=2 * t)
#                                             wait(2 * t, 'e_field1')
#
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t)
#                                             wait(2 * t, 'e_field2')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field2', duration=2 * t)
#                                             wait(2 * t, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         if k > 1:
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t)
#                                             wait(2 * t, 'e_field2')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit')  # pi_x
#
#                                 else:
#                                     for i in range(0, number_of_pulse_blocks):  # for CPMG or spin-echo
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             if k > 1:
#                                                 wait(pi_time, 'e_field1', 'e_field2')
#                                                 if i % 2 == 0:
#                                                     play('trig', 'e_field2', duration=2 * t)
#                                                     wait(2 * t, 'e_field1')
#                                                 else:
#                                                     play('trig', 'e_field1', duration=2 * t)
#                                                     wait(2 * t, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit')
#
#                                 wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                 if k > 1:
#                                     wait(pi_time, 'e_field1', 'e_field2')
#                                     if number_of_pulse_blocks % 2 == 1 and self.settings['decoupling_seq']['type'] in [
#                                         'XY4', 'XY8']:
#                                         play('trig', 'e_field1', duration=t)
#                                         wait(t, 'e_field2')
#                                     else:
#                                         play('trig', 'e_field2', duration=t)
#                                         wait(t, 'e_field1')
#
#                                 if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                     z_rot(-np.pi / 2, 'qubit')
#
#                                 if k % 2 == 1:
#                                     z_rot(np.pi, 'qubit')
#                                 if k > 4:
#                                     z_rot(np.pi / 2, 'qubit')
#
#                                 play('pi2' * amp(IF_amp), 'qubit')
#
#                                 align('qubit', 'laser', 'e_field1', 'e_field2', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                                 align('qubit', 'laser', 'e_field1', 'e_field2', 'readout1', 'readout2')
#                                 wait(laser_off, 'qubit')
#
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(n, rep_num_st)
#                                 save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 6).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(ac_sensing)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(ac_sensing, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
#             # job.get_simulated_samples().con1.plot()
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         job = self.qm.execute(qua_program)
#
#         vec_handle = job.result_handles.get("live_data")
#         progress_handle = job.result_handles.get("live_rep_num")
#         tracking_handle = job.result_handles.get("current_counts")
#
#         vec_handle.wait_for_values(1)
#         self.data = {'t_vec': np.array(self.tau_total), 'ref_cnts': None, 'signal_avg_vec': None, 'pdd_norm': None,
#                      'esig_x_norm': None, 'esig_y_norm': None}
#
#         ref_counts = -1
#         tolerance = self.settings['NV_tracking']['tolerance']
#
#         while vec_handle.is_processing():
#             try:
#                 vec = vec_handle.fetch_all()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 sig_avg = vec * 1e6 / self.meas_len / 2
#                 sig_avg = vec * 1e6 / self.meas_len
#                 self.data.update({'signal_avg_vec': sig_avg / sig_avg[0, 0], 'ref_cnts': sig_avg[0, 0]})
#
#                 self.data['pdd_norm'] = 2 * (
#                     self.data['signal_avg_vec'][:, 0] - self.data['signal_avg_vec'][:, 1]) / \
#                                         (self.data['signal_avg_vec'][:, 0] + self.data['signal_avg_vec'][:, 1])
#
#                 self.data['esig_x_norm'] = 2 * (
#                     self.data['signal_avg_vec'][:, 2] - self.data['signal_avg_vec'][:, 3]) / \
#                                            (self.data['signal_avg_vec'][:, 2] + self.data['signal_avg_vec'][:, 3])
#
#                 self.data['esig_y_norm'] = 2 * (
#                     self.data['signal_avg_vec'][:, 4] - self.data['signal_avg_vec'][:, 5]) / \
#                                            (self.data['signal_avg_vec'][:, 4] + self.data['signal_avg_vec'][:, 5])
#
#             # do fitting
#             # try:
#             #     echo_fits = fit_exp_decay(self.data['t_vec'], self.data['signal_norm'], offset = True, verbose = False)
#             #     self.data['fits'] = echo_fits
#             # except Exception as e:
#             #     print('** ATTENTION **')
#             #     print(e)
#             # else:
#             #     self.data['T2'] = self.data['fits'][1]
#
#             try:
#                 current_rep_num = progress_handle.fetch_all()
#                 current_counts_vec = tracking_handle.fetch_all()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
#                 # print('current kcps', current_counts_kcps)
#                 # print('ref counts', ref_counts)
#                 if ref_counts < 0:
#                     ref_counts = current_counts_kcps
#                 if self.settings['NV_tracking']['on']:
#                     if current_counts_kcps > ref_counts * (1 + tolerance) or current_counts_kcps < ref_counts * (
#                             1 - tolerance):
#                         self.qm.set_io1_value(True)
#                         self.NV_tracking()
#                         self.qm.set_io1_value(False)
#                         job.resume()
#                 self.progress = current_rep_num * 100. / self.settings['rep_num']
#                 self.updateProgress.emit(int(self.progress))
#
#             if self._abort:
#                 # job.halt() # Currently not implemented. Will be implemented in future releases.
#                 self.qm.execute(job_stop)
#                 break
#
#             time.sleep(0.8)
#
#         # full_res = job.result_handles.get("total_counts")
#         # full_res_vec = full_res.fetch_all()
#         # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def NV_tracking(self):
#         # need to put a find_NV script here
#         time.sleep(5)
#
#     def plot(self, figure_list):
#         super(ACSensing2, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#             axes_list[1].clear()
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="pdd +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="pdd -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="esig +x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="esig -x")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="esig +y")
#             axes_list[1].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="esig -y")
#
#             axes_list[1].set_xlabel('Total tau [ns]')
#             axes_list[1].set_ylabel('Normalized Counts')
#             axes_list[1].legend()
#
#             axes_list[0].clear()
#             axes_list[0].plot(data['t_vec'], data['pdd_norm'], label="pdd")
#             axes_list[0].plot(data['t_vec'], data['esig_x_norm'], label="esig x")
#             axes_list[0].plot(data['t_vec'], data['esig_y_norm'], label="esig y")
#             axes_list[0].set_xlabel('Total tau [ns]')
#             axes_list[0].set_ylabel('Contrast')
#
#             # if 'fits' in data.keys():
#             #     tau = data['t_vec']
#             #     fits = data['fits']
#             #     tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
#             #     axes_list[0].plot(tauinterp, exp_offset(tauinterp, fits[0], fits[1], fits[2]), label="exp fit (T2={:2.1f} ns)".format(fits[1]))
#             axes_list[0].legend()
#             axes_list[0].set_title(
#                 'AC Sensing\n{:s} {:d} block(s)\nRef fluor: {:0.1f}kcps\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
#                     self.settings['decoupling_seq']['type'],
#                     self.settings['decoupling_seq']['num_of_pulse_blocks'],
#                     self.data['ref_cnts'], self.settings['mw_pulses']['pi_half_pulse_time'],
#                     self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
#                     self.settings['mw_pulses']['mw_power'],
#                     self.settings['mw_pulses']['mw_frequency'] * 1e-9,
#                     self.settings['mw_pulses']['IF_amp'],
#                     self.settings['mw_pulses']['IF_frequency'] * 1e-6))
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list


# ACSensing is attempting to use a qua real-time variable for the six different measurements. but fpga takes time to do with_if so align command doesn't work.

pass
# class ElectricSensing(Script):
#     """
#         This script is based on the PDDQM script, with additional electric field pulses.
#         At the end of the sequence, you can choose whether or not to do both X and Y readout.
#
#         Ziwei Qiu 8/28/2020
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('mw_pulses', [
#             Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#             Parameter('mw_power', -20.0, float, 'RF power in dBm'),
#             Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
#             Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#             Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 36, float, 'time duration of a pi/2 pulse (in ns)')
#         ]),
#         Parameter('tau_times', [
#             Parameter('min_time', 200, int, 'minimum time between the two pi pulses'),
#             Parameter('max_time', 10000, int, 'maximum time between the two pi pulses'),
#             Parameter('time_step', 100, int, 'time step increment of time between the two pi pulses (in ns)')
#         ]),
#         # Parameter('do_Y_readout', True, bool, 'choose whether or not to do a Y readout'),
#         Parameter('decoupling_seq', [
#             Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
#                       'type of dynamical decoupling sequences'),
#             Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
#         ]),
#         # Parameter('do_normalization', False, bool, 'choose whether or not to do normalization on the data'),
#         Parameter('read_out', [
#             Parameter('meas_len', 420, int, 'measurement time in ns'),
#             Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#             Parameter('delay_readout', 780, int,
#                       '(no effect) delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#             Parameter('laser_off', 1000, int, 'laser off time in ns before applying RF'),
#             Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on'),
#             Parameter('rf_switch_extra_time', 40, int,
#                       '[ns] buffer time of the MW switch (trigger) window on both sides of I/Q pulses (no effect, '
#                       'buffer time is defined in the configuration)')
#         ]),
#         Parameter('rep_num', 500000, int, 'define the repetition number'),
#         Parameter('simulation_duration', 50000, int, 'duration of simulation in ns'),
#     ]
#     _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
#     _SCRIPTS = {}
#
#     def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
#         """
#         Example of a script that emits a QT signal for the gui
#         Args:
#             name (optional): name of script, if empty same as class name
#             settings (optional): settings for this script, if empty same as default settings
#         """
#         Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
#                         log_function=log_function, data_path=data_path)
#
#         self._connect()
#
#     def _connect(self):
#         #####################################
#         # Open communication with the server:
#         #####################################
#         if self.settings['IP_address'] == 'automatic':
#             try:
#                 self.qmm = QuantumMachinesManager()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#         else:
#             try:
#                 self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#     def _function(self):
#         if self.settings['to_do'] == 'reconnection':
#             self._connect()
#         else:
#             try:
#                 self.qm = self.qmm.open_qm(config)
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 rep_num = self.settings['rep_num']
#                 self.meas_len = round(self.settings['read_out']['meas_len'])
#                 delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
#                 nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
#                 rf_switch_extra_time = round(self.settings['read_out']['rf_switch_extra_time'] / 4)
#                 laser_off = round(self.settings['read_out']['laser_off'] / 4)
#                 delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
#                 pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
#                 pi_half_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
#                 # three_pi_half_time = round(self.settings['mw_pulses']['3pi_half_pulse_time']/ 4)
#
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 num_of_evolution_blocks = 0
#                 number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
#                 if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
#                     'type'] == 'CPMG':
#                     num_of_evolution_blocks = 1 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY4':
#                     num_of_evolution_blocks = 4 * number_of_pulse_blocks
#                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                     num_of_evolution_blocks = 8 * number_of_pulse_blocks
#
#                 # tau time between MW pulses
#                 tau_start = round(self.settings['tau_times']['min_time'] / num_of_evolution_blocks)
#                 tau_end = round(self.settings['tau_times']['max_time'] / num_of_evolution_blocks)
#                 tau_step = round(self.settings['tau_times']['time_step'] / num_of_evolution_blocks)
#
#                 # this is half of the each evolution block time
#                 # using 19 instead of 4 is because the actual (or simulated) output wait time is 60ns longer than defined
#                 # 60/4 = 15, 15+4 = 19
#                 self.t_vec = [int(a_) for a_ in
#                               np.arange(np.max([int(np.ceil(tau_start / 8)), 19]), int(np.ceil(tau_end / 8)),
#                                         int(np.ceil(tau_step / 8)))]
#                 self.tau_total = np.array(self.t_vec) * 8 * num_of_evolution_blocks
#
#                 t_num = len(self.t_vec)
#                 print('Half evolution block [ns]: ', np.array(self.t_vec) * 4)
#                 print('Total_tau_times [ns]: ', self.tau_total)
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 # define the qua program
#                 t_vec = [int(a_ - wait_pulse_artifect) for a_ in self.t_vec]
#                 with program() as spin_echo:
#                     update_frequency('qubit', self.settings['mw_pulses']['IF_frequency'])
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#                     t = declare(int)
#                     n = declare(int)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         with for_each_(t, t_vec):
#                             for k in [0, 1]:
#                                 # k=0, +x readout, k=1, -x readout, no electric field
#                                 # k=2, +x readout, k=3, -x readout, with electric field
#                                 # k=4, +y readout, k=5, -y readout, with electric field
#
#                                 reset_frame('qubit')
#                                 align('qubit', 'laser', 'readout1', 'readout2', 'e_field1', 'e_field2')
#                                 wait(laser_off, 'qubit')
#                                 play('pi2' * amp(IF_amp), 'qubit', duration=pi_half_time)
#
#                                 if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                     z_rot(np.pi / 2, 'qubit')
#                                 wait(t, 'qubit') # for the first pulse, only wait half of the evolution block time
#                                 wait(laser_off+pi_half_time, 'e_field1', 'e_field2')
#                                 play('trig', 'e_field1', duration=t+wait_pulse_artifect)
#                                 wait(t+wait_pulse_artifect, 'e_field2')
#
#                                 if self.settings['decoupling_seq']['type'] == 'XY4':
#                                     for i in range(0, number_of_pulse_blocks):
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                             wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#
#                                 elif self.settings['decoupling_seq']['type'] == 'XY8':
#                                     for i in range(0, number_of_pulse_blocks):
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                             wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         z_rot(np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_y
#                                         wait(2 * t, 'qubit')
#                                         wait(pi_time, 'e_field1', 'e_field2')
#                                         play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                         wait(2 * t + wait_pulse_artifect, 'e_field1')
#
#                                         z_rot(-np.pi / 2, 'qubit')
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)  # pi_x
#
#                                 else:
#                                     for i in range(0, number_of_pulse_blocks):  # for CPMG or spin-echo
#                                         if i > 0:
#                                             wait(2 * t, 'qubit')
#                                             wait(pi_time, 'e_field1', 'e_field2')
#                                             if i % 2 == 0:
#                                                 play('trig', 'e_field2', duration=2 * t + wait_pulse_artifect)
#                                                 wait(2 * t + wait_pulse_artifect, 'e_field1')
#                                             else:
#                                                 play('trig', 'e_field1', duration=2 * t + wait_pulse_artifect)
#                                                 wait(2 * t + wait_pulse_artifect, 'e_field2')
#
#                                         play('pi' * amp(IF_amp), 'qubit', duration=pi_time)
#
#
#                                 wait(t, 'qubit')  # for the last pulse, only wait half of the evolution block time
#                                 wait(pi_time, 'e_field1', 'e_field2')
#
#                                 if number_of_pulse_blocks%2 == 1:
#                                     play('trig', 'e_field2', duration=t + wait_pulse_artifect)
#                                     wait(t + wait_pulse_artifect, 'e_field1')
#                                 else:
#                                     play('trig', 'e_field1', duration=t + wait_pulse_artifect)
#                                     wait(t + wait_pulse_artifect, 'e_field2')
#
#
#                                 if self.settings['decoupling_seq']['type'] == 'CPMG':
#                                     z_rot(-np.pi / 2, 'qubit')
#
#                                 if k == 1:
#                                     z_rot(np.pi, 'qubit')
#                                 play('pi2' * amp(IF_amp), 'qubit', duration=pi_half_time)
#                                 # play('pi2', 'qubit', duration=pi_half_time)
#                                 align('qubit', 'laser', 'readout1', 'readout2')
#                                 wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                                 play('trig', 'laser', duration=nv_reset_time)
#                                 # wait(delay_readout, 'readout1', 'readout2')
#                                 measure('readout', 'readout1', None,
#                                         time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                                 measure('readout', 'readout2', None,
#                                         time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#                                 assign(total_counts, counts1 + counts2)
#                                 save(total_counts, total_counts_st)
#                                 save(total_counts, "total_counts")
#                                 save(n, rep_num_st)
#
#                     with stream_processing():
#                         total_counts_st.buffer(t_num, 2).average().save("live_pdd_data")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(spin_echo)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(spin_echo, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         start = time.time()
#         try:
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
#         except Exception as e:
#             print('** ATTENTION **')
#             print(e)
#         else:
#             # job.get_simulated_samples().con1.plot()
#             end = time.time()
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
#
#         self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
#         print('Turned on RF generator SGS100A (IQ on, trigger on).')
#
#         job = self.qm.execute(qua_program)
#
#         vec_handle = job.result_handles.get("live_pdd_data")
#         progress_handle = job.result_handles.get("live_rep_num")
#
#         vec_handle.wait_for_values(1)
#         self.data = {'t_vec': np.array(self.tau_total)}
#
#         current_rep_num = 0
#
#         while vec_handle.is_processing():
#             try:
#                 vec = vec_handle.fetch_all()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#             else:
#                 self.data['signal_avg_vec'] = vec * 1e6 / self.meas_len
#                 # print('shape:', self.data['signal_avg_vec'].shape)
#
#             try:
#                 current_rep_num = progress_handle.fetch_all()
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#
#             if self._abort:
#                 # job.halt() # Currently not implemented. Will be implemented in future releases.
#                 self.qm.execute(job_stop)
#                 break
#
#             self.progress = current_rep_num * 100. / self.settings['rep_num']
#             self.updateProgress.emit(int(self.progress))
#             time.sleep(0.8)
#
#         # full_res = job.result_handles.get("total_counts")
#         # full_res_vec = full_res.fetch_all()
#         # self.data['signal_full_vec'] = full_res_vec
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#     def plot(self, figure_list):
#         super(ElectricSensing, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         """
#             Plots the confocal scan image
#             Args:
#                 axes_list: list of axes objects on which to plot the galvo scan on the first axes object
#                 data: data (dictionary that contains keys image_data, extent) if not provided use self.data
#         """
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[1], data)
#
#         if 't_vec' in data.keys() and 'signal_avg_vec' in data.keys():
#             axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="+x")
#             axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="-x")
#             axes_list[0].set_xlabel('Tau total [ns]')
#             axes_list[0].set_ylabel('Photon Counts [kcps]')
#             axes_list[0].set_title('PDD')
#             axes_list[0].legend()
#
#     def _update_plot(self, axes_list):
#         if self.data is None:
#             return
#
#         if 't_vec' in self.data.keys() and 'signal_avg_vec' in self.data.keys():
#             try:
#                 axes_list[0].lines[0].set_ydata(self.data['signal_avg_vec'][:, 0])
#                 axes_list[0].lines[1].set_ydata(self.data['signal_avg_vec'][:, 1])
#                 axes_list[0].lines[0].set_xdata(self.data['t_vec'])
#                 axes_list[0].relim()
#                 axes_list[0].autoscale_view()
#                 axes_list[0].set_title('PDD live plot\n' + time.asctime())
#             except Exception as e:
#                 print('** ATTENTION **')
#                 print(e)
#                 self._plot(axes_list)
#
#         else:
#             self._plot(axes_list)
#
#     def get_axes_layout(self, figure_list):
#         """
#             returns the axes objects the script needs to plot its data
#             this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
#             Args:
#                 figure_list: a list of figure objects
#             Returns:
#                 axes_list: a list of axes objects
#
#         """
#         axes_list = []
#         if self._plot_refresh is True:
#             for fig in figure_list:
#                 fig.clf()
#             axes_list.append(figure_list[0].add_subplot(111))  # axes_list[0]
#             axes_list.append(figure_list[1].add_subplot(111))  # axes_list[1]
#         else:
#             axes_list.append(figure_list[0].axes[0])
#             axes_list.append(figure_list[1].axes[0])
#         return axes_list
