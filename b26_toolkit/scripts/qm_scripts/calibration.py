from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.qua import frame_rotation as z_rot
import time
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config
import numpy as np

from b26_toolkit.instruments import SGS100ARFSource


class DelayReadoutMeas(Script):
    """
        This script calibrates the delay time needed between turning on the laser and starting APD measurement.
        - Ziwei Qiu 9/1/2020
    """

    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM orchestrator'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('delay_times', [
            Parameter('min_time', 16, int, 'minimum time for rabi oscillations (in ns), >=16ns'),
            Parameter('max_time', 2500, int, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 48, int,
                      'time step increment of rabi pulse duration (in ns), using multiples of 4ns')
        ]),
        Parameter('rep_num', 100000, int, 'define the repetition number'),
        Parameter('simulation_duration', 5000, int, 'duration of simulation [ns]'),
        Parameter('read_out', [
            Parameter('meas_len', 200, int, 'APD measurement time [ns]'),
            Parameter('nv_reset_time', 1000, int, 'laser on time [ns]'),
        ])
    ]
    _INSTRUMENTS = {}
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

        #####################################
        # Open communication with the server:
        #####################################
        if self.settings['IP_address'] == 'automatic':
            self.qmm = QuantumMachinesManager()
        else:
            self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])

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
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rep_num = self.settings['rep_num']
                self.meas_len = round(self.settings['read_out']['meas_len'] / 4)
                nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                self.t_vec = [int(a_) for a_ in
                              np.arange(round(np.ceil(self.settings['delay_times']['min_time'] / 4)),
                                        round(np.ceil(self.settings['delay_times']['max_time'] / 4)),
                                        round(np.ceil(self.settings['delay_times']['time_step'] / 4)))]
                t_vec = self.t_vec
                t_num = len(self.t_vec)

                with program() as delay_readout_meas:

                    m = declare(int)
                    t = declare(int)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(m, 0, m < rep_num, m + 1):
                        with for_each_(t, t_vec):
                            wait(500, 'laser')  # wait for 2us to avoid overlapping with the previous sequence
                            align('laser', 'readout1', 'readout2')
                            play('trig', 'laser', duration=nv_reset_time)
                            wait(t, 'readout1', 'readout2')
                            measure("readout", "readout1", None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure("readout", "readout2", None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))

                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            save(m, rep_num_st)
                            save(total_counts, "total_counts")

                    with stream_processing():
                        total_counts_st.buffer(t_num).average().save("live_data")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(delay_readout_meas)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(delay_readout_meas, job_stop)
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
        job = self.qm.execute(qua_program)
        vec_handle = job.result_handles.get("live_data")
        progress_handle = job.result_handles.get("live_rep_num")

        vec_handle.wait_for_values(1)
        self.data = {'t_vec': np.array(self.t_vec) * 4}  # because in the unit of 4ns

        while vec_handle.is_processing():

            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # print('getting live data')
                self.data['signal_avg_vec'] = vec * 1e6 / self.meas_len

            try:
                current_rep_num = progress_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.progress = current_rep_num * 100. / self.settings['rep_num']
                self.updateProgress.emit(int(self.progress))

            if self._abort:
                # job.halt() # Currently not implemented. Will be implemented in future releases.
                self.qm.execute(job_stop)
                break

            time.sleep(0.5)

        full_res = job.result_handles.get("total_counts")
        full_res_vec = full_res.fetch_all()
        self.data['signal_full_vec'] = full_res_vec

    def plot(self, figure_list):
        super(DelayReadoutMeas, self).plot([figure_list[0], figure_list[1]])

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
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'])
            axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
            axes_list[0].set_xlabel('Delay Time [ns]')
            axes_list[0].set_ylabel('Counts [kcps]')
            axes_list[0].set_title('Readout Delay Calibration')

    def _update_plot(self, axes_list):
        if self.data is None:
            return

        if 't_vec' in self.data.keys() and 'signal_avg_vec' in self.data.keys():
            try:
                axes_list[0].lines[0].set_ydata(self.data['signal_avg_vec'])
                axes_list[0].lines[0].set_xdata(self.data['t_vec'])
                axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
                axes_list[0].relim()
                axes_list[0].autoscale_view()
                axes_list[0].set_title('Readout Delay Calibration (live) \n' + time.asctime())
            except Exception as e:
                print('** ATTENTION **')
                print(e)
                self._plot(axes_list)

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


class IQCalibration(Script):
    """
        This script calibrates the I and Q amplitude and phase symmetry by running echo sequence on an NV center.
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
            Parameter('min_time', 16, int, 'half minimum time between the two pi pulses'),
            Parameter('max_time', 2500, int, 'half maximum time between the two pi pulses'),
            Parameter('time_step', 20, int, 'time step increment of half time between the two pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 250, int, 'measurement time in ns'),
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
                with program() as iq_calibration:
                    update_frequency('qubit', IF_freq)
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    t = declare(int)
                    n = declare(int)
                    k = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(t, t_vec):
                            with for_(k, 0, k < 8, k + 1):
                                reset_frame('qubit')
                                # the following z_rot cannot be separated from play by a with if_ command. otherwise OPX cannot handle it.
                                # z_rot(self.settings['mw_pulses']['phase'] / 180 * np.pi, 'qubit')

                                with if_(k == 0):  # XXX plus
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                with if_(k == 1):  # XXX minus
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                with if_(k == 2):  # XYX plus
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(-np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 3):  # XYX minus
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(-np.pi / 2, 'qubit')
                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 4):  # YYY plus
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 5):  # YYY minus
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 6):  # YXY plus
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(-np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')

                                with if_(k == 7):  # YXY minus
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi2' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
                                    z_rot(np.pi / 2, 'qubit')
                                    play('pi' * amp(IF_amp), 'qubit')
                                    wait(t, 'qubit')
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
                        total_counts_st.buffer(t_num, 8).average().save("live_data")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(iq_calibration)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(iq_calibration, job_stop)
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

        vec_handle = job.result_handles.get("live_data")
        progress_handle = job.result_handles.get("live_rep_num")

        vec_handle.wait_for_values(1)
        self.data = {'t_vec': np.array(self.tau_total), 'signal_avg_vec': None}

        while vec_handle.is_processing():
            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                avg_sig = vec * 1e6 / self.meas_len
                self.data.update({'signal_avg_vec': avg_sig})

            try:
                current_rep_num = progress_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.progress = current_rep_num * 100. / self.settings['rep_num']
                self.updateProgress.emit(int(self.progress))

            if self._abort:
                # job.halt() # Currently not implemented. Will be implemented in future releases.
                self.qm.execute(job_stop)
                break

            time.sleep(0.5)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(IQCalibration, self).plot([figure_list[0], figure_list[1]])

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
            axes_list[0].clear()
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 0], label="XXX (plus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 1], label="XXX (minus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 2], label="XYX (plus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 3], label="XYX (minus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 4], label="YYY (plus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 5], label="YYY (minus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 6], label="YXY (plus)")
            axes_list[0].plot(data['t_vec'], data['signal_avg_vec'][:, 7], label="YXY (minus)")

            axes_list[0].set_xlabel('Total tau [ns]')
            axes_list[0].set_ylabel('Counts')
            axes_list[0].legend()
            axes_list[0].set_title(
                'IQ Calibration\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                    self.settings['mw_pulses']['pi_half_pulse_time'],
                    self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'],
                                                            self.settings['mw_pulses']['IF_frequency'] * 1e-6))

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
