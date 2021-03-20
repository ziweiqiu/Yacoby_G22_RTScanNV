from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.qua import frame_rotation as z_rot
import time
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser
import numpy as np
from b26_toolkit.instruments import SGS100ARFSource

from b26_toolkit.data_processing.esr_signal_processing import fit_esr
from b26_toolkit.plotting.plots_1d import plot_esr
from b26_toolkit.data_processing.fit_functions import fit_rabi_decay, cose_with_decay


class TimeTraceQMsim(Script):
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM orchestrator'),
        Parameter('rep_num', 10000000, int, 'define the repetition number'),
        Parameter('simulation_duration', 5000, int, 'duration of simulation [ns]'),
        Parameter('read_out', [
            Parameter('meas_len', 2000, int, 'APD measurement time [ns]'),
            Parameter('nv_reset_time', 2000, int, 'laser on time [ns]'),
            Parameter('delay_readout', 60, int,
                      '(no effect) delay between laser on and readout (given by spontaneous decay rate) [ns]')
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

    def _function(self):
        ############
        # Open a QM:
        ############
        # from b26_toolkit.scripts.qm_scripts.Configuration import config
        self.qm = self.qmm.open_qm(config)

        # meas_len = 2000
        # rep_num = 1e7

        rep_num = self.settings['rep_num']
        meas_len = round(self.settings['read_out']['meas_len'] / 4)
        delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
        nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)

        with program() as time_trace:
            ##############################
            # Declare real-time variables:
            ##############################

            n = declare(
                int)  # Declare a single QUA variable or QUA vector to be used in subsequent expressions and
            # assignments.
            m = declare(int)
            p = declare(int)
            result1 = declare(int, size=2)
            resultLen1 = declare(int, value=0)
            result2 = declare(int, size=2)
            resultLen2 = declare(int, value=0)

            ###############
            # The sequence:
            ###############

            with for_(m, 0, m < rep_num, m + 1):
                ####################
                # Qubit preparation:
                ####################
                play("pi", "qubit")  # play('pulse_name' * amp(v), 'element')

                ##########
                # Readout:
                ##########
                align("qubit", "laser", "readout1", "readout2")
                # wait(delay_readout, 'readout1', 'readout2')  # time to wait, in multiples of 4nsec
                play('trig', 'laser', duration=nv_reset_time)  # is duration implemented??
                measure("readout", "readout1", None,
                        time_tagging.raw(result1, meas_len, targetLen=resultLen1))  # result1 = [203, 1000]
                measure("readout", "readout2", None,
                        time_tagging.raw(result2, meas_len, targetLen=resultLen2))  # result2 = [222]
                #
                ################
                # Save to client
                ################
                with for_(n, 0, n < resultLen1, n + 1):
                    save(result1[n], "res1")
                with for_(p, 0, p < resultLen2, p + 1):
                    save(result2[p], "res2")

            ############
            # Simulation:
            ############
            start = time.time()
            simulation_duration = int(self.settings['simulation_duration'] / 4)
            job = self.qm.simulate(time_trace, SimulationConfig(simulation_duration))
            # job.get_simulated_samples().con1.plot()
            end = time.time()
            print('QM simulation took {:.1f}s.'.format(end - start))
            self.log('QM simulation took {:.1f}s.'.format(end - start))
            samples = job.get_simulated_samples().con1
            self.data = {'analog': samples.analog,
                         'digital': samples.digital}

            ############
            # Execute:
            ############
            # my_job = qm.execute(time_trace)
            # time.sleep(1.0)
            # time_trace_results = my_job.get_results()

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data
        plot_qmsimulation_samples(axes_list[0], data)

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
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[0]
        else:
            axes_list.append(figure_list[1].axes[0])
        return axes_list


class RabiQM(Script):
    """
        This script applies a microwave pulse at fixed power for varying durations to measure Rabi oscillations.
        Pulses are controlled by a Quantum Machine. Note that the QM clock cycle is 4ns.

        - Ziwei Qiu 8/11/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'execution', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('phase', 0, float, 'starting phase of the RF pulse in deg')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 16, int, 'minimum time for rabi oscillations (in ns), >=16ns'),
            Parameter('max_time', 500, int, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 12, int,
                      'time step increment of rabi pulse duration (in ns), using multiples of 4ns')
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
        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # config['elements']['laser']['digitalInputs']['switch_in']['delay'] = 140
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
                with program() as rabi:
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
                            z_rot(self.settings['mw_pulses']['phase'] / 180 * np.pi, 'qubit')
                            play('const' * amp(IF_amp), 'qubit', duration=t)

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
                    self._qm_simulation(rabi)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(rabi, job_stop)
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
                # rabi_avg = vec * 1e6 / self.meas_len / 2
                rabi_avg = vec * 1e6 / self.meas_len
                self.data.update({'signal_avg_vec': rabi_avg / rabi_avg[0], 'ref_cnts': rabi_avg[0]})

            # do fitting
            try:
                rabi_fits = fit_rabi_decay(self.data['t_vec'], self.data['signal_avg_vec'], variable_phase=True)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                fits = rabi_fits[0]
                RabiT = 2 * np.pi / fits[1]
                phaseoffs = fits[2]

                self.data['fits'] = fits
                self.data['phaseoffs'] = phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_time'] = RabiT / 2 - phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_half_time'] = RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['three_pi_half_time'] = 3 * RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['T2_star'] = fits[4]
                self.data['rabi_freq'] = 1000 * fits[1] / (2 * np.pi)  # Rabi frequency in [MHz]

            try:
                current_rep_num = progress_handle.fetch_all()
                current_counts_vec = tracking_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # Check if tracking is called
                self.data['rep_num'] = float(current_rep_num)
                # print(current_rep_num)
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
        time.sleep(10)

    def plot(self, figure_list):
        super(RabiQM, self).plot([figure_list[0], figure_list[1]])

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
            if 'fits' in data.keys():
                try:
                    pi_time = data['pi_time']
                    pi_half_time = data['pi_half_time']
                    three_pi_half_time = data['three_pi_half_time']
                    fits = data['fits']
                    phaseoffs = data['phaseoffs']

                    axes_list[0].clear()
                    axes_list[0].plot(data['t_vec'], data['signal_avg_vec'], '.-')
                    axes_list[0].plot(data['t_vec'], cose_with_decay(data['t_vec'], *fits), lw=2)

                    axes_list[0].plot(pi_time, cose_with_decay(pi_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi$={:0.1f}ns'.format(pi_time), xy=(pi_time, cose_with_decay(pi_time, *fits)),
                                          xytext=(pi_time + 10., cose_with_decay(pi_time, *fits)), xycoords='data')
                    axes_list[0].plot(pi_half_time, cose_with_decay(pi_half_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi/2$=\n{:0.1f}ns'.format(pi_half_time),
                                          xy=(pi_half_time, cose_with_decay(pi_half_time, *fits)),
                                          xytext=(pi_half_time + 10., cose_with_decay(pi_half_time, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(three_pi_half_time, cose_with_decay(three_pi_half_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$3\pi/2$=\n{:0.1f}ns'.format(three_pi_half_time),
                                          xy=(three_pi_half_time, cose_with_decay(three_pi_half_time, *fits)),
                                          xytext=(three_pi_half_time + 10., cose_with_decay(three_pi_half_time, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(-phaseoffs, cose_with_decay(-phaseoffs, *fits), 'gd', lw=3)
                    axes_list[0].annotate('$start$={:0.1f}ns'.format(-phaseoffs),
                                          xy=(-phaseoffs, cose_with_decay(-phaseoffs, *fits)),
                                          xytext=(-phaseoffs + 10., cose_with_decay(-phaseoffs, *fits)), xycoords='data')
                    if title:
                        axes_list[0].set_title(
                            'Rabi frequency: {:0.2f}MHz\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns \n T2*: {:2.1f}ns, Ref fluor: {:0.1f}kcps\nRepetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                                data['rabi_freq'], pi_half_time, pi_time, three_pi_half_time, data['T2_star'],
                                data['ref_cnts'], int(data['rep_num']),
                                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                                self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6))
                    axes_list[0].set_xlabel('Rabi tau [ns]')
                    axes_list[0].set_ylabel('Contrast')

                except Exception as e:
                    print('** ATTENTION **')
                    print(e)

            else:
                try:
                    axes_list[0].clear()
                    axes_list[0].plot(data['t_vec'], data['signal_avg_vec'])
                    axes_list[0].set_xlabel('Rabi tau [ns]')
                    axes_list[0].set_ylabel('Contrast')
                    if title:
                        axes_list[0].set_title(
                            'Rabi\nRef fluor: {:0.1f}kcps\nRepetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz'.format(
                                data['ref_cnts'], int(data['rep_num']),
                                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                                self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6))

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


class RabiPlusMinus(Script):
    """
        This script implements Rabi between the plus and minus state under a perpendicular magnetic field.
        - Ziwei Qiu 3/7/2021
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'execution', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('pi_frequency', 'esr2', ['esr1', 'esr2'], 'the pi pulse frequency'),
            Parameter('pi_pulse_time', 72, float, 'time duration of a pi pulse (in ns)'),
            Parameter('esr1', 2.0e7, float, 'the minus state IF frequency in Hz'),
            Parameter('esr2', 5.0e7, float, 'the plus state IF frequency in Hz'),
            Parameter('subqubit_IF_amp', 1.0, float, 'amplitude of the subqubit IF pulse, between 0 and 1'),

        ]),
        Parameter('tau_times', [
            Parameter('min_time', 16, int, 'minimum time for rabi oscillations (in ns), >=16ns'),
            Parameter('max_time', 500, int, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 12, int,
                      'time step increment of rabi pulse duration (in ns), using multiples of 4ns')
        ]),
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

    def _function(self):
        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                # unit: cycle of 4ns
                pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
                # unit: ns
                config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rep_num = self.settings['rep_num']
                # tracking_num = self.settings['NV_tracking']['tracking_num']
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

                subqubit_IF_amp = self.settings['mw_pulses']['subqubit_IF_amp']
                if subqubit_IF_amp > 1.0:
                    subqubit_IF_amp = 1.0
                elif subqubit_IF_amp < 0.0:
                    subqubit_IF_amp = 0.0

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

                pi_frequency = self.settings['mw_pulses'][self.settings['mw_pulses']['pi_frequency']]
                subqubit_frequency = round(
                    np.abs(self.settings['mw_pulses']['esr1'] - self.settings['mw_pulses']['esr2']))

                # define the qua program
                with program() as rabi:
                    update_frequency('qubit', pi_frequency)
                    update_frequency('sub_qubit', subqubit_frequency)

                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    t = declare(int)
                    n = declare(int)

                    # # the following two variable are used to flag tracking
                    # assign(IO1, False)
                    # flag = declare(bool, value=False)
                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        # # Check if tracking is called
                        # assign(flag, IO1)
                        # # with while_(flag):
                        # #     play('trig', 'laser', duration=10000)
                        # with if_(flag):
                        #     pause()
                        with for_each_(t, t_vec):
                            reset_frame('qubit')

                            play('pi' * amp(IF_amp), 'qubit')
                            align('qubit', 'sub_qubit')
                            wait(20, 'sub_qubit')
                            play('const' * amp(subqubit_IF_amp), 'sub_qubit', duration=t)
                            align('qubit', 'sub_qubit')
                            wait(20, 'qubit')
                            play('pi' * amp(IF_amp), 'qubit')

                            align('qubit', 'sub_qubit', 'laser', 'readout1', 'readout2')
                            wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
                            play('trig', 'laser', duration=nv_reset_time)
                            wait(delay_readout, 'readout1', 'readout2')
                            measure('readout', 'readout1', None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure('readout', 'readout2', None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))

                            align('qubit', 'sub_qubit', 'laser', 'readout1', 'readout2')
                            wait(laser_off, 'qubit')

                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            save(n, rep_num_st)
                            save(total_counts, "total_counts")

                    with stream_processing():
                        total_counts_st.buffer(t_num).average().save("live_rabi_data")
                        # total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(rabi)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(rabi, job_stop)
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

        vec_handle = job.result_handles.get("live_rabi_data")
        progress_handle = job.result_handles.get("live_rep_num")
        # tracking_handle = job.result_handles.get("current_counts")

        vec_handle.wait_for_values(1)
        progress_handle.wait_for_values(1)
        # tracking_handle.wait_for_values(1)
        self.data = {'t_vec': np.array(self.t_vec) * 4, 'signal_avg_vec': None, 'ref_cnts': None, 'rep_num': None}

        while vec_handle.is_processing():
            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rabi_avg = vec * 1e6 / self.meas_len
                self.data.update({'signal_avg_vec': rabi_avg / rabi_avg[0], 'ref_cnts': rabi_avg[0]})

            # do fitting
            try:
                rabi_fits = fit_rabi_decay(self.data['t_vec'], self.data['signal_avg_vec'], variable_phase=True)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                fits = rabi_fits[0]
                RabiT = 2 * np.pi / fits[1]
                phaseoffs = fits[2]

                self.data['fits'] = fits
                self.data['phaseoffs'] = phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_time'] = RabiT / 2 - phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_half_time'] = RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['three_pi_half_time'] = 3 * RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['T2_star'] = fits[4]
                self.data['rabi_freq'] = 1000 * fits[1] / (2 * np.pi)  # Rabi frequency in [MHz]

            try:
                current_rep_num = progress_handle.fetch_all()
                # current_counts_vec = tracking_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # Check if tracking is called
                self.data['rep_num'] = float(current_rep_num)
                # print(current_rep_num)
                self.progress = current_rep_num * 100. / self.settings['rep_num']
                self.updateProgress.emit(int(self.progress))

            if self._abort:
                self.qm.execute(job_stop)
                break

            time.sleep(0.8)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    def plot(self, figure_list):
        super(RabiPlusMinus, self).plot([figure_list[0], figure_list[1]])

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
            if 'fits' in data.keys():
                try:
                    pi_time = data['pi_time']
                    pi_half_time = data['pi_half_time']
                    three_pi_half_time = data['three_pi_half_time']
                    fits = data['fits']
                    phaseoffs = data['phaseoffs']

                    axes_list[0].clear()
                    axes_list[0].plot(data['t_vec'], data['signal_avg_vec'], '.-')
                    axes_list[0].plot(data['t_vec'], cose_with_decay(data['t_vec'], *fits), lw=2)

                    axes_list[0].plot(pi_time, cose_with_decay(pi_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi$={:0.1f}ns'.format(pi_time), xy=(pi_time, cose_with_decay(pi_time, *fits)),
                                          xytext=(pi_time + 10., cose_with_decay(pi_time, *fits)), xycoords='data')
                    axes_list[0].plot(pi_half_time, cose_with_decay(pi_half_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi/2$=\n{:0.1f}ns'.format(pi_half_time),
                                          xy=(pi_half_time, cose_with_decay(pi_half_time, *fits)),
                                          xytext=(pi_half_time + 10., cose_with_decay(pi_half_time, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(three_pi_half_time, cose_with_decay(three_pi_half_time, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$3\pi/2$=\n{:0.1f}ns'.format(three_pi_half_time),
                                          xy=(three_pi_half_time, cose_with_decay(three_pi_half_time, *fits)),
                                          xytext=(three_pi_half_time + 10., cose_with_decay(three_pi_half_time, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(-phaseoffs, cose_with_decay(-phaseoffs, *fits), 'gd', lw=3)
                    axes_list[0].annotate('$start$={:0.1f}ns'.format(-phaseoffs),
                                          xy=(-phaseoffs, cose_with_decay(-phaseoffs, *fits)),
                                          xytext=(-phaseoffs + 10., cose_with_decay(-phaseoffs, *fits)), xycoords='data')
                    if title:
                        axes_list[0].set_title(
                            'Rabi frequency: {:0.2f}MHz\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns \n T2*: {:2.1f}ns, Ref fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, pi-time: {:2.1f}ns, pi freq: {:s}\nesr1: {:0.2f}MHz, esr2: {:0.2f}MHz, subqubit amp: {:0.1f}'.format(
                                data['rabi_freq'], pi_half_time, pi_time, three_pi_half_time, data['T2_star'],
                                data['ref_cnts'], int(data['rep_num']), self.settings['mw_pulses']['mw_power'],
                                self.settings['mw_pulses']['mw_frequency'] * 1e-9, self.settings['mw_pulses']['IF_amp'],
                                self.settings['mw_pulses']['pi_pulse_time'],
                                self.settings['mw_pulses']['pi_frequency'],self.settings['mw_pulses']['esr1'] * 1e-6,
                                self.settings['mw_pulses']['esr2'] * 1e-6, self.settings['mw_pulses']['subqubit_IF_amp']))
                    axes_list[0].set_xlabel('Rabi tau [ns]')
                    axes_list[0].set_ylabel('Contrast')

                except Exception as e:
                    print('** ATTENTION **')
                    print(e)

            else:
                try:
                    axes_list[0].clear()
                    axes_list[0].plot(data['t_vec'], data['signal_avg_vec'])
                    axes_list[0].set_xlabel('Rabi tau [ns]')
                    axes_list[0].set_ylabel('Contrast')
                    if title:
                        axes_list[0].set_title(
                            'Rabi\nRef fluor: {:0.1f}kcps, Repetition number: {:d}\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, pi-time: {:2.1f}ns, pi freq: {:s}\nesr1: {:0.2f}MHz, esr2: {:0.2f}MHz, subqubit amp: {:0.1f}'.format(
                                data['ref_cnts'], int(data['rep_num']),
                                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                                self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['pi_pulse_time'],
                                self.settings['mw_pulses']['pi_frequency'], self.settings['mw_pulses']['esr1'] * 1e-6,
                                self.settings['mw_pulses']['esr2'] * 1e-6, self.settings['mw_pulses']['subqubit_IF_amp'],
                            ),
                        )

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


class PowerRabi(Script):
    """
        This script applies a microwave pulse at varying amplitudes for fixed tau to measure Rabi oscillations.
        Pulses are controlled by a Quantum Machine. Note that the QM clock cycle is 4ns.

        - Ziwei Qiu 9/25/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -20.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('tau_time', 60, int, 'rabi oscillations (in ns), >=16ns'),
            Parameter('phase', 0, float, 'starting phase of the RF pulse in deg')
        ]),
        Parameter('IF_amplitudes', [
            Parameter('min_amp', 0.1, float, 'minimum IF amplitude, >0'),
            Parameter('max_amp', 1.0, float, 'maximum IF amplitude, <1'),
            Parameter('amp_step', 0.05, float, 'IF amplitude incremental step')
        ]),
        Parameter('read_out', [
            Parameter('meas_len', 170, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('NV_tracking', [
            Parameter('on', False, bool, 'track NV and do a galvo scan if the counts out of the reference range'),
            Parameter('tracking_num', 10000, int, 'number of recent APD windows used for calculating current counts'),
            Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
            Parameter('tolerance', 0.3, float, 'define the reference range (1+/-tolerance)*ref')
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
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # config['elements']['laser']['digitalInputs']['switch_in']['delay'] = 140
                rep_num = self.settings['rep_num']
                tracking_num = self.settings['NV_tracking']['tracking_num']
                self.meas_len = round(self.settings['read_out']['meas_len'])

                delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
                nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
                laser_off = round(self.settings['read_out']['laser_off'] / 4)
                delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
                tau = round(self.settings['mw_pulses']['tau_time'] / 4)

                amp_start = np.max([self.settings['IF_amplitudes']['min_amp'],0])
                amp_end = np.min([self.settings['IF_amplitudes']['max_amp'], 1])
                amp_step = self.settings['IF_amplitudes']['amp_step']
                self.amp_vec = np.arange(amp_start, amp_end, amp_step).tolist()
                #if amp_start == 0:
                 #   self.amp_vec.pop(0)

                amp_num = len(self.amp_vec)
                print('amp_vec: ', self.amp_vec)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                # define the qua program
                with program() as power_rabi:
                    update_frequency('qubit', self.settings['mw_pulses']['IF_frequency'])
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    IF_amp = declare(fixed)
                    n = declare(int)

                    # the following two variable are used to flag tracking
                    assign(IO1, False)
                    flag = declare(bool, value=False)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        # Check if tracking is called
                        assign(flag, IO1)
                        with if_(flag):
                            pause()
                        with for_each_(IF_amp, self.amp_vec):
                            reset_frame('qubit')
                            z_rot(self.settings['mw_pulses']['phase'] / 180 * np.pi, 'qubit')
                            play('const' * amp(IF_amp), 'qubit', duration=tau)

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
                        total_counts_st.buffer(amp_num).average().save("live_rabi_data")
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if amp_num > 0:
                    if self.settings['to_do'] == 'simulation':
                        self._qm_simulation(power_rabi)
                    elif self.settings['to_do'] == 'execution':
                        self._qm_execution(power_rabi, job_stop)
                else:
                    print('No amplitudes to sweep. No action.')
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

        vec_handle = job.result_handles.get("live_rabi_data")
        progress_handle = job.result_handles.get("live_rep_num")
        tracking_handle = job.result_handles.get("current_counts")

        vec_handle.wait_for_values(1)
        progress_handle.wait_for_values(1)
        tracking_handle.wait_for_values(1)
        self.data = {'amp_vec': np.array(self.amp_vec), 'signal_avg_vec': None, 'ref_cnts': None}

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

            # do fitting
            try:
                rabi_fits = fit_rabi_decay(self.data['amp_vec'], self.data['signal_avg_vec'], variable_phase=True)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                fits = rabi_fits[0]
                RabiT = 2 * np.pi / fits[1]
                phaseoffs = fits[2]

                self.data['fits'] = fits
                self.data['phaseoffs'] = phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_amp'] = RabiT / 2 - phaseoffs * RabiT / (2 * np.pi)
                self.data['pi_half_amp'] = RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['three_pi_half_amp'] = 3 * RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
                self.data['T2_star'] = fits[4]
                self.data['rabi_freq'] = fits[1] / (2 * np.pi)  # Rabi frequency in [MHz]

            try:
                current_rep_num = progress_handle.fetch_all()
                current_counts_vec = tracking_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                # Check if tracking is called

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
        time.sleep(5)

    def plot(self, figure_list):
        super(PowerRabi, self).plot([figure_list[0], figure_list[1]])

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

        if 'amp_vec' in data.keys() and 'signal_avg_vec' in data.keys():
            if 'fits' in data.keys():
                try:
                    pi_amp = data['pi_amp']
                    pi_half_amp = data['pi_half_amp']
                    three_pi_half_amp = data['three_pi_half_amp']
                    fits = data['fits']
                    phaseoffs = data['phaseoffs']

                    axes_list[0].clear()
                    axes_list[0].plot(data['amp_vec'], data['signal_avg_vec'], '.-')
                    axes_list[0].plot(data['amp_vec'], cose_with_decay(data['amp_vec'], *fits), lw=2)

                    axes_list[0].plot(pi_amp, cose_with_decay(pi_amp, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi$={:0.1f}ns'.format(pi_amp), xy=(pi_amp, cose_with_decay(pi_amp, *fits)),
                                          xytext=(pi_amp + 10., cose_with_decay(pi_amp, *fits)), xycoords='data')
                    axes_list[0].plot(pi_half_amp, cose_with_decay(pi_half_amp, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$\pi/2$=\n{:0.1f}ns'.format(pi_half_amp),
                                          xy=(pi_half_amp, cose_with_decay(pi_half_amp, *fits)),
                                          xytext=(pi_half_amp + 10., cose_with_decay(pi_half_amp, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(three_pi_half_amp, cose_with_decay(three_pi_half_amp, *fits), 'ro', lw=3)
                    axes_list[0].annotate('$3\pi/2$=\n{:0.1f}ns'.format(three_pi_half_amp),
                                          xy=(three_pi_half_amp, cose_with_decay(three_pi_half_amp, *fits)),
                                          xytext=(three_pi_half_amp + 10., cose_with_decay(three_pi_half_amp, *fits)),
                                          xycoords='data')
                    axes_list[0].plot(-phaseoffs, cose_with_decay(-phaseoffs, *fits), 'gd', lw=3)
                    axes_list[0].annotate('$start$={:0.1f}ns'.format(-phaseoffs),
                                          xy=(-phaseoffs, cose_with_decay(-phaseoffs, *fits)),
                                          xytext=(-phaseoffs + 10., cose_with_decay(-phaseoffs, *fits)), xycoords='data')
                    if title:
                        axes_list[0].set_title(
                            'Power Rabi frequency: {:0.2f}\npi-half amp: {:2.1f}, pi-amp: {:2.1f}, 3pi-half amp: {:2.1f} \n T2*: {:2.1f}, Ref fluor: {:0.1f}kcps\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF freq: {:0.2f}MHz, tau: {:0.2f}ns'.format(
                                self.data['rabi_freq'], pi_half_amp, pi_amp, three_pi_half_amp, self.data['T2_star'],
                                self.data['ref_cnts'],
                                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                                self.settings['mw_pulses']['IF_frequency'] * 1e-6, self.settings['mw_pulses']['tau_time']))
                    axes_list[0].set_xlabel('Rabi tau [ns]')
                    axes_list[0].set_ylabel('Contrast')

                except Exception as e:
                    print('** ATTENTION **')
                    print(e)

            else:
                try:
                    axes_list[0].clear()
                    axes_list[0].plot(data['amp_vec'], data['signal_avg_vec'])
                    axes_list[0].set_xlabel('IF amplitude')
                    axes_list[0].set_ylabel('Contrast')
                    if title:
                        axes_list[0].set_title(
                            'Power Rabi\nRef fluor: {:0.1f}kcps\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF freq: {:0.2f}MHz, tau: {:0.2f}ns'.format(
                                self.data['ref_cnts'],
                                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                                self.settings['mw_pulses']['IF_frequency'] * 1e-6, self.settings['mw_pulses']['tau_time']))

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


class ESRQM(Script):
    """
        This class runs ESR on an NV center. MW frequency is swept by sweeping the IF frequency output by the QM.
        - Ziwei Qiu 8/12/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('power_out', -50.0, float, 'RF power in dBm'),
        Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
        Parameter('IF_center', 0.0, float, 'center of the IF frequency scan'),
        Parameter('IF_range', 1e8, float, 'range of the IF frequency scan'),
        Parameter('freq_points', 100, int, 'number of frequencies in scan'),
        Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
        Parameter('time_per_pt', 20000, int, 'time per frequency point in ns, default 20us'),
        Parameter('read_out',
                  [Parameter('meas_len', 19000, int, 'measurement time in ns')
                   ]),
        Parameter('fit_constants',
                  [Parameter('num_of_peaks', -1, [-1, 1, 2],
                             'specify number of peaks for fitting. if not specifying the number of peaks, choose -1'),
                   Parameter('minimum_counts', 0.9, float, 'minumum counts for an ESR to not be considered noise'),
                   Parameter('contrast_factor', 3.0, float,
                             'minimum contrast for an ESR to not be considered noise'),
                   Parameter('zfs', 2.87E9, float, 'zero-field splitting [Hz]'),
                   Parameter('gama', 2.8028E6, float, 'NV spin gyromagnetic ratio [Hz/Gauss]'),

                   ]),
        Parameter('rep_num', 20000, int, 'define the repetition number'),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in units of ns'),
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
                self.meas_len = round(self.settings['read_out']['meas_len'])
                f_start = self.settings['IF_center'] - self.settings['IF_range'] / 2
                f_stop = self.settings['IF_center'] + self.settings['IF_range'] / 2
                freqs_num = self.settings['freq_points']
                time_per_pt = round(self.settings['time_per_pt'] / 4)
                IF_amp = self.settings['IF_amp']
                if IF_amp > 1.0:
                    IF_amp = 1.0
                elif IF_amp < 0.0:
                    IF_amp = 0.0

                self.f_vec = [int(f_) for f_ in np.linspace(f_start, f_stop, freqs_num)]

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                # define the qua program
                with program() as ODMR_CW:
                    result1 = declare(int, size=res_len)
                    result2 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    f = declare(int)
                    n = declare(int)
                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(f, self.f_vec):
                            update_frequency('qubit', f)
                            play('const' * amp(IF_amp), 'qubit', duration=time_per_pt)
                            play('trig', 'laser', duration=time_per_pt)
                            # wait(delay_readout, 'readout1', 'readout2')
                            measure('readout', 'readout1', None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure('readout', 'readout2', None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))
                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            # save(total_counts, "total_counts")
                            save(n, rep_num_st)

                    with stream_processing():
                        total_counts_st.buffer(freqs_num).average().save('florecence_vs_freq')
                        total_counts_st.buffer(10000).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(ODMR_CW)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(ODMR_CW, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):
        start = time.time()
        job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
        # job.get_simulated_samples().con1.plot()
        end = time.time()
        print('QM simulation took {:.1f}s.'.format(end - start))
        self.log('QM simulation took {:.1f}s.'.format(end - start))
        samples = job_sim.get_simulated_samples().con1
        # self.data['analog'] = samples.analog
        # self.data['digital'] = samples.digital
        self.data = {'analog': samples.analog,
                     'digital': samples.digital}

    def _qm_execution(self, qua_program, job_stop):

        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['power_out']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, no trigger).')

        try:
            job = self.qm.execute(qua_program)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("florecence_vs_freq")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            freq_values = np.array(self.f_vec) + self.settings['mw_frequency']
            self.data = {'f_vec': freq_values, 'avg_cnts': None, 'esr_avg': None,
                         'fit_params': None}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    # esr_avg = vec * 1e6 / self.meas_len / 2
                    esr_avg = vec * 1e6 / self.meas_len # After calibrating the threshold, no need to have 2
                    self.data.update({'esr_avg': esr_avg / esr_avg.mean(), 'avrg_counts': esr_avg.mean()})

                # do fitting
                try:
                    fit_params = fit_esr(freq_values, esr_avg / esr_avg.mean(),
                                         min_counts=self.settings['fit_constants']['minimum_counts'],
                                         contrast_factor=self.settings['fit_constants']['contrast_factor'],
                                         num_of_peaks=self.settings['fit_constants']['num_of_peaks'])
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    self.data.update({'fit_params': fit_params})

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    # current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len / 2
                    current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
                    self.progress = current_rep_num * 100. / self.settings['rep_num']
                    self.updateProgress.emit(int(self.progress))

                if self._abort:
                    # job.halt()
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['esr_full'] = full_res_vec

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    # def plot(self, figure_list):
    #     super(ESRQM, self).plot([figure_list[0], figure_list[1]])

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
            plot_qmsimulation_samples(axes_list[0], data)

        if 'f_vec' in data.keys() and 'esr_avg' in data.keys():
            if 'fit_params' in data.keys():
                plot_esr(axes_list[0], data['f_vec'], data['esr_avg'],
                         data['fit_params'],
                         avg_counts=data['avrg_counts'],
                         mw_power=self.settings['power_out'], D=self.settings['fit_constants']['zfs'],
                         gama=self.settings['fit_constants']['gama'], err=None, LO=self.settings['mw_frequency'])
            else:
                axes_list[0].clear()
                axes_list[0].plot(data['f_vec'] / 1e6, data['esr_avg'])
                axes_list[0].set_xlabel('IF frequency [MHz]')
                axes_list[0].set_ylabel('Photon Counts [kcps]')
                axes_list[0].set_title('ESR\n' + 'LO: {:.4f} GHz'.format(self.settings['mw_frequency'] / 1e9))

    # def _update_plot(self, axes_list):
    #     if self.data is None:
    #         return
    #
    #     if 'f_vec' in self.data.keys() and 'esr_avg' in self.data.keys():
    #         try:
    #             axes_list[0].lines[0].set_ydata(self.data['esr_avg'])
    #             axes_list[0].lines[0].set_xdata(self.data['f_vec'] / 1e6)
    #             axes_list[0].relim()
    #             axes_list[0].autoscale_view()
    #             axes_list[0].set_title(
    #                 'ESR live plot\n' + 'LO: {:.4f} GHz'.format(
    #                     self.settings['mw_frequency'] / 1e9) + '\n' + time.asctime())
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
        the default creates a single axes object on each figure
        This can/should be overwritten in a child script if more axes objects are needed
        Args:
            figure_list: a list of figure objects
        Returns:
            axes_list: a list of axes objects

        """

        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()  # this is to make the plot refresh faster
        new_figure_list = [figure_list[1]]
        return super(ESRQM, self).get_axes_layout(new_figure_list)


class ESRQM_FitGuaranteed(Script):
    """
        This class runs ESR on an NV center. MW frequency is swept by sweeping the IF frequency output by the QM.
        - Ziwei Qiu 9/18/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('esr_avg_min', 2000, int, 'minimum number of esr averages'),
        Parameter('esr_avg_max', 20000, int, 'maximum number of esr averages'),
        Parameter('power_out', -50.0, float, 'RF power in dBm'),
        Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
        Parameter('IF_center', 0.0, float, 'center of the IF frequency scan'),
        Parameter('IF_range', 1e8, float, 'range of the IF frequency scan'),
        Parameter('freq_points', 100, int, 'number of frequencies in scan'),
        Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
        Parameter('time_per_pt', 20000, int, 'time per frequency point in ns, default 20us'),
        Parameter('read_out',
                  [Parameter('meas_len', 19000, int, 'measurement time in ns')
                   ]),
        Parameter('fit_constants',
                  [Parameter('num_of_peaks', -1, [-1, 1, 2],
                             'specify number of peaks for fitting. if not specifying the number of peaks, choose -1'),
                   Parameter('minimum_counts', 0.9, float, 'minumum counts for an ESR to not be considered noise'),
                   Parameter('contrast_factor', 3.0, float,
                             'minimum contrast for an ESR to not be considered noise'),
                   Parameter('zfs', 2.87E9, float, 'zero-field splitting [Hz]'),
                   Parameter('gama', 2.8028E6, float, 'NV spin gyromagnetic ratio [Hz/Gauss]'),
                   ]),

        Parameter('simulation_duration', 10000, int, 'duration of simulation in units of ns'),
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

        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:

                rep_num = self.settings['esr_avg_max']
                self.meas_len = round(self.settings['read_out']['meas_len'])
                f_start = self.settings['IF_center'] - self.settings['IF_range'] / 2
                f_stop = self.settings['IF_center'] + self.settings['IF_range'] / 2
                freqs_num = self.settings['freq_points']
                time_per_pt = round(self.settings['time_per_pt'] / 4)
                IF_amp = self.settings['IF_amp']
                if IF_amp > 1.0:
                    IF_amp = 1.0
                elif IF_amp < 0.0:
                    IF_amp = 0.0

                self.f_vec = [int(f_) for f_ in np.linspace(f_start, f_stop, freqs_num)]

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                # define the qua program
                with program() as ODMR_CW:
                    result1 = declare(int, size=res_len)
                    result2 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    f = declare(int)
                    n = declare(int)
                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        with for_each_(f, self.f_vec):
                            update_frequency('qubit', f)
                            play('const' * amp(IF_amp), 'qubit', duration=time_per_pt)
                            play('trig', 'laser', duration=time_per_pt)
                            # wait(delay_readout, 'readout1', 'readout2')
                            measure('readout', 'readout1', None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure('readout', 'readout2', None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))
                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            # save(total_counts, "total_counts")
                            save(n, rep_num_st)

                    with stream_processing():
                        total_counts_st.buffer(freqs_num).average().save('florecence_vs_freq')
                        total_counts_st.buffer(10000).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(ODMR_CW)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(ODMR_CW, job_stop)
                self._abort = True

    def _qm_simulation(self, qua_program):
        start = time.time()
        job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)))
        # job.get_simulated_samples().con1.plot()
        end = time.time()
        print('QM simulation took {:.1f}s.'.format(end - start))
        self.log('QM simulation took {:.1f}s.'.format(end - start))
        samples = job_sim.get_simulated_samples().con1
        # self.data['analog'] = samples.analog
        # self.data['digital'] = samples.digital
        self.data = {'analog': samples.analog,
                     'digital': samples.digital}

    def _qm_execution(self, qua_program, job_stop):

        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['power_out']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, no trigger).')

        try:
            job = self.qm.execute(qua_program)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            vec_handle = job.result_handles.get("florecence_vs_freq")
            progress_handle = job.result_handles.get("live_rep_num")
            tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            tracking_handle.wait_for_values(1)
            freq_values = np.array(self.f_vec) + self.settings['mw_frequency']
            self.data = {'f_vec': freq_values, 'avg_cnts': None, 'esr_avg': None,
                         'fit_params': None}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    esr_avg = vec * 1e6 / self.meas_len
                    self.data.update({'esr_avg': esr_avg / esr_avg.mean(), 'avrg_counts': esr_avg.mean()})

                # do fitting
                try:
                    fit_params = fit_esr(freq_values, esr_avg / esr_avg.mean(),
                                         min_counts=self.settings['fit_constants']['minimum_counts'],
                                         contrast_factor=self.settings['fit_constants']['contrast_factor'],
                                         num_of_peaks=self.settings['fit_constants']['num_of_peaks'])
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    self.data.update({'fit_params': fit_params})

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION **')
                    print(e)
                else:
                    # current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len / 2
                    current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len

                    self.progress = current_rep_num * 100. / self.settings['esr_avg_min']
                    self.updateProgress.emit(int(self.progress))

                    # Break out of the loop if # of averages is enough and a good fit has been found.
                    if current_rep_num >= self.settings['esr_avg_min'] and fit_params is not None:
                        self._abort = True
                        break

                if self._abort:
                    # job.halt()
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        # full_res = job.result_handles.get("total_counts")
        # full_res_vec = full_res.fetch_all()
        # self.data['esr_full'] = full_res_vec

        self.qm.execute(job_stop)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

    # def plot(self, figure_list):
    #     super(ESRQM, self).plot([figure_list[0], figure_list[1]])

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
            plot_qmsimulation_samples(axes_list[0], data)

        if 'f_vec' in data.keys() and 'esr_avg' in data.keys():
            if 'fit_params' in data.keys():
                plot_esr(axes_list[0], data['f_vec'], data['esr_avg'],
                         data['fit_params'],
                         avg_counts=data['avrg_counts'],
                         mw_power=self.settings['power_out'], D=self.settings['fit_constants']['zfs'],
                         gama=self.settings['fit_constants']['gama'], err=None, LO=self.settings['mw_frequency'])
                axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
            else:
                axes_list[0].clear()
                axes_list[0].plot(data['f_vec'] / 1e6, data['esr_avg'])
                axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
                axes_list[0].set_xlabel('IF frequency [MHz]')
                axes_list[0].set_ylabel('Photon Counts [kcps]')
                axes_list[0].set_title('ESR\n' + 'LO: {:.4f} GHz'.format(self.settings['mw_frequency'] / 1e9))


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

        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf()  # this is to make the plot refresh faster
        new_figure_list = [figure_list[1]]
        return super(ESRQM_FitGuaranteed, self).get_axes_layout(new_figure_list)

#
# class PulsedESR(Script):
#     """
#         This script applies a microwave pulse at fixed power and durations for varying frequencies.
#         - Ziwei Qiu 1/18/2021
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
#                   'IP address of the QM server'),
#         Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
#                   'choose to do output simulation or real experiment'),
#         Parameter('esr_avg_min', 100000, int, 'minimum number of esr averages'),
#         Parameter('esr_avg_max', 200000, int, 'maximum number of esr averages'),
#         Parameter('power_out', -50.0, float, 'RF power in dBm'),
#         Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
#         Parameter('IF_center', 0.0, float, 'center of the IF frequency scan'),
#         Parameter('IF_range', 1e8, float, 'range of the IF frequency scan'),
#         Parameter('freq_points', 100, int, 'number of frequencies in scan'),
#         Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
#         Parameter('mw_tau', 80, float, 'the time duration of the microwaves (in ns)'),
#         Parameter('fit_constants',
#                   [Parameter('num_of_peaks', -1, [-1, 1, 2],
#                              'specify number of peaks for fitting. if not specifying the number of peaks, choose -1'),
#                    Parameter('minimum_counts', 0.9, float, 'minumum counts for an ESR to not be considered noise'),
#                    Parameter('contrast_factor', 3.0, float,
#                              'minimum contrast for an ESR to not be considered noise'),
#                    Parameter('zfs', 2.87E9, float, 'zero-field splitting [Hz]'),
#                    Parameter('gama', 2.8028E6, float, 'NV spin gyromagnetic ratio [Hz/Gauss]'),
#                    ]),
#         Parameter('read_out',
#                   [Parameter('meas_len', 180, int, 'measurement time in ns'),
#                    Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
#                    Parameter('delay_readout', 370, int,
#                              'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
#                    Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
#                    Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
#                    ]),
#         Parameter('NV_tracking',
#                   [Parameter('on', False, bool,
#                              'track NV and do a galvo scan if the counts out of the reference range'),
#                    Parameter('tracking_num', 50000, int,
#                              'number of recent APD windows used for calculating current counts, suggest 50000'),
#                    Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
#                    Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
#                    ]),
#         Parameter('simulation_duration', 10000, int, 'duration of simulation in units of ns'),
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
#                 pi_time = round(self.settings['mw_tau'] / 4) # unit: cycle of 4ns
#                 config['pulses']['pi_pulse']['length'] = int(pi_time * 4) # unit: ns
#                 self.qm = self.qmm.open_qm(config)
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
#                 IF_amp = self.settings['mw_pulses']['IF_amp']
#                 if IF_amp > 1.0:
#                     IF_amp = 1.0
#                 elif IF_amp < 0.0:
#                     IF_amp = 0.0
#
#                 res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2
#
#                 f_start = self.settings['IF_center'] - self.settings['IF_range'] / 2
#                 f_stop = self.settings['IF_center'] + self.settings['IF_range'] / 2
#                 freqs_num = self.settings['freq_points']
#                 self.f_vec = [int(f_) for f_ in np.linspace(f_start, f_stop, freqs_num)]
#
#                 # define the qua program
#                 with program() as pulsed_esr:
#                     result1 = declare(int, size=res_len)
#                     counts1 = declare(int, value=0)
#                     result2 = declare(int, size=res_len)
#                     counts2 = declare(int, value=0)
#                     total_counts = declare(int, value=0)
#
#                     f = declare(int)
#                     n = declare(int)
#
#                     # the following variable is used to flag tracking
#                     assign(IO1, False)
#
#                     total_counts_st = declare_stream()
#                     rep_num_st = declare_stream()
#
#                     with for_(n, 0, n < rep_num, n + 1):
#                         # Check if tracking is called
#                         with while_(IO1):
#                             play('trig', 'laser', duration=10000)
#                         with for_each_(f, self.f_vec):
#                             update_frequency('qubit', f)
#                             play('pi' * amp(IF_amp), 'qubit')
#                             align('qubit', 'laser', 'readout1', 'readout2')
#                             wait(delay_mw_readout, 'laser', 'readout1', 'readout2')
#                             play('trig', 'laser', duration=nv_reset_time)
#                             wait(delay_readout, 'readout1', 'readout2')
#                             measure('readout', 'readout1', None,
#                                     time_tagging.raw(result1, self.meas_len, targetLen=counts1))
#                             measure('readout', 'readout2', None,
#                                     time_tagging.raw(result2, self.meas_len, targetLen=counts2))
#
#                             align('qubit', 'laser', 'readout1', 'readout2')
#                             wait(laser_off, 'qubit')
#
#                             assign(total_counts, counts1 + counts2)
#                             save(total_counts, total_counts_st)
#                             save(n, rep_num_st)
#                             save(total_counts, "total_counts")
#
#                     with stream_processing():
#                         total_counts_st.buffer(freqs_num).average().save("live_data")
#                         total_counts_st.buffer(tracking_num).save("current_counts")
#                         rep_num_st.save("live_rep_num")
#
#                 with program() as job_stop:
#                     play('trig', 'laser', duration=10)
#
#                 if self.settings['to_do'] == 'simulation':
#                     self._qm_simulation(pulsed_esr)
#                 elif self.settings['to_do'] == 'execution':
#                     self._qm_execution(pulsed_esr, job_stop)
#                 self._abort = True
#
#     def _qm_simulation(self, qua_program):
#         try:
#             start = time.time()
#             job_sim = self.qm.simulate(qua_program, SimulationConfig(round(self.settings['simulation_duration'] / 4)),
#                                        flags=['skip-add-implicit-align'])
#             end = time.time()
#         except Exception as e:
#             print('** ATTENTION in QM simulation **')
#             print(e)
#         else:
#             print('QM simulation took {:.1f}s.'.format(end - start))
#             self.log('QM simulation took {:.1f}s.'.format(end - start))
#             samples = job_sim.get_simulated_samples().con1
#             self.data = {'analog': samples.analog,
#                          'digital': samples.digital}
#
#     def _qm_execution(self, qua_program, job_stop):
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
#             print('** ATTENTION in QM execution **')
#             print(e)
#
#         else:
#             vec_handle = job.result_handles.get("live_data")
#             progress_handle = job.result_handles.get("live_rep_num")
#             tracking_handle = job.result_handles.get("current_counts")
#
#             vec_handle.wait_for_values(1)
#             progress_handle.wait_for_values(1)
#             tracking_handle.wait_for_values(1)
#
#             freq_values = np.array(self.f_vec) + self.settings['mw_frequency']
#             self.data = {'f_vec': freq_values, 'avg_cnts': None, 'esr_avg': None, 'fit_params': None}
#
#             ref_counts = -1
#             tolerance = self.settings['NV_tracking']['tolerance']
#
#             while vec_handle.is_processing():
#                 try:
#                     vec = vec_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION in vec_handle.fetch_all() **')
#                     print(e)
#                 else:
#                     esr_avg = vec * 1e6 / self.meas_len
#                     self.data.update({'esr_avg': esr_avg / esr_avg.mean(), 'avrg_counts': esr_avg.mean()})
#
#                 # do fitting
#                 try:
#                     fit_params = fit_esr(freq_values, esr_avg / esr_avg.mean(),
#                                          min_counts=self.settings['fit_constants']['minimum_counts'],
#                                          contrast_factor=self.settings['fit_constants']['contrast_factor'],
#                                          num_of_peaks=self.settings['fit_constants']['num_of_peaks'])
#                 except Exception as e:
#                     print('** ATTENTION in fit_esr **')
#                     print(e)
#                 else:
#                     self.data.update({'fit_params': fit_params})
#
#                 try:
#                     current_rep_num = progress_handle.fetch_all()
#                     current_counts_vec = tracking_handle.fetch_all()
#                 except Exception as e:
#                     print('** ATTENTION in progress_handle / tracking_handle **')
#                     print(e)
#                 else:
#                     current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
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
#                                     print('** ATTENTION in set_io1_value **')
#                                     print(e)
#                                 else:
#                                     counts_out_num = 0
#                                     ref_counts = self.settings['NV_tracking']['ref_counts']
#
#                     self.data['rep_num'] = float(current_rep_num)
#
#                     self.progress = current_rep_num * 100. / self.settings['esr_avg_min']
#                     self.updateProgress.emit(int(self.progress))
#
#                     # Break out of the loop if # of averages is enough and a good fit has been found.
#                     if current_rep_num >= self.settings['esr_avg_min'] and self.data['fit_params'] is not None:
#                         self._abort = True
#                         break
#
#                 if self._abort:
#                     # job.halt()
#                     self.qm.execute(job_stop)
#                     break
#
#                 time.sleep(0.8)
#
#         self.qm.execute(job_stop)
#
#         self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
#         self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
#         print('Turned off RF generator SGS100A (IQ off, trigger off).')
#
#
#     def NV_tracking(self):
#         self.flag_optimize_plot = True
#         self.scripts['optimize'].run()
#
#     def plot(self, figure_list):
#         super(PulsedESR, self).plot([figure_list[0], figure_list[1]])
#
#     def _plot(self, axes_list, data=None):
#         if data is None:
#             data = self.data
#
#         if 'analog' in data.keys() and 'digital' in data.keys():
#             plot_qmsimulation_samples(axes_list[0], data)
#
#         if 'f_vec' in data.keys() and 'esr_avg' in data.keys():
#             if 'fit_params' in data.keys():
#                 axes_list[0].clear()
#                 plot_esr(axes_list[0], data['f_vec'], data['esr_avg'],
#                          data['fit_params'],
#                          avg_counts=data['avrg_counts'],
#                          mw_power=self.settings['power_out'], D=self.settings['fit_constants']['zfs'],
#                          gama=self.settings['fit_constants']['gama'], err=None, LO=self.settings['mw_frequency'])
#                 axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
#             else:
#                 axes_list[0].clear()
#                 axes_list[0].plot(data['f_vec'] / 1e6, data['esr_avg'])
#                 axes_list[0].grid(b=True, which='major', color='#666666', linestyle='--')
#                 axes_list[0].set_xlabel('IF frequency [MHz]')
#                 axes_list[0].set_ylabel('Photon Counts')
#                 axes_list[0].set_title('ESR\n' + 'LO: {:.4f} GHz'.format(self.settings['mw_frequency'] / 1e9))
#
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


class LaserControl(Script):
    """
        This script turns on and off the laser.
        - Ziwei Qiu 8/15/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('laser_on', False, bool, 'turn on or off the laser')
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
            with program() as laser_on:
                with infinite_loop_():
                    play('trig', 'laser', duration=3000)

            with program() as job_stop:
                play('trig', 'laser', duration=10)

            if self.settings['laser_on']:
                self.qm.execute(laser_on)
                print('Laser is on.')
            else:
                self.qm.execute(job_stop)
                print('Laser is off.')
            self._abort = True


if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'TimeTraceQMsim': 'TimeTraceQMsim'}, script, instr)

    print(script)
    print(failed)
    print(instr)
