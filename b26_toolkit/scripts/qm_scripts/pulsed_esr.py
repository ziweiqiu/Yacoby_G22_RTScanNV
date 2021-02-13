import time
import numpy as np

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from pylabcontrol.core import Script, Parameter
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from b26_toolkit.instruments import SGS100ARFSource
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser
from b26_toolkit.data_processing.esr_signal_processing import fit_esr
from b26_toolkit.plotting.plots_1d import plot_esr


class PulsedESR(Script):
    """
        This script applies a microwave pulse at fixed power and durations for varying IF frequencies.
        - Ziwei Qiu 1/18/2021 (newest)
    """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'simulation', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('esr_avg_min', 100000, int, 'minimum number of esr averages'),
        Parameter('esr_avg_max', 200000, int, 'maximum number of esr averages'),
        Parameter('power_out', -50.0, float, 'RF power in dBm'),
        Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
        Parameter('IF_center', 0.0, float, 'center of the IF frequency scan'),
        Parameter('IF_range', 1e8, float, 'range of the IF frequency scan'),
        Parameter('freq_points', 100, int, 'number of frequencies in scan'),
        Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
        Parameter('mw_tau', 80, float, 'the time duration of the microwaves (in ns)'),
        Parameter('fit_constants',
                  [Parameter('num_of_peaks', -1, [-1, 1, 2],
                             'specify number of peaks for fitting. if not specifying the number of peaks, choose -1'),
                   Parameter('minimum_counts', 0.9, float, 'minumum counts for an ESR to not be considered noise'),
                   Parameter('contrast_factor', 3.0, float,
                             'minimum contrast for an ESR to not be considered noise'),
                   Parameter('zfs', 2.87E9, float, 'zero-field splitting [Hz]'),
                   Parameter('gama', 2.8028E6, float, 'NV spin gyromagnetic ratio [Hz/Gauss]'),
                   ]),
        Parameter('read_out',
                  [Parameter('meas_len', 180, int, 'measurement time in ns'),
                   Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
                   Parameter('delay_readout', 370, int,
                             'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
                   Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
                   Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
                   ]),
        Parameter('NV_tracking',
                  [Parameter('on', False, bool,
                             'track NV and do a galvo scan if the counts out of the reference range'),
                   Parameter('tracking_num', 50000, int,
                             'number of recent APD windows used for calculating current counts, suggest 50000'),
                   Parameter('ref_counts', -1, float, 'if -1, the first current count will be used as reference'),
                   Parameter('tolerance', 0.25, float, 'define the reference range (1+/-tolerance)*ref, suggest 0.25')
                   ]),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in units of ns'),
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
                pi_time = round(self.settings['mw_tau'] / 4) # unit: cycle of 4ns
                config['pulses']['pi_pulse']['length'] = int(pi_time * 4) # unit: ns
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                rep_num = self.settings['esr_avg_max']
                tracking_num = self.settings['NV_tracking']['tracking_num']
                self.meas_len = round(self.settings['read_out']['meas_len'])
                nv_reset_time = round(self.settings['read_out']['nv_reset_time'] / 4)
                laser_off = round(self.settings['read_out']['laser_off'] / 4)
                delay_mw_readout = round(self.settings['read_out']['delay_mw_readout'] / 4)
                delay_readout = round(self.settings['read_out']['delay_readout'] / 4)
                IF_amp = self.settings['IF_amp']
                if IF_amp > 1.0:
                    IF_amp = 1.0
                elif IF_amp < 0.0:
                    IF_amp = 0.0

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                f_start = self.settings['IF_center'] - self.settings['IF_range'] / 2
                f_stop = self.settings['IF_center'] + self.settings['IF_range'] / 2
                freqs_num = self.settings['freq_points']
                self.f_vec = [int(f_) for f_ in np.linspace(f_start, f_stop, freqs_num)]

                # define the qua program
                with program() as pulsed_esr:
                    result1 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    result2 = declare(int, size=res_len)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)

                    f = declare(int)
                    n = declare(int)

                    # the following variable is used to flag tracking
                    assign(IO1, False)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        # Check if tracking is called
                        with while_(IO1):
                            play('trig', 'laser', duration=10000)
                        with for_each_(f, self.f_vec):
                            update_frequency('qubit', f)
                            play('pi' * amp(IF_amp), 'qubit')
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
                        total_counts_st.buffer(freqs_num).average().save("live_data")
                        total_counts_st.buffer(tracking_num).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(pulsed_esr)
                elif self.settings['to_do'] == 'execution':
                    self._qm_execution(pulsed_esr, job_stop)
                self._abort = True

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
        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['power_out']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_frequency']})
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

            freq_values = np.array(self.f_vec) + self.settings['mw_frequency']
            self.data = {'f_vec': freq_values, 'avg_cnts': None, 'esr_avg': None, 'fit_params': None}

            ref_counts = -1
            tolerance = self.settings['NV_tracking']['tolerance']

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in vec_handle.fetch_all() **')
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
                    print('** ATTENTION in fit_esr **')
                    print(e)
                else:
                    self.data.update({'fit_params': fit_params})

                try:
                    current_rep_num = progress_handle.fetch_all()
                    current_counts_vec = tracking_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle / tracking_handle **')
                    print(e)
                else:
                    current_counts_kcps = current_counts_vec.mean() * 1e6 / self.meas_len
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

                    self.progress = current_rep_num * 100. / self.settings['esr_avg_min']
                    self.updateProgress.emit(int(self.progress))

                    # Break out of the loop if # of averages is enough and a good fit has been found.
                    if current_rep_num >= self.settings['esr_avg_min'] and self.data['fit_params'] is not None:
                        self._abort = True
                        break

                if self._abort:
                    # job.halt()
                    self.qm.execute(job_stop)
                    break

                time.sleep(0.8)

        self.qm.execute(job_stop)

        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')


    def NV_tracking(self):
        self.flag_optimize_plot = True
        self.scripts['optimize'].run()

    def plot(self, figure_list):
        super(PulsedESR, self).plot([figure_list[0], figure_list[1]])

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        if 'analog' in data.keys() and 'digital' in data.keys():
            plot_qmsimulation_samples(axes_list[1], data)

        if 'f_vec' in data.keys() and 'esr_avg' in data.keys():
            if 'fit_params' in data.keys():
                axes_list[0].clear()
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
                axes_list[0].set_ylabel('Photon Counts')
                axes_list[0].set_title('ESR\n' + 'LO: {:.4f} GHz'.format(self.settings['mw_frequency'] / 1e9))


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




if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'PulsedESR': 'PulsedESR'}, script, instr)

    print(script)
    print(failed)
    print(instr)