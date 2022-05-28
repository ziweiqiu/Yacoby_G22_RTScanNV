import numpy as np
import time
import math
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.set_laser import SetScannerXY_gentle
from b26_toolkit.scripts.galvo_scan.confocal_scan_G22B import AFM1D_qm
from b26_toolkit.scripts.qm_scripts.basic import ESRQM
from b26_toolkit.instruments import SGS100ARFSource, AgilentN9310A
from b26_toolkit.scripts.optimize import optimize
from b26_toolkit.plotting.plots_2d import plot_fluorescence_new, update_fluorescence


from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import frame_rotation as z_rot
from qm.qua import *
from qm import SimulationConfig
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples

class ScanningESR(Script):
    """
        Perform ESR and scan along a 1D line.
        - Ziwei Qiu 2/3/2022
    """
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'initial x-coordinate [V]'),
                   Parameter('y', 0, float, 'initial y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0, float, 'last x-coordinate [V]'),
                   Parameter('y', 1, float, 'last y-coordinate [V]')
                   ]),
        Parameter('num_points', 60, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('do_afm1d',
                  [Parameter('before', False, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int,
                             'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool,
                             'whether to scan back to the original point after the experiment is done'),
                   ]),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 20, int, 'do the optimization once per N points')]),
        Parameter('mw_power_change',
                  [Parameter('on', False, bool, 'choose whether to change the MW power during scan'),
                   Parameter('start_power', -20, float, 'choose the MW power at the first point'),
                   Parameter('end_power', -20, float, 'choose the MW power at the last point'),
                   ]),
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'esr': ESRQM, 'set_scanner': SetScannerXY_gentle, 'afm1d': AFM1D_qm, 'afm1d_before_after': AFM1D_qm,
                'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

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

    def do_afm1d_after(self, verbose = True):
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

    def _function(self):
        scan_pos_1d, dist_array = self._get_scan_array()

        if self.settings['mw_power_change']['on']:
            mw_power_arr = np.linspace(self.settings['mw_power_change']['start_power'],
                                       self.settings['mw_power_change']['end_power'],
                                       self.settings['num_points'], endpoint=True)
        else:
            mw_power_arr = np.linspace(self.scripts['esr'].settings['power_out'],
                                       self.scripts['esr'].settings['power_out'],
                                       self.settings['num_points'], endpoint=True)

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

        f1_MHz = (self.scripts['esr'].settings['mw_frequency'] + self.scripts['esr'].settings['IF_center'] - 0.5 * \
                  self.scripts['esr'].settings['IF_range']) / 1E6
        f2_MHz = (self.scripts['esr'].settings['mw_frequency'] + self.scripts['esr'].settings['IF_center'] + 0.5 * \
                  self.scripts['esr'].settings['IF_range']) / 1E6
        x1 = 0
        x2 = np.sqrt((self.settings['point_b']['x'] - self.settings['point_a']['x']) ** 2 + (
                    self.settings['point_b']['y'] - self.settings['point_a']['y']) ** 2)

        self.data = {'scan_pos_1d': scan_pos_1d, 'afm_dist_array': np.array([]), 'afm_ctr': np.array([]),
                     'afm_analog': np.array([]), 'mw_power_arr': mw_power_arr,
                     'esr_data': 0.9*np.ones((self.settings['num_points'], self.scripts['esr'].settings['freq_points'])),
                     'extent': [f1_MHz, f2_MHz, x2, x1] }

        # do afm before scan
        if self.settings['do_afm1d']['before'] and not self._abort:
            self.do_afm1d_before()

        # do tracking at the initial point
        if not self._abort and self.settings['do_optimize']['on']:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_0'
            self.scripts['optimize'].run()

        # do esr at the initial point
        if not self._abort:
            self.scripts['esr'].settings['tag'] = 'esr_0'
            self.scripts['esr'].settings['power_out'] = float(mw_power_arr[0])
            self.scripts['esr'].run()
            self.data['esr_data'][0] = self.scripts['esr'].data['esr_avg']

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                break

            afm_start = scan_pos_1d[i]
            afm_end = scan_pos_1d[i + 1]
            self.setup_afm(afm_start, afm_end)

            # move to the new pt
            try:
                self.scripts['afm1d'].run()
                afm_dist_array = self.scripts['afm1d'].data['dist_array'] + dist_array[i]
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.data['afm_dist_array'] = np.concatenate((self.data['afm_dist_array'], afm_dist_array))
                afm_ctr = self.scripts['afm1d'].data['data_ctr'][0]
                self.data['afm_ctr'] = np.concatenate((self.data['afm_ctr'], afm_ctr))
                afm_analog = self.scripts['afm1d'].data['data_analog'][0]
                self.data['afm_analog'] = np.concatenate((self.data['afm_analog'], afm_analog))

            # do tracking
            if self.settings['do_optimize']['on'] and (i + 1) % self.settings['do_optimize'][
                'frequency'] == 0 and not self._abort:
                self.scripts['optimize'].settings['tag'] = 'optimize_'+str(i+1)
                self.scripts['optimize'].run()

            # run ESR
            try:
                self.scripts['esr'].settings['tag'] = 'esr_'+str(i+1)
                self.scripts['esr'].settings['power_out'] = float(mw_power_arr[i+1])
                self.scripts['esr'].run()
                self.data['esr_data'][i+1] = self.scripts['esr'].data['esr_avg']
            except Exception as e:
                print('** ATTENTION in running ESR **')
                print(e)

        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()

        plot_fluorescence_new(data['esr_data'], data['extent'], axes_list[0], rotation=0, colorbar_name = 'contrast',
                              axes_labels=['MW Frequency [MHz]', 'Position [V]'], axes_not_voltage=True, cmap='afmhot',
                              title='pta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                                  self.settings['point_a']['x'], self.settings['point_a']['y'],
                                  self.settings['point_b']['x'], self.settings['point_b']['y']),
                              colorbar_labels = [0.9, 1])
        if self._current_subscript_stage['current_subscript'] == self.scripts['esr']:
            self.scripts['esr']._plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[2]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[2], axes_list[3]], title=False)

    def _update_plot(self, axes_list):

        try:
            update_fluorescence(self.data['esr_data'], axes_list[0], colorbar_labels = [0.9, 1])
            if self._current_subscript_stage['current_subscript'] == self.scripts['esr']:
                self.scripts['esr']._update_plot([axes_list[1]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._update_plot([axes_list[2]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[2], axes_list[3]], title=False)

        except Exception as e:
            print('** ATTENTION in scanning_esr _update_plot **')
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
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(122))

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])

        return axes_list


class CountsMWAmpMod(Script):
    """
            This class runs ESR on an NV center. MW frequency is swept by sweeping the IF frequency output by the QM.
            - Ziwei Qiu 9/18/2020
        """
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM server'),
        Parameter('to_do', 'execution', ['simulation', 'execution', 'reconnection'],
                  'choose to do output simulation or real experiment'),
        Parameter('power_out', -15.0, float, 'RF power in dBm'),
        Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
        Parameter('IF_frequency', 0.0, float, 'center of the IF frequency scan'),
        Parameter('IF_amp_1', 1.0, float, 'amplitude of the IF pulse 1, between 0 and 1'),
        Parameter('IF_amp_2', 0.0, float, 'amplitude of the IF pulse 2, between 0 and 1'),
        Parameter('time_per_pt', 20000, int, 'time per point in ns, default 20us'),
        Parameter('rep_num', 20000, int, 'minimum number of esr averages'),
        Parameter('read_out',
                  [Parameter('meas_len', 19000, int, 'measurement time in ns')
                   ]),
        Parameter('simulation_duration', 10000, int, 'duration of simulation in units of ns'),
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
                print('** ATTENTION in self.qmm **')
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print('** ATTENTION in self.qmm **')
                print(e)

    def _function(self):
        if self.settings['to_do'] == 'reconnection':
            self._connect()
        else:
            try:
                self.qm = self.qmm.open_qm(config)
            except Exception as e:
                print('** ATTENTION in self.qm **')
                print(e)
            else:
                rep_num = self.settings['rep_num']
                self.meas_len = round(self.settings['read_out']['meas_len'])
                time_per_pt = round(self.settings['time_per_pt'] / 4)
                IF_amp_1 = self.settings['IF_amp_1']
                IF_amp_2 = self.settings['IF_amp_2']
                IF_amp_1 = min(IF_amp_1, 1.0)
                IF_amp_2 = min(IF_amp_2, 1.0)
                IF_amp_1 = max(IF_amp_1, 0.0)
                IF_amp_2 = max(IF_amp_2, 0.0)

                res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

                # define the qua program
                with program() as AMP_MOD:
                    update_frequency('qubit', self.settings['IF_frequency'])
                    result1 = declare(int, size=res_len)
                    result2 = declare(int, size=res_len)
                    counts1 = declare(int, value=0)
                    counts2 = declare(int, value=0)
                    total_counts = declare(int, value=0)
                    n = declare(int)

                    total_counts_st = declare_stream()
                    rep_num_st = declare_stream()

                    with for_(n, 0, n < rep_num, n + 1):
                        for IF_amp in [IF_amp_1, IF_amp_2]:
                            align('qubit', 'laser', 'readout1', 'readout2')
                            play('const' * amp(IF_amp), 'qubit', duration=time_per_pt)
                            play('trig', 'laser', duration=time_per_pt)
                            wait(90, 'readout1', 'readout2')
                            measure('readout', 'readout1', None,
                                    time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                            measure('readout', 'readout2', None,
                                    time_tagging.raw(result2, self.meas_len, targetLen=counts2))
                            assign(total_counts, counts1 + counts2)
                            save(total_counts, total_counts_st)
                            save(n, rep_num_st)

                    with stream_processing():
                        total_counts_st.buffer(2).average().save('florecence_vs_amp')
                        # total_counts_st.buffer(10000).save("current_counts")
                        rep_num_st.save("live_rep_num")

                with program() as job_stop:
                    play('trig', 'laser', duration=10)

                if self.settings['to_do'] == 'simulation':
                    self._qm_simulation(AMP_MOD)
                elif self.settings['to_do'] == 'execution':
                    # print('start execution')
                    self._qm_execution(AMP_MOD, job_stop)
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

        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['power_out']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, no trigger).')

        try:
            job = self.qm.execute(qua_program)
        except Exception as e:
            print('** ATTENTION in self.qm.execute **')
            print(e)
        else:
            vec_handle = job.result_handles.get("florecence_vs_amp")
            progress_handle = job.result_handles.get("live_rep_num")
            # tracking_handle = job.result_handles.get("current_counts")

            vec_handle.wait_for_values(1)
            progress_handle.wait_for_values(1)
            # tracking_handle.wait_for_values(1)
            self.data = {'f': self.settings['IF_frequency'] + self.settings['mw_frequency'],
                         'power': self.settings['power_out'],
                         'cnts': None, 'sig_norm':None,
                         'IF_amp_1': self.settings['IF_amp_1'], 'IF_amp_2': self.settings['IF_amp_2']}

            while vec_handle.is_processing():
                try:
                    vec = vec_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in vec_handle **')
                    print(e)
                else:

                    cnts = vec * 1e6 / self.meas_len
                    self.data.update({'cnts': cnts, 'sig_norm':cnts[0]/cnts[1]})

                try:
                    current_rep_num = progress_handle.fetch_all()
                except Exception as e:
                    print('** ATTENTION in progress_handle **')
                    print(e)
                else:
                    self.data['rep_num'] = float(current_rep_num)
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
        super(CountsMWAmpMod, self).plot([figure_list[0], figure_list[1]])

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

        if 'cnts' in data.keys():
            try:
                axes_list[0].clear()
                axes_list[0].scatter(data['IF_amp_1'], data['cnts'][0], label="cnts 1")
                axes_list[0].scatter(data['IF_amp_2'], data['cnts'][1], label="cnts 2")
                axes_list[0].set_xlabel('IF amplitude')
                axes_list[0].set_ylabel('kcounts/sec')
                axes_list[0].legend(loc='upper right')
                if title:
                    axes_list[0].set_title(
                        'MW amplitude modulation (Repetition number: {:d})\nRF power: {:0.1f}dBm, LO freq: {:0.2f}GHz, IF freq: {:0.2f}MHz'.format(
                            int(data['rep_num']), self.settings['power_out'],
                            self.settings['mw_frequency'] * 1e-9, self.settings['IF_frequency'] * 1e-6))
                else:
                    axes_list[0].set_title(
                        '{:0.1f}dBm, LO: {:0.2f}GHz, IF: {:0.2f}MHz'.format(
                            self.settings['power_out'],
                            self.settings['mw_frequency'] * 1e-9,
                            self.settings['IF_frequency'] * 1e-6))
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


class ScanningCountsMW(Script):
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'initial x-coordinate [V]'),
                   Parameter('y', 0, float, 'initial y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0, float, 'last x-coordinate [V]'),
                   Parameter('y', 1, float, 'last y-coordinate [V]')
                   ]),
        Parameter('num_points', 50, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 50, int, 'do the optimization once per N points')]),
        Parameter('do_afm1d',
                  [Parameter('before', False, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int,
                             'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool,
                             'whether to scan back to the original point after the experiment is done'),
                   ]),
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'cntsmw': CountsMWAmpMod, 'set_scanner': SetScannerXY_gentle, 'afm1d': AFM1D_qm,
                'afm1d_before_after': AFM1D_qm, 'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

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

    def do_afm1d_after(self, verbose = True):
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

    def _function(self):
        scan_pos_1d, dist_array = self._get_scan_array()

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

        self.data = {'scan_pos_1d': scan_pos_1d, 'afm_dist_array': np.array([]), 'afm_ctr': np.array([]),
                     'afm_analog': np.array([]), 'cnts1': np.array([]), 'cnts2': np.array([]), 'sig_norm': np.array([]),
                     'dist_array': np.array([])}

        if self.settings['do_afm1d']['before'] and not self._abort:
            self.do_afm1d_before()

        if not self._abort and self.settings['do_optimize']['on']:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_0'
            self.scripts['optimize'].run()

        if not self._abort:
            self.scripts['cntsmw'].settings['tag'] = 'cntsmw_0'
            self.scripts['cntsmw'].run()
            cnts1, cnts2 = self.scripts['cntsmw'].data['cnts']
            self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
            self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
            sig_norm = self.scripts['cntsmw'].data['sig_norm']
            self.data['sig_norm'] = np.concatenate((self.data['sig_norm'], np.array([sig_norm])), axis=0)
            self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                break

            afm_start = scan_pos_1d[i]
            afm_end = scan_pos_1d[i + 1]
            self.setup_afm(afm_start, afm_end)

            # move to the new pt
            try:
                self.scripts['afm1d'].run()
                afm_dist_array = self.scripts['afm1d'].data['dist_array'] + dist_array[i]
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.data['afm_dist_array'] = np.concatenate((self.data['afm_dist_array'], afm_dist_array))
                afm_ctr = self.scripts['afm1d'].data['data_ctr'][0]
                self.data['afm_ctr'] = np.concatenate((self.data['afm_ctr'], afm_ctr))
                afm_analog = self.scripts['afm1d'].data['data_analog'][0]
                self.data['afm_analog'] = np.concatenate((self.data['afm_analog'], afm_analog))

            if self.settings['do_optimize']['on'] and (i + 1) % self.settings['do_optimize'][
                'frequency'] == 0 and not self._abort:
                self.scripts['optimize'].settings['tag'] = 'optimize_'+str(i+1)
                self.scripts['optimize'].run()

            # run cnts_mw
            try:
                self.scripts['cntsmw'].settings['tag'] = 'cntsmw_'+str(i+1)
                self.scripts['cntsmw'].run()
                cnts1, cnts2 = self.scripts['cntsmw'].data['cnts']
                self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
                self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
                sig_norm = self.scripts['cntsmw'].data['sig_norm']
                self.data['sig_norm'] = np.concatenate((self.data['sig_norm'], np.array([sig_norm])), axis=0)
                self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([dist_array[i + 1]])))
            except Exception as e:
                print('** ATTENTION in cnts_mw **')
                print(e)


        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()
        axes_list[4].clear()
        axes_list[5].clear()

        try:
            if 'afm_dist_array' in data.keys() and len(data['afm_dist_array']) > 0:
                axes_list[0].plot(data['afm_dist_array'], data['afm_ctr'])
                axes_list[1].plot(data['afm_dist_array'], data['afm_analog'])
            else:
                axes_list[0].plot(np.zeros([10]), np.zeros([10]))
                axes_list[1].plot(np.zeros([10]), np.zeros([10]))

            if 'cnts1' in data.keys() and len(data['dist_array']) > 0:
                axes_list[2].plot(data['dist_array'], data['cnts1'], label="cnts1")
                axes_list[2].plot(data['dist_array'], data['cnts2'], label="cnts2")
                axes_list[3].plot(data['dist_array'], data['sig_norm'])

            else:
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="cnts1")
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="cnts2")
                axes_list[3].plot(np.zeros([10]), np.zeros([10]))
        except Exception as e:
            print('** ATTENTION in _plot 1d **')
            print(e)

        axes_list[0].set_ylabel('Counts [kcps]')
        axes_list[1].set_ylabel('Z_out [V]')
        axes_list[2].set_ylabel('Counts [kcps]')
        axes_list[3].set_ylabel('Norm Sig')

        axes_list[0].xaxis.set_ticklabels([])
        axes_list[1].xaxis.set_ticklabels([])
        axes_list[2].xaxis.set_ticklabels([])

        axes_list[3].set_xlabel('Position [V]')
        axes_list[2].legend(loc='upper right')

        axes_list[0].set_title(
            'MW amplitude modulation (Repetition number: {:d})\nRF power: {:0.1f}dBm, LO freq: {:0.2f}GHz, IF freq: {:0.2f}MHz'.format(
                int(self.scripts['cntsmw'].settings['rep_num']), self.scripts['cntsmw'].settings['power_out'],
                self.scripts['cntsmw'].settings['mw_frequency'] * 1e-9, self.scripts['cntsmw'].settings['IF_frequency'] * 1e-6))

        if self._current_subscript_stage['current_subscript'] == self.scripts['cntsmw']:
            self.scripts['cntsmw']._plot([axes_list[4]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[5]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[4], axes_list[5]], title=False)

    def _update_plot(self, axes_list):

        try:
            if self._current_subscript_stage['current_subscript'] == self.scripts['cntsmw']:
                self.scripts['cntsmw']._update_plot([axes_list[4]], title=False)
            elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._plot([axes_list[5]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[4], axes_list[5]], title=False)
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
                if 'cnts1' in self.data.keys() and len(self.data['dist_array']) > 0:
                    axes_list[2].lines[0].set_xdata(self.data['dist_array'])
                    axes_list[2].lines[0].set_ydata(self.data['cnts1'])
                    axes_list[2].lines[1].set_xdata(self.data['dist_array'])
                    axes_list[2].lines[1].set_ydata(self.data['cnts2'])
                    axes_list[2].relim()
                    axes_list[2].autoscale_view()
                    axes_list[3].lines[0].set_xdata(self.data['dist_array'])
                    axes_list[3].lines[0].set_ydata(self.data['sig_norm'])
                    axes_list[3].relim()
                    axes_list[3].autoscale_view()


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
            axes_list.append(figure_list[0].add_subplot(411))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(412))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(413))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(414))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


class ScanningCountsMW2D(Script):
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
                             [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size',
                  [Parameter('axis1', 1.0, float, 'inner loop [V]'),
                   Parameter('axis2', 1.0, float, 'outer loop [V]')
                   ]),
        Parameter('num_points',
                  [Parameter('axis1', 100, int, 'number of points to scan in each line (inner loop)'),
                   Parameter('axis2', 20, int, 'number of lines to scan (outer loop)'),
                   ]),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'scanning_counts_mw': ScanningCountsMW, 'set_scanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

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

    def _function(self):
        self._get_scan_array()

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:

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
                    self.scripts['scanning_counts_mw'].settings['tag'] = 'sensing1d_ind' + str(i)
                    self.scripts['scanning_counts_mw'].settings['scan_speed'] = self.settings['scan_speed']
                    self.scripts['scanning_counts_mw'].settings['point_a']['x'] = self.scan_pos_1d_ac[i][0]
                    self.scripts['scanning_counts_mw'].settings['point_a']['y'] = self.scan_pos_1d_ac[i][1]
                    self.scripts['scanning_counts_mw'].settings['point_b']['x'] = self.scan_pos_1d_bd[i][0]
                    self.scripts['scanning_counts_mw'].settings['point_b']['y'] = self.scan_pos_1d_bd[i][1]
                    self.scripts['scanning_counts_mw'].settings['num_points'] = self.settings['num_points']['axis1']
                    # we always need to scan back in each line
                    self.scripts['scanning_counts_mw'].settings['do_afm1d']['after'] = True
                    self.scripts['scanning_counts_mw'].run()
                except Exception as e:
                    print('** ATTENTION in scanning_counts_mw 1d **')
                    print(e)

    def _plot(self, axes_list, data=None):

        if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_counts_mw']:
            self.scripts['scanning_counts_mw']._plot(axes_list)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_counts_mw']:

            if self.flag_1d_plot:
                self.scripts['scanning_counts_mw']._plot(axes_list)
                self.flag_1d_plot = False
            else:
                self.scripts['scanning_counts_mw']._update_plot(axes_list)

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
            axes_list.append(figure_list[0].add_subplot(411))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(412))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(413))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(414))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


class ScanMWMod(Script):
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'initial x-coordinate [V]'),
                   Parameter('y', 0, float, 'initial y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0, float, 'last x-coordinate [V]'),
                   Parameter('y', 1, float, 'last y-coordinate [V]')
                   ]),
        Parameter('num_points', 50, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 50, int, 'do the optimization once per N points')]),
        Parameter('do_afm1d',
                  [Parameter('before', False, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int,
                             'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool,
                             'whether to scan back to the original point after the experiment is done'),
                   ]),
        Parameter('MW_mod',
                  [Parameter('power_out', -15.0, float, 'RF power in dBm'),
                   Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
                   Parameter('IF_frequency', 0.0, float, 'center of the IF frequency scan'),
                   Parameter('IF_amp_1', 1.0, float, 'amplitude of the IF pulse 1, between 0 and 1'),
                   Parameter('IF_amp_2', 0.0, float, 'amplitude of the IF pulse 2, between 0 and 1'),
                   Parameter('time_per_pt', 20000, int, 'time per point in ns, default 20us'),
                   Parameter('avg_num', 5000, int, 'number of esr averages'),
                   Parameter('meas_len', 19000, int, 'measurement time in ns'),
                   ]),
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource}
    _SCRIPTS = {'set_scanner': SetScannerXY_gentle, 'afm1d_before_after': AFM1D_qm, 'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.qmm = QuantumMachinesManager()

    def _get_scan_array(self):

        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        N = self.settings['num_points']
        scan_pos_1d = np.linspace(Vstart, Vend, N, endpoint=True)
        dist_array = np.linspace(0, dist, N, endpoint=True)

        return scan_pos_1d, dist_array

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

    def do_afm1d_after(self, verbose = True):
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

    def _function(self):
        scan_pos_1d, dist_array = self._get_scan_array()

        self.data = {'scan_pos_1d': scan_pos_1d, 'cnts1': np.array([]), 'cnts2': np.array([]), 'sig_norm': np.array([]),
                     'dist_array': np.array([])}

        # Move to the initial scanning position
        self.scripts['set_scanner'].update({'to_do': 'set'})
        self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['set_scanner'].update({'step_size': 0.0001})
        self.scripts['set_scanner'].update({'verbose': False})
        self.scripts['set_scanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['set_scanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['set_scanner'].run()

        if self.settings['do_afm1d']['before'] and not self._abort:
            self.do_afm1d_before()

        if not self._abort and self.settings['do_optimize']['on']:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_0'
            self.scripts['optimize'].run()

        # Turn on MWs
        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['MW_mod']['power_out']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['MW_mod']['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, no trigger).')

        rep_num = self.settings['MW_mod']['avg_num']
        self.meas_len = round(self.settings['MW_mod']['meas_len'])
        time_per_pt = round(self.settings['MW_mod']['time_per_pt'] / 4)
        IF_amp_1 = self.settings['MW_mod']['IF_amp_1']
        IF_amp_2 = self.settings['MW_mod']['IF_amp_2']
        IF_amp_1 = min(IF_amp_1, 1.0)
        IF_amp_2 = min(IF_amp_2, 1.0)
        IF_amp_1 = max(IF_amp_1, 0.0)
        IF_amp_2 = max(IF_amp_2, 0.0)

        res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

        # define the qua programs
        with program() as job_stop:
            play('trig', 'laser', duration=10)

        with program() as AMP_MOD:
            update_frequency('qubit', self.settings['MW_mod']['IF_frequency'])
            result1 = declare(int, size=res_len)
            result2 = declare(int, size=res_len)
            counts1 = declare(int, value=0)
            counts2 = declare(int, value=0)
            total_counts = declare(int, value=0)
            n = declare(int)
            total_counts_st = declare_stream()

            with infinite_loop_():
                with for_(n, 0, n < rep_num, n + 1):
                    for IF_amp in [IF_amp_1, IF_amp_2]:
                        align('qubit', 'laser', 'readout1', 'readout2')
                        play('const' * amp(IF_amp), 'qubit', duration=time_per_pt)
                        play('trig', 'laser', duration=time_per_pt)
                        # wait(90, 'readout1', 'readout2')
                        measure('readout', 'readout1', None,
                                time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                        measure('readout', 'readout2', None,
                                time_tagging.raw(result2, self.meas_len, targetLen=counts2))
                        assign(total_counts, counts1 + counts2)
                        save(total_counts, total_counts_st)
                pause()

            with stream_processing():
                total_counts_st.buffer(2*rep_num).save("data")

        self.qm = self.qmm.open_qm(config)
        job = self.qm.execute(AMP_MOD)

        vec_handle = job.result_handles.get("data")
        vec_handle.wait_for_values(1)

        while job.is_paused() is not True:
            time.sleep(0.05)
        try:
            vec = vec_handle.fetch_all()
        except Exception as e:
            print('** ATTENTION in fetching data **')
            print(e)
        else:
            cnts = vec * 1e6 / self.meas_len
            cnts = cnts.reshape([rep_num,2])
            cnts1, cnts2 = cnts[:, 0].mean(), cnts[:, 1].mean()
            self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
            self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
            self.data['sig_norm'] = np.concatenate((self.data['sig_norm'], np.array([cnts1/cnts2])), axis=0)
            self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                self.qm.execute(job_stop)
                break

            # move to the new pt
            self.scripts['set_scanner'].settings['point']['x'] = scan_pos_1d[i + 1][0]
            self.scripts['set_scanner'].settings['point']['y'] = scan_pos_1d[i + 1][1]
            self.scripts['set_scanner'].run()

            # # do optimize (disabled because QUA program needs to be resumed)
            # if self.settings['do_optimize']['on'] and (i + 1) % self.settings['do_optimize'][
            #     'frequency'] == 0 and not self._abort:
            #     self.scripts['optimize'].settings['tag'] = 'optimize_'+str(i+1)
            #     self.scripts['optimize'].run()

            # run qua program
            job.resume()
            while job.is_paused() is not True:
                time.sleep(0.05)
            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION in fetching data **')
                print(e)
            else:
                cnts = vec * 1e6 / self.meas_len
                cnts = cnts.reshape([rep_num, 2])
                cnts1, cnts2 = cnts[:, 0].mean(), cnts[:, 1].mean()
                self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
                self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
                self.data['sig_norm'] = np.concatenate((self.data['sig_norm'], np.array([cnts1 / cnts2])), axis=0)
                self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([dist_array[i + 1]])))

        # turn off MWs
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')

        # do AFM 1d after scan
        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()

        try:
            if 'cnts1' in data.keys() and len(data['dist_array']) > 0:
                axes_list[0].plot(data['dist_array'], data['cnts1'], label="IF amp = {:0.1f}".format(self.settings['MW_mod']['IF_amp_1']))
                axes_list[0].plot(data['dist_array'], data['cnts2'], label="IF amp = {:0.1f}".format(self.settings['MW_mod']['IF_amp_2']))
                axes_list[1].plot(data['dist_array'], data['sig_norm'])
            else:
                axes_list[0].plot(np.zeros([10]), np.zeros([10]), label="IF amp = {:0.1f}".format(self.settings['MW_mod']['IF_amp_1']))
                axes_list[0].plot(np.zeros([10]), np.zeros([10]), label="IF amp = {:0.1f}".format(self.settings['MW_mod']['IF_amp_2']))
                axes_list[1].plot(np.zeros([10]), np.zeros([10]))
        except Exception as e:
            print('** ATTENTION in _plot 1d **')
            print(e)

        axes_list[0].set_ylabel('Counts [kcps]')
        axes_list[1].set_ylabel('Norm Sig')

        axes_list[0].xaxis.set_ticklabels([])

        axes_list[1].set_xlabel('Position [V]')
        axes_list[0].legend(loc='upper right')

        axes_list[0].set_title(
            'MW amplitude modulation (Repetition number: {:d})\nRF power: {:0.1f}dBm, LO freq: {:0.3f}GHz, IF freq: {:0.3f}MHz\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                int(self.settings['MW_mod']['avg_num']), self.settings['MW_mod']['power_out'],
                self.settings['MW_mod']['mw_frequency'] * 1e-9, self.settings['MW_mod']['IF_frequency'] * 1e-6,
                self.settings['point_a']['x'], self.settings['point_a']['y'],
                self.settings['point_b']['x'], self.settings['point_b']['y']
            ), fontsize=9.5)

        if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[2]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[2], axes_list[3]], title=False)

    def _update_plot(self, axes_list):
        try:
            if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._plot([axes_list[2]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[2], axes_list[3]], title=False)
            elif 'cnts1' in self.data.keys() and len(self.data['dist_array']) > 0:
                axes_list[0].lines[0].set_xdata(self.data['dist_array'])
                axes_list[0].lines[0].set_ydata(self.data['cnts1'])
                axes_list[0].lines[1].set_xdata(self.data['dist_array'])
                axes_list[0].lines[1].set_ydata(self.data['cnts2'])
                axes_list[0].relim()
                axes_list[0].autoscale_view()
                axes_list[1].lines[0].set_xdata(self.data['dist_array'])
                axes_list[1].lines[0].set_ydata(self.data['sig_norm'])
                axes_list[1].relim()
                axes_list[1].autoscale_view()

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
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


class ScanMWMod2D(Script):
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
                             [Parameter('x', 1.0, float, 'x-coordinate [V]'),
                              Parameter('y', 0.0, float, 'y-coordinate [V]')
                              ]),
                   Parameter('type', 'parallel', ['perpendicular', 'parallel'],
                             'scan direction perpendicular or parallel to the pt1pt2 line')
                   ]),
        Parameter('scan_size',
                  [Parameter('axis1', 1.0, float, 'inner loop [V]'),
                   Parameter('axis2', 1.0, float, 'outer loop [V]')
                   ]),
        Parameter('num_points',
                  [Parameter('axis1', 100, int, 'number of points to scan in each line (inner loop)'),
                   Parameter('axis2', 20, int, 'number of lines to scan (outer loop)'),
                   ]),
        Parameter('mw_power_change',
                  [Parameter('on', False, bool, 'choose whether to change the MW power during scan'),
                   Parameter('start_power', -20, float, 'choose the MW power at the first point'),
                   Parameter('end_power', -20, float, 'choose the MW power at the last point'),
                   ]),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 15, int, 'do the optimization once per N points')]),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),

    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'scanning_counts_mw': ScanMWMod, 'set_scanner': SetScannerXY_gentle, 'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

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

    def _function(self):
        self._get_scan_array()

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:

            if self.settings['mw_power_change']['on']:
                mw_power_arr = np.linspace(self.settings['mw_power_change']['start_power'],
                                           self.settings['mw_power_change']['end_power'],
                                           self.settings['num_points']['axis2'], endpoint=True)
            else:
                mw_power_arr = np.linspace(self.scripts['scanning_counts_mw'].settings['MW_mod']['power_out'],
                                           self.scripts['scanning_counts_mw'].settings['MW_mod']['power_out'],
                                           self.settings['num_points']['axis2'], endpoint=True)

            # Move to the initial scanning position
            self.scripts['set_scanner'].update({'to_do': 'set'})
            self.scripts['set_scanner'].update({'verbose': False})
            self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['set_scanner'].update({'step_size': 0.0001})

            x2 = np.sqrt((self.ptb[0] - self.pta[0]) ** 2 + (self.ptb[1] - self.pta[1]) ** 2)
            y2 = np.sqrt((self.ptd[0] - self.ptc[0]) ** 2 + (self.ptd[1] - self.ptc[1]) ** 2)
            self.data = {'scan_center': self.scan_center, 'scan_size_1': self.settings['scan_size']['axis1'],
                         'scan_size_2': self.settings['scan_size']['axis2'], 'vector_x': self.vector_x,
                         'contrast_data': 0.9*np.ones((self.settings['num_points']['axis2'],
                                                 self.settings['num_points']['axis1'])),
                         'extent': np.array([0, x2, y2, 0]),
                         'region': np.array([self.pta, self.ptb, self.ptc, self.ptd])}

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

                # do optimize
                if self.settings['do_optimize']['on'] and i % self.settings['do_optimize'][
                    'frequency'] == 0 and not self._abort:
                    self.scripts['optimize'].settings['tag'] = 'optimize_'+str(i)
                    self.scripts['optimize'].run()

                try:
                    self.flag_1d_plot = True
                    self.scripts['scanning_counts_mw'].settings['tag'] = 'sensing1d_ind' + str(i)
                    self.scripts['scanning_counts_mw'].settings['scan_speed'] = self.settings['scan_speed']
                    self.scripts['scanning_counts_mw'].settings['point_a']['x'] = float(self.scan_pos_1d_ac[i][0])
                    self.scripts['scanning_counts_mw'].settings['point_a']['y'] = float(self.scan_pos_1d_ac[i][1])
                    self.scripts['scanning_counts_mw'].settings['point_b']['x'] = float(self.scan_pos_1d_bd[i][0])
                    self.scripts['scanning_counts_mw'].settings['point_b']['y'] = float(self.scan_pos_1d_bd[i][1])
                    self.scripts['scanning_counts_mw'].settings['num_points'] = self.settings['num_points']['axis1']
                    self.scripts['scanning_counts_mw'].settings['MW_mod']['power_out'] = float(mw_power_arr[i])
                    # we always need to scan back in each line
                    self.scripts['scanning_counts_mw'].settings['do_afm1d']['after'] = True
                    self.scripts['scanning_counts_mw'].run()
                    self.data['contrast_data'][i]= self.scripts['scanning_counts_mw'].data['sig_norm']
                except Exception as e:
                    print('** ATTENTION in scanning_counts_mw 1d **')
                    print(e)

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()
        axes_list[4].clear()
        axes_list[5].clear()
        freq_MHz = (self.scripts['scanning_counts_mw'].settings['MW_mod']['mw_frequency'] +
                    self.scripts['scanning_counts_mw'].settings['MW_mod']['IF_frequency']) / 1E6
        power = self.scripts['scanning_counts_mw'].settings['MW_mod']['power_out']
        plot_fluorescence_new(data['contrast_data'], data['extent'], axes_list[0], rotation=0, colorbar_name='contrast',
                              axes_labels=['Y [V]', 'X [V]'], axes_not_voltage=True, cmap='afmhot',
                              title='2D Scan\nMW: {:0.3f}MHz, {:0.1f}dBm'.format(freq_MHz, power),
                              colorbar_labels = [0.9, 1]
                              )
        if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_counts_mw']:
            self.scripts['scanning_counts_mw']._plot(axes_list[2:])

    def _update_plot(self, axes_list):
        try:
            update_fluorescence(self.data['contrast_data'], axes_list[0], colorbar_labels = [0.9, 1])
            if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_counts_mw']:

                if self.flag_1d_plot:
                    self.scripts['scanning_counts_mw']._plot(axes_list[2:])
                    self.flag_1d_plot = False
                else:
                    self.scripts['scanning_counts_mw']._update_plot(axes_list[2:])
        except Exception as e:
            print('** ATTENTION in scanning_esr _update_plot **')
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
            axes_list.append(figure_list[0].add_subplot(221))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(222))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


class ScanACStark(Script):
    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x', 0, float, 'initial x-coordinate [V]'),
                   Parameter('y', 0, float, 'initial y-coordinate [V]')
                   ]),
        Parameter('point_b',
                  [Parameter('x', 0, float, 'last x-coordinate [V]'),
                   Parameter('y', 1.5, float, 'last y-coordinate [V]')
                   ]),
        Parameter('num_points', 50, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 50, int, 'do the optimization once per N points')]),
        Parameter('do_afm1d',
                  [Parameter('before', False, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int,
                             'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool,
                             'whether to scan back to the original point after the experiment is done'),
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
        Parameter('mw_pulses_2', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -10.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
        ]),
        Parameter('tau', 1000, int, 'total time between the two pi/2 pulses (in ns)'),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('sensing_type', 'both', ['cosine', 'sine', 'both'], 'choose the sensing type'),
        Parameter('read_out', [
            Parameter('meas_len', 180, int, 'measurement time in ns'),
            Parameter('nv_reset_time', 2000, int, 'laser on time in ns'),
            Parameter('delay_readout', 370, int,
                      'delay between laser on and APD readout (given by spontaneous decay rate) in ns'),
            Parameter('laser_off', 500, int, 'laser off time in ns before applying RF'),
            Parameter('delay_mw_readout', 600, int, 'delay between mw off and laser on')
        ]),
        Parameter('rep_num', 200000, int, 'define the repetition number')

    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource, 'mw_gen_iq_2':AgilentN9310A}
    _SCRIPTS = {'set_scanner': SetScannerXY_gentle, 'afm1d_before_after': AFM1D_qm, 'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.qmm = QuantumMachinesManager()

    def _get_scan_array(self):

        Vstart = np.array([self.settings['point_a']['x'], self.settings['point_a']['y']])
        Vend = np.array([self.settings['point_b']['x'], self.settings['point_b']['y']])
        dist = np.linalg.norm(Vend - Vstart)
        N = self.settings['num_points']
        scan_pos_1d = np.linspace(Vstart, Vend, N, endpoint=True)
        dist_array = np.linspace(0, dist, N, endpoint=True)
        return scan_pos_1d, dist_array

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

    def do_afm1d_after(self, verbose = True):
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

    def _function(self):
        scan_pos_1d, dist_array = self._get_scan_array()

        self.data = {'scan_pos_1d': scan_pos_1d, 'cnts1': np.array([]), 'cnts2': np.array([]), 'cnts3': np.array([]),
                     'cnts4': np.array([]),  'cos_sig': np.array([]), 'sin_sig': np.array([]),
                     'coherence': np.array([]), 'phase': np.array([]), 'dist_array': np.array([])}

        # Move to the initial scanning position
        self.scripts['set_scanner'].update({'to_do': 'set'})
        self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
        self.scripts['set_scanner'].update({'step_size': 0.0001})
        self.scripts['set_scanner'].update({'verbose': False})
        self.scripts['set_scanner'].settings['point']['x'] = self.settings['point_a']['x']
        self.scripts['set_scanner'].settings['point']['y'] = self.settings['point_a']['y']
        self.scripts['set_scanner'].run()

        if self.settings['do_afm1d']['before'] and not self._abort:
            self.do_afm1d_before()

        if not self._abort and self.settings['do_optimize']['on']:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_0'
            self.scripts['optimize'].run()


        # Turn on MWs
        self.instruments['mw_gen_iq']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen_iq']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': True})
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': True})
        print('Turned on RF generator SGS100A (IQ on, trigger on).')

        self.instruments['mw_gen_iq_2']['instance'].update({'enable_IQ': True})
        self.instruments['mw_gen_iq_2']['instance'].update({'enable_modulation': True})
        self.instruments['mw_gen_iq_2']['instance'].update({'freq_mode': 'CW'})
        self.instruments['mw_gen_iq_2']['instance'].update({'power_mode': 'CW'})
        self.instruments['mw_gen_iq_2']['instance'].update({'amplitude': self.settings['mw_pulses_2']['mw_power']})
        self.instruments['mw_gen_iq_2']['instance'].update({'frequency': self.settings['mw_pulses_2']['mw_frequency']})
        self.instruments['mw_gen_iq_2']['instance'].update({'enable_output': True})
        print('Turned on RF generator N9310A (IQ on).')

        # define the qua programs
        # unit: cycle of 4ns
        pi_time = round(self.settings['mw_pulses']['pi_pulse_time'] / 4)
        pi2_time = round(self.settings['mw_pulses']['pi_half_pulse_time'] / 4)
        pi32_time = round(self.settings['mw_pulses']['3pi_half_pulse_time'] / 4)

        # unit: ns
        config['pulses']['pi_pulse']['length'] = int(pi_time * 4)
        config['pulses']['pi2_pulse']['length'] = int(pi2_time * 4)
        config['pulses']['pi32_pulse']['length'] = int(pi32_time * 4)

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

        IF_amp2 = self.settings['mw_pulses_2']['IF_amp']
        if IF_amp2 > 1.0:
            IF_amp2 = 1.0
        elif IF_amp2 < 0.0:
            IF_amp2 = 0.0

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
        self.data['tau']= float(self.tau_total)
        res_len = int(np.max([round(self.meas_len / 200), 2]))  # result length needs to be at least 2

        IF_freq = self.settings['mw_pulses']['IF_frequency']
        IF_freq2 = self.settings['mw_pulses_2']['IF_frequency']

        def xy4_block(is_last_block, IF_amp=IF_amp, efield=False):
            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('trig', 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                play('const' * amp(IF_amp2), 'qubit2', duration=t)
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(-np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            if is_last_block:
                wait(t, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                    wait(t, 'qubit2')  # e field 2 off
            else:
                wait(2 * t + pi2_time, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                    wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

        def xy8_block(is_last_block, IF_amp=IF_amp, efield=False):

            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

            z_rot(-np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(-np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

            z_rot(np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_y
            wait(2 * t + pi2_time, 'qubit')
            if efield:
                wait(pi_time, 'qubit2')  # e field 1, 2 off
                play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                # wait(2 * t + pi2_time, 'e_field1')  # e field 1 off

            z_rot(-np.pi / 2, 'qubit')
            play('pi' * amp(IF_amp), 'qubit')  # pi_x
            if is_last_block:
                wait(t, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    # play('trig', 'e_field1', duration=t)  # e field 1 on
                    wait(t, 'qubit2')  # e field 2 off
            else:
                wait(2 * t + pi2_time, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    # play('trig', 'e_field1', duration=2 * t + pi2_time)  # e field 1 on
                    wait(2 * t + pi2_time, 'qubit2')  # e field 2 off

        def spin_echo_block(is_last_block, IF_amp=IF_amp, efield=False, on='e_field2', off='e_field1'):
            play('pi' * amp(IF_amp), 'qubit')
            if is_last_block:
                wait(t, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    if on == 'e_field2':
                        play('const' * amp(IF_amp2), 'qubit2', duration=t)  # e field 2 on
                    else:
                        wait(t, 'qubit2')  # e field  off
            else:
                wait(2 * t + pi2_time, 'qubit')
                if efield:
                    wait(pi_time, 'qubit2')  # e field 1, 2 off
                    if on == 'e_field2':
                        play('const' * amp(IF_amp2), 'qubit2', duration=2 * t + pi2_time)  # e field 2 on
                    else:
                        wait(2 * t + pi2_time, 'qubit2')  # e field  off

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
        with program() as acstark:

            update_frequency('qubit', IF_freq)
            update_frequency('qubit2', IF_freq2)

            result1 = declare(int, size=res_len)
            counts1 = declare(int, value=0)
            result2 = declare(int, size=res_len)
            counts2 = declare(int, value=0)
            total_counts = declare(int, value=0)
            # t = declare(int)
            n = declare(int)
            k = declare(int)
            i = declare(int)

            total_counts_st = declare_stream()
            # rep_num_st = declare_stream()

            with infinite_loop_():
                with for_(n, 0, n < rep_num, n + 1):
                    # note that by having only 4 k values, I can do at most XY8-2 seq.
                    # by having 6 k values, I can do at most XY8 seq.
                    # the plot needs to be changed accordingly.
                    with for_(k, 0, k < 4, k + 1):

                        reset_frame('qubit', 'qubit2')

                        with if_(k == 0):  # +x readout, no E field
                            align('qubit', 'qubit2')
                            play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(np.pi / 2, 'qubit')

                            wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'qubit2')  # e field 2 off

                            if self.settings['sensing_type'] == 'both':
                                pi_pulse_train(efield=True)
                            else:
                                pi_pulse_train()

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(-np.pi / 2, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                        with if_(k == 1):  # -x readout, no E field
                            align('qubit', 'qubit2')
                            play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(np.pi / 2, 'qubit')

                            wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'qubit2')  # e field 2 off

                            if self.settings['sensing_type'] == 'both':
                                pi_pulse_train(efield=True)
                            else:
                                pi_pulse_train()

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(-np.pi / 2, 'qubit')

                            z_rot(np.pi, 'qubit')
                            play('pi2' * amp(IF_amp), 'qubit')
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                        with if_(k == 2):  # +x readout, with E field
                            align('qubit', 'qubit2')
                            play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(np.pi / 2, 'qubit')

                            wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'qubit2')  # e field 2 off

                            pi_pulse_train(efield=True)

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(-np.pi / 2, 'qubit')

                            if self.settings['sensing_type'] != 'cosine':
                                z_rot(np.pi / 2, 'qubit')

                            play('pi2' * amp(IF_amp), 'qubit')
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                        with if_(k == 3):  # -x readout, with E field
                            align('qubit', 'qubit2')
                            play('pi2' * amp(IF_amp), 'qubit')  # pi/2 x
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(np.pi / 2, 'qubit')

                            wait(t, 'qubit')  # for the first pulse, only wait half of the evolution block time
                            # play('trig', 'e_field1', duration=t)  # e field 1 on
                            wait(t, 'qubit2')  # e field 2 off
                            pi_pulse_train(efield=True)

                            if self.settings['decoupling_seq']['type'] == 'CPMG':
                                z_rot(-np.pi / 2, 'qubit')

                            z_rot(np.pi, 'qubit')

                            if self.settings['sensing_type'] != 'cosine':
                                z_rot(np.pi / 2, 'qubit')

                            play('pi2' * amp(IF_amp), 'qubit')
                            # wait(pi2_time, 'e_field1')  # e field 1 off
                            wait(pi2_time, 'qubit2')  # e field 2 off

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

                pause()

            with stream_processing():
                total_counts_st.buffer(4*rep_num).save("data")

        with program() as job_stop:
            play('trig', 'laser', duration=10)

        self.qm = self.qmm.open_qm(config)
        job = self.qm.execute(acstark)

        vec_handle = job.result_handles.get("data")
        vec_handle.wait_for_values(1)

        while job.is_paused() is not True:
            time.sleep(0.05)
        try:
            vec = vec_handle.fetch_all()
        except Exception as e:
            print('** ATTENTION in fetching data **')
            print(e)
        else:
            cnts = vec * 1e6 / self.meas_len
            cnts = cnts.reshape([rep_num,4])
            cnts1, cnts2, cnts3, cnts4 = cnts[:, 0].mean(), cnts[:, 1].mean(), cnts[:, 2].mean(), cnts[:, 3].mean()
            self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
            self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
            self.data['cnts3'] = np.concatenate((self.data['cnts3'], np.array([cnts3])), axis=0)
            self.data['cnts4'] = np.concatenate((self.data['cnts4'], np.array([cnts4])), axis=0)
            cos_sig = np.array([2 * (cnts2 - cnts1) / (cnts2 + cnts1)])
            sin_sig = np.array([2 * (cnts4 - cnts3) / (cnts4 + cnts3)])
            coherence = np.sqrt(cos_sig ** 2 + sin_sig ** 2)
            phase = np.arccos(cos_sig / coherence) * np.sign(sin_sig)
            self.data['cos_sig'] = np.concatenate((self.data['cos_sig'], cos_sig), axis=0)
            self.data['sin_sig'] = np.concatenate((self.data['sin_sig'], sin_sig), axis=0)
            self.data['coherence'] = np.concatenate((self.data['coherence'], coherence), axis=0)
            self.data['phase'] = np.concatenate((self.data['phase'], phase), axis=0)
            self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                self.qm.execute(job_stop)
                break

            # move to the new pt
            self.scripts['set_scanner'].settings['point']['x'] = scan_pos_1d[i + 1][0]
            self.scripts['set_scanner'].settings['point']['y'] = scan_pos_1d[i + 1][1]
            self.scripts['set_scanner'].run()

            # run qua program
            job.resume()
            while job.is_paused() is not True:
                time.sleep(0.05)
            try:
                vec = vec_handle.fetch_all()
            except Exception as e:
                print('** ATTENTION in fetching data **')
                print(e)
            else:
                cnts = vec * 1e6 / self.meas_len
                cnts = cnts.reshape([rep_num, 4])
                cnts1, cnts2, cnts3, cnts4 = cnts[:, 0].mean(), cnts[:, 1].mean(), cnts[:, 2].mean(), cnts[:, 3].mean()
                self.data['cnts1'] = np.concatenate((self.data['cnts1'], np.array([cnts1])), axis=0)
                self.data['cnts2'] = np.concatenate((self.data['cnts2'], np.array([cnts2])), axis=0)
                self.data['cnts3'] = np.concatenate((self.data['cnts3'], np.array([cnts1])), axis=0)
                self.data['cnts4'] = np.concatenate((self.data['cnts4'], np.array([cnts2])), axis=0)
                cos_sig = np.array([2 * (cnts2 - cnts1) / (cnts2 + cnts1)])
                sin_sig = np.array([2 * (cnts4 - cnts3) / (cnts4 + cnts3)])
                coherence = np.sqrt(cos_sig ** 2 + sin_sig ** 2)
                phase = np.arccos(cos_sig / coherence) * np.sign(sin_sig)
                self.data['cos_sig'] = np.concatenate((self.data['cos_sig'], cos_sig), axis=0)
                self.data['sin_sig'] = np.concatenate((self.data['sin_sig'], sin_sig), axis=0)
                self.data['coherence'] = np.concatenate((self.data['coherence'], coherence), axis=0)
                self.data['phase'] = np.concatenate((self.data['phase'], phase), axis=0)
                self.data['dist_array'] = np.concatenate((self.data['dist_array'], np.array([dist_array[i + 1]])))

        # turn off MWs
        self.instruments['mw_gen_iq']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq']['instance'].update({'ext_trigger': False})
        self.instruments['mw_gen_iq']['instance'].update({'enable_IQ': False})
        print('Turned off RF generator SGS100A (IQ off, trigger off).')
        time.sleep(0.5)
        self.instruments['mw_gen_iq_2']['instance'].update({'enable_output': False})
        self.instruments['mw_gen_iq_2']['instance'].update({'enable_IQ': False})
        self.instruments['mw_gen_iq_2']['instance'].update({'enable_modulation': False})
        print('Turned off RF generator N9310A (IQ off).')

        # do AFM 1d after scan
        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

    def _plot(self, axes_list, data=None, title = True):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()

        try:
            if 'cnts1' in data.keys() and len(data['dist_array']) > 0:
                axes_list[0].plot(data['dist_array'], data['cos_sig'], label="cosine")
                axes_list[0].plot(data['dist_array'], data['sin_sig'], label="sine")
                axes_list[0].plot(data['dist_array'], data['coherence'], '--', label="coherence")
                axes_list[1].plot(data['dist_array'], data['phase'])
            else:
                axes_list[0].plot(np.zeros([10]), np.zeros([10]), label="cosine")
                axes_list[0].plot(np.zeros([10]), np.zeros([10]), label="sine")
                axes_list[0].plot(np.zeros([10]), np.zeros([10]), '--', label="coherence")
                axes_list[1].plot(np.zeros([10]), np.zeros([10]))
        except Exception as e:
            print('** ATTENTION in _plot 1d **')
            print(e)

        axes_list[0].set_ylabel('Contrast')
        axes_list[1].set_ylabel('Phase [rad]')
        axes_list[0].xaxis.set_ticklabels([])
        axes_list[1].set_xlabel('Position [V]')
        axes_list[0].legend(loc='upper right')

        # axes_list[0].set_title(
        #     'AC Stark Shift (type: {:s})\n{:s} {:d} block(s), tau = {:0.3}us, {:0.1f}kcps, {:d} repetitions\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nRF power2: {:0.1f}dBm, LO freq2: {:0.4f}GHz, IF amp2: {:0.1f}, IF freq2: {:0.2f}MHz\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
        #         self.settings['sensing_type'], self.settings['decoupling_seq']['type'],
        #         self.settings['decoupling_seq']['num_of_pulse_blocks'], data['tau'] / 1000,
        #         (data['cnts1'][-1] + data['cnts2'][-1]) / 2, self.settings['rep_num'],
        #         self.settings['mw_pulses']['pi_half_pulse_time'],
        #         self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
        #         self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
        #         self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
        #         self.settings['mw_pulses_2']['mw_power'], self.settings['mw_pulses_2']['mw_frequency'] * 1e-9,
        #         self.settings['mw_pulses_2']['IF_amp'], self.settings['mw_pulses_2']['IF_frequency'] * 1e-6,
        #         self.settings['point_a']['x'], self.settings['point_a']['y'],
        #         self.settings['point_b']['x'], self.settings['point_b']['y']
        #     ), fontsize=9.5)

        if title:
            axes_list[0].set_title(
                'AC Stark Shift (type: {:s})\n{:s} {:d} block(s), tau = {:0.3}us, {:d} repetitions\npi-half time: {:2.1f}ns, pi-time: {:2.1f}ns, 3pi-half time: {:2.1f}ns\nRF power: {:0.1f}dBm, LO freq: {:0.4f}GHz, IF amp: {:0.1f}, IF freq: {:0.2f}MHz\nRF power2: {:0.1f}dBm, LO freq2: {:0.4f}GHz, IF amp2: {:0.1f}, IF freq2: {:0.2f}MHz\npta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
                    self.settings['sensing_type'], self.settings['decoupling_seq']['type'],
                    self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.settings['tau']/1000, self.settings['rep_num'],
                    self.settings['mw_pulses']['pi_half_pulse_time'],
                    self.settings['mw_pulses']['pi_pulse_time'], self.settings['mw_pulses']['3pi_half_pulse_time'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses']['IF_amp'], self.settings['mw_pulses']['IF_frequency'] * 1e-6,
                    self.settings['mw_pulses_2']['mw_power'], self.settings['mw_pulses_2']['mw_frequency'] * 1e-9,
                    self.settings['mw_pulses_2']['IF_amp'], self.settings['mw_pulses_2']['IF_frequency'] * 1e-6,
                    self.settings['point_a']['x'], self.settings['point_a']['y'],
                    self.settings['point_b']['x'], self.settings['point_b']['y']
                ), fontsize=9.5)
        else:
            axes_list[0].set_title(
                'pta: x={:0.3f}V, y={:0.3f}V\nptb: x={:0.3f}V, y={:0.3f}V'.format(
                    self.settings['point_a']['x'], self.settings['point_a']['y'],
                    self.settings['point_b']['x'], self.settings['point_b']['y']
                ), fontsize=9.5)

        if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[2]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[2], axes_list[3]], title=False)

    def _update_plot(self, axes_list):
        try:
            if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._plot([axes_list[2]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[2], axes_list[3]], title=False)
            elif 'cnts1' in self.data.keys() and len(self.data['dist_array']) > 0:
                axes_list[0].lines[0].set_xdata(self.data['dist_array'])
                axes_list[0].lines[0].set_ydata(self.data['cos_sig'])
                axes_list[0].lines[1].set_xdata(self.data['dist_array'])
                axes_list[0].lines[1].set_ydata(self.data['sin_sig'])
                axes_list[0].lines[2].set_xdata(self.data['dist_array'])
                axes_list[0].lines[2].set_ydata(self.data['coherence'])
                axes_list[0].relim()
                axes_list[0].autoscale_view()
                axes_list[1].lines[0].set_xdata(self.data['dist_array'])
                axes_list[1].lines[0].set_ydata(self.data['phase'])
                axes_list[1].relim()
                axes_list[1].autoscale_view()

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
            axes_list.append(figure_list[0].add_subplot(211))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(212))  # axes_list[1]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


class ScanACStark2D(Script):
    _DEFAULT_SETTINGS = [
        Parameter('to_do', 'print_info', ['print_info', 'execution'],
                  'choose to print information of the scanning settings or do real scanning'),
        Parameter('scan_center',
                  [Parameter('x', 2, float, 'x-coordinate [V]'),
                   Parameter('y', 2, float, 'y-coordinate [V]')
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
                  [Parameter('axis1', 4.0, float, 'inner loop [V]'),
                   Parameter('axis2', 4.0, float, 'outer loop [V]')
                   ]),
        Parameter('num_points',
                  [Parameter('axis1', 105, int, 'number of points to scan in each line (inner loop)'),
                   Parameter('axis2', 75, int, 'number of lines to scan (outer loop)'),
                   ]),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 12, int, 'do the optimization once per N points')]),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),

    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'scan_ac_stark': ScanACStark, 'set_scanner': SetScannerXY_gentle, 'optimize': optimize}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)

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

    def _function(self):
        self._get_scan_array()

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif np.max([self.pta, self.ptb, self.ptc, self.ptd]) <= 8 and np.min(
                [self.pta, self.ptb, self.ptc, self.ptd]) >= 0:
            # Move to the initial scanning position
            self.scripts['set_scanner'].update({'to_do': 'set'})
            self.scripts['set_scanner'].update({'verbose': False})
            self.scripts['set_scanner'].update({'scan_speed': self.settings['scan_speed']})
            self.scripts['set_scanner'].update({'step_size': 0.0001})

            x2 = np.sqrt((self.ptb[0] - self.pta[0]) ** 2 + (self.ptb[1] - self.pta[1]) ** 2)
            y2 = np.sqrt((self.ptd[0] - self.ptc[0]) ** 2 + (self.ptd[1] - self.ptc[1]) ** 2)
            self.data = {'scan_center': self.scan_center, 'scan_size_1': self.settings['scan_size']['axis1'],
                         'scan_size_2': self.settings['scan_size']['axis2'], 'vector_x': self.vector_x,
                         'cos_sig': 0 * np.ones((self.settings['num_points']['axis2'],
                                                   self.settings['num_points']['axis1'])),
                         'sin_sig': 0 * np.ones((self.settings['num_points']['axis2'],
                                                   self.settings['num_points']['axis1'])),
                         'coherence': 0 * np.ones((self.settings['num_points']['axis2'],
                                                   self.settings['num_points']['axis1'])),
                         'phase': 0 * np.ones((self.settings['num_points']['axis2'],
                                                   self.settings['num_points']['axis1'])),
                         'extent': np.array([0, x2, y2, 0]),
                         'region': np.array([self.pta, self.ptb, self.ptc, self.ptd])}

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

                # do optimize
                if self.settings['do_optimize']['on'] and i % self.settings['do_optimize'][
                    'frequency'] == 0 and not self._abort:
                    self.scripts['optimize'].settings['tag'] = 'optimize_' + str(i)
                    # self.flag_optimize_plot = True
                    self.scripts['optimize'].run()

                try:
                    self.flag_1d_plot = True
                    self.scripts['scan_ac_stark'].settings['tag'] = 'sensing1d_ind' + str(i)
                    self.scripts['scan_ac_stark'].settings['scan_speed'] = self.settings['scan_speed']
                    self.scripts['scan_ac_stark'].settings['point_a']['x'] = float(self.scan_pos_1d_ac[i][0])
                    self.scripts['scan_ac_stark'].settings['point_a']['y'] = float(self.scan_pos_1d_ac[i][1])
                    self.scripts['scan_ac_stark'].settings['point_b']['x'] = float(self.scan_pos_1d_bd[i][0])
                    self.scripts['scan_ac_stark'].settings['point_b']['y'] = float(self.scan_pos_1d_bd[i][1])
                    self.scripts['scan_ac_stark'].settings['num_points'] = self.settings['num_points']['axis1']

                    # we always need to scan back in each line
                    self.scripts['scan_ac_stark'].settings['do_afm1d']['after'] = True
                    self.scripts['scan_ac_stark'].run()
                    self.data['cos_sig'][i] = self.scripts['scan_ac_stark'].data['cos_sig']
                    self.data['sin_sig'][i] = self.scripts['scan_ac_stark'].data['sin_sig']
                    self.data['coherence'][i] = self.scripts['scan_ac_stark'].data['coherence']
                    self.data['phase'][i] = self.scripts['scan_ac_stark'].data['phase']

                except Exception as e:
                    print('** ATTENTION in scan_ac_stark 1d **')
                    print(e)

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()
        axes_list[4].clear()
        axes_list[5].clear()

        detuning = (self.scripts['scan_ac_stark'].settings['mw_pulses_2']['mw_frequency'] + \
                   self.scripts['scan_ac_stark'].settings['mw_pulses_2']['IF_frequency'] - \
                   self.scripts['scan_ac_stark'].settings['mw_pulses']['mw_frequency'] - \
                   self.scripts['scan_ac_stark'].settings['mw_pulses']['IF_frequency']) / 1E6
        power = self.scripts['scan_ac_stark'].settings['mw_pulses_2']['mw_power']
        tau_time = self.scripts['scan_ac_stark'].settings['tau']

        plot_fluorescence_new(data['phase'], data['extent'], axes_list[0], rotation=0, colorbar_name='phase\n[rad]',
                              axes_labels=['Y [V]', 'X [V]'], axes_not_voltage=True, cmap='afmhot',
                              title='AC Stark Shift - Phase\n' + r'$\Delta=$' + '{:0.1f}MHz,{:0.1f}dBm,'.format(detuning,
                                                                                                               power) + r'$\tau=$' + '{:0.1f}us'.format(
                                  tau_time / 1000), colorbar_labels=[-3.14, 3.14] )
        # plot_fluorescence_new(data['cos_sig'], data['extent'], axes_list[1], rotation=0, colorbar_name='contrast',
        #                       axes_labels=['Y [V]', 'X [V]'], axes_not_voltage=True, cmap='afmhot',
        #                       title='cosine', colorbar_labels=[-0.2,0.2] )
        #
        # plot_fluorescence_new(data['sin_sig'], data['extent'], axes_list[3], rotation=0, colorbar_name='contrast',
        #                       axes_labels=['Y [V]', 'X [V]'], axes_not_voltage=True, cmap='afmhot',
        #                       title='sine', colorbar_labels=[-0.2, 0.2])
        if self._current_subscript_stage['current_subscript'] == self.scripts['scan_ac_stark']:
            self.scripts['scan_ac_stark']._plot(axes_list[2:], title = False)

    def _update_plot(self, axes_list):
        try:
            update_fluorescence(self.data['phase'], axes_list[0], colorbar_labels=[-3.14, 3.14])
            # update_fluorescence(self.data['cos_sig'], axes_list[1], colorbar_labels=[-0.2,0.2])
            # update_fluorescence(self.data['sin_sig'], axes_list[2], colorbar_labels=[-0.2, 0.2])
            if self._current_subscript_stage['current_subscript'] == self.scripts['scan_ac_stark']:
                if self.flag_1d_plot:
                    self.scripts['scan_ac_stark']._plot(axes_list[2:], title = False)
                    self.flag_1d_plot = False
                else:
                    self.scripts['scan_ac_stark']._update_plot(axes_list[2:])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._plot([axes_list[4]])
        except Exception as e:
            print('** ATTENTION in scan_ac_stark _update_plot **')
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
            axes_list.append(figure_list[0].add_subplot(221))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(223))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(222))  # axes_list[2]
            axes_list.append(figure_list[0].add_subplot(224))  # axes_list[3]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[4]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[5]
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[1].axes[1])
        return axes_list


