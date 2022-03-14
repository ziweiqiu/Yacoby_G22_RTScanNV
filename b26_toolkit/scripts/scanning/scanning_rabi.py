import numpy as np

from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.set_laser import SetScannerXY_gentle
from b26_toolkit.scripts.galvo_scan.confocal_scan_G22B import AFM1D_qm
from b26_toolkit.scripts.qm_scripts.basic import RabiQM


class ScanningRabi(Script):
    """
        Measure Rabi when scanning along a 1D line.
        - Ziwei Qiu 9/12/2020
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
        Parameter('num_points', 20, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('mw_pulses', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -10.0, float, 'RF power in dBm'),
            Parameter('IF_frequency', 100e6, float, 'IF frequency in Hz from the QM'),
            Parameter('IF_amp', 1.0, float, 'amplitude of the IF pulse, between 0 and 1'),
            Parameter('phase', 0, float, 'starting phase of the RF pulse in deg')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 16, int, 'minimum time for rabi oscillations (in ns), >=16ns'),
            Parameter('max_time', 500, int, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 16, int,
                      'time step increment of rabi pulse duration (in ns), using multiples of 4ns')
        ]),
        Parameter('rabi_rep_num', 500000, int, 'define the repetition number for rabi')
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'rabi': RabiQM, 'set_scanner': SetScannerXY_gentle, 'afm1d': AFM1D_qm}

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

    def setup_rabi(self):
        self.scripts['rabi'].settings['to_do'] = 'execution'
        self.scripts['rabi'].settings['mw_pulses']['mw_frequency'] = self.settings['mw_pulses']['mw_frequency']
        self.scripts['rabi'].settings['mw_pulses']['mw_power'] = self.settings['mw_pulses']['mw_power']
        self.scripts['rabi'].settings['mw_pulses']['IF_frequency'] = self.settings['mw_pulses']['IF_frequency']
        self.scripts['rabi'].settings['mw_pulses']['IF_amp'] = self.settings['mw_pulses']['IF_amp']
        self.scripts['rabi'].settings['mw_pulses']['phase'] = self.settings['mw_pulses']['phase']
        self.scripts['rabi'].settings['tau_times']['min_time'] = self.settings['tau_times']['min_time']
        self.scripts['rabi'].settings['tau_times']['max_time'] = self.settings['tau_times']['max_time']
        self.scripts['rabi'].settings['tau_times']['time_step'] = self.settings['tau_times']['time_step']
        self.scripts['rabi'].settings['rep_num'] = self.settings['rabi_rep_num']

    def _function(self):

        scan_pos_1d, dist_array = self._get_scan_array()

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

        self.data = {'scan_pos_1d': scan_pos_1d, 'rabi_dist_array': np.array([]), 'rabi_freq': np.array([]),
                     'afm_dist_array': np.array([]), 'afm_ctr': np.array([]), 'afm_analog': np.array([])}

        self.setup_rabi()
        try:
            self.scripts['rabi'].run()
            rabi_freq = np.array([self.scripts['rabi'].data['rabi_freq']])
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            self.data['rabi_freq'] = np.concatenate((self.data['rabi_freq'], rabi_freq))
            self.data['rabi_dist_array'] = np.concatenate((self.data['rabi_dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):

            if self._abort:
                break

            afm_start = scan_pos_1d[i]
            afm_end = scan_pos_1d[i + 1]
            self.setup_afm(afm_start, afm_end)
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

            try:
                self.scripts['rabi'].run()
                rabi_freq = np.array([self.scripts['rabi'].data['rabi_freq']])
            except Exception as e:
                print('** ATTENTION **')
                print(e)
            else:
                self.data['rabi_freq'] = np.concatenate((self.data['rabi_freq'], rabi_freq))
                self.data['rabi_dist_array'] = np.concatenate((self.data['rabi_dist_array'], np.array([dist_array[i + 1]])))

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        if self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
            self.scripts['rabi']._plot([axes_list[3]], title=False)

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()

        if len(data['afm_dist_array']) > 0:
            axes_list[0].plot(data['afm_dist_array'], data['afm_ctr'])
            axes_list[1].plot(data['afm_dist_array'], data['afm_analog'])
        else:
            axes_list[0].plot(np.zeros([10]), np.zeros([10]))
            axes_list[1].plot(np.zeros([10]), np.zeros([10]))

        if len(data['rabi_dist_array']) > 0:
            axes_list[2].plot(data['rabi_dist_array'], data['rabi_freq'])
        else:
            axes_list[2].plot(np.zeros([10]), np.zeros([10]))

        axes_list[0].set_ylabel('Counts [kcps]')
        axes_list[1].set_ylabel('Z_out [V]')
        axes_list[2].set_ylabel('Rabi Freq. [MHz]')
        axes_list[2].set_xlabel('Position [V]')
        axes_list[0].set_title('pta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V'.format(
            self.settings['point_a']['x'], self.settings['point_a']['y'],
            self.settings['point_b']['x'], self.settings['point_b']['y']))

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
            self.scripts['rabi']._update_plot([axes_list[3]], title=False)
        else:
            if len(self.data['afm_dist_array']) > 0:
                axes_list[0].lines[0].set_xdata(self.data['afm_dist_array'])
                axes_list[0].lines[0].set_ydata(self.data['afm_ctr'])
                axes_list[0].relim()
                axes_list[0].autoscale_view()

                axes_list[1].lines[0].set_xdata(self.data['afm_dist_array'])
                axes_list[1].lines[0].set_ydata(self.data['afm_analog'])
                axes_list[1].relim()
                axes_list[1].autoscale_view()

            if len(self.data['rabi_dist_array']) > 0:
                axes_list[2].lines[0].set_xdata(self.data['rabi_dist_array'])
                axes_list[2].lines[0].set_ydata(self.data['rabi_freq'])
                axes_list[2].relim()
                axes_list[2].autoscale_view()

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
            axes_list.append(figure_list[0].add_subplot(311))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(312))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(313))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(111))  # axes_list[3]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[1].axes[0])

        return axes_list
