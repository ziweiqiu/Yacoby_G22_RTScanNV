import numpy as np
import math
from pyanc350.v2 import Positioner

from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.set_laser import SetScannerXY_gentle
from b26_toolkit.scripts.galvo_scan.confocal_scan_G22B import AFM1D_qm
from b26_toolkit.scripts.qm_scripts.echo import AC_DGate_SingleTau, AC_AGate_SingleTau
from b26_toolkit.scripts.qm_scripts.basic import RabiQM
from b26_toolkit.scripts.optimize import optimize
from b26_toolkit.instruments import NI6733, NI6220, NI6210

import smtplib, ssl

receiver_email = "ziweiqiu@g.harvard.edu"
def send_email(receiver_email, message):
    port = 465  # For SSL
    password = "diamond2020"
    sender_email = "nv.scanning.alert@gmail.com"
    # Create a secure SSL context
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        server.quit()


class ScanningACSensingV0(Script):
    """
        Perform AC sensing measurements and scan along a 1D line.
        - Ziwei Qiu 10/19/2020
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
        Parameter('gate_type', 'digital', ['analog', 'digital'],
                  'define the gate type. if digital, the gate is output from OPX D5, if analog, the gate is output from OPX A5')
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {'ac_sensing_digital': AC_DGate_SingleTau, 'ac_sensing_analog': AC_AGate_SingleTau,
                'set_scanner': SetScannerXY_gentle, 'afm1d': AFM1D_qm}

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

    def setup_ac_sensing(self):
        self.ac_sensing_script = 'ac_sensing_' + self.settings['gate_type']
        self.scripts[self.ac_sensing_script].settings['to_do'] = 'execution'

    def do_ac_sensing(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts[self.ac_sensing_script].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts[self.ac_sensing_script].settings['tag'] = label
        elif index >= 0:
            self.scripts[self.ac_sensing_script].settings['tag'] = 'ac_' + self.settings['gate_type'] + '_ind' + str(
                index)
        else:
            self.scripts[self.ac_sensing_script].settings['tag'] = 'ac_' + self.settings['gate_type']

        if verbose:
            print('==> Executing AC sensing' + self.settings['gate_type'] + '...')

        self.scripts[self.ac_sensing_script].run()
        ac_sig = self.scripts[self.ac_sensing_script].data['signal_avg_vec']
        norm_sig = np.array([[2 * (ac_sig[1] - ac_sig[0]) / (ac_sig[1] + ac_sig[0]),
                              2 * (ac_sig[3] - ac_sig[2]) / (ac_sig[3] + ac_sig[2])]])
        ac_sig = np.array([[ac_sig[0], ac_sig[1], ac_sig[2], ac_sig[3]]])

        return ac_sig, norm_sig

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

        # Initialize the data structure
        self.data = {'scan_pos_1d': scan_pos_1d, 'ac_dist_array': np.array([]),
                     'afm_dist_array': np.array([]), 'afm_ctr': np.array([]), 'afm_analog': np.array([])}

        self.setup_ac_sensing()
        # The first point
        try:
            ac_sig, norm_sig = self.do_ac_sensing(index=0)
        except Exception as e:
            print('** ATTENTION do_ac_sensing(index=0) **')
            print(e)
        else:
            self.data['ac_data'] = ac_sig
            self.data['norm_data'] = norm_sig
            self.data['ac_dist_array'] = np.concatenate((self.data['ac_dist_array'], np.array([0])))

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
                ac_sig, norm_sig = self.do_ac_sensing(index=i+1)
            except Exception as e:
                print('** ATTENTION in do_ac_sensing **')
                print(e)
            else:
                self.data['ac_data'] = np.concatenate((self.data['ac_data'], ac_sig), axis=0)
                self.data['norm_data'] = np.concatenate((self.data['norm_data'], norm_sig), axis=0)
                self.data['ac_dist_array'] = np.concatenate((self.data['ac_dist_array'], np.array([dist_array[i + 1]])))

    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        ac_script = 'ac_sensing_' + self.settings['gate_type']
        if self._current_subscript_stage['current_subscript'] == self.scripts[ac_script]:
            self.scripts[ac_script]._plot([axes_list[3]], title=False)

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()

        if len(data['afm_dist_array']) > 0:
            axes_list[0].plot(data['afm_dist_array'], data['afm_ctr'])
            axes_list[1].plot(data['afm_dist_array'], data['afm_analog'])
        else:
            axes_list[0].plot(np.zeros([10]), np.zeros([10]))
            axes_list[1].plot(np.zeros([10]), np.zeros([10]))

        if 'ac_data' in data.keys() and len(data['ac_dist_array']) > 0:
            axes_list[2].plot(data['ac_dist_array'], data['norm_data'][:, 0], label="norm sig1")
            axes_list[2].plot(data['ac_dist_array'], data['norm_data'][:, 1], label="norm sig2")
        else:
            axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig1")
            axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig2")

        axes_list[0].set_ylabel('Counts [kcps]')
        axes_list[1].set_ylabel('Z_out [V]')
        axes_list[2].set_ylabel('Contrast')
        axes_list[2].set_xlabel('Position [V]')
        axes_list[2].legend(loc='upper right')

        if self.settings['gate_type'] == 'analog':
            ac_title = "analog gate, type: {:s}, {:s} {:d} block(s), {:d} repetitions\ngate1: {:0.3f}V, gate2: {:0.3f}V, offset: {:0.3f}V".format(
                self.scripts['ac_sensing_analog'].settings['sensing_type'],
                self.scripts['ac_sensing_analog'].settings['decoupling_seq']['type'],
                self.scripts['ac_sensing_analog'].settings['decoupling_seq']['num_of_pulse_blocks'],
                self.scripts['ac_sensing_analog'].settings['rep_num'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['gate1'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['gate2'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['offset'])
        else:
            ac_title = "digital gate, type: {:s}, {:s} {:d} block(s), {:d} repetitions".format(
                self.scripts['ac_sensing_digital'].settings['sensing_type'],
                self.scripts['ac_sensing_digital'].settings['decoupling_seq']['type'],
                self.scripts['ac_sensing_digital'].settings['decoupling_seq']['num_of_pulse_blocks'],
                self.scripts['ac_sensing_digital'].settings['rep_num'])

        axes_list[0].set_title('pta:x={:0.3f}V, y={:0.3f}V. ptb:x={:0.3f}V, y={:0.3f}V\n'.format(
            self.settings['point_a']['x'], self.settings['point_a']['y'],
            self.settings['point_b']['x'], self.settings['point_b']['y']) + ac_title)

    def _update_plot(self, axes_list):
        if self._current_subscript_stage['current_subscript'] == self.scripts['ac_sensing_digital']:
            self.scripts['ac_sensing_digital']._update_plot([axes_list[3]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['ac_sensing_analog']:
            self.scripts['ac_sensing_analog']._update_plot([axes_list[3]], title=False)
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
            if 'ac_data' in self.data.keys() and len(self.data['ac_dist_array']) > 0:
                axes_list[2].lines[0].set_xdata(self.data['ac_dist_array'])
                axes_list[2].lines[0].set_ydata(self.data['norm_data'][:, 0])
                axes_list[2].lines[1].set_xdata(self.data['ac_dist_array'])
                axes_list[2].lines[1].set_ydata(self.data['norm_data'][:, 1])
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


class ScanningACSensing(Script):
    """
        Perform AC sensing measurements and scan along a 1D line.
        One can also choose to do Rabi and fluorescence optimize every several points.
        - Ziwei Qiu 10/24/2020
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
        Parameter('num_points', 100, int, 'number of points for NV measurement'),
        Parameter('scan_speed', 0.04, [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
                  '[V/s] scanning speed on average. suggest 0.04V/s, i.e. 200nm/s.'),
        Parameter('gate_type', 'analog', ['analog', 'digital'],
                  'define the gate type. if digital, the gate is output from OPX D5, if analog, the gate is output from OPX A5'),
        Parameter('do_afm1d',
                  [Parameter('before', True, bool, 'whether to do a round-trip afm 1d before the experiment starts'),
                   Parameter('num_of_cycles', 1, int, 'number of forward and backward scans before the experiment starts'),
                   Parameter('after', True, bool, 'whether to scan back to the original point after the experiment is done'),
                   ]),
        Parameter('do_rabi',
                  [Parameter('on', True, bool, 'choose whether to do Rabi calibration'),
                   Parameter('frequency', 5, int, 'do the Rabi once per N points'),
                   Parameter('start_power', -10, float, 'choose the MW power at the first point'),
                   Parameter('end_power', -20, float, 'choose the MW power at the last point'),
                   ]),
        Parameter('do_optimize',
                  [Parameter('on', True, bool, 'choose whether to do fluorescence optimization'),
                   Parameter('frequency', 50, int, 'do the optimization once per N points'),
                   Parameter('range_xy', 0.25, float, 'the range [V] in which the xy focus can fluctuate reasonably'),
                   Parameter('range_z', 0.5, float, 'the range [V] in which the z focus can fluctuate reasonably')
                   ]),

        Parameter('monitor_AFM', False, bool,
                  'monitor the AFM Z_out voltage and retract the tip when the feedback loop is out of control'),

        Parameter('DAQ_channels',
                  [Parameter('x_ao_channel', 'ao5', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for x voltage analog output'),
                   Parameter('y_ao_channel', 'ao6', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for y voltage analog output'),
                   Parameter('z_ao_channel', 'ao7', ['ao5', 'ao6', 'ao7'],
                             'Daq channel used for z voltage analog output'),
                   Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring obj x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring obj y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring obj z voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for measuring scanner z voltage')
                   ])
    ]
    _INSTRUMENTS = {'NI6733': NI6733, 'NI6220': NI6220, 'NI6210': NI6210}
    _SCRIPTS = {'ac_sensing_digital': AC_DGate_SingleTau, 'ac_sensing_analog': AC_AGate_SingleTau, 'rabi': RabiQM,
                'optimize': optimize, 'set_scanner': SetScannerXY_gentle, 'afm1d': AFM1D_qm, 'afm1d_before_after': AFM1D_qm}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.daq_in_AI = self.instruments['NI6220']['instance']
        self.daq_out = self.instruments['NI6733']['instance']
        self.daq_in_usbAI = self.instruments['NI6210']['instance']

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

    def setup_ac_sensing(self):
        self.ac_sensing_script = 'ac_sensing_' + self.settings['gate_type']
        self.scripts[self.ac_sensing_script].settings['to_do'] = 'execution'

    def do_ac_sensing(self, label=None, index=-1, verbose=True):
        # update the tag of pdd script
        if label is not None and index >= 0:
            self.scripts[self.ac_sensing_script].settings['tag'] = label + '_ind' + str(index)
        elif label is not None:
            self.scripts[self.ac_sensing_script].settings['tag'] = label
        elif index >= 0:
            self.scripts[self.ac_sensing_script].settings['tag'] = 'ac_' + self.settings['gate_type'] + '_ind' + str(
                index)
        else:
            self.scripts[self.ac_sensing_script].settings['tag'] = 'ac_' + self.settings['gate_type']

        if self.settings['do_optimize']['on'] and index >= 0 and index % self.settings['do_optimize'][
            'frequency'] == 0 and not self._abort:
            self.scripts['optimize'].settings['tag'] = 'optimize' + '_ind' + str(index)

            pos_before = self.daq_in_AI.get_analog_voltages(
                [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
                 self.settings['DAQ_channels']['z_ai_channel']])

            if verbose:
                print('==> Executing optimize ...')
            self.scripts['optimize'].run()

            pos_after = self.daq_in_AI.get_analog_voltages(
                [self.settings['DAQ_channels']['x_ai_channel'], self.settings['DAQ_channels']['y_ai_channel'],
                 self.settings['DAQ_channels']['z_ai_channel']])

            # if np.abs(pos_after[0]- pos_before[0]) > self.settings['do_optimize']['range_xy']:
            #     self.daq_out.set_analog_voltages(
            #         {self.settings['DAQ_channels']['x_ao_channel']: pos_before[0]})
            #     print('---> Optimize X does not make sense. Set laser X back.')
            # if np.abs(pos_after[1] - pos_before[1]) > self.settings['do_optimize']['range_xy']:
            #     self.daq_out.set_analog_voltages(
            #         {self.settings['DAQ_channels']['y_ao_channel']: pos_before[1]})
            #     print('---> Optimize Y does not make sense. Set laser Y back.')
            # if np.abs(pos_after[2] - pos_before[2]) > self.settings['do_optimize']['range_z']:
            #     self.daq_out.set_analog_voltages(
            #         {self.settings['DAQ_channels']['z_ao_channel']: pos_before[2]})
            #     print('---> Optimize Z does not make sense. Set laser Z back.')
            if np.abs(pos_after[0] - pos_before[0]) > self.settings['do_optimize']['range_xy'] or np.abs(
                    pos_after[1] - pos_before[1]) > self.settings['do_optimize']['range_xy'] or np.abs(
                    pos_after[2] - pos_before[2]) > self.settings['do_optimize']['range_z']:
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['x_ao_channel']: pos_before[0]})
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['y_ao_channel']: pos_before[1]})
                self.daq_out.set_analog_voltages({self.settings['DAQ_channels']['z_ao_channel']: pos_before[2]})
                print('---> Optimize does not make sense. Set laser back.')

        if self.settings['do_rabi']['on'] and index >= 0 and index % self.settings['do_rabi'][
            'frequency'] == 0 and not self._abort:

            self.scripts['rabi'].settings['tag'] = 'rabi' + '_ind' + str(index)
            self.scripts['rabi'].settings['mw_pulses']['phase'] = 0

            mw_power = self.settings['do_rabi']['start_power'] + (
                        self.settings['do_rabi']['end_power'] - self.settings['do_rabi']['start_power']) * index / \
                    self.settings['num_points']
            self.scripts['rabi'].settings['mw_pulses']['mw_power'] = mw_power

            try:
                if verbose:
                    print('==> Executing Rabi ...')
                self.scripts['rabi'].run()
                pi_time = self.scripts['rabi'].data['pi_time']
                pi_half_time = self.scripts['rabi'].data['pi_half_time']
                three_pi_half_time = self.scripts['rabi'].data['three_pi_half_time']
                rabi_freq = np.array([self.scripts['rabi'].data['rabi_freq']])
                rabi_power = np.array([self.scripts['rabi'].settings['mw_pulses']['mw_power']])

            except Exception as e:
                print('** ATTENTION in Rabi **')
                print(e)
            else:
                if pi_time >= 16 and pi_half_time >= 16 and three_pi_half_time >= 16:
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['mw_frequency'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['mw_frequency'])
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['mw_power'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['mw_power'])
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['IF_frequency'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['IF_frequency'])
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['IF_amp'] = \
                        float(self.scripts['rabi'].settings['mw_pulses']['IF_amp'])
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['pi_pulse_time'] = float(pi_time)
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['pi_half_pulse_time'] = float(pi_half_time)
                    self.scripts[self.ac_sensing_script].settings['mw_pulses']['3pi_half_pulse_time'] = float(
                        three_pi_half_time)

                # save the rabi data
                if 'rabi_freq' in self.data.keys():
                    self.data['rabi_freq'] = np.concatenate((self.data['rabi_freq'], rabi_freq))
                else:
                    self.data['rabi_freq'] = rabi_freq

                if 'rabi_power' in self.data.keys():
                    self.data['rabi_power'] = np.concatenate((self.data['rabi_power'], rabi_power))
                else:
                    self.data['rabi_power'] = rabi_power

                if 'rabi_dist_array' in self.data.keys():
                    self.data['rabi_dist_array'] = np.concatenate((self.data['rabi_dist_array'], np.array([self.dist_array[index]])))
                else:
                    self.data['rabi_dist_array'] = np.array([self.dist_array[index]])


        if verbose:
            print('==> Executing AC sensing' + self.settings['gate_type'] + '...')

        if not self._abort:
            self.scripts[self.ac_sensing_script].run()

            ac_sig = self.scripts[self.ac_sensing_script].data['signal_avg_vec']
            norm_sig = np.array([[2 * (ac_sig[1] - ac_sig[0]) / (ac_sig[1] + ac_sig[0]),
                                  2 * (ac_sig[3] - ac_sig[2]) / (ac_sig[3] + ac_sig[2])]])
            ac_sig = np.array([[ac_sig[0], ac_sig[1], ac_sig[2], ac_sig[3]]])
            exact_tau = np.array([self.scripts[self.ac_sensing_script].data['tau']])


            if self.scripts[self.ac_sensing_script].settings['sensing_type'] == 'both':
                coherence = np.sqrt(norm_sig[0,0] ** 2 + norm_sig[0,1] ** 2)
                phase = np.arccos(norm_sig[0,0] / coherence) * np.sign(norm_sig[0,1])
            elif self.scripts[self.ac_sensing_script].settings['sensing_type'] == 'cosine':
                coherence = norm_sig[0,0]
                phase = np.arccos(norm_sig[0,1] / coherence)
            else:
                coherence = norm_sig[0,0]
                phase = np.arcsin(norm_sig[0,1] / coherence)

            coherence = np.array([coherence])
            phase = np.array([phase])

            return ac_sig, norm_sig, exact_tau, coherence, phase
        else:
            return None, None, None, None, None

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

    def _setup_anc(self):
        self.anc_sample_connected = False
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_last = z_out[0]

        if self.settings['monitor_AFM']:
            try:
                self.anc_sample = Positioner()
                self.anc_sample_connected = self.anc_sample.is_connected
                state = self.anc_sample.getDcInEnable(5)
                print('Z scanner dcInEnable is ' + str(state))
            except Exception as e:
                print('** ATTENTION in creating ANC_sample **')
                print(e)

    def _check_AFM(self):
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_now = z_out[0]
        if np.abs(self.Z_scanner_now - self.Z_scanner_last) > 0.35:
            try:
                self.anc_sample.dcInEnable(5, False)
                state = self.anc_sample.getDcInEnable(5)
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                print('Z scanner dcInEnable is ' + str(state))
                self.log('Z scanner dcInEnable is ' + str(state))
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed and the tip has been retracted. Be relaxed and try again!"""

                send_email(receiver_email, message)
                self._abort = True

            except Exception as e:
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                print('** But the tip CANNOT be Retracted!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                self.log('** But the tip CANNOT be Retracted!! **')
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed BUT the tip CANNOT be retracted. Please take action!"""

                send_email(receiver_email, message)
                self._abort = True

            self._abort = True

        else:
            self.Z_scanner_last = self.Z_scanner_now

    def _function(self):
        self._setup_anc()
        if self.settings['monitor_AFM'] and not self.anc_sample_connected:
            print('** Attention ** ANC350 v2 (sample) is not connected. No scanning started.')
            self._abort = True

        scan_pos_1d, dist_array = self._get_scan_array()
        self.dist_array = dist_array

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

        # Initialize the data structure
        self.data = {'scan_pos_1d': scan_pos_1d, 'ac_dist_array': np.array([]),
                     'afm_dist_array': np.array([]), 'afm_ctr': np.array([]), 'afm_analog': np.array([])}

        if self.settings['do_afm1d']['before']:
            self.do_afm1d_before()

        self.setup_ac_sensing()
        if not self._abort:
            # The first point
            try:
                ac_sig, norm_sig, exact_tau, coherence, phase = self.do_ac_sensing(index=0)
            except Exception as e:
                print('** ATTENTION do_ac_sensing(index=0) **')
                print(e)
            else:
                self.data['ac_data'] = ac_sig
                self.data['norm_data'] = norm_sig
                self.data['exact_tau'] = exact_tau
                self.data['coherence'] = coherence
                self.data['phase'] = phase
                self.data['ac_dist_array'] = np.concatenate((self.data['ac_dist_array'], np.array([0])))

        for i in range(self.settings['num_points'] - 1):
            if self._abort:
                break

            afm_start = scan_pos_1d[i]
            afm_end = scan_pos_1d[i + 1]
            self.setup_afm(afm_start, afm_end)

            try:
                self.scripts['afm1d'].run()
                afm_dist_array = self.scripts['afm1d'].data['dist_array'] + dist_array[i]
                self.data['afm_dist_array'] = np.concatenate((self.data['afm_dist_array'], afm_dist_array))
                afm_ctr = self.scripts['afm1d'].data['data_ctr'][0]
                self.data['afm_ctr'] = np.concatenate((self.data['afm_ctr'], afm_ctr))
                afm_analog = self.scripts['afm1d'].data['data_analog'][0]
                self.data['afm_analog'] = np.concatenate((self.data['afm_analog'], afm_analog))
            except Exception as e:
                print('** ATTENTION **')
                print(e)


            try:
                ac_sig, norm_sig, exact_tau, coherence, phase = self.do_ac_sensing(index=i+1)
                self.data['ac_data'] = np.concatenate((self.data['ac_data'], ac_sig), axis=0)
                self.data['norm_data'] = np.concatenate((self.data['norm_data'], norm_sig), axis=0)
                self.data['exact_tau'] = np.concatenate((self.data['exact_tau'], exact_tau))
                self.data['coherence'] = np.concatenate((self.data['coherence'], coherence))
                self.data['phase'] = np.concatenate((self.data['phase'], phase))
                self.data['ac_dist_array'] = np.concatenate((self.data['ac_dist_array'], np.array([dist_array[i + 1]])))
            except Exception as e:
                print('** ATTENTION in do_ac_sensing **')
                print(e)


        if self.settings['do_afm1d']['after'] and not self._abort:
            self.do_afm1d_after()

        if self.anc_sample_connected:

            try:
                self.anc_sample.dcInEnable(5, False)
                state = self.anc_sample.getDcInEnable(5)
                print('****************************')
                print('** Scanning 2D is done **')
                self.log('** Scanning 2D is done **')
                print('Z scanner dcInEnable is ' + str(state))
                self.log('Z scanner dcInEnable is ' + str(state))
                print('****************************')
                message = """\
            Subject: Horray! NV Scanning Done

            Scanning 2D is done and AFM is retracted. Cheers!"""

                send_email(receiver_email, message)

            except Exception as e:
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                print('** But the tip CANNOT be Retracted!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                self.log('** But the tip CANNOT be Retracted!! **')
                print('****************************')
                message = """\
            Subject: NV Scanning Alert

            Attention! Scanning 2D is done but AFM cannot be retracted."""

                send_email(receiver_email, message)

            try:
                self.anc_sample.close()
                print('ANC350 v2 (sample) is closed.')
                self.log('ANC350 v2 (sample) is closed.')
            except Exception as e:
                print('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')
                self.log('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')

    def _plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        axes_list[0].clear()
        axes_list[1].clear()
        axes_list[2].clear()
        axes_list[3].clear()
        axes_list[4].clear()
        axes_list[5].clear()
        axes_list[6].clear()
        axes_list[7].clear()

        ac_script = 'ac_sensing_' + self.settings['gate_type']

        try:
            if 'afm_dist_array' in data.keys() and len(data['afm_dist_array']) > 0:
                axes_list[0].plot(data['afm_dist_array'], data['afm_ctr'])
                axes_list[1].plot(data['afm_dist_array'], data['afm_analog'])
            else:
                axes_list[0].plot(np.zeros([10]), np.zeros([10]))
                axes_list[1].plot(np.zeros([10]), np.zeros([10]))

            if 'ac_data' in data.keys() and len(data['ac_dist_array']) > 0:
                axes_list[2].plot(data['ac_dist_array'], data['norm_data'][:, 0], label="norm sig1")
                axes_list[2].plot(data['ac_dist_array'], data['norm_data'][:, 1], label="norm sig2")
                axes_list[2].plot(data['ac_dist_array'], data['coherence'],'--', label="coherence")
                axes_list[2].axhline(y=0.0, color='r', ls='--', lw=1.3)
                axes_list[7].plot(data['ac_dist_array'], data['phase'])
                axes_list[7].axhline(y=3.14159, color='r', ls='--', lw=1.1)
                axes_list[7].axhline(y=-3.14159, color='r', ls='--', lw=1.1)
            else:
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig1")
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), label="norm sig2")
                axes_list[2].plot(np.zeros([10]), np.zeros([10]), '--', label="coherence")
                axes_list[2].axhline(y=0.0, color='r', ls='--', lw=1.3)
                axes_list[7].plot(np.zeros([10]), np.zeros([10]))
                axes_list[7].axhline(y=3.14159, color='r', ls='--', lw=1.1)
                axes_list[7].axhline(y=-3.14159, color='r', ls='--', lw=1.1)

            if 'rabi_freq' in data.keys() and 'rabi_power' in data.keys() and 'rabi_dist_array' in data.keys():
                axes_list[4].plot(data['rabi_dist_array'], data['rabi_freq'], label="freq")
                axes_list[5].plot(data['rabi_dist_array'], data['rabi_power'], '--', label="pwr")
            else:
                axes_list[4].plot(np.zeros([10]), np.zeros([10]), label="freq")
                axes_list[5].plot(np.zeros([10]), np.zeros([10]), '--', label="pwr")
        except Exception as e:
            print('** ATTENTION in _plot 1d **')
            print(e)

        axes_list[0].set_ylabel('Counts [kcps]')
        axes_list[1].set_ylabel('Z_out [V]')
        axes_list[2].set_ylabel('Contrast')
        axes_list[4].set_ylabel('Rabi freq. [MHz]')
        axes_list[5].set_ylabel('RF power [dB]')
        axes_list[7].set_ylabel('Phase [rad]')

        # axes_list[0].xaxis.set_visible(False)
        # axes_list[1].xaxis.set_visible(False)
        # axes_list[2].xaxis.set_visible(False)
        # axes_list[4].xaxis.set_visible(False)
        # axes_list[5].xaxis.set_visible(False)

        axes_list[0].xaxis.set_ticklabels([])
        axes_list[1].xaxis.set_ticklabels([])
        axes_list[2].xaxis.set_ticklabels([])
        axes_list[4].xaxis.set_ticklabels([])
        axes_list[5].xaxis.set_ticklabels([])

        axes_list[7].set_xlabel('Position [V]')

        axes_list[2].legend(loc='upper right')
        axes_list[4].legend(loc='upper left')
        axes_list[5].legend(loc='upper right')

        if self.settings['gate_type'] == 'analog':
            ac_title = "analog gate, type: {:s}, {:s} {:d} block(s), {:d} repetitions\ngate1: {:0.3f}V, gate2: {:0.3f}V, offset: {:0.3f}V, tau = {:0.3}us".format(
                self.scripts['ac_sensing_analog'].settings['sensing_type'],
                self.scripts['ac_sensing_analog'].settings['decoupling_seq']['type'],
                self.scripts['ac_sensing_analog'].settings['decoupling_seq']['num_of_pulse_blocks'],
                self.scripts['ac_sensing_analog'].settings['rep_num'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['gate1'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['gate2'],
                self.scripts['ac_sensing_analog'].settings['gate_voltages']['offset'],
                self.scripts['ac_sensing_analog'].settings['tau']/1000)
        else:
            ac_title = "digital gate, type: {:s}, {:s} {:d} block(s), {:d} repetitions\ntau = {:0.3}us".format(
                self.scripts['ac_sensing_digital'].settings['sensing_type'],
                self.scripts['ac_sensing_digital'].settings['decoupling_seq']['type'],
                self.scripts['ac_sensing_digital'].settings['decoupling_seq']['num_of_pulse_blocks'],
                self.scripts['ac_sensing_digital'].settings['rep_num'],
                self.scripts['ac_sensing_digital'].settings['tau']/1000)

        axes_list[0].set_title('1D AC Sensing: pta=({:0.3f}V, {:0.3f}V), ptb=({:0.3f}V, {:0.3f}V)\n'.format(
            self.settings['point_a']['x'], self.settings['point_a']['y'],
            self.settings['point_b']['x'], self.settings['point_b']['y']) + ac_title)

        if self._current_subscript_stage['current_subscript'] == self.scripts[ac_script]:
            self.scripts[ac_script]._plot([axes_list[3]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
            self.scripts['rabi']._plot([axes_list[3]], title=False)
        elif self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
            self.scripts['optimize']._plot([axes_list[3]])
        elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
            self.scripts['afm1d_before_after']._plot([axes_list[3], axes_list[6]], title=False)

    def _update_plot(self, axes_list, monitor_AFM=True):
        # idea: do tip monitoring in the current script, and turn off that function in child scripts
        if monitor_AFM and self.anc_sample_connected:
            self._check_AFM()
        try:
            if self._current_subscript_stage['current_subscript'] == self.scripts['optimize']:
                self.scripts['optimize']._update_plot([axes_list[3]])
            elif self._current_subscript_stage['current_subscript'] == self.scripts['ac_sensing_digital']:
                self.scripts['ac_sensing_digital']._update_plot([axes_list[3]], title=False)
            elif self._current_subscript_stage['current_subscript'] == self.scripts['ac_sensing_analog']:
                self.scripts['ac_sensing_analog']._update_plot([axes_list[3]], title=False)
            elif self._current_subscript_stage['current_subscript'] == self.scripts['rabi']:
                self.scripts['rabi']._update_plot([axes_list[3]], title=False)
            elif self._current_subscript_stage['current_subscript'] == self.scripts['afm1d_before_after']:
                self.scripts['afm1d_before_after']._update_plot([axes_list[3], axes_list[6]], title=False,
                                                                monitor_AFM=False)

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
                if 'ac_data' in self.data.keys() and len(self.data['ac_dist_array']) > 0:
                    axes_list[2].lines[0].set_xdata(self.data['ac_dist_array'])
                    axes_list[2].lines[0].set_ydata(self.data['norm_data'][:, 0])
                    axes_list[2].lines[1].set_xdata(self.data['ac_dist_array'])
                    axes_list[2].lines[1].set_ydata(self.data['norm_data'][:, 1])
                    axes_list[2].lines[2].set_xdata(self.data['ac_dist_array'])
                    axes_list[2].lines[2].set_ydata(self.data['coherence'])
                    axes_list[2].relim()
                    axes_list[2].autoscale_view()

                    axes_list[7].lines[0].set_xdata(self.data['ac_dist_array'])
                    axes_list[7].lines[0].set_ydata(self.data['phase'])
                    axes_list[7].relim()
                    axes_list[7].autoscale_view()

                if 'rabi_freq' in self.data.keys() and 'rabi_power' in self.data.keys() and 'rabi_dist_array' in self.data.keys():
                    axes_list[4].lines[0].set_xdata(self.data['rabi_dist_array'])
                    axes_list[4].lines[0].set_ydata(self.data['rabi_freq'])
                    axes_list[5].lines[0].set_xdata(self.data['rabi_dist_array'])
                    axes_list[5].lines[0].set_ydata(self.data['rabi_power'])
                    axes_list[4].relim()
                    axes_list[4].autoscale_view()
                    axes_list[5].relim()
                    axes_list[5].autoscale_view()

        except Exception as e:
            print('** ATTENTION in scanning_ac_sensing 1d _update_plot **')
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
            axes_list.append(figure_list[0].add_subplot(511))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(512))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(513))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[3]
            axes_list.append(figure_list[0].add_subplot(514))  # axes_list[4]
            axes_list.append(axes_list[4].twinx())             # axes_list[5]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[6]
            axes_list.append(figure_list[0].add_subplot(515))  # axes_list[7]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[0].axes[4])
            axes_list.append(figure_list[1].axes[1])
            axes_list.append(figure_list[0].axes[5])

        return axes_list


class ScanningACSensing2D(Script):
    """
        Perform AC sensing measurements and perform AFM scan in 2D.
        One can also choose to do Rabi and fluorescence optimize every several points.
        One needs to go to child subscripts to define the settings.
        - Ziwei Qiu 11/16/2020
    """
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
        Parameter('do_rabi',
                  [Parameter('pta_power', -15, float, 'choose the MW power at pta, first point of first line'),
                   Parameter('ptb_power', -15, float, 'choose the MW power at ptb, last point of first line'),
                   Parameter('ptc_power', -15, float, 'choose the MW power at ptc, first point of last line'),
                   Parameter('ptd_power', -15, float, 'choose the MW power at ptd, last point of last line'),
                   ]),
        Parameter('monitor_AFM', False, bool,
                  'monitor the AFM Z_out voltage and retract the tip when the feedback loop is out of control'),
        Parameter('DAQ_channels',
                  [Parameter('x_ai_channel', 'ai5', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring x voltage'),
                   Parameter('y_ai_channel', 'ai6', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring y voltage'),
                   Parameter('z_ai_channel', 'ai7', ['ai5', 'ai6', 'ai7'], 'Daq channel used for measuring z voltage'),
                   Parameter('z_usb_ai_channel', 'ai1', ['ai0', 'ai1', 'ai2', 'ai3'],
                             'Daq channel used for measuring scanner z voltage')
                   ])

        ]
    _INSTRUMENTS = {'NI6210': NI6210}
    _SCRIPTS = {'scanning_acsensing_1d': ScanningACSensing, 'set_scanner': SetScannerXY_gentle}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)
        self.daq_in_usbAI = self.instruments['NI6210']['instance']

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
        self.start_power_ac = np.linspace(self.settings['do_rabi']['pta_power'], self.settings['do_rabi']['ptc_power'],
                                          self.settings['num_points']['axis2'], endpoint=True)
        self.end_power_bd = np.linspace(self.settings['do_rabi']['ptb_power'], self.settings['do_rabi']['ptd_power'],
                                        self.settings['num_points']['axis2'], endpoint=True)

    def _setup_anc(self):
        self.anc_sample_connected = False
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_last = z_out[0]

        if self.settings['monitor_AFM']:
            try:
                self.anc_sample = Positioner()
                self.anc_sample_connected = self.anc_sample.is_connected
            except Exception as e:
                print('** ATTENTION in creating ANC_sample **')
                print(e)


    def _check_AFM(self):
        z_out = self.daq_in_usbAI.get_analog_voltages([self.settings['DAQ_channels']['z_usb_ai_channel']])
        self.Z_scanner_now = z_out[0]
        if np.abs(self.Z_scanner_now - self.Z_scanner_last) > 0.35:
            try:
                self.anc_sample.dcInEnable(5, False)
                state = self.anc_sample.getDcInEnable(5)

                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                print('Z scanner dcInEnable is ' + str(state))
                self.log('Z scanner dcInEnable is ' + str(state))
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed and the tip has been retracted. Be relaxed and try again!"""

                send_email(receiver_email, message)

            except Exception as e:
                print('****************************')
                print('** ATTENTION: AFM Fails!! **')
                print('** But the tip CANNOT be Retracted!! **')
                self.log('** ATTENTION: AFM Fails!! **')
                self.log('** But the tip CANNOT be Retracted!! **')
                print('****************************')
                message = """\
Subject: NV Scanning Alert

Attention! AFM just failed and the tip has been retracted. Be relaxed and try again!"""

                send_email(receiver_email, message)

            self._abort = True

        else:
            self.Z_scanner_last = self.Z_scanner_now

    def _function(self):
        self._setup_anc()

        self._get_scan_array()

        if self.settings['to_do'] == 'print_info':
            print('No scanning started.')

        elif self.settings['monitor_AFM'] and not self.anc_sample_connected:
            print('** Attention ** ANC350 v2 (sample) is not connected. No scanning started.')
            self._abort = True

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
                    self.scripts['scanning_acsensing_1d'].settings['tag'] = 'sensing1d_ind' + str(i)
                    self.scripts['scanning_acsensing_1d'].settings['monitor_AFM'] = False
                    self.scripts['scanning_acsensing_1d'].settings['scan_speed'] = self.settings['scan_speed']
                    self.scripts['scanning_acsensing_1d'].settings['point_a']['x'] = self.scan_pos_1d_ac[i][0]
                    self.scripts['scanning_acsensing_1d'].settings['point_a']['y'] = self.scan_pos_1d_ac[i][1]
                    self.scripts['scanning_acsensing_1d'].settings['point_b']['x'] = self.scan_pos_1d_bd[i][0]
                    self.scripts['scanning_acsensing_1d'].settings['point_b']['y'] = self.scan_pos_1d_bd[i][1]
                    self.scripts['scanning_acsensing_1d'].settings['num_points'] = self.settings['num_points']['axis1']
                    # we always need to scan back in each line
                    self.scripts['scanning_acsensing_1d'].settings['do_afm1d']['after'] = True
                    self.scripts['scanning_acsensing_1d'].settings['do_rabi']['start_power'] = float(
                        self.start_power_ac[i])
                    self.scripts['scanning_acsensing_1d'].settings['do_rabi']['end_power'] = float(self.end_power_bd[i])
                    self.scripts['scanning_acsensing_1d'].run()

                # else:
                    norm_sig1 = self.scripts['scanning_acsensing_1d'].data['norm_data'][:, 0]
                    norm_sig2 = self.scripts['scanning_acsensing_1d'].data['norm_data'][:, 1]
                    if 'norm_sig1_data' in self.data.keys():
                        self.data['norm_sig1_data'] = np.concatenate((self.data['norm_sig1_data'], norm_sig1), axis=0)
                    else:
                        self.data['norm_sig1_data'] = norm_sig1
                    if 'norm_sig2_data' in self.data.keys():
                        self.data['norm_sig2_data'] = np.concatenate((self.data['norm_sig2_data'], norm_sig2), axis=0)
                    else:
                        self.data['norm_sig2_data'] = norm_sig2

                    exact_tau = self.scripts['scanning_acsensing_1d'].data['exact_tau']
                    if 'exact_tau' in self.data.keys():
                        self.data['exact_tau'] = np.concatenate((self.data['exact_tau'], exact_tau), axis=0)
                    else:
                        self.data['exact_tau'] = exact_tau

                    phase = self.scripts['scanning_acsensing_1d'].data['phase']
                    if 'phase' in self.data.keys():
                        self.data['phase'] = np.concatenate((self.data['phase'], phase), axis=0)
                    else:
                        self.data['phase'] = phase
                except Exception as e:
                    print('** ATTENTION in scanning_acsensing_1d **')
                    print(e)

        if self.anc_sample_connected:
            try:
                self.anc_sample.close()
                print('ANC350 v2 (sample) is closed.')
                self.log('ANC350 v2 (sample) is closed.')
            except Exception as e:
                print('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')
                self.log('** ATTENTION: ANC350 v2 (sample) CANNOT be closed. **')

    def _plot(self, axes_list, data=None):

        if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_acsensing_1d']:
            self.scripts['scanning_acsensing_1d']._plot(axes_list)

    def _update_plot(self, axes_list, monitor_AFM=True):
        if monitor_AFM and self.anc_sample_connected:
            self._check_AFM()

        if self._current_subscript_stage['current_subscript'] == self.scripts['scanning_acsensing_1d']:

            if self.flag_1d_plot:
                self.scripts['scanning_acsensing_1d']._plot(axes_list)
                self.flag_1d_plot = False
            else:
                self.scripts['scanning_acsensing_1d']._update_plot(axes_list, monitor_AFM=False)

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
            axes_list.append(figure_list[0].add_subplot(511))  # axes_list[0]
            axes_list.append(figure_list[0].add_subplot(512))  # axes_list[1]
            axes_list.append(figure_list[0].add_subplot(513))  # axes_list[2]
            axes_list.append(figure_list[1].add_subplot(121))  # axes_list[3]
            axes_list.append(figure_list[0].add_subplot(514))  # axes_list[4]
            axes_list.append(axes_list[4].twinx())  # axes_list[5]
            axes_list.append(figure_list[1].add_subplot(122))  # axes_list[6]
            axes_list.append(figure_list[0].add_subplot(515))  # axes_list[7]

        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
            axes_list.append(figure_list[0].axes[2])
            axes_list.append(figure_list[1].axes[0])
            axes_list.append(figure_list[0].axes[3])
            axes_list.append(figure_list[0].axes[4])
            axes_list.append(figure_list[1].axes[1])
            axes_list.append(figure_list[0].axes[5])

        return axes_list
