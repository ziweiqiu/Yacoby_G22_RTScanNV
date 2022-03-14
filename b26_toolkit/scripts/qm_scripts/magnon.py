from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import time
from b26_toolkit.plotting.plots_1d import plot_qmsimulation_samples
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config
from b26_toolkit.scripts.optimize import OptimizeNoLaser
import numpy as np
from b26_toolkit.instruments import SGS100ARFSource, R8SMicrowaveGenerator
from b26_toolkit.data_processing.fit_functions import fit_exp_decay, exp_offset
from qm.qua import frame_rotation as z_rot

wait_pulse_artifect = 68 / 4
# analog_digital_align_artifect = 0
analog_digital_align_artifect = int(140 / 4)

# The following script still has memory problem.
class ACStarkShift(Script):
    """
        This script implements AC Stark Shift experiments. Tau is swept.
        Due to limited OPX memory, at most output XY8-2 sequences.
        Modulating field is controlled by DIGITAL pulses, named as 'e_field2' (digital 5).
        Rhode Schwarz SGS100A is used and it has IQ modulation.
        Tau is the total evolution time between the center of the first and last pi/2 pulses.

        - Ziwei Qiu 1/16/2022
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
        Parameter('mw_pulses_2', [
            Parameter('mw_frequency', 2.87e9, float, 'LO frequency in Hz'),
            Parameter('mw_power', -10.0, float, 'RF power in dBm'),
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

        Parameter('sensing_type', 'cosine', ['cosine', 'sine', 'both'], 'choose the sensing type'),
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
        Parameter('simulation_duration', 50000, int, 'duration of simulation in ns')
    ]
    _INSTRUMENTS = {'mw_gen_iq': SGS100ARFSource,
                    'microwave_generator': R8SMicrowaveGenerator,
                    }
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

        self.instruments['microwave_generator']['instance'].update({'amplitude': self.settings['mw_pulses_2']['mw_power']})
        self.instruments['microwave_generator']['instance'].update({'frequency': self.settings['mw_pulses_2']['mw_frequency']})
        self.instruments['microwave_generator']['instance'].update({'power_mode': 'CW'})
        self.instruments['microwave_generator']['instance'].update({'freq_mode': 'CW'})
        self.instruments['microwave_generator']['instance'].update({'display_on': False})
        self.instruments['microwave_generator']['instance'].update({'enable_output': True})
        print('Turned on RF generator SMB100A (no IQ).')

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
        super(ACStarkShift, self).plot([figure_list[0], figure_list[1]])

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
            axes_list[1].clear()
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

