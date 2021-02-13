# works with 0.3.65

import numpy as np
import b26_toolkit.scripts.qm_scripts.Helpers as helpers

####################
# The Configuration:
####################

# IF_freq = 50e6
IF_freq = 50e6
LO_freq = 2.87e9  # why do we need to specify these frequency values at all?
# spcm_pulse = -np.load("C:/Users/NV/QuantumMachines/NV_Amir_QUA/spcm_pulse.npy")
# spcm_pulse2 = -np.load("C:/Users/NV/QuantumMachines/NV_Amir_QUA/spcm_pulse2.npy")
rf_switch_extra_time = 40  # extra time is 32ns
delay = 136  # the digital pulse starts earlier than the analog pulse
IF_amp = 0.4
time_of_flight = 400  # setting time_of_flight = 948ns, no need to have a wait time between laser-on and APD.
# but setting shorter time gives more flexibility.

gate1_voltage = -0.2
gate2_voltage = 0.2

pi_time = 88
pi2_time = 44
pi32_time = 132

config = {

    'version': 1,

    'controllers': {

        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  # I
                2: {'offset': 0.0},  # Q
                3: {'offset': 0.0},  # Current 1 for Berry's phase detection
                4: {'offset': 0.0},  # Current 2 for Berry's phase detection
                5: {'offset': 0.01},  # analog gate for ac sensing (offset is assuming 50ohm load)
                # 6: {'offset': 0.0},  # galvo y
                # 7: {'offset': 0.0},  # galvo z
                # 10: {'offset': 0.0},  # AOM for charge probe laser
            },
            'digital_outputs': {},  # laser
            'analog_inputs': {
                1: {'offset': 0.0},  # readout1
                2: {'offset': 0.0},  # readout2
            }
        },
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': LO_freq,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': IF_freq,
            'operations': {
                'const': 'const_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'pi32': 'pi32_pulse',
            },
            'digitalInputs': {
                "  rf_switch": {
                    "port": ("con1", 3),
                    "delay": delay,
                    "buffer": rf_switch_extra_time,
                },
            },

        },

        'laser': {
            'digitalInputs': {
                'switch_in': {
                    'port': ('con1', 1),
                    'delay': 0,
                    # 'delay': 35 * 4,  # Digital pulse arrives later than the analog ones by 140ns
                    'buffer': 0,
                },
            },
            'operations': {
                'trig': 'trig_pulse',
            }
        },

        # 'rf_switch': {
        #     'digitalInputs': {
        #         'laser_in': {
        #             'port': ('con1', 3),
        #             'delay': 0,
        #             'buffer': 0,
        #         },
        #     },
        #     'operations': {
        #         'trig': 'trig_pulse',
        #     }
        # },

        'gate': {
            "singleInput": {"port": ("con1", 5)},
            'intermediate_frequency':0,
            'digitalInputs': {
                'e_field1': {
                    'port': ('con1', 5),
                    'delay': 35 * 4,  # Digital pulse arrives later than the analog ones by 140ns
                    'buffer': 0,
                },
            },
            'operations': {
                'gate1': 'gate_pulse1',
                'gate2': 'gate_pulse2',
            }
        },

        'e_field1': { # only digital outputs
            'digitalInputs': {
                'e_field1': {
                    'port': ('con1', 4),
                    'delay': 35 * 4,  # Digital pulse arrives later than the analog ones by 140ns
                    'buffer': 0,
                },
            },
            'operations': {
                'trig': 'trig_pulse',
            }
        },

        'e_field2': { # only digital outputs
            'digitalInputs': {
                'e_field2': {
                    'port': ('con1', 5),
                    'delay': 35 * 4,  # Digital pulse arrives later than the analog ones by 140ns
                    'buffer': 0,
                },
            },
            'operations': {
                'trig': 'trig_pulse',
            }
        },

        "qe1": {
                    "singleInput": {
                        "port": ("con1", 3)
                    },
                    'intermediate_frequency': 0,
                    'hold_offset': {'duration': 10}
                },

        "qe2": {
                    "singleInput": {
                        "port": ("con1", 4)
                    },
                    'intermediate_frequency': 0,
                    'hold_offset': {'duration': 10}
                },

        'readout1': {

            "outputs": {
                'out1': ("con1", 1),
            },
            'time_of_flight': time_of_flight,
            'smearing': 0,

            'operations': {
                'readout': 'readout',
            },

            # 'outputPulse': [int(arg) for arg in spcm_pulse],

            'digitalInputs': {
                "laser_in": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "singleInput": {
                "port": ("con1", 1)
            },
            'outputPulseParameters': {
                'signalThreshold': -700,
                'signalPolarity': 'Descending',
                'derivativeThreshold': -300,
                'derivativePolarity': 'Descending'
            },

        },

        'readout2': {

            "outputs": {
                'out1': ("con1", 2)
            },

            'time_of_flight': time_of_flight,
            'smearing': 0,
            # 'outputPulse': [int(arg) for arg in spcm_pulse2],

            'operations': {
                'readout': 'readout',
            },

            'digitalInputs': {
                "laser_in": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "singleInput": {
                "port": ("con1", 1)
            },
            'outputPulseParameters': {
                'signalThreshold': -700,
                'signalPolarity': 'Descending',
                'derivativeThreshold': -300,
                'derivativePolarity': 'Descending'
            },
        },

        # 'charge_probe_laser': {
        #     'singleInput': {
        #         'port': ("con1", 10)
        #     },
        #     'intermediate_frequency': 80e6,
        #     'operations': {
        #         'probe': 'probe_pulse',
        #     },
        # },
        #
        # 'galvo_x': {
        #     'singleInput': {
        #         'port': ("con1", 5)
        #     },
        #     'operations': {
        #         'pulse': 'probe_pulse',
        #     },
        # },
        #
        # 'galvo_y': {
        #     'singleInput': {
        #         'port': ("con1", 6)
        #     },
        #     'operations': {
        #         'pulse': 'probe_pulse',
        #     },
        # },
        #
        # 'material': {
        #     'singleInput': {
        #         'port': ("con1", 5)
        #     },
        #     'intermediate_frequency': 3e6,
        #     'operations': {
        #         'probe': 'probe_pulse',
        #     },
        #     'outputs': {
        #         'out1': ("con1", 1)
        #     },
        #     'time_of_flight': 28,
        #     'smearing': 0
        # },

    },

    "pulses": {

        'const_pulse': {
            'operation': "control",
            'length': 120,
            'waveforms': {
                "I": "const_wf",
                "Q": "zero_wf"
            },
            'digital_marker': 'ON'
        },

        'pi_pulse': {
            'operation': "control",
            'length': pi_time,
            'waveforms': {
                "I": "const_wf",
                "Q": "zero_wf"
            },
            'digital_marker': 'ON'
        },

        'pi2_pulse': {  # pi-half pulse
            'operation': "control",
            'length': pi2_time,
            'waveforms': {
                "I": "const_wf",
                # "I": "gauss_wf_0", # this will be better, but not implemented now
                "Q": "zero_wf"
            },
            'digital_marker': 'ON'
        },

        'pi32_pulse': {  # three-pi-half pulse
            'operation': "control",
            'length': pi32_time,
            'waveforms': {
                "I": "const_wf",
                # "I": "gauss_wf_0", # this will be better, but not implemented now
                "Q": "zero_wf"
            },
            'digital_marker': 'ON'
        },

        'trig_pulse': { # used for digital ports
            'operation': "control",
            'length': 1600,
            'digital_marker': 'ON'
        },

        'gate_pulse1': {
                        'operation': "control",
                        'length': 500,
                        'waveforms': {'single':"const_gate1"},
                        'digital_marker': 'ON'
                },

        'gate_pulse2': {
                        'operation': "control",
                        'length': 500,
                        'waveforms': {'single':"const_gate2"},
                        'digital_marker': 'ON'
                        },

        'readout': {
            'operation': "measurement",
            "length": 252,
            "waveforms": {
                "single": "zero_wf"  # fake!
            },
            'digital_marker': 'ON',
        },

        'probe_pulse': {
            'operation': "measurement",
            'length': 1000,
            'waveforms': {
                "single": "const_wf",
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',

            },
            'digital_marker': 'ON',
        },

    },

    'waveforms': {

        "const_wf": {
            "type": "constant",
            "sample": IF_amp
        },

        "const_gate1": {"type": "constant",
                        "sample": gate1_voltage
                        },

        "const_gate2": {"type": "constant",
                        "sample": gate2_voltage
                        },

        "zero_wf": {"type": "constant",
                    "sample": 0.0
                    },

        'gauss_wf_0': {
            'type': 'arbitrary',
            'samples': helpers.gauss(0.36, 0.0, 10.0, 60),
        },

    },

    "digital_waveforms": {

        "ON": {
            "samples": [(1, 0)]
        }
    },

    "mixers": {  # post-shape the pulse to compensate for imperfections
        'mixer_qubit': [
            {'intermediate_frequency': IF_freq, 'lo_frequency': LO_freq, 'correction': [1, 0, 0, 1]}
        ]
    },

    'integration_weights': {

        'integW1': {
            'cosine': [1.0] * int(1000 / 4),
            'sine': [0.0] * int(1000 / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(1000 / 4),
            'sine': [1.0] * int(1000 / 4),
        },

    },

}
