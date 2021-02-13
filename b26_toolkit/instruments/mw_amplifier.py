import visa
import pyvisa.errors
from pylabcontrol.core import Parameter, Instrument
import time


class MWAmplifier(Instrument):
    """
        This class implements the AR Microwave Amplifier 30S1G6. The class commuicates with the
        device over USB.
        Manual: https://www.arworld.us/post/opMan/30S1G6-rev%20A.pdf
        -Ziwei 8/16/2020
    """

    _DEFAULT_SETTINGS = Parameter([
        Parameter('VISA_address', 'USB0::0x0547::0x1B58::0349484::0::INSTR', ['USB0::0x0547::0x1B58::0349484::0::INSTR'],
                  'VISA address of the instrument'),
        Parameter('power_on', False, bool, 'Control the power on/off state of the amplifier'),
        Parameter('gain', 0.8, float, 'Set the gain level of the amplifier with 4095 steps of resolution (from 0 to 1)')
    ])

    def __init__(self, name=None, settings=None):

        super(MWAmplifier, self).__init__(name, settings)

        # XXXXX MW ISSUE = START =========================================== Issue where visa.ResourceManager() takes
        # 4 minutes no longer happens after using pdb to debug (??? not sure why???)
        try:
            self._connect()
        except pyvisa.errors.VisaIOError:
            print('No Microwave Amplifier Detected! Check that you are using the correct communication type')
            # raise
        except Exception as e:
            print(e)
            print('Please control the MW amplifier manually')
            # raise e

    def _connect(self, verbose=False):
        rm = visa.ResourceManager()
        self.srs = rm.open_resource(self.settings['VISA_address'])
        if verbose:
            print('MW amplifier 30S1G6 connected: ' + self.srs.query('*IDN?'))

    @property
    def is_connected(self):
        try:
            self.srs.query('*IDN?')  # arbitrary call to check connection, throws exception on failure to get response
            return True
        except pyvisa.errors.VisaIOError:
            return False

    def update(self, settings):
        """
        Updates the internal settings of the MicrowaveGenerator, and then also updates physical parameters such as
        frequency, amplitude, modulation type, etc in the hardware
        Args:
            settings: a dictionary in the standard settings format

        """

        super(MWAmplifier, self).update(settings)

        for key, value in settings.items():
            if key == 'VISA_address':
                time.sleep(0.1)
                self._connect()

            elif key == 'power_on':
                if value:
                    self.srs.write('P1')
                else:
                    self.srs.write('P0')

            elif key == 'gain':
                if value > 1.0:
                    value = 1.0
                elif value < 0.0:
                    value = 0.0
                self.srs.write('G'+str(int(value*4095)))

    @property
    def _PROBES(self):
        return None

    def read_probes(self, key):
        pass



