import visa
import pyvisa.errors
from pylabcontrol.core import Parameter, Instrument

class YokogawaGS200(Instrument):
    """
        This class implements the very basic functions of Yokogawa DC Voltage/Current Source GS200.
        The class communicates with the device over USB or GPIB using pyvisa.
        Manual: https://cdn.tmi.yokogawa.com/1/6218/files/IMGS210-01EN.pdf
        - Ziwei Qiu 10/3/2020
    """

    _DEFAULT_SETTINGS = Parameter([
        Parameter('VISA_address', 'USB0::0x0B21::0x0039::91U225587::0::INSTR',
                  ['USB0::0x0B21::0x0039::91U225587::0::INSTR'],
                  'VISA address of the instrument'),
        Parameter('enable_output', False, bool, 'Sets the output state (on/off). '),
        Parameter('source', 'VOLT', ['VOLT', 'CURR'], 'Sets the source function (voltage/current)'),
        Parameter('level', 0.0, float, 'Sets the source level in V/A in terms of the most appropriate automatic range'),
        Parameter('voltage_limit', 1.0, float,
                  'Sets the voltage limiter level in V (between 1V and 30V) when the source is current'),
        Parameter('current_limit', 1E-3, float,
                  'Sets the current limiter level in A (between 1E-3A and 200E-3A) when the source is voltage'),
    ])

    def __init__(self, name=None, settings=None):

        super(YokogawaGS200, self).__init__(name, settings)

        try:
            self._connect()
        except pyvisa.errors.VisaIOError:
            print('No Yokogawa GS200 Detected!. Check that you are using the correct communication type.')
            raise
        except Exception as e:
            raise(e)

    def _connect(self):
        rm = visa.ResourceManager()
        self.srs = rm.open_resource(self.settings['VISA_address'])
        self.srs.query('*IDN?')
        print('Connected: ' + self.srs.query('*IDN?'))

    @property
    def is_connected(self):
        try:
            self.srs.query('*IDN?')  # arbitrary call to check connection, throws exception on failure to get response
            return True
        except pyvisa.errors.VisaIOError:
            return False

    def update(self, settings):
        super(YokogawaGS200, self).update(settings)

        if self._settings_initialized:
            for key, value in settings.items():
                if key == 'VISA_address':
                    self._connect()
                else:
                    if key == 'enable_output':
                        if value:
                            self.srs.write(':OUTP 1')
                        else:
                            self.srs.write(':OUTP 0')
                        print('The output is ' + self.srs.query(':OUTP?'))
                    elif key == 'source':
                        self.srs.write(':SOUR:FUNC ' + str(value))
                        print('The function being used is ' + self.srs.query(':SOUR:FUNC?'))
                    elif key == 'level':
                        self.srs.write(':SOUR:LEV:AUTO ' + str(value))
                        print('The source level that is being used is ' + self.srs.query(':SOUR:LEV:AUTO?'))
                    elif key == 'voltage_limit':
                        if value < 1:
                            value = 1
                        elif value > 30:
                            value = 30
                        self.srs.write(':SOUR:PROT:VOLT ' + str(value))
                        print('The voltage limiter level that is being used is' + self.srs.query(
                            ':SOUR:PROT:VOLT?'))
                    elif key == 'current_limit':
                        if value < 1E-3:
                            value = 1E-3
                        elif value > 200E-3:
                            value = 200E-3
                        self.srs.write(':SOUR:PROT:CURR ' + str(value))
                        print('The current limiter level that is being used is' + self.srs.query(
                            ':SOUR:PROT:CURR?'))

    @property
    def _PROBES(self):
        return None

    def read_probes(self, key):
        pass

if __name__ == '__main__':
    dc_source = YokogawaGS200()