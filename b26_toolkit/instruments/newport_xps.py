from pylabcontrol.core import Parameter, Instrument
from newportXpsQ8 import driver


class XPSQ8(Instrument):
    """
    This class implements the Newport XPS Q8 motion controller. The class communicates with the device over ethernet.
    Programmer's Manual: ftp://download.newport.com/MotionControl/Current/MotionControllers/XPS%20Unified/ Driver is
    downloaded from Github: https://github.com/H-Plus-Time/newport-xps-q8 -Ziwei 8/24/2020
    """

    _DEFAULT_SETTINGS = Parameter([
        Parameter('IP_address', '192.168.254.254', str, 'IP address of the instrument'),
        Parameter('group', 'Group1', str, 'name of the group'),
        Parameter('enable_motion', False, bool, 'enable or disable motion'),
        Parameter('position', 0.0, float, 'move to the absolute position in mm. allowed range: -12mm to 12mm'),
        Parameter('lower_limit', -12.5, float, 'minimum allowed position not to bump into anything'),
        Parameter('upper_limit', 12.5, float, 'maximum allowed position not to bump into anything'),
    ])

    def __init__(self, name=None, settings=None):

        super(XPSQ8, self).__init__(name, settings)
        try:
            self._connect()
        except Exception as e:
            print(e)
            print('Failed to connect to the controller ' + self.settings['group'])

    def _connect(self, verbose=False):
        try:
            self.xps = driver.XPS()
            self.socketId = self.xps.TCP_ConnectToServer(IP=self.settings['IP_address'], port=5001, timeOut=20)
        except Exception as e:
            print('** ATTENTION **')
            print(e)
        else:
            # self.kill()
            # self.initialize()
            # self.home()
            self.disable_motion()

    def update(self, settings):
        super(XPSQ8, self).update(settings)

        for key, value in settings.items():
            if key == 'enable_motion':
                if value:
                    self.enable_motion()
                else:
                    self.disable_motion()
            elif key == 'position':
                if self.settings['lower_limit'] <= self.settings['position'] and self.settings['upper_limit'] >= self.settings['position']:
                    self.absolute_move(target=self.settings['position'])
                    self.get_current_position()
                else:
                    print('Exceeds limits. No action.')

    def kill(self, verbose=False):
        [errorCode, returnString] = self.xps.GroupKill(self.socketId, self.settings['group'])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)

    def initialize(self, verbose=False):
        [errorCode, returnString] = self.xps.GroupInitialize(self.socketId, self.settings['group'])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)

    def home(self, verbose=False):
        [errorCode, returnString] = self.xps.GroupHomeSearch(self.socketId, self.settings['group'])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)

    def enable_motion(self, verbose=False):
        [errorCode, returnString] = self.xps.GroupMotionEnable(self.socketId, self.settings['group'])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)

    def absolute_move(self, target, verbose=False):
        [errorCode, returnString] = self.xps.GroupMoveAbsolute(self.socketId, self.settings['group'], [target])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)
        return errorCode

    def relative_move(self, displacement, verbose=False):
        [errorCode, returnString] = self.xps.GroupMoveRelative(self.socketId, self.settings['group'], [displacement])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)


    def disable_motion(self, verbose=False):
        [errorCode, returnString] = self.xps.GroupMotionDisable(self.socketId, self.settings['group'])
        if verbose:
            [errorCode2, errorString] = self.xps.ErrorStringGet(self.socketId, errorCode)
            print(errorString)

    def get_current_position(self, verbose = True):
        current_position = self.xps.GroupPositionCurrentGet(self.socketId, self.settings['group'], 1)[1]
        if verbose:
            print(self.settings['group'] + ' pos: ' + str(current_position) + 'mm')
        return current_position

    @property
    def _PROBES(self):
        return None

    def read_probes(self, key):
        pass


class MagnetX(XPSQ8):
    '''
        Same as XPSQ8 except the IP address and group name are fixed
        - Ziwei Qiu 8/24/2020
    '''
    _DEFAULT_SETTINGS = Parameter([
        Parameter('IP_address', '192.168.254.254', ['192.168.254.254'], 'IP address of the instrument'),
        Parameter('group', 'Group1', ['Group1'], 'name of the group'),
        Parameter('enable_motion', False, bool, 'enable or disable motion'),
        Parameter('position', 0.0, float, 'move to the absolute position in mm. allowed range: -12mm to 12mm'),
        Parameter('lower_limit', -12.5, float, 'minimum allowed position not to bump into anything'),
        Parameter('upper_limit', 12.5, float, 'maximum allowed position not to bump into anything'),
    ])

    def __init__(self, name=None, settings=None):
        super(MagnetX, self).__init__()


class MagnetY(XPSQ8):
    '''
        Same as XPSQ8 except the IP address and group name are fixed
        - Ziwei Qiu 8/24/2020
    '''
    _DEFAULT_SETTINGS = Parameter([
        Parameter('IP_address', '192.168.254.254', ['192.168.254.254'], 'IP address of the instrument'),
        Parameter('group', 'Group2', ['Group2'], 'name of the group'),
        Parameter('enable_motion', False, bool, 'enable or disable motion'),
        Parameter('position', 0.0, float, 'move to the absolute position in mm. allowed range: -12mm to 12mm'),
        Parameter('lower_limit', -12.5, float, 'minimum allowed position not to bump into anything'),
        Parameter('upper_limit', 12.5, float, 'maximum allowed position not to bump into anything'),
    ])

    def __init__(self, name=None, settings=None):
        super(MagnetY, self).__init__()


class MagnetZ(XPSQ8):
    '''
        Same as XPSQ8 except the IP address and group name are fixed
        - Ziwei Qiu 8/24/2020
    '''
    _DEFAULT_SETTINGS = Parameter([
        Parameter('IP_address', '192.168.254.254', ['192.168.254.254'], 'IP address of the instrument'),
        Parameter('group', 'Group3', ['Group3'], 'name of the group'),
        Parameter('enable_motion', False, bool, 'enable or disable motion'),
        Parameter('position', 0.0, float, 'move to the absolute position in mm. allowed range: -12mm to 12mm'),
        Parameter('lower_limit', -12.5, float, 'minimum allowed position not to bump into anything'),
        Parameter('upper_limit', 12.5, float, 'maximum allowed position not to bump into anything'),
    ])

    def __init__(self, name=None, settings=None):
        super(MagnetZ, self).__init__()
