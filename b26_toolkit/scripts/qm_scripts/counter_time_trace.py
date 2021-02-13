from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import time
from pylabcontrol.core import Script, Parameter
from b26_toolkit.scripts.qm_scripts.Configuration import config
import numpy as np
from collections import deque
from b26_toolkit.plotting.plots_1d import plot_counts_vs_pos,update_counts_vs_pos


class CounterTimeTrace(Script):
    _DEFAULT_SETTINGS = [
        Parameter('IP_address', 'automatic', ['140.247.189.191', 'automatic'],
                  'IP address of the QM orchestrator'),
        Parameter('integration_time', 0.3, float, 'Time per data point (s)'),
        Parameter('total_int_time', 10.0, float, 'Total time to integrate (s) (if -1 then it will go indefinitely)')
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
                print(e)
        else:
            try:
                self.qmm = QuantumMachinesManager(host=self.settings['IP_address'])
            except Exception as e:
                print(e)

    def _function(self):
        self.meas_len = 40000  # 50us
        buffer_num = int(0.8 * self.settings['integration_time'] * 1e9 / self.meas_len)
        res_len = int(np.max([int(self.meas_len/200), 2])) # result length needs to be at least 2

        try:
            self.qm = self.qmm.open_qm(config)
        except Exception as e:
            print(e)
        else:
            with program() as counter_time_trace:
                result1 = declare(int, size=res_len) # note that this says the measured counts saturate at 5000kcps
                counts1 = declare(int)
                result2 = declare(int, size=res_len)
                counts2 = declare(int)

                counts1_st = declare_stream()
                counts2_st = declare_stream()

                with infinite_loop_():
                    play('trig', 'laser', duration=int(self.meas_len / 4))
                    measure("readout", "readout1", None,
                            time_tagging.raw(result1, self.meas_len, targetLen=counts1))
                    measure("readout", "readout2", None,
                            time_tagging.raw(result2, self.meas_len, targetLen=counts2))

                    save(counts1, counts1_st)
                    save(counts2, counts2_st)

                with stream_processing():
                    counts1_st.buffer(buffer_num).save("live_counts1")
                    counts2_st.buffer(buffer_num).save("live_counts2")

            with program() as job_stop:
                play('trig', 'laser', duration=10)

            self._qm_execution(counter_time_trace, job_stop)
            self._abort = True
            # the following three commands will still run after self._abort is set to be True
            self.data['counts1'] = list(self.data['counts1'])
            self.data['counts2'] = list(self.data['counts2'])
            self.data['time'] = list(self.data['time'])


    def _qm_execution(self, qua_program, job_stop):
        self.data = {'counts1': deque(), 'counts2': deque(), 'time': deque()}

        job = self.qm.execute(qua_program)
        counts1_handle = job.result_handles.get("live_counts1")
        counts2_handle = job.result_handles.get("live_counts2")
        counts1_handle.wait_for_values(1)
        counts2_handle.wait_for_values(1)

        start_time = time.time()

        while counts1_handle.is_processing() and counts2_handle.is_processing():

            try:
                counts1 = counts1_handle.fetch_all()
                counts2 = counts2_handle.fetch_all()
            except Exception as e:
                print(e)
            else:
                self.data['time'].append(time.time() - start_time)

                self.data['counts1'].append(counts1.mean() * 1e6 / self.meas_len)
                self.data['counts2'].append(counts2.mean() * 1e6 / self.meas_len)

            if self._abort:
                # job.halt()
                self.qm.execute(job_stop)
                break

            if self.settings['total_int_time'] == -1:
                self.progress = 50.
                self.updateProgress.emit(int(self.progress))
            else:
                self.progress = 100. * (time.time() - start_time) / self.settings['total_int_time']
                self.updateProgress.emit(int(self.progress))

            if self.progress > 100:
                self._abort = True
            else:
                time.sleep(self.settings['integration_time'])

    # def plot(self, figure_list):
    #     super(CounterTimeTrace, self).plot([figure_list[1]])
        # super(CounterTimeTrace, self).plot([figure_list[0], figure_list[1]])
    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data

        axes_list[0].clear()
        pos = np.array(data['time'])
        array1 = np.array(data['counts1'])
        array2 = np.array(data['counts2'])
        data_length = np.min([len(data['counts1']), len(data['counts2']), len(data['time'])])

        if data_length > 0:
            try:
                axes_list[0].plot(pos[0:data_length], array1[0:data_length], '.', color ='#1f77b4',label='APD 1', linestyle='dashed')
                axes_list[0].plot(pos[0:data_length], array2[0:data_length], '.', color ='#ff7f0e', label='APD 2', linestyle='dashed')
                axes_list[0].plot(pos[0:data_length], array1[0:data_length] + array2[0:data_length], '.', color ='r', label='Total', linestyle='solid')
                axes_list[0].set_xlabel('time [sec]', fontsize=9.5)
                axes_list[0].set_ylabel('Counts [kcps]', fontsize=9.5)
                axes_list[0].set_title('Counter Time Trace', fontsize=10)
                # axes_list[0].legend()
                axes_list[0].legend(loc='upper center', bbox_to_anchor=(1.1, 1.2), ncol=1, fontsize=9)
            except Exception as e:
                print('** ATTENTION **')
                print(e)
        # plot_counts_vs_pos(axes_list[0], np.array(data['counts1']), np.array(data['time']), x_label='time [sec]')

    def _update_plot(self, axes_list, data=None):

        if data is None:
            data = self.data

        if len(data['counts1']) > 0:
            try:
                pos = np.array(data['time'])
                array1 = np.array(data['counts1'])
                array2 = np.array(data['counts2'])
                data_length = np.min([len(data['counts1']), len(data['counts2']), len(data['time'])])

                axes_list[0].lines[0].set_ydata(array1[0:data_length])
                axes_list[0].lines[0].set_xdata(pos[0:data_length])
                axes_list[0].lines[1].set_ydata(array2[0:data_length])
                axes_list[0].lines[1].set_xdata(pos[0:data_length])
                axes_list[0].lines[2].set_ydata(array1[0:data_length]+array2[0:data_length])
                axes_list[0].lines[2].set_xdata(pos[0:data_length])

                axes_list[0].relim()
                axes_list[0].autoscale_view()
            except Exception as e:
                print('** ATTENTION **')
                print(e)

        # if data:
        #   update_counts_vs_pos(axes_list[0], np.array(data['counts1']), np.array(data['time']))

    def get_axes_layout(self, figure_list):
        """
            returns the axes objects the script needs to plot its data
            this overwrites the default get_axis_layout in PyLabControl.src.core.scripts
            Args:
                figure_list: a list of figure objects
            Returns:
                axes_list: a list of axes objects

        """
        # axes_list = []
        # if self._plot_refresh is True:
        #     for fig in figure_list:
        #         fig.clf()
        #     axes_list.append(figure_list[1].add_subplot(111))  # axes_list[0]
        # else:
        #     axes_list.append(figure_list[1].axes[0])
        # return axes_list
        if self._plot_refresh is True:
            for fig in figure_list:
                fig.clf() # this is to make the plot refresh faster
        new_figure_list = [figure_list[1]]
        return super(CounterTimeTrace, self).get_axes_layout(new_figure_list)



if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'CounterTimeTrace': 'CounterTimeTrace'}, script, instr)

    print(script)
    print(failed)
    print(instr)
