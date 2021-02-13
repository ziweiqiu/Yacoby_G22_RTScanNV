from pylabcontrol.core import Script, Parameter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from numpy.linalg import norm
import time
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

class PDDESEEM(Script):
    """
        This script calculates the expected PDD signal when the magnetic field angle is clse to 90 deg.
        ESEEM effects from 15N and 13C are taken into account.
        No other noise is considered. No electric field is present.
        - Ziwei Qiu 10/14/2020
    """
    _DEFAULT_SETTINGS = [
        Parameter('bias_fields', [
            Parameter('B_magnitude',  81.0, float, 'bias magnetic field magnitude in [Gauss]'),
            Parameter('phi_B', 0.0, float, 'B field azimuth angle in [degree], measured from x'),
            Parameter('E_magnitude', 0.0, float, 'bias electric field magnitude in [V/um]'),
            Parameter('theta_E', 0.0, float, 'E field zenith angle in [degree], measured from z'),
            Parameter('phi_E', 0.0, float, 'E field azimuth angle in [degree], measured from x')
        ]),
        Parameter('B_zenith_angles', [
            Parameter('min_angle', 89, float, 'minimum B field zenith angle (theta_B, measured from z) in degree'),
            Parameter('max_angle', 91, float, 'maximum B field zenith angle (theta_B, measured from z) in degree'),
            Parameter('angle_step', 0.1, float, 'angle step increment in degree')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 200, float, 'minimum time between the two pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between the two pi pulses'),
            Parameter('time_step', 100, float, 'time step increment of time between the two pi pulses (in ns)')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'CPMG', 'XY4', 'XY8'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.'),
        ]),
        Parameter('B_carbon_rms', 50, float, 'specify the carbon fluctuating field RMS in Gauss'),
    ]
    _INSTRUMENTS = {}
    _SCRIPTS = {}

    def __init__(self, instruments=None, scripts=None, name=None, settings=None, log_function=None, data_path=None):

        Script.__init__(self, name, settings=settings, instruments=instruments, scripts=scripts,
                        log_function=log_function, data_path=data_path)


    def _function(self):

        start = time.time()
        def NV_Hamiltonian(B, theta_B, phi_B, E, theta_E, phi_E):
            Bx = B * np.sin(theta_B * np.pi / 180) * np.cos(phi_B * np.pi / 180)
            By = B * np.sin(theta_B * np.pi / 180) * np.sin(phi_B * np.pi / 180)
            Bz = B * np.cos(theta_B * np.pi / 180)
            Ex = E * np.sin(theta_E * np.pi / 180) * np.cos(phi_E * np.pi / 180)
            Ey = E * np.sin(theta_E * np.pi / 180) * np.sin(phi_E * np.pi / 180)
            Ez = E * np.cos(theta_E * np.pi / 180)

            me = 0.00054858  # electron mass in amu
            mp = 1.007  # proton mass in amu
            miuB = 1.4 * 2 * np.pi  # Bohr magneton in rad/s/Gauss
            miuN = miuB * me / mp  # Nuclear magneton in rad/s/Gauss
            gNV = 2
            g_matrix = gNV * np.identity(3)  # g-factor matrix of NV spin
            d_long = 35E-4 * 2 * np.pi  # longitundinal coupling to E field in [Mrad/s/V/um]
            d_trans = 17E-2 * 2 * np.pi  # transverse coupling to E field in [Mrad/s/V/um]
            D = 2.87 * 1000 * 2 * np.pi  # zero field splitting in [Mrad/s]
            A_N = np.array([[3.36, 0, 0], [0, 3.36, 0], [0, 0, 3.03]]) * 2 * np.pi

            gN15 = -0.56637768  # (*N15 nuclear g-factor*) (http://easyspin.org/easyspin/documentation/isotopetable.html)
            gN = gN15 * np.identity(3)

            SxNV = Qobj([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
            SyNV = Qobj([[0, -1.0j, 0], [1.0j, 0, -1.0j], [0, 1.0j, 0]]) / np.sqrt(2)
            SzNV = Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
            IxN = sigmax() / 2
            IyN = sigmay() / 2
            IzN = sigmaz() / 2
            Sx = tensor([qeye(2), SxNV])
            Sy = tensor([qeye(2), SyNV])
            Sz = tensor([qeye(2), SzNV])
            Ix = tensor([IxN, qeye(3)])
            Iy = tensor([IyN, qeye(3)])
            Iz = tensor([IzN, qeye(3)])

            H_ZFS = D * Sz * Sz
            H_Zeeman = miuB * (g_matrix[0, 0] * Bx * Sx + g_matrix[1, 1] * By * Sy + g_matrix[2, 2] * Bz * Sz)
            H_stark_long = d_long * Ez * Sz * Sz
            H_stark_trans = d_trans * Ex * (Sy * Sy - Sx * Sx) + d_trans * Ey * (Sx * Sy + Sy * Sx)
            H_hyperfine = A_N[0, 0] * Ix.trans() * Sx + A_N[1, 1] * Iy.trans() * Sy + A_N[2, 2] * Iz.trans() * Sz
            H_NZeeman = miuN * (gN[0, 0] * Bx * Ix + gN[1, 1] * By * Iy + gN[2, 2] * Bz * Iz)
            H0 = H_ZFS + H_Zeeman + H_stark_long + H_stark_trans + H_hyperfine + H_NZeeman  # The Full Hamiltonian

            return H0

        # bias fields:
        B = self.settings['bias_fields']['B_magnitude']
        phi_B = self.settings['bias_fields']['phi_B']
        E = self.settings['bias_fields']['E_magnitude']  # magnitude of E-field in [V/um]
        theta_E = self.settings['bias_fields']['theta_E']  # misalignment zenith angle in [deg], measured from z
        phi_E = self.settings['bias_fields'][
            'phi_E']  # azimuth angle in [deg] in the perpendicular plan, measured from x
        Bn = self.settings['B_carbon_rms']

        number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']

        if self.settings['decoupling_seq']['type'] == 'spin_echo' or self.settings['decoupling_seq'][
            'type'] == 'CPMG':
            num_of_evolution_blocks = 1 * number_of_pulse_blocks
        elif self.settings['decoupling_seq']['type'] == 'XY4':
            num_of_evolution_blocks = 4 * number_of_pulse_blocks
        else: # XY8
            num_of_evolution_blocks = 8 * number_of_pulse_blocks

        taulist_0 = np.arange(self.settings['tau_times']['min_time']/1000, self.settings['tau_times']['max_time']/1000,
                             self.settings['tau_times']['time_step']/1000)  # in us
        taulist = taulist_0 / num_of_evolution_blocks

        theta_B_list = np.arange(self.settings['B_zenith_angles']['min_angle'], self.settings['B_zenith_angles']['max_angle'],
                                 self.settings['B_zenith_angles']['angle_step'])
        num_of_theta_B = len(theta_B_list)
        num_of_tau = len(taulist)

        Ix = tensor([sigmax() / 2, qeye(3)])
        Iy = tensor([sigmay() / 2, qeye(3)])
        Iz = tensor([sigmaz() / 2, qeye(3)])

        Ix_exp = np.zeros([6, num_of_theta_B])
        Iy_exp = np.zeros([6, num_of_theta_B])
        Iz_exp = np.zeros([6, num_of_theta_B])
        theta_I = np.zeros([6, num_of_theta_B])
        Freq = np.zeros([6, num_of_theta_B])

        SxNV = Qobj([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        SyNV = Qobj([[0, -1.0j, 0], [1.0j, 0, -1.0j], [0, 1.0j, 0]]) / np.sqrt(2)
        SzNV = Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        Sx = tensor([qeye(2), SxNV])
        Sy = tensor([qeye(2), SyNV])
        Sz = tensor([qeye(2), SzNV])
        Sx_exp = np.zeros([6, num_of_theta_B])
        Sy_exp = np.zeros([6, num_of_theta_B])
        Sz_exp = np.zeros([6, num_of_theta_B])

        i = 0
        for theta_B in theta_B_list:
            H0 = NV_Hamiltonian(B, theta_B, phi_B, E, theta_E, phi_E)
            eigenstates = H0.eigenstates()
            V = eigenstates[1][:]
            Energies = H0.eigenenergies()
            Ix_exp[:, i] = expect(Ix, V).transpose()
            Iy_exp[:, i] = expect(Iy, V).transpose()
            Iz_exp[:, i] = expect(Iz, V).transpose()
            theta_I[:, i] = np.sign(Ix_exp[:, i]) * np.arctan(Iz_exp[:, i] / Ix_exp[:, i]) * 180 / np.pi
            Freq[:, i] = Energies / (2 * np.pi)
            Sx_exp[:, i] = expect(Sx, V).transpose()
            Sy_exp[:, i] = expect(Sy, V).transpose()
            Sz_exp[:, i] = expect(Sz, V).transpose()
            i += 1

        # Calculate the hyperfine splittings (in MHz)
        hf_plus = Freq[5, :] - Freq[4, :]
        hf_minus = Freq[3, :] - Freq[2, :]
        hf_zero = Freq[1, :] - Freq[0, :]

        hf_zero_vector = np.zeros([len(hf_zero), 3])
        hf_zero_vector[:, 0] = np.cos(theta_I[0, :] * np.pi / 180)
        hf_zero_vector[:, 2] = np.sin(theta_I[0, :] * np.pi / 180)

        hf_minus_vector = np.zeros([len(hf_minus), 3])
        hf_minus_vector[:, 0] = np.cos(theta_I[2, :] * np.pi / 180)
        hf_minus_vector[:, 2] = np.sin(theta_I[2, :] * np.pi / 180)

        hf_plus_vector = np.zeros([len(hf_plus), 3])
        hf_plus_vector[:, 0] = np.cos(theta_I[4, :] * np.pi / 180)
        hf_plus_vector[:, 2] = np.sin(theta_I[4, :] * np.pi / 180)

        # The coupling strength to the carbon nuclear spin bath
        Sx_exp_m = (Sx_exp[2, :] + Sx_exp[3, :] - Sx_exp[0, :] - Sx_exp[1, :]) / 2
        Sy_exp_m = (Sy_exp[2, :] + Sy_exp[3, :] - Sy_exp[0, :] - Sy_exp[1, :]) / 2
        Sz_exp_m = (Sz_exp[2, :] + Sz_exp[3, :] - Sz_exp[0, :] - Sz_exp[1, :]) / 2
        Stot_minus = np.sqrt(Sx_exp_m ** 2 + Sy_exp_m ** 2 + Sz_exp_m ** 2)

        Sx_exp_p = (Sx_exp[4, :] + Sx_exp[5, :] - Sx_exp[0, :] - Sx_exp[1, :]) / 2
        Sy_exp_p = (Sy_exp[4, :] + Sy_exp[5, :] - Sy_exp[0, :] - Sy_exp[1, :]) / 2
        Sz_exp_p = (Sz_exp[4, :] + Sz_exp[5, :] - Sz_exp[0, :] - Sz_exp[1, :]) / 2
        Stot_plus = np.sqrt(Sx_exp_p ** 2 + Sy_exp_p ** 2 + Sz_exp_p ** 2)

        # gyromagnetic ratio of nuclear spins
        miuB = 1.4 * 2 * np.pi  # Bohr magneton in rad/s/Gauss
        gama_N14 = 3.077 * 0.0001 * 2 * np.pi
        gama_N15 = -4.316 * 0.0001 * 2 * np.pi
        gama_C13 = 10.7084 * 0.0001 * 2 * np.pi
        gNV = 2
        gama_NV = miuB * gNV

        pdd_minus = np.zeros((len(theta_B_list), len(taulist)))
        pdd_plus = np.zeros((len(theta_B_list), len(taulist)))

        # calculate minus state pdd
        prefactor_list = []
        for ii in np.arange(num_of_theta_B):
            prefactor = (norm(np.cross(hf_zero_vector[ii, :], hf_minus_vector[ii, :]))) ** 2
            prefactor_list.append(prefactor)

            jj = 0
            for tau in taulist:
                signal = 1 - prefactor * ((np.sin(2 * np.pi * hf_zero[ii] * tau / 4)) ** 2) * (
                            (np.sin(2 * np.pi * hf_minus[ii] * tau / 4)) ** 2)
                LamorFreq = gama_C13 * B  # in Mrad/sec
                carbon_revival = np.exp(-8 * (gama_C13 * Bn * Stot_minus[ii] / LamorFreq) ** 2 * ((np.sin(LamorFreq * tau / 4)) ** 4))
                # see Sushkov PRL paper supplementary
                signal = (signal - 0.5) * carbon_revival + 0.5
                pdd_minus[[ii], [jj]] = signal
                jj += 1

        # calculate plus state pdd
        prefactor_list = []
        for ii in np.arange(num_of_theta_B):

            prefactor = (norm(np.cross(hf_zero_vector[ii, :], hf_plus_vector[ii, :]))) ** 2
            prefactor_list.append(prefactor)

            jj = 0
            for tau in taulist:
                signal = 1 - prefactor * ((np.sin(2 * np.pi * hf_zero[ii] * tau / 4)) ** 2) * (
                        (np.sin(2 * np.pi * hf_plus[ii] * tau / 4)) ** 2)
                LamorFreq = gama_C13 * B  # in Mrad/sec
                carbon_revival = np.exp(-8 * (gama_C13 * Bn * Stot_plus[ii] / LamorFreq) ** 2 * ((np.sin(LamorFreq * tau / 4)) ** 4))

                signal = (signal - 0.5) * carbon_revival + 0.5
                pdd_plus[[ii], [jj]] = signal
                jj += 1

        end = time.time()
        print('PDD calculation took {:.1f}s.'.format(end - start))
        self.log('PDD calculation took {:.1f}s.'.format(end - start))
        self.data = {'pdd_minus': pdd_minus, 'pdd_plus': pdd_plus, 'taulist': taulist_0, 'theta_B_list': theta_B_list}

    def _plot(self, axes_list, data=None):
        """
            Plots the confocal scan image
            Args:
                axes_list: list of axes objects on which to plot the galvo scan on the first axes object
                data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """

        if data is None:
            data = self.data


        extent = [data['taulist'][0], data['taulist'][-1], data['theta_B_list'][-1], data['theta_B_list'][0]]
        colornorm = mpl.colors.Normalize(vmin=0, vmax=1.0)

        if 'pdd_minus' in data.keys():
            axes_list[0].clear()
            implot = axes_list[0].imshow(data['pdd_minus'], cmap='seismic', extent=extent, norm=colornorm, aspect=1)
            # axes_list[0].set_xlabel(r'$\tau$ ($\mu$s)')
            axes_list[0].axhline(y=90.0, color='w', ls='--', lw=1.3)
            axes_list[0].set_ylabel(r'$\theta_B$ (deg)', fontsize=13)
            divider = make_axes_locatable(axes_list[0])
            cax = divider.append_axes("right", size="2.2%", pad=0.1)
            fig = axes_list[0].get_figure()
            cbar = fig.colorbar(implot, cax=cax, orientation='vertical')
            # cbar.set_label("pdd signal", labelpad=22, y=0.5, rotation=-90, size=15)
            axes_list[0].set_aspect('auto')
            axes_list[0].set_title(
                '|B|={:2.1f}G, {:s} {:d} block(s)\n'.format(self.settings['bias_fields']['B_magnitude'],
                                                            self.settings['decoupling_seq']['type'],
                                                            self.settings['decoupling_seq'][
                                                                'num_of_pulse_blocks']) + r'$|0\rangle$ and $|-\rangle$',
                fontsize=13)

        if 'pdd_plus' in data.keys():
            axes_list[1].clear()
            implot = axes_list[1].imshow(data['pdd_plus'], cmap='seismic', extent=extent, norm=colornorm, aspect=1)
            axes_list[1].axhline(y=90.0, color='w', ls='--', lw=1.3)
            axes_list[1].set_xlabel(r'$\tau$ ($\mu$s)', fontsize=13)
            axes_list[1].set_ylabel(r'$\theta_B$ (deg)', fontsize=13)
            divider = make_axes_locatable(axes_list[1])
            cax = divider.append_axes("right", size="2.2%", pad=0.1)
            fig = axes_list[1].get_figure()
            cbar = fig.colorbar(implot, cax=cax, orientation='vertical')
            # cbar.set_label("pdd signal", labelpad=22, y=0.5, rotation=-90, size=15)
            axes_list[1].set_aspect('auto')
            axes_list[1].set_title(r'$|0\rangle$ and $|+\rangle$', fontsize=13)

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
        else:
            axes_list.append(figure_list[0].axes[0])
            axes_list.append(figure_list[0].axes[1])
        return axes_list









