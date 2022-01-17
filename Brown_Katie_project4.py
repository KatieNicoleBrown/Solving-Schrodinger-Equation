# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
# Based off code from NM4P


def sch_eqn(nspace, ntime, tau, method='crank', length=200, potential=[],
            wparam=[10, 0, 0.5]):
    """
    Solves the 1-D time-dependent Schroedinger Equation using either the FTCS
    or Crank Nicolson scheme and calculates the total probability of finding
    the particle at each time.
    :param nspace: number of spacial grid points (positive integer)
    :param ntime: number of time points (positive integer)
    :param tau: time-step (positive integer or float)
    :param method: method of solving, either foward-time-centre-step ('ftcs')
    or Crank-Nicolson ('crank')
    :param length: size of spatial grid (grid extends from -L/2 to L/2)
    (float or int)
    :param potential: spatial index values at which potential is set to 1
    (1-D array)
    :param wparam: parameters for initial condition [sigma0, x0, k0]
    (3-element list)
    :return:
    psi_sol: values of wave equation solution on space-time grid (2-D array)
    x: coordinates of spacial grid points (1-D array)
    t: times at which solutions were calculates (1-D array)
    probs: total probability at each time step (1-D array)
    """

    # Assertions to check that all input parameters are of correct type
    assert isinstance(nspace, int) and nspace > 0, 'nspace must be a positive ' \
                                                   'integer'
    assert isinstance(ntime, int) and ntime > 0, 'ntime must be a positive ' \
                                                 'integer'
    assert isinstance(tau, float) or isinstance(tau, int), 'tau must be a ' \
                                                           'float or integer'
    assert tau > 0, 'time-step must be positive'
    assert method == 'ftcs' or method == 'crank', "Method must be either ftcs" \
                                                  " or crank"
    assert isinstance(length, int) or isinstance(length, float) and length > 0, \
        "length must be a float/integer"
    assert isinstance(length, float) or isinstance(length, int), \
        'length must be a float or int'
    assert length > 0, 'length must be positive'
    assert isinstance(potential, list), 'potential must be a list'
    assert len(potential) <= nspace, 'potential cannot have more elements ' \
                                     'than the number of spatial points.'
    assert all(x < nspace for x in potential), \
        "spatial indices of potential must lie within spatial grid"
    assert isinstance(wparam, list) and len(wparam) == 3, \
        'wparams must be a 3-element list'
    assert wparam[0] > 0, "sigma0 must be positive"
    assert -length / 2 <= wparam[1] <= length / 2, \
        "x0 must lie within +/- length/2"
    assert wparam[2] > 0, "k0 must be positive"

    # Define constants and parameters
    i_imag = 1j  # Imaginary i
    h = length / (nspace - 1)  # Grid size
    x = np.arange(nspace) * h - length / 2.  # Coordinates  of grid points
    t = np.arange(ntime + 1) * tau  # Time points
    h_bar = 1.  # Natural units
    mass = 1 / 2.  # Natural units
    potent = np.zeros(nspace)  # Initialize potential array
    for i in potential:
        potent[i] = 1

        # * Set up the Hamiltonian operator matrix
    ham = np.zeros((nspace, nspace))  # Set all elements to zero
    coeff = -h_bar ** 2 / (2 * mass * h ** 2)
    for i in range(1, nspace - 1):
        ham[i, i - 1] = coeff
        ham[i, i] = -2 * coeff + potent[i]  # Set interior rows
        ham[i, i + 1] = coeff

    # First and last rows for periodic boundary conditions
    ham[0, -1] = coeff
    ham[0, 0] = -2 * coeff + potent[0]
    ham[0, 1] = coeff
    ham[-1, -2] = coeff
    ham[-1, -1] = -2 * coeff + potent[-1]
    ham[-1, 0] = coeff

    if method == 'crank':
        # Compute the Crank-Nicolson matrix
        matrix = np.dot(np.linalg.inv(np.identity(nspace) + .5 * i_imag *
                                      tau / h_bar * ham),
                        (np.identity(
                            nspace) - .5 * i_imag * tau / h_bar * ham))

    elif method == 'ftcs':
        # Compute the ftcs scheme matrix
        matrix = np.identity(nspace) - i_imag * tau / h_bar * ham
        eig = np.linalg.eigvals(matrix)
        spectral_rad = np.absolute(max(eig))
        assert spectral_rad <= 1, 'Solution will be unstable'

    # Initialize the wavefunction
    sigma0, x0, k0 = wparam[0], wparam[1], wparam[2]
    Norm_psi = 1. / (np.sqrt(sigma0 * np.sqrt(np.pi)))  # Normalization
    psi = np.empty(nspace, dtype=complex)
    for i in range(nspace):
        psi[i] = Norm_psi * np.exp(i_imag * k0 * x[i]) * \
                 np.exp(-(x[i] - x0) ** 2 / (2 * sigma0 ** 2))

    # Initialize loop and plot variables
    probs = np.zeros(ntime + 1)  # Array of probabilities
    psi_sol = np.zeros((nspace, ntime + 1),
                       dtype=complex)  # psi solutions array
    psi_sol[:, 0] = psi

    prob_matrix = np.zeros((nspace, ntime + 1))  # Note that P(x,t) is real
    prob_matrix[:, 0] = np.absolute(psi[:]) ** 2  # Record initial condition

    # Loop over desired number of time steps
    for iter in range(ntime):
        # Compute new wave function using the Crank-Nicolson scheme
        psi = np.dot(matrix, psi)
        psi_sol[:,
        iter + 1] = psi  # Add solution at t to array of all solutions

        prob_matrix[:, iter + 1] = np.absolute(psi[:]) ** 2

    for i in range(ntime + 1):
        probs[i] = np.sum(prob_matrix[:, i])

    return psi_sol, x, t, probs


def sch_plot(psi_sol, x, t, t_plot, plot_choice, save_plot=False):
    """
    Plots the solution (either wavefunction or probability distribution) at
    a specified time - should be used with output from sch_eqn function
    :param psi_sol: solution to schro. eqn (amplitude of wavefunction at x & t)
    (2-D Array)
    :param x: spatial grid on which to plot solutions (1-D array)
    :param t: time grid on which solutions were calculated (1-D array)
    :param t_plot: time at which to plot solutions (must be in t) (int/float)
    :param plot_choice: 'psi' prompts wavefunction to be plotted, 'prob'
    prompts probability density to be plotted
    :param save_plot: True prompts figure to be saved (Boolean)
    """
    # Assertions to ensure correct input:
    assert plot_choice == 'psi' or plot_choice == 'prob', \
        'plot_choice must either be psi or prob'
    assert t_plot in t, 't_plot must be in given time array'

    t_plot_index = np.where(t == t_plot)[0]

    if plot_choice == 'psi':
        # * Plot the wavefunction at specified time
        to_plot = psi_sol[:, t_plot_index]
        plt.plot(x, np.real(to_plot))
        plt.xlabel('x')
        plt.ylabel(r'$\psi(x)$')
        plt.title('Real part of wave function')
        if save_plot:
            plt.savefig('WavefunctionReal.png')
        plt.show()

    elif plot_choice == 'prob':
        # * Plot probability versus position at specified time
        p_plot = np.absolute(psi_sol[:, t_plot_index]) ** 2
        plt.plot(x, p_plot)
        plt.xlabel('x')
        plt.ylabel('P(x,t)')
        plt.title('Probability density at time t')
        if save_plot:
            plt.savefig('ProbabilityPlot.png')
        plt.show()
