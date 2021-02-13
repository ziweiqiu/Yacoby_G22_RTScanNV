import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame

# todo: Calibrate demodulation function for the ONIX


def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


def demodulation(SSB, Wc, Ws, S, ts0):

    '''
    demodulation
    :param SSB: IF frequency in Hz
    :param Wc:  list of cosine weights
    :param Ws: list of sine weights
    :param S: The signal
    :param ts0: The time stamp of the first sample
    :return:
    '''

    phi0 = 2 * np.pi * SSB * (ts0 - 32) * 1e-9
    Sum = 0
    delta_phi = 2 * np.pi * SSB * 1e-9
    Phi = [i * delta_phi + phi0 for i in range(4)]

    for i in range(int(len(S) / 4)):
        Svec = np.array([S[4 * i + s] for s in range(4)])
        Sum = Sum + np.sum(np.multiply(np.multiply(Wc[i], np.cos(Phi)) + np.multiply(Ws[i], np.sin(Phi)), Svec))
        for j in range(4):
            Phi[j] = Phi[j] + 4 * delta_phi

    return 16 * Sum * 2 ** -28


def extract_envelope(S, TS0_ns, SSB):

    period_ns = int(1 / SSB * 1e9)
    period_ts = int(period_ns / 4)
    envI = []
    envQ = []

    for i in range(int(len(S)/4) - period_ts + 1):
        envI.append(demodulation(SSB, [1.0] * period_ts, [0.0] * period_ts, S[i * 4: i * 4 + period_ns], TS0_ns + i * 4))
        envQ.append(demodulation(SSB, [0.0] * period_ts, [1.0] * period_ts, S[i * 4: i * 4 + period_ns], TS0_ns + i * 4))
    for j in range(period_ts - 1):
        if j % 2 == 0:
            envI.append(np.float64(0.0))
            envQ.append(np.float64(0.0))
        if j % 2 == 1:
            envI = [0.0] + envI
            envQ = [0.0] + envQ

    return envI, envQ


def weights_opt_basic(t_g, t_e, adc_g, adc_e, Ng, Ne, SSB):

    # RESHAPE PARAMETERS:
    L = int(len(adc_g)/Ng/4)
    adc_g = np.reshape(adc_g, (Ng, -1))
    t_g = np.reshape(t_g, (Ng, -1))
    adc_e = np.reshape(adc_e, (Ne, -1))
    t_e = np.reshape(t_e, (Ne, -1))

    # EXTRACT AVERAGED ENVELOPES:
    envIg = np.zeros(L)
    envQg = np.zeros(L)
    envIe = np.zeros(L)
    envQe = np.zeros(L)
    for i in range(Ng):
        envIg_, envQg_ = extract_envelope(adc_g[i], int(t_g[i][0]), SSB)
        envIg = envIg + np.array(envIg_)
        envQg = envQg + np.array(envQg_)
    for i in range(Ne):
        envIe_, envQe_ = extract_envelope(adc_e[i], int(t_e[i][0]), SSB)
        envIe = envIe + np.array(envIe_)
        envQe = envQe + np.array(envQe_)
    envIg = envIg / Ng
    envQg = envQg / Ng
    envIe = envIe / Ne
    envQe = envQe / Ne

    # NORMALIZATION:
    deltaI = envIe - envIg
    deltaQ = envQe - envQg
    W = [list(- deltaI), list(- deltaQ), list(+ deltaQ), list(- deltaI)]

    exp_gI = demodulation(SSB, W[0], W[1], adc_g[0], t_g[0][0])
    exp_gQ = demodulation(SSB, W[2], W[3], adc_g[0], t_g[0][0])
    exp_eI = demodulation(SSB, W[0], W[1], adc_e[0], t_e[0][0])
    exp_eQ = demodulation(SSB, W[2], W[3], adc_e[0], t_e[0][0])

    demod_max = 2**(16-1)
    exp_demod_max = np.max([np.abs(exp_gI), np.abs(exp_gQ), np.abs(exp_eI), np.abs(exp_eQ)])
    C = 0.5 * demod_max / exp_demod_max

    exp_weight_max = np.max(np.abs(list(C * deltaI) + list(C * deltaQ)))
    weight_max = 2**(12-1)
    r = weight_max / exp_weight_max
    if r < 1:
        C = C * r * 0.9

    W = [list(-C * deltaI), list(-C * deltaQ), list(+C * deltaQ), list(-C * deltaI)]
    env = [(envIg, envQg), (envIe, envQe)]

    # todo: In the real googMorningQM func the C should be returned

    return W, env, C


def miss_prep_identification(t, adc, Ng, Ne, SSB, miss_prep, seq0, W):

    # GET THE K-MEANS SEQ:
    I = []
    Q = []
    for i in range(Ng + Ne):
        I.append(demodulation(SSB, W[0], W[1], adc[i], t[i][0]))
        Q.append(demodulation(SSB, W[2], W[3], adc[i], t[i][0]))
    Data = {'x': I, 'y': Q}
    df = DataFrame(Data, columns=['x', 'y'])
    kmeans = KMeans(n_clusters=2).fit(df)
    seq_k = []
    match_counter = 0
    j = 0
    for arg in kmeans.labels_.astype(float):
        if arg == 0:
            seq_k.append(0)
        else:
            seq_k.append(1)
        if seq0[j] == arg:
            match_counter += 1
        j += 1
    if match_counter < 0.5 * (Ng + Ne):
        seq_k = [1 - arg for arg in seq_k]

    # COMPARE SEQ0 AND K-MEANS SEQ:
    comp = np.array(seq_k) - np.array(seq0)
    ind_suspected_g2e = [i for i, x in enumerate(comp) if x == +1]
    ind_suspected_e2g = [i for i, x in enumerate(comp) if x == -1]

    # GET THE NEW SEQ:
    [[I1, Q1], [I2, Q2]] = kmeans.cluster_centers_
    dI1 = np.array(I) - I1
    dQ1 = np.array(Q) - Q1
    dI2 = np.array(I) - I2
    dQ2 = np.array(Q) - Q2
    D = np.abs(np.sqrt(abs(dI1)**2+abs(dQ1)**2) - np.sqrt(abs(dI2)**2+abs(dQ2)**2))

    D_suspected_g2e = []
    for ind in ind_suspected_g2e:
        D_suspected_g2e.append(D[ind])

    D_suspected_e2g = []
    for ind in ind_suspected_e2g:
        D_suspected_e2g.append(D[ind])

    g2e_ = np.flip(np.argsort(D_suspected_g2e))[:int(miss_prep * Ng)]
    e2g_ = np.flip(np.argsort(D_suspected_e2g))[:int(miss_prep * Ne)]

    ind_g2e = [ind_suspected_g2e[i] for i in g2e_]
    ind_e2g = [ind_suspected_e2g[i] for i in e2g_]

    new_seq = []
    for ind in range(len(seq0)):
        if ind in ind_g2e:
            new_seq.append(1)
        elif ind in ind_e2g:
            new_seq.append(0)
        else:
            new_seq.append(seq0[ind])

    return new_seq


def avg_envelopes(L, new_seq, t, adc, SSB):
    envIg = np.zeros(L)
    envQg = np.zeros(L)
    envIe = np.zeros(L)
    envQe = np.zeros(L)
    Ng = 0
    Ne = 0
    for i in range(len(new_seq)):
        envI_, envQ_ = extract_envelope(adc[i], t[i][0], SSB)
        if new_seq[i] == 0:
            envIg = envIg + np.array(envI_)
            envQg = envQg + np.array(envQ_)
            Ng += 1
        elif new_seq[i] == 1:
            envIe = envIe + np.array(envI_)
            envQe = envQe + np.array(envQ_)
            Ne += 1
    envIg = envIg / Ng
    envQg = envQg / Ng
    envIe = envIe / Ne
    envQe = envQe / Ne

    return [(envIg, envQg), (envIe, envQe)]


def weights_normalization(env, new_seq, t, adc, SSB):

    [(envIg, envQg), (envIe, envQe)] = env
    deltaI = envIe - envIg
    deltaQ = envQe - envQg
    W = [list(- deltaI), list(- deltaQ), list(+ deltaQ), list(- deltaI)]

    ind_g = new_seq.index(0)
    ind_e = new_seq.index(1)
    exp_gI = demodulation(SSB, W[0], W[1], adc[ind_g], t[ind_g][0])
    exp_gQ = demodulation(SSB, W[2], W[3], adc[ind_g], t[ind_g][0])
    exp_eI = demodulation(SSB, W[0], W[1], adc[ind_e], t[ind_e][0])
    exp_eQ = demodulation(SSB, W[2], W[3], adc[ind_e], t[ind_e][0])

    demod_max = 2 ** (16 - 1)
    exp_demod_max = np.max([np.abs(exp_gI), np.abs(exp_gQ), np.abs(exp_eI), np.abs(exp_eQ)])
    C = 0.5 * demod_max / exp_demod_max

    exp_weight_max = np.max(np.abs(list(C * deltaI) + list(C * deltaQ)))
    weight_max = 2 ** (12 - 1)
    r = weight_max / exp_weight_max
    if r < 1:
        C = C * r * 0.9

    return C


def weights_opt(t_g, t_e, adc_g, adc_e, Ng, Ne, SSB, miss_prep):

    # RESHAPE PARAMETERS:
    miss_prep = float(miss_prep)
    Ng = int(Ng)
    Ne = int(Ne)
    adc = np.reshape(list(adc_g) + list(adc_e), (Ng + Ne, -1))
    t = np.reshape(list(t_g) + list(t_e), (Ng + Ne, -1))
    seq0 = [0] * Ng + [1] * Ne
    L = int(len(adc_g)/Ng/4)

    # IDENTIFICATION OF MISS PREPARATIONS:
    W_square = [[1.0] * L, [0.0] * L, [0.0] * L, [1.0] * L]
    new_seq = miss_prep_identification(t, adc, Ng, Ne, SSB, miss_prep, seq0, W_square)

    # EXTRACT AVERAGED ENVELOPES
    env = avg_envelopes(L, new_seq, t, adc, SSB)

    # NORMALIZATION
    C = weights_normalization(env, new_seq, t, adc, SSB)

    # NEW WEIGHTS
    [(envIg, envQg), (envIe, envQe)] = env
    deltaI = envIe - envIg
    deltaQ = envQe - envQg
    W = [list(-C * deltaI), list(-C * deltaQ), list(+C * deltaQ), list(-C * deltaI)]

    return W, [(envIg, envQg), (envIe, envQe)]

def get_optimal_weights(filename):
    return [[1.0] * int(600 / 4), [0.0] * int(600 / 4)]


def simulate_pulse(IF_freq, chi, k, Ts, Td, power):

    I = [0]; Q = [0]

    for t in range(Ts):
        I.append(I[-1]+( power - k*I[-1]+Q[-1]*chi))
        Q.append(Q[-1]+( - k*Q[-1]-I[-1]*chi))

    for t in range(Td - 1):
        I.append(I[-1]+(-k*I[-1]+Q[-1]*chi))
        Q.append(Q[-1]+(-k*Q[-1]-I[-1]*chi))

    I = np.array(I)
    Q = np.array(Q)
    t = np.arange(len(I))

    S = I * np.cos(2 * np.pi * IF_freq * t * 1e-9) + Q * np.sin(2 * np.pi * IF_freq * t * 1e-9)

    return(t, I, Q, S)

IF_freq = 50e6
Ts = 120
Td = 80
power = 0.2
k = 0.04
chi = 0.025

[tg_, Ig_, Qg_, Sg_] = simulate_pulse(IF_freq, chi, k, Ts, Td, power)
[te_, Ie_, Qe_, Se_] = simulate_pulse(IF_freq, -chi, k, Ts, Td, power)