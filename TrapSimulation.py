import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import broyden1
from semiconductor.material.thermal_velocity import ThermalVelocity as Vel_th
from semiconductor.material.ni import IntrinsicCarrierDensity as NI
from scipy.integrate import odeint
from semiconductor.electrical.mobility import Mobility as mob
from semiconductor.electrical.ionisation import Ionisation as Ion
import semiconductor

kb = const.k / const.e


def getsomeparam(Ndop, doptype, temp, **kwarg):
    T = temp
    ni = NI().update(temp=T)
    ni = ni[0]
    ve, vp = Vel_th().update(temp=T)
    ve = ve[0]
    if doptype == 'p':
        Na = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='boron')
        Na = Na[0]
        Nd = 0
    elif doptype == 'n':
        Nd = Ion(temp=T).update_dopant_ionisation(
            N_dop=Ndop, nxc=0, impurity='phosphorous')
        Nd = Nd[0]
        Na = 0
    miu_tot = mob(temp=T, Na=Na, Nd=Nd).mobility_sum()
    miu_h = mob(temp=T, Na=Na, Nd=Nd).hole_mobility()
    miu_e = mob(temp=T, Na=Na, Nd=Nd).electron_mobility()
    return ni, Na, Nd, ve, vp, miu_tot, miu_h, miu_e

# Solve for Equilibrium


def SolveEq(Ndop, doptype, temp, defect, **kwarg):
    Ndop = Ndop
    doptype = doptype
    T = temp
    defectlist = defect
    ni, Na, Nd, ve, vp, miu_tot, miu_h, miu_e = getsomeparam(
        Ndop=Ndop, doptype=doptype, temp=T)

    tol = 1e-12
    f0list = [0 if x['type'] == 'A' else 1 for x in defectlist]
    diff = 100

    while diff > tol:
        Ndtot1 = 0
        Natot1 = 0
        for d in defectlist:
            if d['type'] == 'D':
                Ndtot1 += (1 - f0list[defectlist.index(d)]) * d['Nt']
            elif d['type'] == 'A':
                Natot1 += (f0list[defectlist.index(d)]) * d['Nt']
        Ndtot = Nd + Ndtot1
        Natot = Na + Natot1
        if Ndtot >= Natot:
            n00 = 0.5 * ((Ndtot - Natot) +
                         np.sqrt((Ndtot - Natot)**2 + 4 * ni**2))
            p00 = ni**2 / n00
        elif Ndtot < Natot:
            p00 = 0.5 * ((Natot - Ndtot) +
                         np.sqrt((Natot - Ndtot)**2 + 4 * ni**2))
            n00 = ni**2 / p00
        fnlist = [(x['se'] * ve * n00 +
                   x['sp'] * vp * ni * np.exp(-x['Et'] / kb / T)) /
                  (x['se'] * ve * (n00 + ni * np.exp(x['Et'] / kb / T)) +
                   x['sp'] * vp * (p00 + ni * np.exp(-x['Et'] / kb / T)))
                  for x in defectlist]
        # fnlist = [1/(1+np.exp(x['Et'] / kb / T)*ni/n0) for x in defectlist]
        nt00 = 0
        diff = 0
        for fn, f, d in zip(fnlist, f0list, defectlist):
            nt00 += fn * d['Nt']
            diff += abs(fn - f)
        f0list = fnlist
    return n00, p00, nt00, f0list


# Solve for Steady state
def SolveSS(Ndop, doptype, temp, defect, dplist, **kwarg):
    Ndop = Ndop
    doptype = doptype
    T = temp
    defectlist = defect
    dplist = dplist
    ni, Na, Nd, ve, vp, miu_tot, miu_h, miu_e = getsomeparam(
        Ndop=Ndop, doptype=doptype, temp=T)
    n00, p00, nt00, f0list = SolveEq(
        Ndop=Ndop, doptype=doptype, temp=T, defect=defectlist)

    nlist = []
    plist = []
    allntlist = []
    tauefflist = []
    dnapplist = []
    tauapplist = []
    ntchargelist = []

    for i in range(dplist.shape[0]):
        dp = dplist[i]
        diff = 100
        tol = 1e-2
        n = n00 + dp
        p = p00 + dp
        ntcharge = 0
        while diff > tol:
            fnnlist = [(x['se'] * ve * n +
                        x['sp'] * vp * ni * np.exp(-x['Et'] / kb / T)) /
                       (x['se'] * ve * (n + ni * np.exp(x['Et'] / kb / T)) +
                        x['sp'] * vp * (p + ni * np.exp(-x['Et'] / kb / T)))
                       for x in defectlist]
            ntlist = [fnn * d['Nt'] for fnn, d in zip(fnnlist, defectlist)]

            if doptype == 'p':
                pp = (p00 + dp + (sum(ntlist) - nt00))
                diff = abs(pp - p) / p
                p = pp
            elif doptype == 'n':
                nn = (n00 + dp - (sum(ntlist) - nt00))
                diff = abs(nn - n) / p
                n = nn
        for d, nt in zip(defectlist, ntlist):
            if d['type'] == 'D':
                ntcharge += d['Nt'] - nt
            elif d['type'] == 'A':
                ntcharge += -nt
        ntchargelist.append(ntcharge)
        nlist.append(n)
        plist.append(p)
        allntlist.append(ntlist)
        USRHlist = [(n * p - ni**2) /
                    ((n + ni * np.exp(d['Et'] / kb / T)) / d['Nt'] / d['sp'] / vp +
                     (p + ni * np.exp(-d['Et'] / kb / T)) / d['Nt'] / d['se'] / ve)
                    for d in defectlist]
        Utot = sum(USRHlist)
        if doptype == 'p':
            tau_eff = (n - n00) / Utot
        elif doptype == 'n':
            tau_eff = (p - p00) / Utot
        dn_app = ((n - n00) * miu_e + (p - p00) * miu_h) / miu_tot
        tau_app = dn_app / Utot
        tauefflist.append(tau_eff)
        dnapplist.append(dn_app)
        tauapplist.append(tau_app)

    # Plot SS
    plt.figure('Carrier concentration in Steady state')
    plt.xlabel('Minoroty carrier concentration [cm-3]')
    plt.ylabel('Carrier concentrations [cm-3]')
    ax = plt.gca()
    plt.loglog()
    plt.plot([dplist[0], dplist[-1]], [n00, n00], 'r--')
    plt.plot([dplist[0], dplist[-1]], [p00, p00], 'b--')
    plt.plot(dplist, nlist, 'r', label='n')
    plt.plot(dplist, plist, 'b', label='p')
    for i in range(len(defectlist)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot([dplist[0], dplist[-1]], [f0list[i] * defectlist[i]['Nt'],
                                           f0list[i] * defectlist[i]['Nt']],
                 '--', color=color)
        plt.plot(dplist, np.array(allntlist)[
                 :, i], color=color, label='nt' + str(i + 1))
    # plt.plot(dplist, dnapplist, label='dnapp')
    plt.plot(dplist, Nd - Na - nlist + plist +
             ntchargelist, '.', label='charge neutrality')
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('Excess carrier concentration in Steady state')
    plt.xlabel('Minoroty carrier concentration [cm-3]')
    plt.ylabel('Excess carrier concentrations [cm-3]')
    plt.loglog()
    plt.plot(dplist, nlist - n00, 'r', label='delta n')
    plt.plot(dplist, plist - n00, 'b', label='delta p')
    for i in range(len(defectlist)):
        plt.plot(dplist, np.array(allntlist)[
                 :, i] - f0list[i] * defectlist[i]['Nt'], label='delta nt' +
                 str(i + 1))
    plt.plot(dplist, dnapplist, label='delta napp')
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('Recombination and Generation rate')
    plt.xlabel('Minoroty carrier concentration [cm-3]')
    plt.ylabel('Recombination and Generation rate [cm-3s-1]')
    for i in range(len(defectlist)):
        x = defectlist[i]
        Reimp = x['se'] * ve * \
            np.array(nlist) * (x['Nt'] - np.array(allntlist)[:, i])
        Geimp = x['se'] * ve * ni * \
            np.exp(x['Et'] / kb / T) * np.array(allntlist)[:, i]
        Rhimp = x['sp'] * vp * np.array(plist) * np.array(allntlist)[:, i]
        Ghimp = x['sp'] * vp * ni * \
            np.exp(-x['Et'] / kb / T) * (x['Nt'] - np.array(allntlist)[:, i])
        plt.plot(dplist, Geimp, '--', label='Geimp nt' + str(i + 1))
        plt.plot(dplist, Reimp, '--', label='Reimp nt' + str(i + 1))
        plt.plot(dplist, Ghimp, '--', label='Ghimp nt' + str(i + 1))
        plt.plot(dplist, Rhimp, '--', label='Rhimp nt' + str(i + 1))
    plt.loglog()
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('PC and PL Lifetime')
    plt.loglog()
    plt.xlabel('Minoroty carrier concentration [cm-3]')
    plt.ylabel('PC / PL lifetime [s]')
    plt.plot(dplist, tauefflist, 'b--', label='PL lifetine SS')
    plt.plot(dnapplist, tauapplist, 'r--', label='PC lifetine SS')
    plt.grid(True)
    plt.legend(loc=0)


# Solve for transient

# Solve for initial state
def SolveTrans(Ndop, doptype, temp, defect, t, d0, tG, **kwarg):
    Ndop = Ndop
    doptype = doptype
    T = temp
    defectlist = defect
    d0 = d0
    tG = tG
    t = t
    ni, Na, Nd, ve, vp, miu_tot, miu_h, miu_e = getsomeparam(
        Ndop=Ndop, doptype=doptype, temp=T)
    n00, p00, nt00, f0list = SolveEq(
        Ndop=Ndop, doptype=doptype, temp=T, defect=defectlist)
    diff = 100
    tol = 1e-10
    n = n00 + d0
    p = p00 + d0
    while diff > tol:
        fnnlist = [(x['se'] * ve * n +
                    x['sp'] * vp * ni * np.exp(-x['Et'] / kb / T)) /
                   (x['se'] * ve * (n + ni * np.exp(x['Et'] / kb / T)) +
                    x['sp'] * vp * (p + ni * np.exp(-x['Et'] / kb / T)))
                   for x in defectlist]
        ntlist = [fnn * d['Nt'] for fnn, d in zip(fnnlist, defectlist)]
        for fn, d in zip(fnnlist, defectlist):
            if doptype == 'p':
                pp = (p00 + d0 + (sum(ntlist) - nt00))
                diff = abs(pp - p) / p
                p = pp
            elif doptype == 'n':
                nn = (n00 + d0 - (sum(ntlist) - nt00))
                diff = abs(nn - n) / p
                n = nn
    p0 = p
    n0 = n
    nt0 = [fnn * d['Nt'] for fnn, d in zip(fnnlist, defectlist)]
    G0 = np.sum([(n0 * p0 - ni**2) /
                 ((n0 + ni * np.exp(d['Et'] / kb / T)) / d['Nt'] / d['sp'] / vp +
                  (p0 + ni * np.exp(-d['Et'] / kb / T)) / d['Nt'] / d['se'] / ve)
                 for d in defectlist])

    # ODE

    def trapode(y, t, tG):
        n = y[0]
        p = y[1]
        if tG == 0:
            Gen = 0 * t
        elif tG == 'inf':
            Gen = G0 + 0 * t
        else:
            Gen = G0 * np.exp(-t / tG)
        Reimp = [x['se'] * ve * n *
                 (x['Nt'] - y[defectlist.index(x) + 2]) for x in defectlist]
        Geimp = [x['se'] * ve * ni *
                 np.exp(x['Et'] / kb / T) * y[defectlist.index(x) + 2]
                 for x in defectlist]
        Rhimp = [x['sp'] * vp * p *
                 y[defectlist.index(x) + 2] for x in defectlist]
        Ghimp = [x['sp'] * vp * ni * np.exp(-x['Et'] / kb / T) * (
            x['Nt'] - y[defectlist.index(x) + 2]) for x in defectlist]

        dydt = [Gen - sum(Reimp) + sum(Geimp), Gen - sum(Rhimp) + sum(Ghimp)]
        for Re, Ge, Rh, Gh in zip(Reimp, Geimp, Rhimp, Ghimp):
            dydt.append(Re - Rh - Ge + Gh)

        return dydt

    y0 = [n0, p0]
    y0.extend(nt0)

    sol = odeint(trapode, y0, t, args=(tG,))

    n = sol[:, 0]
    p = sol[:, 1]
    nt = sol[:, 2:]

    # Calculate lifetime
    if doptype == 'n':
        nxc = p - p00
    elif doptype == 'p':
        nxc = n - n00

    if tG == 0:
        gen = 0 * t
    elif tG == 'inf':
        gen = G0 + 0 * t
    else:
        gen = G0 * np.exp(-t / tG)
    Gen = gen[1:]

    nxcpc = ((n - n00) * miu_e + (p - p00) * miu_h) / miu_tot
    DiffPL = np.diff(nxc) / np.diff(t)
    DiffPC = np.diff(nxcpc) / np.diff(t)
    nxc = nxc[1:]
    nxcpc = nxcpc[1:]
    tauPL = nxc / (Gen - DiffPL)
    tauPC = nxcpc / (Gen - DiffPC)
    ntcharge = -np.sum(nt, axis=1)
    for d in defectlist:
        if d['type'] == 'D':
            ntcharge += d['Nt']

    # plot for Transient
    plt.figure('Carrier concentration as a function of time')
    plt.xlabel('Time [s]')
    plt.ylabel('Carrier concentration [cm-3]')
    plt.plot([t[1], t[-1]], [n00, n00], 'r--')
    plt.plot([t[1], t[-1]], [p00, p00], 'b--')
    plt.plot(t, n, 'r', label='n')
    plt.plot(t, p, 'b', label='p')
    plt.plot(t, gen, 'k', label='Gen')
    ax = plt.gca()
    for i in range(len(defectlist)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot([t[1], t[-1]], [f0list[i] * defectlist[i]['Nt'],
                                 f0list[i] * defectlist[i]['Nt']], '--',
                 color=color)
        plt.plot(t, nt[:, i], color=color, label='nt' + str(i + 1))
    plt.plot(t, Nd - n - Na + p + ntcharge, '.', label='Charge neutrality')
    plt.loglog()
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('Excess carrier concentration as a function of time')
    plt.xlabel('Time [s]')
    plt.ylabel('Excess carrier concentration [cm-3]')
    plt.plot(t, n - n00, 'r', label='delta n')
    plt.plot(t, p - p00, 'b', label='delta p')
    plt.plot(t, gen, 'k', label='Gen')
    ax = plt.gca()
    for i in range(len(defectlist)):
        plt.plot(t, nt[:, i] - f0list[i] * defectlist[i]
                 ['Nt'], label='delta nt' + str(i + 1))
    plt.plot(t[1:], nxcpc, label='delta napp')
    plt.loglog()
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('Recombination and Generation rate as a function of time')
    plt.xlabel('Time [s]')
    plt.ylabel('Recombination and Generation rate [cm-3s-1]')
    for i in range(len(defectlist)):
        x = defectlist[i]
        Reimp = x['se'] * ve * n * (x['Nt'] - nt[:, i])
        Geimp = x['se'] * ve * ni * np.exp(x['Et'] / kb / T) * nt[:, i]
        Rhimp = x['sp'] * vp * p * nt[:, i]
        Ghimp = x['sp'] * vp * ni * \
            np.exp(-x['Et'] / kb / T) * (x['Nt'] - nt[:, i])
        plt.plot(t, Geimp, '--', label='Geimp nt' + str(i + 1))
        plt.plot(t, Reimp, '--', label='Reimp nt' + str(i + 1))
        plt.plot(t, Ghimp, '--', label='Ghimp nt' + str(i + 1))
        plt.plot(t, Rhimp, '--', label='Rhimp nt' + str(i + 1))
    plt.loglog()
    plt.grid(True)
    plt.legend(loc=0)

    plt.figure('PC and PL Lifetime')
    plt.xlabel('Minoroty carrier concentration [cm-3]')
    plt.ylabel('PC / PL lifetime [s]')
    plt.plot(nxc, tauPL, 'b-.', label='PL lifetime Transient')
    plt.plot(nxcpc, tauPC, 'r-.', label='PC lifetime Transient')
    plt.loglog()
    plt.grid(True)
    plt.legend(loc=0)


# Measurement parameters
T = 300
# Steady state parameters
dplist = np.logspace(1, 16, 100)
# Transient parameters
d0 = 1e16
t = np.linspace(0, 0.001, 100000)
tG = 0
# Trap parameters
defect1 = {'Nt': 1e14, 'Et': 0, 'se': 1e-15, 'sp': 1e-15, 'type': 'D'}
defect2 = {'Nt': 1e14, 'Et': -0.3, 'se': 1e-15, 'sp': 1e-19, 'type': 'D'}
defectlist = [defect1, defect2]
# Sample parameters
Ndop = 1.5e16
doptype = 'p'

SolveSS(Ndop=Ndop, doptype=doptype, temp=T, defect=defectlist, dplist=dplist)
SolveTrans(Ndop=Ndop, doptype=doptype, temp=T,
           defect=defectlist, t=t, d0=d0, tG=tG)

plt.show()
