# Black-Derman-Toy Interest Rates Model
# =====================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def make_bdt_tree(yields, n, sigma, delta=0.5, prob=0.5):
    # Construct the recombining tree
    yields = yields[:n]
    ratesTreeLn = np.nan * np.ones((n, n))
    bondsTree = np.nan * np.ones((n, n)) # prob weighted prices of bonds
    thetasOptim = np.zeros(n)

    # Calculate zero coupon prices implied by the yields
    times = delta * np.arange(1, n+1)
    bondsMkt = np.exp(-yields * times)
    
    for i in range(n):
        if i == 0:
            # Define tree root
            ratesTreeLn[0,0] = np.log(yields[0])
            bondsTree[0,0] = np.exp(-np.exp(ratesTreeLn[0,0]) * delta)
            
        else:
            # Construct bond price vector weighted by node prob for current time i
            def get_bond_price(theta):
                bondsCol = np.zeros(i+1)
                for j in range(i): # loop through nodes in the time step before
                    rateUp = ratesTreeLn[j,i-1] + theta*delta + sigma*np.sqrt(delta)
                    rateDn = ratesTreeLn[j,i-1] + theta*delta - sigma*np.sqrt(delta)
                    discUp = np.exp(-np.exp(rateUp) * delta)
                    discDn = np.exp(-np.exp(rateDn) * delta)
                    bondsCol[j] += bondsTree[j,i-1] * discUp * prob
                    bondsCol[j+1] += bondsTree[j,i-1] * discDn * (1 - prob)
                return bondsCol
            
            # Objective function returnning error for market defined price
            def objective_function(theta):
                return (get_bond_price(theta.item()).sum() - bondsMkt[i])**2
            
            # Find optimal theta
            thetaGuess = 0
            thetaOptim = minimize(
                objective_function,
                thetaGuess,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-12,   # tolerance in θ
                    'fatol': 1e-16,   # tolerance in objective value
                    'maxiter': 10000,
                    'disp': False
                }
            ).x[0]
            thetasOptim[i] = thetaOptim

            # Update bond prices with theta shifts
            bondsTree[0:i+1,i] = get_bond_price(thetaOptim)

            # # Update rates with optimal theta shifts
            # ratesTreeLn[0,i] = ratesTreeLn[0,i-1] + thetaOptim*delta + sigma*np.sqrt(delta)
            # ratesTreeLn[1:i+1,i] = ratesTreeLn[0,i] - 2*sigma*np.sqrt(delta)*np.arange(1, i+1)

            # Update rates with optimal theta shifts
            ratesTreeLn[0:i,i] = ratesTreeLn[0:i,i-1] + thetaOptim*delta + sigma*np.sqrt(delta)
            ratesTreeLn[i,i]   = ratesTreeLn[i-1,i-1] + thetaOptim*delta - sigma*np.sqrt(delta)

    # convert log rates to linear
    ratesTree = np.exp(ratesTreeLn)
    return (ratesTree, thetasOptim)

if __name__ == '__main__':
    # set printing options
    np.set_printoptions(precision=4, linewidth=200)
    
    # import data
    data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
    years = np.array(data.iloc[2,1:].to_list())
    yields = np.array(data.iloc[4,1:].to_list()) / 100

    # Ho-Lee model parameters
    delta = 0.5    # compounding period
    sigma = 0.2142 # estimated volatility
    prob  = 0.5    # probability of going up

    n = 5
    assert(yields.size >= n)
    ratesTree, thetaOptim = make_bdt_tree(yields, n, sigma, delta, prob)
    
    print('\n Rates tree (%):\n', 100*ratesTree)
    print('\n Calibration theta (x100):\n', 100*thetaOptim)

