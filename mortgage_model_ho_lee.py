import pandas as pd
import numpy as np
from scipy.optimize import minimize
import rates_model_ho_lee as ho

def calculate_mortgage(ratem, yields, n, sigma, delta, mortg):
    ratesTree, _ = ho.make_ho_lee_tree(yields, n, sigma, delta)
    bondTree     = np.nan * np.ones((n+1, n+1))
    optionTree   = np.nan * np.ones((n+1, n+1))
    prepayTree   = np.zeros((n+1, n+1), dtype=bool)

    # Calculate semi-annual payments
    discounts = 1 / (1 + ratem/2)**(1 + np.arange(n))  # i = 1..n
    coup = mortg / discounts.sum()

    # Calculate loan outstanding balance
    interestPaid   = np.zeros(n+1)
    principalPaid  = np.zeros(n+1)
    outstPrincipal = np.zeros(n+1)

    outstPrincipal[0] = mortg
    for i in range(1, n+1):
        interestPaid[i]   = outstPrincipal[i-1] * ratem / 2
        principalPaid[i]  = coup - interestPaid[i]
        outstPrincipal[i] = outstPrincipal[i-1] - principalPaid[i]

    # Value of mortgage without prepayment (bond with no face value repayment)
    for i in np.flip(np.arange(n+1)):
        for j in np.arange(i+1):
            if i == n:
                bondTree[j,i] = 0.0
            else:
                rate = ratesTree[j,i]
                up   = bondTree[j,i+1]
                down = bondTree[j+1,i+1]
                bondTree[j,i] = np.exp(-delta*rate) * ((up + down)/2 + coup)

    # Value of option to prepay
    for i in np.flip(np.arange(n+1)):
        for j in np.arange(i+1):
            if i == n:
                # Option not valueable at the end & beginning
                optionTree[j,i] = 0.0
            else:
                rate = ratesTree[j,i]
                # Value of prepaying = exercising the option
                if i > 0:
                    optionExer = max(bondTree[j,i] - outstPrincipal[i], 0.0)
                else: # do not exercise the option at t=0
                    optionExer = 0.0
                
                # Value to continue holding the option
                up = optionTree[j,i+1]
                down = optionTree[j+1,i+1]
                optionCont = np.exp(-delta*rate) * 0.5 * (up + down)

                # We decide to stop only if more valuable than the option
                optionTree[j,i] = max(optionCont, optionExer)

                # Update prepayment tree
                if i > 0 and optionExer > optionCont:
                    prepayTree[j,i] = True

    return (ratesTree, bondTree, optionTree, 
            interestPaid, principalPaid, outstPrincipal, prepayTree, coup)

def cost_function(ratem):
    _, bondTree, optionTree, *_ = calculate_mortgage(
        ratem.item(), yields, n, sigma, delta, mortg)
    squareError = (bondTree[0,0] - mortg - optionTree[0,0])**2
    return squareError

if __name__ == '__main__':
    np.set_printoptions(linewidth=200, precision=4)

    # import data
    data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
    yields = np.array(data.iloc[4][1:].to_list()) / 100

    # BDT model parameters
    delta = 0.5    # compounding period
    sigma = 0.0173 #461844550287 # estimated volatility for log rates
    mortg = 100e3  # mortgage value
    n     = 5 # number of periods to model for (times steps = n+1)

    # Find optimal mortgage rate
    ratemGuess = 0.07
    optim = minimize(
                    cost_function,
                    ratemGuess,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-12,   # tolerance in θ
                        'fatol': 1e-16,   # tolerance in objective value
                        'maxiter': 10000,
                        'disp': False
                    }
                )
    ratemOptim = optim.x[0]

    # Calculate optimal values
    (ratesTree, bondTree, optionTree,
    interestPaid, principalPaid,
    outstPrincipal, *_) = calculate_mortgage(
        ratemOptim, yields, n, sigma, delta, mortg)

    print('Rates model: \n', 100*ratesTree)
    print('\nMortgage value: \n', bondTree)
    print('\nOption to prepay: \n', optionTree)
    print('\nOptimal mortgage rate: \n', 100*ratemOptim)
    print('\nInterest paid: \n', interestPaid)
    print('\nPrincipal paid: \n', principalPaid)
    print('\nOutstanding principal: \n', outstPrincipal)
