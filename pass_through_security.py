import pandas as pd
import numpy as np
import rates_model_bdt as bdt
import mortgage_model as mm

def calculate_passthrough_security(rateSecur, rateMort, ratesTree, 
                                   yields, n, sigma, delta, mortg):
    securityTree = np.nan * np.ones((n+1, n+1))

    # Calculate semi-annual payments
    discounts = 1 / (1 + rateMort/2)**(1 + np.arange(n))  # i = 1..n
    coup = mortg / discounts.sum()

    # Calculate loan outstanding balance
    interestPaid    = np.zeros(n+1)
    principalPaid   = np.zeros(n+1)
    outstPrincipal  = np.zeros(n+1)
    interestPaidSec = np.zeros(n+1)

    outstPrincipal[0] = mortg
    for i in range(1, n+1):
        interestPaid[i]    = outstPrincipal[i-1] * rateMort/2
        interestPaidSec[i] = outstPrincipal[i-1] * rateSecur/2
        principalPaid[i]   = coup - interestPaid[i]
        outstPrincipal[i]  = outstPrincipal[i-1] - principalPaid[i]
    
    # Get the prepayment decision tree from mortgage model
    # (also uses yields, n, sigma, delta as global vars)
    _, _, _, _, _, _, prepayTree, *_ = mm.calculate_mortgage(
        rateMort, yields, n, sigma, delta, mortg)

    # Value of mortgage without prepayment (bond with no face value repayment)
    for i in np.flip(np.arange(n+1)):
        for j in np.arange(i+1):
            if i == n:
                # At the last time step, both streams are 0
                securityTree[j,i] = 0.0
            else:
                rate = ratesTree[j,i]
                up = securityTree[j,i+1]
                down = securityTree[j+1,i+1]
                nextCashFlows = interestPaidSec[i+1] + principalPaid[i+1]
                securityValue = np.exp(-delta*rate) * (0.5*(up + down) + nextCashFlows)

                # Option to prepay or continue holding mortgage
                if i > 0 and prepayTree[j,i]:
                    securityTree[j,i] = outstPrincipal[i]  # prepay
                else:
                    securityTree[j,i] = securityValue  # hold

    return (securityTree, interestPaidSec, principalPaid, outstPrincipal)

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    # import data
    data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
    yields = np.array(data.iloc[4][1:].to_list()) / 100

    # BDT model parameters
    delta = 0.5    # compounding period
    sigma = 0.2142 # estimated volatility for log rates
    mortg = 100e3  # mortgage value
    n     = 5      # number of periods to model for (times steps = n+1)

    # Calculate rates tree
    ratesTree, _ = bdt.make_bdt_tree(yields, n, sigma, delta)

    # Use predetermined mortgage rate
    rateMortg = 0.0397181398698594 # optimal value from 'mortgage_model.py'
    rateSecur = rateMortg - 50e-4 # 50 bp lower for securitisation costs

    # Calculate optimal values
    securityTree, interPaidSec, princPaid, outstPrinc = calculate_passthrough_security(
        rateSecur, rateMortg, ratesTree, yields, n, sigma, delta, mortg)

    print('Security value: \n',              securityTree)
    print('\nInterest paid to security: \n', interPaidSec)
    print('\nPrincipal paid: \n',            princPaid)
    print('\nOutstanding principal: \n',     outstPrinc)
