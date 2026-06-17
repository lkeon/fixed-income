import pandas as pd
import numpy as np
import rates_model_ho_lee as rm
import mortgage_model_ho_lee as mm

def calculate_passthrough_security(rateSecur, rateMort, ratesTree,
                                   yields, n, sigma, delta, mortg):
    princStripTree = np.nan * np.ones((n+1, n+1))
    interStripTree = np.nan * np.ones((n+1, n+1))

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
                princStripTree[j,i] = 0.0
                interStripTree[j,i] = 0.0
            else:
                rate = ratesTree[j,i]
                disc = np.exp(-delta*rate)

                # Principal strip
                up = princStripTree[j,i+1]
                down = princStripTree[j+1,i+1]
                princStripValue = disc * (0.5*(up + down) + principalPaid[i+1])

                # Interest strip
                up = interStripTree[j,i+1]
                down = interStripTree[j+1,i+1]
                interStripValue = disc * (0.5*(up + down) + interestPaidSec[i+1])

                # Option to prepay or continue holding mortgage
                if i > 0 and prepayTree[j,i]:
                    princStripTree[j,i] = outstPrincipal[i]  # prepay
                    interStripTree[j,i] = 0.0
                else:
                    princStripTree[j,i] = princStripValue  # hold
                    interStripTree[j,i] = interStripValue

    return (princStripTree, interStripTree, principalPaid, outstPrincipal)

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)

    # import data
    data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
    yields = np.array(data.iloc[4][1:].to_list()) / 100

    # BDT model parameters
    delta = 0.5    # compounding period
    sigma = 0.0173 # estimated volatility for log rates
    mortg = 100e3  # mortgage value
    n     = 5      # number of periods to model for (times steps = n+1)

    # Calculate rates tree
    ratesTree, _ = rm.make_ho_lee_tree(yields, n, sigma, delta)

    # Use predetermined mortgage rate
    rateMortg = 0.04380494174484919 # optimal value from 'mortgage_model.py'
    rateSecur = rateMortg - 50e-4 # 50 bp lower for securitisation costs

    # Calculate optimal values
    princStripTree, interStripTree, princPaid, outstPrinc = calculate_passthrough_security(
        rateSecur, rateMortg, ratesTree, yields, n, sigma, delta, mortg)

    print('Principal strip: \n', princStripTree)
    print('\nInterest strip: \n', interStripTree)
    print('\nPrincipal paid: \n', princPaid)
    print('\nOutstanding principal: \n', outstPrinc)
