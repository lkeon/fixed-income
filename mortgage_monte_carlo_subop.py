# Calculate mortgage value using Monte Carlo with suboptimal prepayment
# =====================================================================

import numpy as np
import rates_model_bdt as bdt

np.set_printoptions(precision=2, linewidth=200)

def get_suboptimal_prepay(iter, delta):
    if iter == 0:
        return False
    else:
        # Suboptimal probability
        cpr = np.min((12*iter*delta*0.2, 6)) / 100
        seasonIndex = 1 if np.mod(iter, 2) == 0 else 2
        prepayProbThresh = seasonIndex * (1 - (1 - 0.5*cpr)**delta)
        
        # Randomly decide whether to prepay
        randNum = np.random.uniform()
        suboptimPrepay = randNum < prepayProbThresh
        return suboptimPrepay


def get_optimal_prepay(rate, value, iter, a=0.8, b=20.0):
    # Calculate random prepayment decision
    prepayProbThresh = a * np.exp(-rate * b)
    randNum = np.random.uniform()
    optimPrepayRand = randNum < prepayProbThresh

    # Calculate whether prepayment is optimal
    optimPrepay = value > outstPrincipal[iter]

    # Only prepay if both conditions are true
    return optimPrepayRand and optimPrepay

# import data
# data = pd.read_excel('../yieldcurve2025.xlsx', sheet_name='4. spot curve')
# yields = np.array(data.iloc[4][1:].to_list()) / 100
yields = np.array([5.86, 6.19, 6.39, 6.52, 6.60, 6.64, 6.64, 6.67, 6.65, 6.65]) / 100

# BDT model parameters
delta  = 0.5    # compounding period
sigma  = 0.2148461844550287 # estimated volatility for log rates
mortg  = 100e3  # mortgage value
n      = 10 # number of periods to model for (times steps = n+1)
ratem  = 7.676816472182068/100 - 50e-4 # discounted mortgage security rate
numSim = 10000

# Construct and calibrate BDT model
ratesTree, _, _ = bdt.make_bdt_tree(yields, n, delta, sigma)

# Calculate semi-annual payments
coup = mortg / np.array([1/(1 + ratem/2)**(i) for i in 1 + np.arange(n-1)]).sum()

# Calculate loan outstanding balance
interestPaid = np.zeros(n+1)
principalPaid = np.zeros(n+1)
outstPrincipal = np.zeros(n+1)
for i in np.arange(n+1):
    if i == 0:
        interestPaid[i] = 0.0
        principalPaid[i] = 0.0
        outstPrincipal[i] = mortg
    elif i == n:
        interestPaid[i] = 0.0
        principalPaid[i] = 0.0
        outstPrincipal[i] = 0.0
    else:
        interestPaid[i] = outstPrincipal[i-1] * ratem / 2
        principalPaid[i] = coup - interestPaid[i]
        outstPrincipal[i] = outstPrincipal[i-1] - principalPaid[i]

mortgVal = np.zeros(numSim)
for i in np.arange(numSim):
    # select random rates path
    randArray = np.random.randint(2, size=(n-1))
    randSteps = np.cumsum(randArray)
    randSteps = np.concat(([0], randSteps))
    rates = ratesTree[randSteps, np.arange(n)]

    # discount CF backwards
    value = 0
    for j in np.flip(np.arange(n)):
        # discount coupons + previous time back in time
        value = np.exp(-delta*rates[j]) * (coup + value)

        # optimal prepayment condition
        optimPrepay = get_optimal_prepay(rates[j], value, j)
        optimPrepay = j != 0 and optimPrepay

        # suboptimal prepayment condition
        suboptimPrepay = get_suboptimal_prepay(j, delta)

        if optimPrepay or suboptimPrepay:
            # if value of mortgage greater than liability, prepay
            value = outstPrincipal[j]

    # save mortgage PV
    mortgVal[i] = value

mortgMean = np.average(mortgVal)
mortgStd = np.std(mortgVal)
mortgStError = mortgStd / np.sqrt(numSim)

print('Mortgage mean: \n', mortgMean)
print('\nMortgage standard error: \n', mortgStError)
print('\n95% confidence interval: \n[{:.2f}, {:.2f}]'.format(
    mortgMean-2*mortgStError, mortgMean+2*mortgStError))
