import numpy as np
import mortgage_model as mm
import pandas as pd

# Calculate mortgage value using Monte Carlo
np.set_printoptions(precision=2, linewidth=200)

def get_suboptimal_prepay(iter, delta):
    if iter == 0:
        return False
    else:
        # Suboptimal probability
        cpr = np.min((12*iter*delta*0.2, 6)) / 100
        seasonIndex = 2 if np.mod(iter, 2) == 0 else 1
        prepayProbThresh = seasonIndex * (1 - (1 - 0.5*cpr)**delta)
        
        # Randomly decide whether to prepay
        randNum = np.random.uniform()
        suboptimPrepay = randNum < prepayProbThresh
        return suboptimPrepay


def get_optimal_prepay(rate, a=0.8, b=20.0):
    # Calculate random prepayment decision
    prepayProbThresh = a * np.exp(-rate * b)
    randNum = np.random.uniform()
    optimPrepayRand = randNum < prepayProbThresh
    return optimPrepayRand

# import data
data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
yields = np.array(data.iloc[4][1:].to_list()) / 100

# BDT model parameters
delta  = 0.5    # compounding period
sigma  = 0.2142 # estimated volatility for log rates
mortg  = 100e3  # mortgage value
n      = 20 # number of periods to model for
ratem  = 4.6818353753114925/100 # optimal mortgage rate
numSim = 100000

# Precalculate mortgage params
(ratesTree, _, _, interestPaid,
 principalPaid, outstPrincipal,
 prepayTree, coup) = mm.calculate_mortgage(
     ratem, yields, n, sigma, delta, mortg)

# Calculate mortgage value for each random path realisation
mortgVal = np.zeros(numSim)
for i in np.arange(numSim):
    # select random rates path
    randArray = np.random.randint(2, size=(n-1))
    randSteps = np.cumsum(randArray)
    randSteps = np.concatenate(([0], randSteps))
    rates = ratesTree[randSteps, np.arange(n)]

    # discount CF along the random path
    value = 0
    for j in np.flip(np.arange(n)):
        # discount coupons + previous time back in time
        value = np.exp(-delta*rates[j]) * (coup + value)

        # Check whether we are in optimal prepayment node
        isOptimal = prepayTree[randSteps[j], j]

        if isOptimal:
            # Optimal prepayment adjustment to rates (conditional on optimal prep.)
            # i.e. prepayment from mortg model and rates adjusted behaviour
            optimPrepay = get_optimal_prepay(rates[j])
            suboptimPrepay = False

        else:
            # Suboptimal prepayment (conditional on not optimal prep.)
            suboptimPrepay = get_suboptimal_prepay(j, delta)
            optimPrepay = False

        if optimPrepay or suboptimPrepay:
            # if value of mortgage greater than liability, prepay
            value = outstPrincipal[j]
    
    # save mortgage PV
    mortgVal[i] = value

mortgMean = np.average(mortgVal)
mortgStd = np.std(mortgVal)
mortgStError = mortgStd / np.sqrt(numSim)

print('Mortgage mean: \n', mortgMean)
print('Mortgage STD: \n', mortgStd)
print('\nMortgage standard error: \n', mortgStError)
print('\n95% confidence interval: \n[{:.2f}, {:.2f}]'.format(
    mortgMean-1.96*mortgStError, mortgMean+1.96*mortgStError))
