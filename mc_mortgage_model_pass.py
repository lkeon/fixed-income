import numpy as np
import pandas as pd
import mortgage_model as mm

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
delta   = 0.5    # compounding period
sigma   = 0.2142 # estimated volatility for log rates
mortg   = 100e3  # mortgage value
n       = 20 # number of periods to model for
rateMrt = 4.6818353753114925/100 # optimal mortgage rate
numSim  = 100000
rateSec = rateMrt - 50e-4 # 50 bp lower

# Precalculate mortgage params
(ratesTree, _, _, interestPaid,
 principalPaid, outstPrincipal,
 prepayTree, coup) = mm.calculate_mortgage(
     rateMrt, yields, n, sigma, delta, mortg)

# Precalculate interest payments to security
interestPaidSec = np.zeros(n+1)
for i in range(1, n+1):
    interestPaidSec[i] = outstPrincipal[i-1] * rateSec/2

# Preallocate
mortgVal  = np.zeros(numSim)
secVal    = np.zeros(numSim)
secPriVal = np.zeros(numSim)
secIntVal = np.zeros(numSim)

# Calculate mortgage value for each random path realisation
for i in np.arange(numSim):
    # select random rates path
    randArray = np.random.randint(2, size=(n-1))
    randSteps = np.cumsum(randArray)
    randSteps = np.concatenate(([0], randSteps))
    rates = ratesTree[randSteps, np.arange(n)]

    # Preallocate
    valueM  = 0 # current MC sample value of mortgage
    valueS  = 0 # current MC sample value of security
    valueSp = 0 # current MC value of security principal strip
    valueSi = 0 # current MC value of security interest strip

    # Discount CF along the random path
    for j in np.flip(np.arange(n)):
        # mortgage: discount coupons + previous time back in time
        disc = np.exp(-delta*rates[j])
        valueM = disc * (coup + valueM)

        # security: discount future value + future CF
        nextCf = interestPaidSec[j+1] + principalPaid[j+1]
        valueS = disc * (valueS + nextCf)

        # separate security strips
        valueSp = disc * (valueSp + principalPaid[j+1])
        valueSi = disc * (valueSi + interestPaidSec[j+1])

        # Check whether we are in optimal prepayment node
        isOptimalPrep = prepayTree[randSteps[j], j]

        if isOptimalPrep:
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
            valueM  = outstPrincipal[j]
            valueS  = outstPrincipal[j]
            valueSp = outstPrincipal[j]
            valueSi = 0 # no future interests if preapying
    
    # Save mortgage/security PV into MC vector
    mortgVal[i]  = valueM
    secVal[i]    = valueS
    secPriVal[i] = valueSp
    secIntVal[i] = valueSi

# Estimate mortgage
mortgMean = np.average(mortgVal)
mortgStd = np.std(mortgVal)
mortgStError = mortgStd / np.sqrt(numSim)

# Estimate security
secMean = np.average(secVal)
secStd = np.std(secVal)
secStError = secStd / np.sqrt(numSim)

print('Mortgage mean: \n', mortgMean)
print('\nMortgage standard error: \n', mortgStError)
print('\n95% confidence interval: \n[{:.2f}, {:.2f}]'.format(
    mortgMean-2*mortgStError, mortgMean+2*mortgStError))

print('\n', 60 * '=')
print('Security mean: \n', secMean)
print('\nSecurity standard error: \n', secStError)
print('\n95% confidence interval: \n[{:.2f}, {:.2f}]'.format(
    secMean-2*secStError, secMean+2*secStError))

print('\nSecurity principal strip mean: \n', np.mean(secPriVal))
print('\nSecurity interest strip mean: \n', np.mean(secIntVal))
print('\nSecurity strip summation sanity check: \n',
      np.mean(secPriVal) + np.mean(secIntVal))
