import numpy as np
import pandas as pd
import mortgage_model as mm
import matplotlib.pyplot as plt

# Calculate mortgage value using Monte Carlo
np.set_printoptions(precision=2, linewidth=300)

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
 prepayTree, coup) = mm.calculate_mortgage(ratem, yields, n, sigma, delta, mortg)

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
        
        # get the optimal prepayment logic from the mortgage model
        if prepayTree[randSteps[j], j]:
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

cumMean = np.cumsum(mortgVal) / np.arange(1, mortgVal.size+1)
plt.figure(3, figsize=(8, 4))
plt.plot(range(1,cumMean.size+1), cumMean, linewidth=1)
plt.xlabel('MC Iteration')
plt.ylabel('Cumulative Mean Value')
plt.grid(True, alpha=0.3)
plt.show()
