import pandas as pd
import numpy as np
import rates_model_bdt as bdt

# import data
# data = pd.read_excel('../yieldcurve2025.xlsx', sheet_name='4. spot curve')
# yields = np.array(data.iloc[4][1:].to_list()) / 100
yields = np.array([1.74, 2.13, 2.62, 3.04, 3.46, 3.80, 4.04, 4.21, 4.36, 4.52, 4.65]) / 100
yields = yields[:3]

# Ho-Lee model parameters
n = 3 # number of yield steps
delta = 0.5    # compounding period
sigma = 0.2142 # estimated volatility for log rates
coupon = .03   # Coupon rate (yearly)
facev = 100    # bond face value
strike = facev

ratesTree, _ = bdt.make_bdt_tree(yields, n, sigma, delta)
bondTree = np.nan * np.ones(ratesTree.shape)
optionTree = np.nan * np.ones(ratesTree.shape)
coup = facev * coupon * delta

# Price the coupon bond
for i in np.flip(np.arange(n)):
    for j in np.arange(i+1):
        rate = ratesTree[j,i]
        if i == n - 1:
            bondTree[j,i] = np.exp(-delta*rate) * (facev + coup)
        else:
            up = bondTree[j,i+1]
            down = bondTree[j+1,i+1]
            bondTree[j,i] = np.exp(-delta*rate) * ((up + down)/2 + coup)

# Price an option callable at bond face value
for i in np.flip(np.arange(n)):
    for j in np.arange(i+1):
        bond = bondTree[j,i]
        rate = ratesTree[j,i]
        if i == n - 1:
            optionTree[j,i] = np.max((bond - strike, 0))
        else:
            up = optionTree[j,i+1]
            down = optionTree[j+1,i+1]
            optionConti = np.exp(-delta*rate) * (up + down) / 2
            optionStop = np.max((bond - strike, 0)) if i > 0 else 0
            optionTree[j,i] = np.max((optionConti, optionStop))


callBond = bondTree[0,0] - optionTree[0,0]

print('Rates model (%): \n', 100*ratesTree)
print('\nBond prices: \n', bondTree)
print('\nCallable option: \n', optionTree)
print('\nPrice of the callale bond: {:.2f}.'.format(callBond))
