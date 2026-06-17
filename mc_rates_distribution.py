# Monte Carlo Interest Rate Distribution Model
# ============================================

import pandas as pd
import numpy as np
import rates_model_bdt as bdt
import rates_model_ho_lee as hl
from scipy.special import comb
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=200)

# import data
data = pd.read_excel('yieldcurve2025.xlsx', sheet_name='4. spot curve')
years = np.array(data.iloc[2,1:].to_list())
yields = np.array(data.iloc[4,1:].to_list()) / 100
print(100*yields)

# Model parameters
delta    = 0.5    # compounding period
sigmaLog = 0.2148 # estimated volatility for log rates
sigmaLin = 0.0173 # estimated vol for linear interest rates
n        = 20     # number of periods for model construction
numSims  = 100000 # num of MC runs

# Construct a calibrated BDT model
ratesTree, *_ = bdt.make_bdt_tree(yields, n, sigmaLog, delta)
finalRatesDbt = np.flip(ratesTree[:,-1])

# Produce 9 randomly distributed binary vars to end in one of the ten final rates
randArray = np.random.randint(2, size=(numSims, n-1), dtype=np.uint8)
randIndex = np.sum(randArray, axis=1)

uniqueMc, countsMc = np.unique(randIndex, sorted=True, return_counts=True)
ratioMc = countsMc / randIndex.size  # distribution in final nodes

# Some nodes with very small proabbility ay be missing
ratioAllBdt = np.zeros(n)
ratioAllBdt[uniqueMc] = ratioMc

print('\n Rates DBT model (%):\n', 100*ratesTree)
print('\n Final indexes reached:\n', uniqueMc)
print('\n Final rates distribution for all nodes (%):\n', 100*ratioAllBdt)
print('\n Final rates distribution (%):\n', 100*ratioMc)
print('\n Final rates values (%):\n', 100*finalRatesDbt)

# Test the results using binomial distribution
prob = 0.5
steps = n - 1
ratioTheory = np.array([
    comb(steps, k) * prob**(k) * (1 - prob)**(steps - k)
    for k in range(n)])
print('\nTheoretical distribution (%): \n', ratioTheory)

# Plot
bins = np.arange(steps + 2) - 0.5
histMc, _ = np.histogram(randIndex, bins=bins)
probMc = histMc / histMc.sum()

print(probMc)

plt.figure(1, figsize=(8, 4))
plt.bar(range(n), 100*ratioAllBdt, width=0.6, alpha=0.6, label='MC Distribution (BDT)')
plt.plot(range(n), 100*ratioTheory, 'o-', linewidth=1, label='Theoretical Binomial')
plt.xlabel('Number of Up Moves (k)')
plt.ylabel('Probability (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Construct a calibrated HL model
ratesTree, *_ = hl.make_ho_lee_tree(yields, n, sigmaLin, delta)
finalRatesHl = np.flip(ratesTree[:,-1])

# Produce 9 randomly distributed binary vars to end in one of the ten final rates
randArray = np.random.randint(2, size=(numSims, n-1), dtype=np.uint8)
randIndex = np.sum(randArray, axis=1)

uniqueMc, countsMc = np.unique(randIndex, sorted=True, return_counts=True)
ratioMc = countsMc / randIndex.size  # distribution in final nodes

# Some nodes with very small proabbility ay be missing
ratioAllHl = np.zeros(n)
ratioAllHl[uniqueMc] = ratioMc

print('\n', 60 * '=')
print('\n Rates HL model (%):\n', 100*ratesTree)
print('\n Final indexes:\n', uniqueMc)
print('\n Final rates distribution for all nodes (%):\n', 100*ratioAllHl)
print('\n Final rates distribution (%):\n', 100*ratioMc)
print('\n Final rates values (%):\n', 100*finalRatesHl)

plt.figure(2, figsize=(8, 4))
plt.bar(range(n), 100*ratioAllHl, width=0.6, alpha=0.6, label='MC Distribution')
plt.plot(range(n), 100*ratioTheory, 'o-', linewidth=1, label='Theoretical Binomial')
plt.xlabel('Number of Up Moves (k)')
plt.ylabel('Probability (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(3, figsize=(8, 4))
plt.plot(100*finalRatesHl, 100*ratioAllHl, 'o-', linewidth=1, label='Ho-Lee Tree')
plt.plot(100*finalRatesDbt, 100*ratioAllBdt, 'o-', linewidth=1, label='BDT Tree')
plt.xlabel('Rate in the Final Node (%)')
plt.ylabel('Probability (%)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
