# Fixed Income Securities and Credit Markets

## Introduction

This repository contains Python models for valuing fixed-rate mortgages and mortgage-backed securities under stochastic interest rates.

The project calibrates Ho-Lee and Black-Derman-Toy interest rate trees to the October 2025 Bank of England yield curve, values a fixed-rate mortgage with an embedded borrower prepayment option, and extends the framework to pass-through MBS, principal-only strips, and interest-only strips.

The main methods are:

* Binomial interest rate trees
* Backward induction
* Optimal prepayment modelling
* Monte Carlo simulation
* Suboptimal behavioural prepayment
* MBS and IO/PO strip valuation

Main assumptions:

* Mortgage notional: 100,000
* Term: 10 years
* Semi-annual time steps: `delta = 0.5`
* Number of periods: `n = 20`
* Monte Carlo paths: `100,000`
* Security coupon: mortgage rate minus 50 basis points

---

## Data

The models use:

```text
yieldcurve2025.xlsx
```

Specifically, the sheet:

```text
4. spot curve
```

The spot curve is used to compute market zero-coupon bond prices, which are the calibration targets for the Ho-Lee and BDT trees.

---

## Requirements

```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

# (a) Constructing Interest Rate Trees

## `rates_model_ho_lee.py`

Builds and calibrates a Ho-Lee interest rate tree.

The Ho-Lee model uses additive rate dynamics:

```text
r(i+1,j) = r(i,j) + theta_i * delta ± sigma * sqrt(delta)
```

Main features:

* Arithmetic short-rate model
* Calibrated to market zero-coupon bond prices
* Uses sequential calibration of `theta_i`
* Simple and transparent
* Can generate negative interest rates

Main function:

```python
make_ho_lee_tree(yields, n, sigma, delta=0.5, prob=0.5)
```

Returns:

* Calibrated rates tree
* Calibrated drift parameters

## `rates_model_bdt.py`

Builds and calibrates a Black-Derman-Toy interest rate tree.

BDT models log-rates:

```text
z = ln(r)
z(i+1,j) = z(i,j) + theta_i * delta ± sigma * sqrt(delta)
r = exp(z)
```

Main features:

* Lognormal short-rate model
* Rates remain strictly positive
* Calibrated in log-rate space
* More realistic for mortgage pricing than Ho-Lee when negative rates are undesirable

Main function:

```python
make_bdt_tree(yields, n, sigma, delta=0.5, prob=0.5)
```

Returns:

* Calibrated BDT rates tree
* Calibrated log-space drift parameters

---

# (b) Prepayment Logic

No separate script is used for this section.

Prepayment logic is embedded in:

```text
mortgage_model.py
mortgage_model_ho_lee.py
mc_mortgage_model.py
mc_mortgage_model_suboptim.py
mc_mortgage_model_pass.py
```

The borrower owns an embedded prepayment option. At each node, the model compares:

```text
Exercise value = max(mortgage value - outstanding principal, 0)
```

against the continuation value.

If exercise value is greater than continuation value, the model records prepayment in:

```python
prepayTree[j, i] = True
```

This prepayment tree is later reused in the MBS models.

---

# (c) Valuing the Mortgage Contract

## `mortgage_model.py`

Values the mortgage using the BDT tree.

The model:

1. Builds the calibrated BDT tree.
2. Calculates the fixed mortgage payment.
3. Builds the amortisation schedule.
4. Values the mortgage without prepayment.
5. Values the borrower’s prepayment option.
6. Solves for the fair mortgage rate.

The fair mortgage rate satisfies:

```text
Mortgage value without prepayment - Prepayment option value = Mortgage principal
```

Main function:

```python
calculate_mortgage(ratem, yields, n, sigma, delta, mortg)
```

Returns:

* Rates tree
* Mortgage value tree
* Prepayment option tree
* Interest payments
* Principal payments
* Outstanding principal
* Prepayment decision tree
* Fixed coupon payment

## `mortgage_model_ho_lee.py`

Ho-Lee version of the mortgage valuation model.

It follows the same structure as `mortgage_model.py`, but uses the Ho-Lee interest rate tree instead of BDT.

The main difference is that Ho-Lee can produce negative rates, which can increase the value of the prepayment option.

---

# (d) Constructing MBSs

## `pass_through_security.py`

Values a pass-through MBS using the BDT model.

The security receives:

* Scheduled principal
* Interest at the security rate
* Outstanding principal if prepayment occurs

The security rate is:

```text
security rate = mortgage rate - 50 basis points
```

Main function:

```python
calculate_passthrough_security(rateSecur, rateMort, ratesTree, yields, n, sigma, delta, mortg)
```

Returns:

* Pass-through security value tree
* Interest paid to security holders
* Principal paid
* Outstanding principal

## `pass_through_security_ho_lee.py`

Ho-Lee version of the pass-through MBS model.

It uses:

* Ho-Lee rates tree
* Ho-Lee mortgage prepayment tree
* Same pass-through valuation logic

## `pass_through_security_strip.py`

Decomposes the BDT pass-through security into:

```text
Pass-through = Principal-only strip + Interest-only strip
```

The PO strip receives principal cash flows. If prepayment occurs, the PO holder receives outstanding principal.

The IO strip receives interest cash flows. If prepayment occurs, future interest is lost and the IO value falls to zero.

Returns:

* Principal-only strip tree
* Interest-only strip tree
* Principal schedule
* Outstanding principal schedule

## `pass_through_security_strip_ho_lee.py`

Ho-Lee version of the IO/PO strip model.

It uses the same strip logic, but with Ho-Lee rates and Ho-Lee prepayment decisions.

---

# (e) Interest Rate Trees with Monte Carlo

## `mc_rates_distribution.py`

Simulates random paths through the calibrated BDT and Ho-Lee trees.

Each path consists of random up/down moves. The terminal node is determined by the number of up moves.

The script compares the simulated terminal-node distribution with the theoretical binomial distribution.

Main purpose:

* Validate Monte Carlo path generation
* Compare final-node rate distributions under Ho-Lee and BDT
* Show that both models have the same binomial path probabilities but different rate distributions

---

# (f) Valuing the Mortgage with Monte Carlo

## `mc_mortgage_model.py`

Values the BDT mortgage using Monte Carlo simulation.

The script:

1. Precomputes the BDT mortgage model and optimal prepayment tree.
2. Simulates random interest-rate paths.
3. Discounts mortgage cash flows along each path.
4. Applies the precomputed prepayment decision.
5. Reports the mean value, standard error, confidence interval, and convergence plot.

This acts as a Monte Carlo validation of the backward-induction mortgage valuation.

---

# (g) Suboptimal Prepayment

## `mc_mortgage_model_suboptim.py`

Adds realistic borrower behaviour to the Monte Carlo mortgage model.

Instead of assuming perfectly optimal prepayment, the model allows:

## Probabilistic optimal prepayment

When prepayment is financially optimal, the borrower prepays with probability:

```text
q = a * exp(-b * rate)
```

with:

```text
a = 0.8
b = 20
```

## Suboptimal behavioural prepayment

When prepayment is not financially optimal, the borrower may still prepay due to non-financial reasons such as moving house.

This is modelled using a PSA-style prepayment probability with seasonality.

The script reports:

* Mortgage mean value
* Standard deviation
* Standard error
* 95% confidence interval

---

# (h) MBS with Monte Carlo

## `mc_mortgage_model_pass.py`

Values the mortgage, pass-through MBS, PO strip, and IO strip using Monte Carlo simulation.

For each path, the script tracks four values:

```python
valueM   # mortgage
valueS   # pass-through security
valueSp  # principal-only strip
valueSi  # interest-only strip
```

If prepayment occurs:

```text
Mortgage value = outstanding principal
Pass-through value = outstanding principal
PO value = outstanding principal
IO value = 0
```

The script reports:

* Mortgage value
* Pass-through MBS value
* PO strip value
* IO strip value
* Standard errors and confidence intervals
* Sanity check that PO + IO equals the pass-through value

This is the most complete model because it combines:

* BDT rates
* Monte Carlo simulation
* Optimal prepayment
* Suboptimal prepayment
* Pass-through valuation
* IO/PO decomposition

---

# Additional Script

## `callable_bond_model.py`

A small standalone example of callable bond pricing using a BDT tree.

The model prices:

```text
Callable bond = Straight bond - Embedded call option
```

This is conceptually related to mortgage valuation because a mortgage with prepayment rights also contains an embedded option.

---

# Suggested Run Order

```bash
python rates_model_ho_lee.py
python rates_model_bdt.py
python mortgage_model_ho_lee.py
python mortgage_model.py
python pass_through_security_ho_lee.py
python pass_through_security.py
python pass_through_security_strip_ho_lee.py
python pass_through_security_strip.py
python mc_rates_distribution.py
python mc_mortgage_model.py
python mc_mortgage_model_suboptim.py
python mc_mortgage_model_pass.py
```

Optional:

```bash
python callable_bond_model.py
```

---

# Reference

Veronesi, P. (2010). *Fixed Income Securities: Valuation, Risk, and Risk Management*. John Wiley & Sons.
