# Kelly Optimization, Drawdown Constraints, and Out-of-Sample Testing

This project compares Full Kelly log-growth optimization with drawdown-constrained portfolio optimization using historical daily financial market data. The objective is to study how leverage, risk limits, and asset correlations impact long-term growth and portfolio drawdowns.

## 1. Data

Daily total returns are used for the following asset classes:

- S&P 500 Index
- Treasury Bills
- U.S. Government Bonds
- Corporate High-Yield Bonds

(Data sources can be updated as needed — e.g., Wharton, Bloomberg, Federal Reserve Bank of St. Louis.)

All asset returns are converted into excess returns relative to the T-bill rate.

## 2. Optimization Methods

### 2.1 Full Kelly Optimization

Maximizes expected log-growth:

```
maximize    E[ log(1 + w' r) ]
```

- Solved numerically via SLSQP
- Allows leverage (weights 0 → 20)
- Produces aggressive, growth-optimal portfolios

### 2.2 Drawdown-Constrained Optimization (PSG)

Maximizes expected uncompounded return under a max-drawdown constraint:

```
maximize    linear(matrix_annualized_returns)
subject to  drawdown_dev_max(matrix_scenarios) ≤ a
            sum(weights) = 1
            weights ≥ lower bounds
```

- α ranges from 5% to 50%
- Uses daily scenarios to estimate drawdown behavior
- Produces safer, risk-controlled portfolios

## 3. Out-of-Sample Rolling Algorithm

The OOS test is performed monthly:

### Step 1 — Training Window
Use the previous 157 months (~13 years) of daily data.

### Step 2 — Optimization
Run Full Kelly or Drawdown-Constrained optimization and store the optimal weights.

### Step 3 — Testing
Apply the weights to the next month's returns.

### Step 4 — Accumulate Results
Store the realized monthly return.

### Step 5 — Slide Forward
Move the window ahead one month and repeat.

This produces a realistic, walk-forward backtest of each strategy.

## 4. Outputs

- Out-of-sample cumulative return curves
- Maximum drawdown comparison
- Kelly vs. drawdown-constrained performance
- Efficient frontiers under varying drawdown limits
- Leverage levels implied by each optimization method
- Benchmark comparison vs. S&P 500
