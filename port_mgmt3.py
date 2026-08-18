import numpy as np
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import cvxpy as cp # for convex optimization, clearer than linprog
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta


def load_data(path: str):
    with open(path, "r") as f:
        #lines = f.readlines()
        df = pd.read_csv(path)
        # make sure Date is a datetime index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

        for col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "")
                .str.replace(",", "")
                .astype(float)
            )

        return df

    

def log_returns(prices): # calculates log returns
    returns = np.log(prices) - np.log(prices.shift(1))
    returns = returns.dropna()
    return returns

def split(returns): # seperates data into testing and training data
    split_num = round(len(returns)*0.6)
    train_d = returns[0:split_num]
    test_d = returns[split_num:len(returns)]
    return train_d, test_d

def find_betas(stock_returns, sp500_returns):
# "The most straightforward way to estimate the betas is to use linear regression vs the S&P."
    betas = {}

    aligned = stock_returns.join(sp500_returns, how='inner') #inner joins sp500 returns w the stock returns
    market = aligned[sp500_returns.name]   # IDs sp500 from aligned
    
    # Add constant for alpha calculation
    market_with_const = sm.add_constant(market)

    for ticker in stock_returns.columns:
        y = aligned[ticker]    # dependent variable = stock return
        model = sm.OLS(y, market_with_const).fit()

        alpha = model.params['const']
        beta = model.params[sp500_returns.name]
        
        betas[ticker] = {"alpha": alpha, "beta": beta}

    return pd.DataFrame(betas).T
'''
old - expected returns only historical
def find_expected_returns(stock_returns):
    mu_daily = stock_returns.mean() # average log return
    mu_annual = mu_daily * 252 # annualized for 252 trading days
    return mu_annual
'''

def find_expected_returns(stock_returns, lookback_days=60, kappa=0.5):
    """
new - tweak by momentum
    """

    # baseline: historical mean returns (annualized)
    mu_daily = stock_returns.mean()          # average daily log return
    mu_annual = mu_daily * 252               # annualized baseline
    mu_annual = mu_annual.astype(float)

    # momentum signal over the last `lookback_days` days
    L = min(lookback_days, len(stock_returns))
    if L <= 1:
        # not enough data for a momentum signal; just return baseline
        return mu_annual

    momentum_window = stock_returns.tail(L)

    # cumulative log return over the lookback window (≈ log of total return)
    momentum_raw = momentum_window.sum(axis=0)

    # standardize momentum cross-sectionally (z-score)
    momentum_mean = momentum_raw.mean()
    momentum_std = momentum_raw.std(ddof=0) 

    if momentum_std == 0 or np.isnan(momentum_std):
        # no cross-sectional variation; no tilt
        momentum_z = pd.Series(0.0, index=mu_annual.index)
    else:
        momentum_z = (momentum_raw - momentum_mean) / momentum_std 

    # 3) Scale the tilt by the cross-sectional volatility of baseline mu
    mu_std = mu_annual.std(ddof=0)
    if mu_std == 0 or np.isnan(mu_std):
        # degenerate case: just use baseline
        return mu_annual

    # final expected returns: baseline + momentum tilt
    mu_final = mu_annual + kappa * mu_std * momentum_z

    return mu_final

        
def port_long(cov, beta, mu, gamma):

    tickers = mu.index
    n = len(tickers)

    mu_vec = mu.values
    cov_mat = cov.values
    beta_vec = beta["beta"].values
    beta_cvx = cp.Constant(beta_vec)
    # i added this line because cvxpy was tweaking about the shape of beta
    
    x = cp.Variable(n) # values, weights, decision variable
        # "vector variable with shape [n]"

    objective = cp.Maximize(mu_vec @ x - (gamma/2)*cp.quad_form(x, cov_mat))
                    # max mu T x  - 1/2 gamma x T V x

    # constraints below
    constraints = [
        cp.sum(x) == 1, # long only so the raw weights have to add to one, basis 1
        x >= 0, # no short
        x <= 0.1, # no one position can be greater than 10% of my portfolio
        beta_cvx.T @ x == 1 # portfolio beta = 1
        ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(x.value, index=tickers)

def port_130_30(cov, beta, mu, gamma):

    tickers = mu.index
    n = len(tickers)

    mu_vec = mu.values
    cov_mat = cov.values
    beta_vec = beta["beta"].values
    beta_cvx = cp.Constant(beta_vec)

    x_long  = cp.Variable(n)
    x_short = cp.Variable(n)    
    x = x_long - x_short # recall leverage constraints

    objective = cp.Maximize(mu_vec @ x - (gamma/2)*cp.quad_form(x, cov_mat))
            # same objective max mu T x  - 1/2 gamma x T V x

    constraints = [
        cp.sum(x_long) == 1.3,      # 130 long
        cp.sum(x_short) == 0.3,     # 30 short
        x_long >= 0,        # they both pos cuz that makes most sense
        x_short >= 0,
        beta_cvx.T @ x == 1,# portfolio beta = 1
        cp.sum(x) == 1,     # net exposure = 1, this is the basis I think from notes
        x_long <= 0.1,
        x_short <= 0.1
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(x.value, index=tickers)

def port_long_short(cov, beta, mu, gamma):
    
    tickers = mu.index
    n = len(tickers)

    mu_vec = mu.values
    cov_mat = cov.values
    beta_vec = beta["beta"].values
    beta_cvx = cp.Constant(beta_vec)
    # i added this line because cvxpy was tweaking about the shape of beta
    
    x = cp.Variable(n) # values, weights, decision variable
        # "vector variable with shape [n]"

    objective = cp.Maximize(mu_vec @ x - (gamma/2)*cp.quad_form(x, cov_mat))
            # same objective max mu T x  - 1/2 gamma x T V x

    gross_limit = 2.0   # 200% total exposure

    constraints = [
        beta_cvx.T @ x == 1,
        cp.sum(x) == 1,
        -1 <= x, x <= 1,     # narrower individual bounds
        cp.norm1(x) <= gross_limit
]


    problem = cp.Problem(objective, constraints)
    problem.solve()

    return pd.Series(x.value, index=tickers)

def backtest_static(stock_returns, sp500_returns, cov, beta, mu, gamma, port_funcs):
    """
    Static train/test backtest using precomputed cov, beta, mu.
    Does NOT recompute any model inputs.
    """

    # split into train/test
    train_d, test_d = split(stock_returns)
    sp500_train, sp500_test = split(sp500_returns)

    # compute weights for each strategy
    weights = {}
    for name, fn in port_funcs.items():
        w = fn(cov, beta, mu, gamma)
        weights[name] = w

    # align test data to tickers
    any_w = next(iter(weights.values()))
    test_d = test_d[any_w.index]

    # convert log returns to simple returns
    test_simple = np.exp(test_d) - 1.0

    wealth = {}

    # compute wealth paths
    for name, w in weights.items():
        r = test_simple @ w
        w_series = (1 + r).cumprod()
        w_series /= w_series.iloc[0]
        wealth[name] = w_series

    # SPX OOS
    sp500_test_simple = np.exp(sp500_test) - 1.0
    w_spx = (1 + sp500_test_simple).cumprod()
    w_spx /= w_spx.iloc[0]
    wealth["SPX"] = w_spx

    return wealth


def backtest_rolling(stock_returns, sp500_returns, gamma, port_funcs,
                     window_days):
    """
    Rolling backtest:
      - rolling window of `window_days` observations (~2 years)
      - rebalance at quarter-end dates
      - Apply the selected weight vector DAILY until the next rebalance
    """
    returns_all = stock_returns.sort_index()
    sp500_all   = sp500_returns.sort_index()

    # Rebalance dates (quarter ends)
    #rebal_dates = returns_all.resample("Q").last().index
    #rebal_dates = returns_all.index[returns_all.resample("Q").last().index]
    # Quarter-end dates in the universe of TRADING days, not calendar days
    rebal_dates = returns_all.resample("Q").last().index

    # Map each quarter-end to the nearest previous trading day
   # calendar quarter-end dates
    calendar_q_end = returns_all.resample("Q").last().index

    # map calendar dates to nearest *previous* trading day
    idx = returns_all.index.get_indexer(calendar_q_end, method="pad")
    rebal_dates = returns_all.index[idx]

    rebal_dates = pd.to_datetime(rebal_dates)


    # containers for DAILY portfolio returns
    roll_returns = {
        name: pd.Series(index=returns_all.index, dtype=float)
        for name in port_funcs.keys()
    }

    n_dates = len(rebal_dates)

    for i in range(1, n_dates):
        reb_date = rebal_dates[i]
        prev_reb_date = rebal_dates[i - 1]

        if reb_date not in returns_all.index:
            continue

        reb_idx = returns_all.index.get_loc(reb_date)
        train_start_idx = reb_idx - window_days
        if train_start_idx < 0:
            continue

        # Training window for cov, betas, mu
        train_window = returns_all.iloc[train_start_idx:reb_idx]
        sp500_train  = sp500_all.loc[train_window.index]

        cov  = train_window.cov()
        beta = find_betas(train_window, sp500_train)
        mu   = find_expected_returns(train_window)

        # Compute PORTFOLIO WEIGHTS at this rebalance
        weights = {name: fn(cov, beta, mu, gamma)
                   for name, fn in port_funcs.items()}

        # Out-of-sample window: from reb_date (EXCLUDED) until next rebalance date
        if i == n_dates - 1:
            # last interval ends at the end of the dataset
            next_window = returns_all.loc[reb_date:]
        else:
            next_reb_date = rebal_dates[i + 1]
            next_window = returns_all.loc[reb_date:next_reb_date].iloc[1:]

        if next_window.empty:
            continue

        # simple returns for next_window
        test_simple = np.exp(next_window) - 1.0

        # compute DAILY returns for each strategy
        for name, w in weights.items():
            aligned = test_simple[w.index]
            daily_r = aligned @ w
            roll_returns[name].loc[daily_r.index] = daily_r

    # Build wealth curves
    wealth = {}
    for name, r in roll_returns.items():
        r = r.fillna(0.0)
        w_series = (1 + r).cumprod()
        first_valid = w_series.first_valid_index()
        if first_valid is not None:
            w_series /= w_series.loc[first_valid]
        wealth[name] = w_series

    # SPX wealth
    sp500_simple_all = np.exp(sp500_all) - 1.0
    wealth_spx = (1 + sp500_simple_all).cumprod()
    first_valid = wealth_spx.first_valid_index()
    if first_valid is not None:
        wealth_spx /= wealth_spx.loc[first_valid]
    wealth["SPX"] = wealth_spx

    return wealth

def compute_summary_stats(wealth_dict, rf=0.0):
    """
    wealth_dict: dict of wealth Series for each strategy (including SPX)
    rf: daily risk-free rate (0 for simplicity)

    Returns a DataFrame of summary metrics for each portfolio.
    """

    results = []

    # Extract SPX daily returns (used as market return)
    spx = wealth_dict["SPX"]
    spx_ret = spx.pct_change().dropna()

    for name, wealth in wealth_dict.items():
        if name == "SPX":
            continue  # skip SPX as a strategy entry (we use it as benchmark)

        # daily returns
        ret = wealth.pct_change().dropna()

        # annualized mean return
        ann_return = (1 + ret.mean()) ** 252 - 1

        # annualized volatility
        ann_vol = ret.std() * np.sqrt(252)

        # sharpe ratio (risk-free assumed 0 unless specified)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

        # max drawdown
        running_max = wealth.cummax()
        drawdown = (wealth - running_max) / running_max
        max_dd = drawdown.min()

        # CAPM regression vs SPX
        aligned = pd.concat([ret, spx_ret], axis=1).dropna()
        aligned.columns = ["ret", "mkt"]

        X = sm.add_constant(aligned["mkt"])
        y = aligned["ret"]
        model = sm.OLS(y, X).fit()

        alpha_daily = model.params["const"]
        beta = model.params["mkt"]

        # translate daily α to annualized α
        alpha_annual = (1 + alpha_daily) ** 252 - 1

        r2 = model.rsquared

        results.append({
            "Strategy": name,
            "Ann Return": ann_return,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Beta": beta,
            "Alpha (ann)": alpha_annual,
            "R^2": r2,
            "Max Drawdown": max_dd,
        })

    return pd.DataFrame(results).set_index("Strategy")

    
def main():
    #this path patch below I was forced to do because somehow python doesnt like windows file paths which is shocking
    #dont know if this is most efficient
    base_dir = Path.home() / "Desktop" / "Programming" / "Python" / "fnce4820 Python" / "active portfolio management"

    stock_path = base_dir / "stock_data.csv"
    sp500_path = base_dir / "SP500.csv"

    stock_data = load_data(stock_path)
    sp500_data = load_data(sp500_path)


    stock_returns = log_returns(stock_data) #log returns of stocks
    
    sp500_returns = log_returns(sp500_data) # log returns of S&P500
    sp500_returns.drop(sp500_returns.tail(1).index, inplace=True) # drop last row
    sp500_returns = sp500_returns.iloc[:, 0] # make into a series for lin reg => betas
    
    train_d, test_d = split(stock_returns) # train data, test data
    sp500_train, sp500_test = split(sp500_returns)
    
    # inputs to problem found from training data
    cov = train_d.cov() # V covariance matrix
    beta = find_betas(train_d, sp500_train) # beta vector
    mu = find_expected_returns(train_d) # mu vector
    gamma = 15.0 # risk aversion, you pick this
    
    w_long_only = port_long(cov, beta, mu, gamma)
    w_130_30 = port_130_30(cov, beta, mu, gamma)
    w_long_short = port_long_short(cov, beta, mu, gamma)

    port_funcs = {
        "Long-only":  port_long,
        "130/30":     port_130_30,
        "Long/Short": port_long_short,
    }

    # static backtest
    
    static_wealth = backtest_static(
        stock_returns,
        sp500_returns,
        cov,
        beta,
        mu,
        gamma,
        port_funcs
    )

    plt.figure(figsize=(10, 6))
    for name, w in static_wealth.items():
        w.plot(label=name)
    plt.title("Out-of-sample Wealth (Static Train/Test)")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # rolling backtest

    
    rolling_wealth = backtest_rolling(stock_returns, sp500_returns, gamma, port_funcs,
                                      window_days=504)

    plt.figure(figsize=(10, 6))
    for name, w in rolling_wealth.items():
        w.plot(label=name)
    plt.title("Wealth – Rolling 2-year Window, Quarterly Rebalancing")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # summary stats
    stats_table = compute_summary_stats(rolling_wealth)
    print(stats_table.applymap(lambda x: f"{x:.4f}"))


    
main()
