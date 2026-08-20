# main is down at the bottom
from datetime import date, datetime, timedelta
from calendar import monthcalendar, FRIDAY
import re

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import math

def load_options(path: str):

    with open(path, "r") as f:
        lines = f.readlines()

    # Find the first line containing the keyword (case-insensitive)
    keyword = "Expiration"
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start_row = i
            break
    df = pd.read_csv(path, skiprows=start_row)
    return df

def friday(d: pd.Timestamp):
    if pd.isna(d): return False # returns false if missing
    cal = monthcalendar(d.year, d.month)
    third_fri = [wk[FRIDAY] for wk in cal if wk[FRIDAY] != 0][2] #finds 3rd week friday
    return (d.weekday() == 4) and (d.day == third_fri) #returns true if is the 3rd friday of the month

def kbid_ask(df, expiry):
    '''
    only want strike k, bids, asks

    from left to right (and our investor perspective), the data should be call bids(short call) call ask(long call), strike,
    put bid(short put) put ask (long put)
    
    '''
    cols = ['Expiration Date','Bid', 'Ask', 'Strike', 'Bid.1', 'Ask.1']
    out = df[cols].rename(columns={
        'Expiration Date':'Expiration',
        'Bid': 'Call Bid',
        'Ask': 'Call Ask',
        'Strike': 'Strike',
        'Bid.1': 'Put Bid',
        'Ask.1': 'Put Ask'
    })


    out['Expiration'] = pd.to_datetime(out['Expiration'], errors='coerce').dt.normalize() #makes sure they all actually dates in datetime
    out['Expiry Type'] = out['Expiration'].apply(lambda d: 'Monthly' if friday(d) else 'Weekly')
 
    #creates a column of expiry type that applies my function to check whether 3rd friday, returns true if it is 

                    
    return out[out['Expiry Type'].str.lower() == expiry.lower()] #returns either the monthly or weekly options based on input expiry


'''
building A
A has a column for each option, aka long calls at strike k0, long calls at strike k1, short calls at k0,
short calls k1, so on and so forth. So the rows of A should be equal to 4x the # of strikes
(4 option for each strike). the rows of A should be 0, the strikes(representing end prices) ,
and then an extra row for the slope of the payoff of each option (1 or -1).
The elements of a are going to be the value of each option at each strike.
AKA what is a long call with strike 1000 worth if the price at the end of the options life is 500? it should be 0.
'''
def _call_payoff(S, K):  # long call payoff
    return np.maximum(S - K, 0.0)

def _put_payoff(S, K):   # long put payoff
    return np.maximum(K - S, 0.0)

def _slope_row_for_strike(Ks):
    # for one K: [call_long=1, call_short=-1, put_long=-1, put_short=+1]
    ones = np.ones(len(Ks))
    zeros = np.zeros(len(Ks))
    return np.concatenate([ +ones, -ones, zeros, zeros])


def _make_scenarios(Ks, include_zero=True):
    # Your spec: rows at S=0 and S=each strike
    S = []
    if include_zero:
        S.append(0.0)
    S.extend(Ks)  # S = [0, K1, K2, ..., KN]
    return np.array(S, dtype=float)

def _build_payoff_rows(S, Ks):
    """
    builds pay off rows dbrown style aka blocks of similar options for every strike
    """
    N = len(Ks)
    #long call block
    C_plus = np.column_stack([_call_payoff(S, K) for K in Ks])
    #short call = negative of long call
    C_minus = -C_plus
    #long put block
    P_plus = np.column_stack([_put_payoff(S, K) for K in Ks])
    #short put = negative of long put
    P_minus = -P_plus

    #put together blocks horizontally
    A = np.hstack([C_plus, C_minus, P_plus, P_minus])  # (|S|) x (4N)
    return A


def _build_cost_vector(df_ordered_by_strike):
    K = df_ordered_by_strike['Strike'].to_numpy(float)
    ca = df_ordered_by_strike['Call Ask'].to_numpy(float)
    cb = df_ordered_by_strike['Call Bid'].to_numpy(float)
    pa = df_ordered_by_strike['Put Ask'].to_numpy(float)
    pb = df_ordered_by_strike['Put Bid'].to_numpy(float)

    # [C+, C-, P+, P-]
    C_plus_cost  = +ca      # pay ask
    C_minus_cost = -cb      # receive bid
    P_plus_cost  = +pa
    P_minus_cost = -pb

    p = np.concatenate([C_plus_cost, C_minus_cost, P_plus_cost, P_minus_cost])
    return p

def _column_labels(Ks):
    lab = []
    for K in Ks: lab.append(f"C+_{K:g}")
    for K in Ks: lab.append(f"C-_{K:g}")
    for K in Ks: lab.append(f"P+_{K:g}")
    for K in Ks: lab.append(f"P-_{K:g}")
    return lab

def build_Apb(df: pd.DataFrame, include_slope=True):
    """
    Build A (payoff matrix), p (cost coeffs), S (scenario prices), and column labels.
    df must have: call_bid, call_ask, strike, put_bid, put_ask
    Rows = scenarios (S=0 plus each strike). Optional last row = slope row.
    Cols = [C+ K1..KN | C- K1..KN | P+ K1..KN | P- K1..KN]
    """
    #ensure sorted by strike and numeric
    work = df.copy()
    work = work[['Call Bid','Call Ask','Strike','Put Bid','Put Ask']].dropna()
    work['Strike'] = pd.to_numeric(work['Strike'], errors='coerce')
    work = work.sort_values('Strike').reset_index(drop=True)

    Ks = work['Strike'].to_numpy(float)
    #scenarios
    S = _make_scenarios(Ks, include_zero=True)         # shape (M,)
    #payoff rows
    A = _build_payoff_rows(S, Ks)                      # shape (M, 4N)
    #optional slope row
    if include_slope:
        slope = _slope_row_for_strike(Ks)[None, :]     # shape (1, 4N)
        A = np.vstack([A, slope])
    #cost vector
    p = _build_cost_vector(work)                     # shape (4N,)
    #labels
    cols = _column_labels(Ks)

    return A, p, S, cols


'''
end of A building
'''

def solve_min_cost(A, p, cols=None, include_slope=True, tol=1e-8):
    """
    Minimize p^T x
    s.t.  A_pay x >= 1     (sure $1 payoff in every scenario row)
          x >= 0
    Arbitrage if optimal cost <= 0 (within tol).
    """
    A = np.asarray(A, float)
    p = np.asarray(p, float)
    M, N = A.shape

    # Use only payoff rows (drop slope row if present)
    A_pay = A[:-1, :] if include_slope else A
    m = A_pay.shape[0]

    # linprog uses <= form:  -A_pay x <= -1
    G = -A_pay
    h = -np.zeros(m) # this is messed up


    '''
    if include_slope:
        slope = A[-1, :]                  # slope row
        G = np.vstack([G, -slope[None, :]])
        h = np.concatenate([h, [0.0]])
    '''

    if include_slope:
    # True slope = change in payoff between last two stock-price scenarios
        slope = A[-1, :] - A[-2, :]

        G = np.vstack([G, -slope[None, :]])  # enforce slope ≥ 0
        h = np.concatenate([h, [0.0]])

        
    G = np.vstack([G, -p[None, :]])
    h = np.concatenate([h, [1.0]])
    # row of prices >= -1
    
    # Objective: minimize p^T x
    c = p.copy()

    # Bounds: x >= 0
    bounds = [(0, None)] * N

    res = linprog(c, A_ub=G, b_ub=h, bounds=bounds, method="highs")

    if not res.success or res.x is None:
        print("LP status:", res.status, res.message)
        return res, "LP failed", {"message": res.message}

    x = res.x
    cost = float(p @ x)
    worst = float((A_pay @ x).min())  # should be >= 1


    # find positions
    active = [(name, qty) for name, qty in zip(cols, x) if abs(qty) > 1e-8] #array of positions, active

    if active:
        #normalize by smallest position
        min_pos = min(qty for _, qty in active if qty > 1e-8)
        norm_factor = 1.0 / min_pos
        cost_norm = float(p @ (x * norm_factor))

        print("\n--- Normalized Arbitrage Portfolio (min position = 1) ---")
        for name, qty in active:
            if qty > 1e-8:
                side = "LONG" if "+" in name else "SHORT"
                print(f"{side:6s} {name:10s}  {qty * norm_factor:10.0f}")

        verdict = "Arb"
    else:
        verdict = "No Arb"
        

    
    return res, verdict
"""
Part 2
"""

def create_position(position):
    
    if len(position) != 4:
        raise ValueError("position must have 4 elements: [side, type, strike, expiry_type]")
        # not sure if I actually need this given it can fail ungracefully

    side, opt_type, strike, expiry_type = position
    # load shit in to the high iq


      # normalize inputs
    side = side.strip().lower() # I believe this is taking off spaces n shit 
    opt_type = opt_type.strip().lower()
    expiry_type = expiry_type.strip().lower()
     # map shorthand C/P to full names
    if opt_type in ["c", "call"]:
        opt_type = "call"
    elif opt_type in ["p", "put"]:
        opt_type = "put"
    else:
        raise ValueError(f"Unknown option type: {opt_type}")

    if side not in ["long", "short"]:
        raise ValueError(f"Unknown side: {side}")
    
    if expiry_type not in ["monthly", "daily"]:
        raise ValueError(f"Unknown expiry: {side}")


    if side == "long":
        e_side = "short"
    else:
        e_side = "long"
        
    #direct_exit = [e_side, opt_type, float(strike), expiry_type]
    

    return {
        "side": side,
        "type": opt_type,
        "strike": float(strike),
        "expiry_type": expiry_type
        }

def spread_cross(position): # i'm too dumb to build this into the function above
    
    e_side, opt_type, strike, expiry_type = position
    # load shit in to the high iq
     # normalize inputs
    e_side = e_side.strip().lower() # I believe this is taking off spaces n shit 
    opt_type = opt_type.strip().lower()
    expiry_type = expiry_type.strip().lower()
    
     # map shorthand C/P to full names
    if opt_type in ["c", "call"]:
        opt_type = "C"
    elif opt_type in ["p", "put"]:
        opt_type = "P"

    if e_side == "long":
        side = "short"
    else:
        side = "long"
        
    direct_exit = [side, opt_type, float(strike), expiry_type]
    

    return direct_exit


# build vector w all 0s except for my position
def position_vector(A_cols, pos):
    side = pos[0].upper()      # "LONG" or "SHORT"
    t    = pos[1].upper()      # "C" or "P"
    k    = int(float(pos[2]))  # 3000 (ensure int to match labels)
    leg  = '+' if side == 'LONG' else '-'
    target = f"{t}{leg}_{k}"
    return [1 if col == target else 0 for col in A_cols]

def add_zcb(A, c, r, days, K=1.0):
    zcb_col = np.ones((A.shape[0], 1))
    zcb_col[-1, 0] = 0
    A = np.hstack((A, zcb_col))

    # ZCB cost (present value of K at maturity)
    c_zcb = K * np.exp(-r * (days / 365))
    c = np.append(c, c_zcb)

    print(' ')
    print(f'{r*100}% interest over {days} days')
    print(f"Zero-coupon bond cost: {c_zcb:.6f}")
    print(' ')
    return A, c


def solve_exit(A, c, x, e_vec):
    
    b_target = -A @ x
    A_eq = A
    b_eq = b_target

    #b_eq = np.zeros(A_eq.shape[0])#zeros b
  
    # Slight cost regularizer
    c_reg = c + 1e-9
    
    value = c.T @ x
    e_value = c.T @ e_vec
    
    # Variable bounds: all nonneg except ZCB free
    n_vars = len(c_reg)
    bounds = [(0, None)] * (n_vars - 1) + [(None, None)]  # last var = ZCB



    #scale bound
    #A_eq = np.vstack([A_eq, np.ones((1, A_eq.shape[1]))])
    #b_eq = np.append(b_eq, 100.0)

    res = linprog(c_reg, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    # Output diagnostics
    if not res.success:
        print(f"LP failed: {res.message}")
    
    return res, value, e_value


    
def pretty_results(res, A_cols, c, valu, e_val):

    print("\nSynthetic Exit")
    print(f"{'Instrument':<14}{'Position':>14}{'Cost':>18}")
    print("-" * 48)

    total_cost = 0.0

    for i, (name, val) in enumerate(zip(A_cols, res.x)):
        if abs(val) > 1e-6:  # skip zeros
            cost = c[i] * val if (c is not None and i < len(c)) else 0.0
            total_cost += cost
            print(f"{name:<14}{val:14.6f}{cost:18.6f}")

    print("-" * 48)
    print(f"{'Exit Cost:':<28}{total_cost:18.6f}")
    #print(f"{'Position Cost:':<28}{valu:18.6f}")
    print(" ")
    print(f"{'Bid/Ask Spread:':<28}{abs(abs(valu)-abs(e_val)):18.6f}")
    print(f"{'Synthetic Spread:':<28}{abs(abs(valu)-abs(total_cost)):18.6f}")
    print(" ")

    if not hasattr(res, "x") or res.x is None or np.isscalar(res.x):
        print("(No valid solution — LP failed or unbounded)")
        print("-" * 48)
        return

'''
main
'''
def main():
    print(" --- Part 1 ---")
    print(" ")
    #INPUTS BELOW
    cboe_path = r"C:\Users\colin\Desktop\Programming\Python\FNCE4820 Python\options arb\SPX_Options1_NoArb.csv"
    expiry = "Monthly"
      
    cboe_data = load_options(cboe_path)
    options = kbid_ask(cboe_data, expiry)
    
    print(options)
    #print(' ')

    #need weekly/monthly expiry

    A, p, S, cols = build_Apb(options,include_slope=True)
    #print(p)
    print(' ')

    #solve 
    res, verdict = solve_min_cost(A, p, include_slope=False, cols=cols)
    print("Verdict:", verdict)
    print(' ')

    """
    part 2
    """
    print(" --- Part 2 ---")
    print(" ")
    
    # INPUTS BELOW
    rate = 0.045
    position = ["long","p","7500","Monthly"]
    parsed = create_position(position)
    e_pos = spread_cross(position) 
    
    print(f'Position: {parsed}')
    
    x = position_vector(cols, position)
    x = np.append(x, 0)
    
    e_vec = position_vector(cols, e_pos)
    e_vec = np.append(e_vec, 0)
    
    newA, newC= add_zcb(A, p, rate, days=128)

    res2, val, e_val = solve_exit(newA, newC, x, e_vec)
    cols.append('ZCB')
    pretty_results(res2, cols, newC, val, e_val)
    # https://www.cboe.com/delayed_quotes/spx/quote_table
    # https://www.cboe.com/delayed_quotes/mbtx/quote_table
main()
