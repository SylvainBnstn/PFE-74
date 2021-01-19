import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Environment simulator
def plus(x):
  return x if x > 0 else 0

def minus(x):
  return 0 if x > 0 else -x

def shock(x):
  return np.sqrt(x)

###############################################################################
# Demand at time step t for current price p_t and previous price p_t_1
# d_0 - k*p_t                   --> linear demand
# - a*shock(plus(p_t - p_t_1))  --> when price increase, demand decrease
# + b*shock(minus(p_t - p_t_1)) --> when price decrease, demand increase
def d_t(p_t, p_t_1, d_0, k, a, b):
    d = plus(d_0 - k*p_t - a*shock(plus(p_t - p_t_1)) + b*shock(minus(p_t - p_t_1))) 
    return d if d<=30 else 30 #30 because the seller cannot sell over 30 nights a month

###############################################################################
# Profit at time step t
# = demand * margin
def profit_t(p_t, p_t_1, d_0, k, a, b, unit_cost):
  return d_t(p_t, p_t_1, d_0, k, a, b)*(p_t - unit_cost) 

def profit_t_d(p_t, demand):
    unit_cost = 10
    return demand*(p_t - unit_cost) 
# Total profit for price vector p over len(p) time steps
# = profit at time t=0 + profit from time t=1 to len(p)
def profit_total(p, unit_cost, d_0, k, a, b):
  return profit_t(p[0], p[0], d_0, k, 0, 0, unit_cost) + sum(map(lambda t: profit_t(p[t], p[t-1], d_0, k, a, b, unit_cost), range(len(p))))

###############################################################################
# Environment parameters

"""
#les 2 dim sont egales que lorsque les states cest juste le prix
price_grid = testwithdata.get_data()[0]
#price_grid = price_grid.transpose()
state_dim = len(price_grid)
action_dim = len(price_grid)
"""

###############################################################################
# Partial bindings for readability
def profit_t_response(p_t, p_t_1):
    d_0 = 250 #according to average price in the market because : d_0 - k*(((pt)))
    k = 1
    unit_cost = 10
    a_q = 1
    b_q = 3
    return profit_t(p_t, p_t_1, d_0, k, a_q, b_q, unit_cost)

def profit_response(p):
    d_0 = 250 #according to average price in the market because : d_0 - k*(((pt)))
    k = 1
    unit_cost = 10
    a_q = 1
    b_q = 3
    return profit_total(p, unit_cost, d_0, k, a_q, b_q)


"""
T=12
###############################################################################
# Find the optimal constante price
def find_opti_cst_price():
    profits = np.array([ profit_response(np.repeat(p, T)) for p in price_grid ])
    p_idx = np.argmax(profits)
    price_opt_const = price_grid[p_idx]
    
    print(f'Optimal price is {price_opt_const}, achieved profit is {profits[p_idx]}')
    
    plt.figure(figsize=(16,7))
    plt.plot(price_grid, profits)
    plt.xlabel("Price")
    plt.ylabel("Profit")
    plt.grid()
    plt.show()
    return price_opt_const

price_opt_const = find_opti_cst_price()

###############################################################################
# Find optimal sequence of prices using greedy search
def find_optimal_price_t(p_baseline, price_grid, t):   # evalutes all possible price schedules 
  p_grid = np.tile(p_baseline, (len(price_grid), 1))   # derived from the baseline by  
  p_grid[:, t] = price_grid                            # changing the price at time t
  profit_grid = np.array([ profit_response(p) for p in p_grid ])
  p_idx = np.argmax(profit_grid)
  price_opt_dynamic = price_grid[p_idx]
  return price_opt_dynamic

p_opt = np.repeat(price_opt_const, T)                  # start with the constant price schedule
for t in range(T):                                     # and optimize one price at a time
  price_t = find_optimal_price_t(p_opt, price_grid, t)
  p_opt[t] = price_t

print(p_opt)
print(f'Achieved profit is {profit_response(p_opt)}')
plt.figure(figsize=(16,7))
plt.plot(range(T), p_opt, c='red')
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid()
plt.show()
"""

