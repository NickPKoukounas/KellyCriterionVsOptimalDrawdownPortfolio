import sys
import os
import pandas as pd
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.io 
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
pyo.init_notebook_mode()
from psgutilities import *
import nbformat
sys.path.insert(1, os.path.join(os.environ.get('PSG25_HOME'), 'Python'))
from psg_preload_libraries import initialize_psg_environment
initialize_psg_environment()
import psgpython as psg
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

sp500srsp = pd.read_excel('SP500crspDailyReturn1926_2024_forStonybrook.xlsx',header=1)
# check missing data 
missing_counts = sp500srsp.isnull().sum()
print(missing_counts) #first row has missing values

#drop the missing data row 
df = sp500srsp.dropna().copy()
missing_counts = df.isnull().sum()
print("Removed the Missing rows, check again:") 
print(missing_counts)

sp500srsp.dropna()
df_1996 = df[df['caldt'] >= '1996-12-01']
df = df_1996
df

df = df.sort_values('caldt').reset_index(drop=True)
df['caldt'] = pd.to_datetime(df['caldt'])

# Compute cumulative (uncompounded) portfolio rate of return 
df['CumulativeRet'] = df['CRSP SP500 TotRet'].cumsum()  

# Compute running max and drawdown
df['RunningMax'] = df['CumulativeRet'].cummax()
df['Drawdown'] = df['RunningMax'] - df['CumulativeRet']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),sharex=True)
# PLot cumulative portfolio rate of return
ax1.plot(df['caldt'], df['CumulativeRet'], color='blue', markersize=1)
ax1.set_title("Uncompounded Cumulative Portfolio Rate of Return", fontsize=13)
ax1.grid(True)
ax1.set_ylabel("Portfolio rate of return", fontsize=11, color='black')

# Plot Drawdown
ax2.plot(df['caldt'], df['Drawdown'], color='orange', linewidth=1)
ax2.grid(True)
ax2.set_xlabel("caldt", fontsize=11)
ax2.set_ylabel("Drawdown", fontsize=11, color='black')

plt.show()


# Low boounds
header_lowerbounds = list(['CRSPSP500TotRet', 'TBillTotRet'])
point_lowerbounds_body = np.array([0.0,-10e5], order='C')
point_lowerbounds = [header_lowerbounds, point_lowerbounds_body]
print(point_lowerbounds)

# matrix_annualized_returns
dailymean_sp500 = df['CRSP SP500 TotRet'].mean()
dailymean_tbill = df['TBillTotRet'].mean()  
#Uncompounded
uncom_ann_mean_sp500 = dailymean_sp500*252
uncom_ann_mean_tbill = dailymean_tbill*252
header_matrix_annualized_returns = list(['CRSPSP500TotRet', 'TBillTotRet'])
uncom_matrix_annualized_returns_body = np.array(np.column_stack((uncom_ann_mean_sp500, uncom_ann_mean_tbill)),order='C')
uncom_matrix_annualized_returns = [header_matrix_annualized_returns, uncom_matrix_annualized_returns_body]
print(uncom_matrix_annualized_returns)

#Compounded
com_ann_mean_sp500 = (1 + dailymean_sp500)**252 - 1
com_ann_mean_tbill = (1 + dailymean_tbill)**252 - 1
com_matrix_annualized_returns_body = np.array(np.column_stack((com_ann_mean_sp500, com_ann_mean_tbill)),order='C')
com_matrix_annualized_returns = [header_matrix_annualized_returns, com_matrix_annualized_returns_body]
print(com_matrix_annualized_returns)

# matrix_scenarios
daily_sp500 = sp500srsp['CRSP SP500 TotRet'].dropna().tolist()
daily_tbill = sp500srsp['TBillTotRet'].dropna().tolist()
header_matrix_scenarios = list(['CRSPSP500TotRet', 'TBillTotRet'])
matrix_scenarios_body = np.array(np.column_stack((daily_sp500, daily_tbill)),order='C')
matrix_scenarios = [header_matrix_scenarios, matrix_scenarios_body]
print(matrix_scenarios)

# budget
header_budget = ['CRSPSP500TotRet', 'TBillTotRet']
matrix_budget_body = np.array([1.0, 1.0])  # each column corresponds to a weight coefficient = 1
matrix_budget = [header_budget, matrix_budget_body]
print(matrix_budget)


#using analitical mu/sig**2

sd = np.std(df['RfreeAdjSP500'])
np.mean(df['RfreeAdjSP500']) / sd**2
np.mean(df['RfreeAdjSP500'])


allowExternal = True
suppressMessages = False
np.random.seed(42)

drawdown_limits = np.arange(.2, 6.0, .2)


objectives = []
opt_weights = []
sp500_weights = []
tbill_weights = []
F = []
Port_returns = []


for limit in drawdown_limits:
    problem_name = f'problem_maxdd_{limit:.2f}'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= {limit:.2f}
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary = {
        'problem_name': problem_name,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': uncom_matrix_annualized_returns,
        'matrix_scenarios': matrix_scenarios,
        'point_lowerbounds': point_lowerbounds
    }

    # Solve the optimal solution
    result = psg.psg_solver(problem_dictionary, allowExternal, suppressMessages)

    # Get the objective values
    objective_text = result['output'][3]
    objective = result['point_problem_1'][1] @ uncom_matrix_annualized_returns[1].flatten()
    objectives.append(objective)

    # Get the optimal weights
    opt_x = result['point_problem_1'][1]
    opt_weights.append(opt_x)
    sp500_weight = opt_x[0]
    sp500_weights.append(sp500_weight)
    tbill_weight = opt_x[1]
    tbill_weights.append(tbill_weight)

    port_return = df[['CRSP SP500 TotRet','TBillTotRet']] @ opt_x
    Port_returns.append(port_return)


    Ftmp = 1 - tbill_weight
    F.append(Ftmp)
    

uncom_results_df = pd.DataFrame({"Drawdown_Limit": drawdown_limits,"Uncompounded Return": objectives, "Uncompounded SP500 Weights": sp500_weights, "Uncompounded Tbill Weights": tbill_weights})

allowExternal = True
suppressMessages = False
np.random.seed(42)



com_objectives = []
com_opt_weights = []
com_sp500_weights = []
com_tbill_weights = []
com_F = []
com_Port_returns = []


for limit in drawdown_limits:
    problem_name = f'problem_maxdd_{limit:.2f}'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= {limit:.2f}
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary = {
        'problem_name': problem_name,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': com_matrix_annualized_returns,
        'matrix_scenarios': matrix_scenarios,
        'point_lowerbounds': point_lowerbounds
    }

    # Solve the optimal solution
    result = psg.psg_solver(problem_dictionary, allowExternal, suppressMessages)

    # Get the objective values
    objective_text = result['output'][3]
    objective = result['point_problem_1'][1] @ com_matrix_annualized_returns[1].flatten()
    com_objectives.append(objective)

    # Get the optimal weights
    opt_x = result['point_problem_1'][1]
    com_opt_weights.append(opt_x)
    sp500_weight = opt_x[0]

    com_sp500_weights.append(sp500_weight)
    tbill_weight = opt_x[1]
    com_tbill_weights.append(tbill_weight)

    port_return = df[['CRSP SP500 TotRet','TBillTotRet']] @ opt_x
    com_Port_returns.append(port_return)


    Ftmp = 1 - tbill_weight
    com_F.append(Ftmp)

    

com_results_df = pd.DataFrame({"Drawdown_Limit": drawdown_limits,"Compounded Return": com_objectives, "Compounded SP500 Weights": com_sp500_weights, "Compounded Tbill Weights": com_tbill_weights})


results_df = pd.DataFrame({"Drawdown_Limit": drawdown_limits,
                           "Compounded Return": com_objectives, "Compounded SP500 Weights": com_sp500_weights, "Compounded Tbill Weights": com_tbill_weights,"Compounded f(Leverage)": com_F,
                           "Uncompounded Return": objectives, "Uncompounded SP500 Weights": sp500_weights, "Uncompounded Tbill Weights": tbill_weights,"Uncompounded f(Levergae)": F
                          
                          })
results_df

'''
Getting the f fraction g(f) graph uncompounded
'''
tbill = df['TBillTotRet'].to_numpy()
F_uncom = results_df['Uncompounded f(Levergae)']

Freturns = []
for i in F_uncom:
    Freturns.append(i * df['RfreeAdjSP500'])

Gf = np.mean(np.log(1 + tbill*1 + np.array(Freturns)), axis = 1) *250

loc = np.argmax(Gf) 
FullKelly = F_uncom[loc] 
maxGF = Gf[loc]


plt.title("Log Compounded Return", fontsize=14, weight='bold')
plt.plot(F_uncom, Gf, color = 'r')
plt.vlines(FullKelly, ymin=min(Gf), ymax=maxGF, color='blue', linestyle='--')
plt.xlabel("f fraction")
plt.ylabel("Expected Log Return")
plt.grid()
print("Full Kelly:", FullKelly)

# Made a change 
plt.figure(figsize=(8,5))
#plt.plot(drawdown_limits, objectives, 'r-', linewidth=2, label="Uncompounded Efficient Frontier")
plt.plot(drawdown_limits, Gf*250, 'b-', linewidth=2, label="Compounded Efficient Frontier")
#plt.scatter(drawdown_limits, objectives, color='k', marker='d')  
plt.scatter(drawdown_limits, Gf * 250)
plt.title("Log Compounded Return", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Expected Log Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()

# Made a change 
plt.figure(figsize=(8,5))
#plt.plot(drawdown_limits, objectives, 'r-', linewidth=2, label="Uncompounded Efficient Frontier")
plt.plot(drawdown_limits, objectives, 'b-', linewidth=2, label="Efficient Frontier")
#plt.scatter(drawdown_limits, objectives, color='k', marker='d')  
plt.scatter(drawdown_limits, objectives)
plt.title("Efficient Frontier", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Uncompounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()


# Made a change 
plt.figure(figsize=(8,5))
#plt.plot(drawdown_limits, objectives, 'r-', linewidth=2, label="Uncompounded Efficient Frontier")
plt.plot(drawdown_limits, com_objectives, 'b-', linewidth=2, label="Compounded Efficient Frontier")
#plt.scatter(drawdown_limits, objectives, color='k', marker='d')  
plt.scatter(drawdown_limits, com_objectives)
plt.title("Efficient Frontier: Compounded Return", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Copounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()


equity_curves = []

for port_return in com_Port_returns:
    wealth = (1 + port_return).cumprod() * 100  # initial wealth = 100
    equity_curves.append(wealth)

# Plot
plt.figure(figsize=(12,6))

# plot each portfolio curve
for i, wealth in enumerate(equity_curves):
    plt.plot(df['caldt'], wealth, linewidth=1, alpha=0.8)

# plot SP500 curve
sp500_wealth = (1 + df['CRSP SP500 TotRet'].dropna()).cumprod() * 100
plt.plot(df.loc[df['CRSP SP500 TotRet'].notna(), 'caldt'], sp500_wealth, color='r', label='S&P 500')

# log scale & labels
plt.yscale('log')
plt.title("Logarithmic Scale - Portfolios by Drawdown Limit")
plt.xlabel("Date")
plt.ylabel("Portfolio Wealth ($100 Start)")
plt.legend()
plt.tight_layout()
plt.show()


equity_curves = []

for port_return in Port_returns:
    wealth = 100 + port_return.cumsum()  # initial wealth = 100
    equity_curves.append(wealth)

# Plot
plt.figure(figsize=(12,6))

# plot each portfolio curve
for i, wealth in enumerate(equity_curves):
    plt.plot(df['caldt'], wealth, linewidth=1, alpha=0.8)

# plot SP500 curve
sp500_wealth_uncom = 100 + df['CRSP SP500 TotRet'].dropna().cumsum()
plt.plot(df.loc[df['CRSP SP500 TotRet'].notna(), 'caldt'], sp500_wealth_uncom, color='r', label='S&P 500')

# log scale & labels

plt.title("Uncompounded Scale - Portfolios by Drawdown Limit")
plt.xlabel("Date")
plt.ylabel("Portfolio Wealth ($100 Start)")
plt.legend()
plt.tight_layout()
plt.show()


# uncompounded
optWeight = sp500_weights[loc], tbill_weights[loc]
FullKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight
FullKellySeries = 100 + (FullKellyReturns).cumsum()

SP500_uncom = 100 + df['CRSP SP500 TotRet'].cumsum()

plt.title("Uncompouned", fontsize=14, weight='bold')
plt.plot(df['caldt'], FullKellySeries, color = 'blue', label='FullKellySeries')
plt.plot(df['caldt'], SP500_uncom, color = 'red', label = 'S&P500')
#plt.plot(sp500_wealth, color = 'red', label='sp500_wealth') # redo this , should be linear
plt.legend()
plt.show()
sp500_wealth


'''
Getting the f fraction g(f) graph Compounded
'''
F_com = results_df['Compounded f(Leverage)']

Freturns_com = []
for i in F_com:
    Freturns_com.append(i * df['RfreeAdjSP500']) #(mu - r)

Gf = np.mean(np.log(1 + tbill*1 + np.array(Freturns_com)), axis = 1)

loc = np.argmax(Gf) #find the index of maximum log return
FullKelly_comp = F_com[loc] # find the maximum log return f fraction which is full kelly 
maxGF = Gf[loc] # find the maximum log return 


plt.title("Compounded", fontsize=14, weight='bold')
plt.plot(F_com, Gf, color = 'r')
plt.xlabel("f fraction")
plt.ylabel("Mean of log capital")
plt.vlines(FullKelly_comp, ymin=min(Gf), ymax=maxGF, color='blue', linestyle='--')
plt.grid()
print("Full Kelly:", FullKelly_comp)
print("maxGF:", maxGF)


# compounded
optWeight = sp500_weights[loc], tbill_weights[loc]
FullKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight
FullKellySeries = (1 + FullKellyReturns).cumprod() * 100
sp500_wealth_com = (1 + df['CRSP SP500 TotRet'].dropna()).cumprod() * 100

plt.title("Compounded", fontsize=14, weight='bold')
plt.plot(df['caldt'], FullKellySeries, color = 'blue', label='FullKellySeries')
plt.plot(df['caldt'], sp500_wealth_com, color = 'red', label='sp500_wealth')
plt.legend()
plt.show()

AvgDD = []
for i in range(len(opt_weights)):
    tmpdata = df[['CRSP SP500 TotRet', 'TBillTotRet']] @ opt_weights[i]
    cum_value = tmpdata.cumsum()
    drawdown = cum_value - cum_value.cummax()
    AvgDD.append(drawdown.mean())

plt.title("Uncompounded", fontsize=14, weight='bold')
plt.plot(results_df['Uncompounded f(Levergae)'], np.array(AvgDD)* -1 , label='AvgDD')
plt.plot(results_df['Uncompounded f(Levergae)'], drawdown_limits, label = 'MaxDD')
plt.xlabel('Leverage F')
plt.ylabel('*100% Scale')
plt.plot(results_df['Uncompounded f(Levergae)'], objectives, label = 'Annual Return') #in excel using monthly
plt.grid()
plt.legend()
plt.show()

AvgDD_com = []
for i in range(len(opt_weights)):
    tmpdata = df[['CRSP SP500 TotRet','TBillTotRet']] @ opt_weights[i]
    cum_value = (1 + tmpdata).cumprod()
    drawdown = (cum_value / cum_value.cummax()) - 1
    AvgDD_com.append(drawdown.mean())


plt.title("Compounded", fontsize=14, weight='bold')
plt.plot(results_df['Compounded f(Leverage)'], np.array(AvgDD_com)* -1 , label='AvgDD')
plt.plot(results_df['Compounded f(Leverage)'], drawdown_limits, label = 'MaxDD')
plt.plot(results_df['Compounded f(Leverage)'], com_objectives, label = 'Annual Return') #in excel using monthly
plt.grid()
plt.xlabel('Leverage F')
plt.ylabel('% Scale')
plt.legend()
plt.show()


# how to find half kelly's drawdown_limit?
# based on the halfKelly find the corresponding drawdown_limit
# FullKelly = 1 - tbill_weight
# tbill_weight = 1 - FullKelly -> -0.17735324066563818
print("FullKelly:", FullKelly)
print("HalfKelly:",FullKelly/2)
print("HalfKelly Tbill weight:", 1-(FullKelly/2))

# create tbill weight constraint
header_budget = ['CRSPSP500TotRet', 'TBillTotRet']
matrix_tbill_body = np.array([0, 1.0]) 
matrix_tbill = [header_budget, matrix_tbill_body]
print(matrix_tbill)

problem_name = 'problem_HalfKelly_drawdown_limit'
problem_statement = f"""
minimize
  drawdown_dev_max(matrix_scenarios)
Constraint: = -0.17735324
  linear(matrix_tbill)
Constraint: = 1
  linear(matrix_budget)
Box: >= point_lowerbounds
"""

problem_dictionary = {
    'problem_name': problem_name,
    'problem_statement': problem_statement,
    'matrix_budget' : matrix_budget,
    'matrix_scenarios': matrix_scenarios,
    'matrix_tbill': matrix_tbill,
    'point_lowerbounds': point_lowerbounds
}

# Solve the problem
result_halfdraw = psg.psg_solver(problem_dictionary, allowExternal, suppressMessages)
result_halfdraw['output'][3] #half Kelly Drawdown Limit

Gf_halfkelly = np.mean(np.log(1 + df['TBillTotRet'] + np.array((FullKelly/2) * df['RfreeAdjSP500'])))

# plot FullKelly & Half Kelly 
plt.title("Uncompouned", fontsize=14, weight='bold')
plt.plot(F_uncom, Gf, color = 'r')
plt.vlines(FullKelly, ymin=min(Gf), ymax=maxGF, color='blue', linestyle='--', label='FullKelly')
plt.vlines(FullKelly/2, ymin=min(Gf), ymax=Gf_halfkelly, color='green', linestyle='--', label='HalfKelly')
plt.xlabel("f fraction")
plt.ylabel("Mean of log capital")
plt.legend()
plt.grid()

# compounded
optWeight = sp500_weights[loc], tbill_weights[loc]
FullKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight
FullKellySeries = (1 + FullKellyReturns).cumprod() * 100

sp500_weight_half = FullKelly/2
tbill_weight_half = 1-FullKelly/2
optWeight_half = sp500_weight_half, tbill_weight_half
HalfKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight_half 
HalfKellySeries = (1 + HalfKellyReturns).cumprod() * 100

plt.title("Compouned", fontsize=14, weight='bold')
plt.plot(FullKellySeries, color = 'blue', label='FullKellySeries')
plt.plot(HalfKellySeries, color = 'green', label='HalfKellySeries')
plt.plot(sp500_wealth_com, color = 'red', label='sp500_wealth')
plt.legend()
plt.show()


# uncompounded
optWeight = sp500_weights[loc], tbill_weights[loc]
FullKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight
FullKellySeries_uncom = FullKellyReturns.cumsum() + 100

sp500_weight_half = FullKelly/2
tbill_weight_half = 1-FullKelly/2
optWeight_half = sp500_weight_half, tbill_weight_half
HalfKellyReturns = df[['CRSP SP500 TotRet','TBillTotRet']] @ optWeight_half 
HalfKellySeries_uncom = HalfKellyReturns.cumsum() + 100

plt.title("uncompouned", fontsize=14, weight='bold')
plt.plot(df['caldt'],FullKellySeries_uncom, color = 'blue', label='FullKellySeries')
plt.plot(df['caldt'], HalfKellySeries_uncom, color = 'green', label='HalfKellySeries')
plt.plot(df['caldt'], SP500_uncom, color = 'red', label = 'S&P500')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return')
plt.show()


USBond = pd.read_csv('USGovBondsVUSTX.csv',header=0).dropna()
USBond['Date'] = pd.to_datetime(USBond['Date'], format="%m/%d/%y")
USBond = USBond[::-1]
USBond['USGovBonds(VUSTX)'] = USBond['USGovBonds(VUSTX)'].pct_change()
USBond = USBond.dropna()
USBond = USBond[USBond['Date'] <= '2024-12-31']
USBond = USBond
merged = pd.merge(df, USBond, left_on='caldt', right_on='Date', how='inner')
merged.dropna()

x = merged['CRSP SP500 TotRet']
y = merged['TBillTotRet']
z = merged['USGovBonds(VUSTX)']


header_budget2 = ['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)']
matrix_datanew = np.array(np.column_stack((x,y,z)), order = 'C')
data = [header_budget2, matrix_datanew]
data


# Low boounds
header_lowerbounds = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)'])
point_lowerbounds_body = np.array([0.0,-10e5, 0], order='C')
point_lowerbounds_3 = [header_lowerbounds, point_lowerbounds_body]
print(point_lowerbounds)

# matrix_annualized_returns
dailymean_sp500 = merged['CRSP SP500 TotRet'].mean()
dailymean_tbill = merged['TBillTotRet'].mean() 
dailymean_corpbond = merged['USGovBonds(VUSTX)'].mean() 
#Uncompounded
uncom_ann_mean_sp500 = dailymean_sp500*252
uncom_ann_mean_tbill = dailymean_tbill*252
uncom_ann_mean_corpbond = dailymean_corpbond*252
header_matrix_annualized_returns = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)'])
uncom_matrix_annualized_returns_body = np.array(np.column_stack((uncom_ann_mean_sp500, uncom_ann_mean_tbill,uncom_ann_mean_corpbond)),order='C')
uncom_matrix_annualized_returns_3 = [header_matrix_annualized_returns, uncom_matrix_annualized_returns_body]
print(uncom_matrix_annualized_returns)

#Compounded
com_ann_mean_sp500 = (1 + dailymean_sp500)**252 - 1
com_ann_mean_tbill = (1 + dailymean_tbill)**252 - 1
com_ann_mean_corpbond = (1 + dailymean_corpbond)**252 - 1
com_matrix_annualized_returns_body = np.array(np.column_stack((com_ann_mean_sp500, com_ann_mean_tbill, com_ann_mean_corpbond)),order='C')
com_matrix_annualized_returns = [header_matrix_annualized_returns, com_matrix_annualized_returns_body]
print(com_matrix_annualized_returns)

# budget
header_budget = ['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)']
matrix_budget_body = np.array([1.0, 1.0, 1.0])  # each column corresponds to a weight coefficient = 1
matrix_budget = [header_budget, matrix_budget_body]
print(matrix_budget)

#matrix scenarios 
data

allowExternal = True
suppressMessages = False
np.random.seed(42)

drawdown_limits2 = np.arange(.2, 4, .1)
problem_name_3 = 'problem_maxdd_3'

objectives_3 = []
opt_weights_3 = []
sp500_weights_3 = []
tbill_weights_3 = []
USGOV_weights = []
F_3 = []
Port_returns_3 = []


for limit in drawdown_limits2:
    problem_name = f'problem_maxdd_{limit:.2f}'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= {limit:.2f}
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary_3 = {
        'problem_name': problem_name_3,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': uncom_matrix_annualized_returns_3,
        'matrix_scenarios': data,
        'point_lowerbounds': point_lowerbounds_3
        
    }

    # Solve the optimal solution
    result = psg.psg_solver(problem_dictionary_3, allowExternal, suppressMessages)
    # Get the objective values
    objective_text = result['output'][3]
    objective = result['point_problem_1'][1] @ uncom_matrix_annualized_returns_3[1].flatten()
    objectives_3.append(objective)

    # Get the optimal weights
    opt_x = result['point_problem_1'][1]
    opt_weights_3.append(opt_x)
    sp500_weight = opt_x[0]
    sp500_weights_3.append(sp500_weight)
    tbill_weight = opt_x[1]
    tbill_weights_3.append(tbill_weight)
    USGOV_weight = opt_x[2]
    USGOV_weights.append(USGOV_weight)


    port_return = merged[['CRSP SP500 TotRet','TBillTotRet', 'USGovBonds(VUSTX)']] @ opt_x
    Port_returns.append(port_return)


    Ftmp = (sp500_weight + USGOV_weight) - 1
    F_3.append(Ftmp)
    

uncom_results_df = pd.DataFrame({"Drawdown_Limit": drawdown_limits2,"Uncompounded Return": objectives_3, "Uncompounded SP500 Weights": sp500_weights_3, "Uncompounded Tbill Weights": tbill_weights_3, 'US Bond wieghts': USGOV_weights, 'F uncom leverage': F_3})

# Made a change 
plt.figure(figsize=(8,5))
#plt.plot(drawdown_limits, objectives, 'r-', linewidth=2, label="Uncompounded Efficient Frontier")
plt.plot(drawdown_limits2, objectives_3, 'b-', linewidth=2, label="Uncompounded Efficient Frontier")
#plt.scatter(drawdown_limits, objectives, color='k', marker='d')  
plt.scatter(drawdown_limits2, objectives_3)
plt.title("Efficient Frontier", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Uncompounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()



'''
Getting the f fraction g(f) graph uncompounded 3 asset
'''

USBondExcess = merged['USGovBonds(VUSTX)'] - merged['TBillTotRet']
tbill = merged['TBillTotRet'].to_numpy()
sp_w = sp500_weights_3
US_w = USGOV_weights


weight_3 = np.column_stack((sp_w, US_w))
port_daily = np.column_stack((merged['RfreeAdjSP500'], USBondExcess))

ports = []
for k in range(len(weight_3)):
    r_tmp = port_daily @ weight_3[k]
    ports.append(r_tmp)


Gf_3 = []
for _ in range(len(ports)):
    gf_tmp = np.mean(np.log(1 + tbill*1 +  ports[_]))
    Gf_3.append(gf_tmp)

loc3 = np.argmax(Gf_3)
maxGF3 = Gf_3[loc3]
FullKelly3 = F_3[loc3]
plt.plot(F_3, Gf_3, color = 'red')
plt.vlines(FullKelly3, ymin=min(Gf_3), ymax=maxGF3, color='blue', linestyle='--')
plt.xlabel('F Leverage')
plt.ylabel('Log return')
plt.title('2 Asset Portfolio Expected Log Growth')
plt.grid(True)
plt.show()
print("2 Asset Portfolio Expected Log Growth:",FullKelly3)



equity_curves3 = []

for port_return in ports:
    wealth = 100 + port_return.cumsum()  # initial wealth = 100
    equity_curves3.append(wealth)

# Plot
plt.figure(figsize=(12,6))

# plot each portfolio curve
for i, wealth in enumerate(equity_curves3):
    plt.plot(merged['Date'], wealth, linewidth=1, alpha=0.8)

# plot SP500 curve
sp500_wealth_uncom = 100 + merged['CRSP SP500 TotRet'].dropna().cumsum()
plt.plot(merged.loc[merged['CRSP SP500 TotRet'].notna(), 'caldt'], sp500_wealth_uncom, color='r', label='S&P 500')

# log scale & labels

plt.title("Uncompounded Scale - 3 asset Portfolios by Drawdown Limit")
plt.xlabel("Date")
plt.ylabel("Portfolio Wealth ($100 Start)")
plt.legend()
plt.tight_layout()
plt.show()

#uncompounded
optWeight = sp500_weights_3[loc3], tbill_weights_3[loc3], USGOV_weights[loc3]
FullKellyReturns = merged[['CRSP SP500 TotRet','TBillTotRet','USGovBonds(VUSTX)']] @ optWeight
FullKellySeries = 100 + FullKellyReturns.cumsum() # 100 capital

plt.title("Uncompouned", fontsize=14, weight='bold')
plt.plot(FullKellySeries, color = 'blue', label='FullKellySeries')
plt.plot(sp500_wealth_uncom, color = 'red', label='sp500_wealth')
plt.legend()
plt.show()

f1 = uncom_results_df['Uncompounded SP500 Weights']
f2 = uncom_results_df['US Bond wieghts']
scatter = plt.scatter(f1, f2, c=np.array(Gf_3)*250, cmap='viridis', s=80)
plt.title('Optimal Portfolio Curve')
plt.xlabel('Stock Weight')
plt.ylabel('Bond Weight')
cbar = plt.colorbar(scatter)
cbar.set_label('Expected Log Growth')
plt.show()

SP = pd.DataFrame({'SP return' : merged['CRSP SP500 TotRet']})
SP.to_excel("SPY.Return.xlsx", index = False)


TBill = pd.DataFrame({'Tbill return' : merged['TBillTotRet']})
TBill.to_excel("TBill.Return.xlsx", index = False)


USBond = pd.DataFrame({'USBond return' : merged['USGovBonds(VUSTX)']})
USBond.to_excel("USBond.Return.xlsx", index = False)

Numrows = len(merged)
Startrow = 1

TbillRet = pd.read_excel("TBill.Return.xlsx", header=0).squeeze("columns").astype(float).to_numpy()
Rt1 = pd.read_excel("SPY.Return.xlsx", header=0).squeeze("columns").astype(float).to_numpy()
Rt2 = pd.read_excel("USBond.Return.xlsx", header=0).squeeze("columns").astype(float).to_numpy()

Returns1 = Rt1 - TbillRet
Returns2 = Rt2 - TbillRet
NoReturns1 = len(Returns1)
NoReturns2 = len(Returns2)

def Loggrowth(f1, f2):
    rp = TbillRet + f1 * Returns1 + f2 * Returns2
    port_ret = 1 + rp
    if np.any(port_ret <= 0): return np.nan
    return (250 / NoReturns1) * np.sum(np.log(port_ret))

def maxdrawdown1(f1, f2, Returns1, Returns2, TbillRet):
    rp = TbillRet + f1 * Returns1 + f2 * Returns2
    Nav, Maxv, PercDD, MaxDD = [100.0], [100.0], [0.0], [0.0]
    for j in range(len(rp)):
        Nav.append(Nav[-1] * (1 + rp[j]))
        Maxv.append(max(Nav[-1], Maxv[-1]))
        PercDD.append((Nav[-1] - Maxv[-1]) / Maxv[-1])
        MaxDD.append(max(-PercDD[-1], MaxDD[-1]))
    return MaxDD[-1]


N1, N2 = 100, 100
Z = np.zeros((N1, N2))
f1min, f1max, f2min, f2max = 0.0, 7.0, 0.0, 9.0
df1, df2 = (f1max - f1min) / (N1 - 1), (f2max - f2min) / (N2 - 1)
F1, F2 = np.zeros((N1, N2)), np.zeros((N1, N2))

f1 = f1min
for j in range(N1):
    f2 = f2min
    for i in range(N2):
        val = Loggrowth(f1, f2)
        Z[j, i] = val if np.isfinite(val) else -10
        F1[j, i], F2[j, i] = f1, f2
        f2 += df2
    f1 += df1

V = np.zeros((N1, N2))
f1 = f1min
for j in range(N1):
    f2 = f2min
    for i in range(N2):
        V[j, i] = maxdrawdown1(f1, f2, Returns1, Returns2, TbillRet)
        F1[j, i], F2[j, i] = f1, f2
        f2 += df2
    f1 += df1

mu1, sigma1 = np.mean(Returns1) * 250, np.std(Returns1) * np.sqrt(250)
print(f"rfree adj mu1={mu1:.3f} sigma1={sigma1:.3f} mu1/sigma1^2={mu1/sigma1**2:.4f}")
mu2, sigma2 = np.mean(Returns2) * 250, np.std(Returns2) * np.sqrt(250)
print(f"rfree adj mu2={mu2:.3f} sigma2={sigma2:.3f} mu2/sigma2^2={mu2/sigma2**2:.4f}")
A = np.corrcoef(Returns1, Returns2)
print(f"asset 1 and 2 correlation={A[0,1]:.2f}")

f1_pts = uncom_results_df["Uncompounded SP500 Weights"].to_numpy()
f2_pts = uncom_results_df['US Bond wieghts'].to_numpy()

#Plot for Log Return COntour
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
cs1 = axs[0].contourf(F1, F2, Z, levels=[0,0.04,0.08,0.12,0.14,0.18,0.21,0.218], alpha=0.5)

#Plot of Optimal leverage protfolios

axs[0].set_xlabel("f1 stocks"); axs[0].set_ylabel("f2 bonds"); axs[0].set_title("Log Growth Contour Plot")
fig.colorbar(cs1, ax=axs[0])

#Plot for Drawdowns
cs2 = axs[1].contourf(F1, F2, V, levels=np.linspace(0.1, np.nanmax(V), 15), alpha=0.5)
axs[1].set_xlabel("f1 stocks"); axs[1].set_ylabel("f2 bonds"); axs[1].set_title("Max Drawdown Contour")
fig.colorbar(cs2, ax=axs[1])

ax3 = fig.add_subplot(3, 1, 3, projection="3d")
ax3.set_xlabel("f1 stocks"); ax3.set_ylabel("f2 bonds"); ax3.set_zlabel("Log Growth"); ax3.set_title("3D Log Growth Surface")
ax3.plot_surface(F1, F2, Z, cmap="viridis", edgecolor="none")
plt.tight_layout()
plt.show()


# using analytical formula
DF_data2 = pd.DataFrame({'SP500': merged['RfreeAdjSP500'], 'GovBond': merged['USGovBonds(VUSTX)']-merged['TBillTotRet']})
cov2 = DF_data2.cov()
mu2 = DF_data2.mean()

F2_kelly = np.linalg.inv(cov2) @ mu2
F2_kelly


fig, ax = plt.subplots(figsize=(10,5))

cs1 = ax.contourf(F1, F2, Z, 
                  levels=[0,0.04,0.08,0.12,0.14,0.18,0.21,0.218], 
                  alpha=0.5)


cs2 = axs[1].contourf(F1, F2, V, levels=np.linspace(0.1, np.nanmax(V), 15), alpha=0.5)
axs[1].set_xlabel("f1 stocks"); axs[1].set_ylabel("f2 bonds"); axs[1].set_title("Max Drawdown Contour")
fig.colorbar(cs2, ax=axs[1])

ax.scatter(f1_pts, f2_pts, c=(np.array(Gf_3)*250),
           cmap="viridis", s=40, edgecolor="k", norm=cs1.norm)
plt.scatter(F2_kelly[0], F2_kelly[1], color = 'red', label = 'Analytical Kelly')
plt.scatter(uncom_results_df.iloc[loc3][2],uncom_results_df.iloc[loc3][4], label = 'Optimal Drawdown')
ax.set_xlabel("f1 stocks")
ax.set_ylabel("f2 bonds")
ax.set_title("Log Growth Contour with Full Kelly and Optimal Portfolios")
plt.legend()
fig.colorbar(cs1, ax=ax)
plt.show()

F2_kelly[0] + F2_kelly[1]


growthF2 = np.mean(np.log(1 + tbill*1 +  F2_kelly[0]*(merged['RfreeAdjSP500']) + F2_kelly[1] *(merged['USGovBonds(VUSTX)']-merged['TBillTotRet'])))*250
growthF2 


fig, ax = plt.subplots(figsize=(10,5))

# --- Log Growth Contour (Z) ---
cs1 = ax.contourf(
    F1, F2, Z, 
    levels=[0, 0.04, 0.08, 0.12, 0.14, 0.18, 0.21, 0.218],
    alpha=0.6,
    cmap="viridis"
)

# --- Max Drawdown Contour (V) with LOW OPACITY (same cmap) ---
cs2 = ax.contourf(
    F1, F2, V,
    levels=np.linspace(0.1, np.nanmax(V), 15),
    alpha=0.40,       # low opacity so it overlays softly
    cmap="viridis"    # SAME colormap as Z
)

# Colorbars
fig.colorbar(cs1, ax=ax, label="Log Growth")
fig.colorbar(cs2, ax=ax, label="Max Drawdown")

# Scatter points
ax.scatter(
    f1_pts, f2_pts,
    c=(np.array(Gf_3)*250),
    cmap="viridis",
    s=40, edgecolor="k",
    norm=cs1.norm
)

# Analytical Kelly
ax.scatter(F2_kelly[0], F2_kelly[1], color='red',
           label='Analytical Kelly', s=80)

# Optimal Drawdown
ax.scatter(
    uncom_results_df.iloc[loc3][2],
    uncom_results_df.iloc[loc3][4],
    color="blue", label='Optimal Drawdown', s=80
)

ax.set_xlabel("f1 (stocks)")
ax.set_ylabel("f2 (bonds)")
ax.set_title("Log Growth Contour with Max Drawdown (Overlay)")
plt.legend()
plt.show()


#weight of analytical kelly
w_k = np.array([F2_kelly[0], F2_kelly[1]])

#weight of optimized 
w_o = np.array([uncom_results_df.iloc[loc3][2], uncom_results_df.iloc[loc3][4]])

#returns of instruments
R = np.column_stack([
    merged['CRSP SP500 TotRet'].values - merged['TBillTotRet'],
    merged['USGovBonds(VUSTX)'].values - merged['TBillTotRet']
])

#Full kelly return series
Kelly_ret2 = R @ w_k
Optimal_ret2 = R @ w_o


plt.plot(merged['Date'],Kelly_ret2.cumsum(), label = 'Full Kelly')
plt.plot(merged['Date'],Optimal_ret2.cumsum(), label = 'Optimal Portfolio')
plt.title('Uncompounded Return')
plt.xlabel('Date')
plt.ylabel('Cummulative Return')
plt.legend()


#weight of analytical kelly
w_k = np.array([F2_kelly[0], F2_kelly[1]])

#weight of optimized 
w_o = np.array([uncom_results_df.iloc[loc3][2], uncom_results_df.iloc[loc3][4]])

#returns of instruments
R = np.column_stack([
    merged['CRSP SP500 TotRet'].values - merged['TBillTotRet'],
    merged['USGovBonds(VUSTX)'].values - merged['TBillTotRet']
])

#Full kelly return series
Kelly_ret2 = R @ w_k
Optimal_ret2 = R @ w_o


plt.plot(merged['Date'], (1 + Kelly_ret2).cumprod(), label = 'Full Kelly')
plt.plot(merged['Date'], (1+ Optimal_ret2).cumprod(), label = 'Optimal Portfolio')
plt.title('Compounded Return')
plt.xlabel('Date')
plt.ylabel('Return (Log Scale)')
plt.legend()
plt.yscale('log')

'''Uncompooudned drawdown comparison'''
FK2_DD_Cum = pd.DataFrame(Kelly_ret2).cumsum()
FKD_DD = FK2_DD_Cum.cummax() - FK2_DD_Cum

Opt_ret2_Cum = pd.DataFrame(Optimal_ret2).cumsum()
Opt_ret2_DD = Opt_ret2_Cum.cummax() - Opt_ret2_Cum

plt.figure(figsize=(14,8))
plt.plot(merged['Date'], FKD_DD, label = 'Full Kelly DD')
plt.plot(merged['Date'], Opt_ret2_DD, label = 'Optimal portfolio DD', alpha = .65)
plt.legend()
plt.title('Uncompouded Drawdowns of Return Series')
plt.xlabel('Dates')
plt.ylabel('Drawdowns')
plt.show()

import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Surface(
            x=F1,
            y=F2,
            z=Z,
            colorscale="Viridis",
            showscale=True,
            opacity=0.9
        )
    ]
)

fig.update_layout(
    title="Interactive 3D Log Growth Surface",
    scene=dict(
        xaxis_title="f1 stocks",
        yaxis_title="f2 bonds",
        zaxis_title="Log Growth",
        camera=dict(eye=dict(x=1.3, y=1.3, z=0.8))
    ),
    width=900,
    height=700,
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.add_trace(
    go.Scatter3d(
        x=f1_pts,
        y=f2_pts,
        z=np.array(Gf_3)*250,   # scale to same annualized level
        mode="markers",
        marker=dict(size=4, color="Blue"),
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[F2_kelly[0]],
        y=[F2_kelly[1]],
        z=[growthF2],   # scale to same annualized level
        mode="markers",
        marker=dict(size=4, color="red"),
    )
)

fig.show()


lft = pd.read_csv('BloombergHighYieldIndex.csv', header = 0)
lft1 = lft[::-1]['LF98TRUU high yield Index (before Aug 7 1998 returns of Fidelity fund FAHYX)'].reset_index()
lft_r = lft1['LF98TRUU high yield Index (before Aug 7 1998 returns of Fidelity fund FAHYX)'].pct_change().dropna()
plt.plot(lft_r.cumsum())
np.mean(lft_r) *252


lft = pd.read_csv('BloombergHighYieldIndex.csv', header = 0)
lft = lft[::-1]
lft['CorpBond'] = lft['LF98TRUU high yield Index (before Aug 7 1998 returns of Fidelity fund FAHYX)'].pct_change()
lft['Unnamed: 0'] = pd.to_datetime(lft['Unnamed: 0'], errors='coerce')
df['caldt'] = pd.to_datetime(df['caldt'])
lft_2024 = lft[lft['Unnamed: 0'] <= '2024-12-31']
df_lft = lft_2024
merged_4 = pd.merge(merged, df_lft, left_on='caldt', right_on='Unnamed: 0', how='inner')
merged_4 = merged_4.dropna()
merged_4 

plt.plot(merged_4['Date'],merged_4 ['CorpBond'].cumsum(), label = 'Corp')
plt.plot(merged_4['Date'],merged_4['USGovBonds(VUSTX)'].cumsum(), label = 'Gov')
plt.title('Uncompounded Returns: Corp Bond vs Gov Bond')
plt.xlabel('Dates')
plt.ylabel('Return')
plt.legend()



plt.plot(merged_4['Date'],(1+ merged_4 ['CorpBond']).cumprod(), label = 'Corp')
plt.plot(merged_4['Date'],(1 +merged_4['USGovBonds(VUSTX)']).cumprod(), label = 'Gov')
plt.yscale('log')

# Low boounds
header_lowerbounds = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
point_lowerbounds_body = np.array([0.0,-10e5, 0.0, 0], order='C')
point_lowerbounds_4 = [header_lowerbounds, point_lowerbounds_body]
print(point_lowerbounds_4)

# matrix_annualized_returns
dailymean_sp500 = merged_4['CRSP SP500 TotRet'].mean()
dailymean_tbill = merged_4['TBillTotRet'].mean() 
dailymean_gov = merged_4['USGovBonds(VUSTX)'].mean() 
dailymean_corpbond = merged_4['CorpBond'].mean()

#Uncompounded
uncom_ann_mean_sp500 = dailymean_sp500*252
uncom_ann_mean_tbill = dailymean_tbill*252
uncom_ann_mean_gov = dailymean_gov*252
uncom_ann_mean_corpbond = dailymean_corpbond*252
header_matrix_annualized_returns = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
uncom_matrix_annualized_returns_body = np.array(np.column_stack((uncom_ann_mean_sp500, uncom_ann_mean_tbill,uncom_ann_mean_gov,uncom_ann_mean_corpbond)),order='C')
uncom_matrix_annualized_returns_4 = [header_matrix_annualized_returns, uncom_matrix_annualized_returns_body]
#print(uncom_matrix_annualized_returns_4)

# matrix_scenarios
daily_sp500 = merged_4['CRSP SP500 TotRet'].dropna().tolist()
daily_tbill = merged_4['TBillTotRet'].dropna().tolist()
daily_gov = merged_4['USGovBonds(VUSTX)'].dropna().tolist()
daily_corpbond = merged_4['CorpBond'].dropna().tolist()

header_matrix_scenarios = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
matrix_scenarios_body = np.array(np.column_stack((daily_sp500, daily_tbill,daily_gov,daily_corpbond)),order='C')
matrix_scenarios = [header_matrix_scenarios, matrix_scenarios_body]
print(matrix_scenarios)

# budget
header_budget = ['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond']
matrix_budget_body = np.array([1.0, 1.0, 1.0, 1.0])  # each column corresponds to a weight coefficient = 1
matrix_budget = [header_budget, matrix_budget_body]
#print(matrix_budget)


allowExternal = True
suppressMessages = False
np.random.seed(42)

drawdown_limits = np.arange(.01,2, .05)
problem_name_4 = 'problem_maxdd_4'

objectives_4 = []
opt_weights_4 = []
sp500_weights_4 = []
tbill_weights_4 = []
gov_weights_4 = []
corpbond_weights_4 = []
F_4 = []
Port_returns_4 = []


for limit in drawdown_limits:
    problem_name = f'problem_maxdd_{limit:.2f}'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= {limit:.2f}
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary_4 = {
        'problem_name': problem_name_4,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': uncom_matrix_annualized_returns_4,
        'matrix_scenarios': matrix_scenarios,
        'point_lowerbounds': point_lowerbounds_4
        
    }

    # Solve the optimal solution
    result = psg.psg_solver(problem_dictionary_4, allowExternal, suppressMessages)
    
    # Get the objective values
    objective_text = result['output'][3]
    objective = result['point_problem_1'][1] @ uncom_matrix_annualized_returns_4[1].flatten() #error there  weight * daily return
    objectives_4.append(objective)

    # Get the optimal weights
    opt_x = result['point_problem_1'][1]
    opt_weights_4.append(opt_x)
    
    sp500_weight = opt_x[0]
    sp500_weights_4.append(sp500_weight)
    
    tbill_weight = opt_x[1]
    tbill_weights_4.append(tbill_weight)
    
    gov_weight = opt_x[2]
    gov_weights_4.append(gov_weight)
    
    corpbond_weight = opt_x[3]
    corpbond_weights_4.append(corpbond_weight)

    port_return = merged_4[['CRSP SP500 TotRet','TBillTotRet', 'USGovBonds(VUSTX)', 'CorpBond']] @ opt_x
    Port_returns_4.append(port_return)


    Ftmp = (sp500_weight + gov_weight + corpbond_weight) -1 
    F_4.append(Ftmp)
    

uncom_results_df = pd.DataFrame({"Drawdown_Limit": drawdown_limits,"Uncompounded Return": objectives_4, 
                                 "Uncompounded SP500 Weights": sp500_weights_4, 
                                 "Uncompounded Tbill Weights": tbill_weights_4,
                                 "Uncompounded USGov Weights": gov_weights_4,
                                 "Uncompounded Corpbond Weights": corpbond_weights_4,
                                 "Uncompounded f fraction":F_4})

uncom_results_df


# Made a change 
plt.figure(figsize=(8,5))
#plt.plot(drawdown_limits, objectives, 'r-', linewidth=2, label="Uncompounded Efficient Frontier")
#plt.plot(drawdown_limits, objectives_4, 'b-', linewidth=2, label="Compounded Efficient Frontier")
plt.plot(drawdown_limits, uncom_results_df['Uncompounded Return'], color = 'Blue')
#plt.scatter(drawdown_limits, objectives, color='k', marker='d')  
plt.scatter(drawdown_limits, objectives_4)
plt.title("Efficient Frontier: Uncompounded Return", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Uncompounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()



'''
Getting the f fraction g(f) graph uncompounded
'''
Ftmp = uncom_results_df['Uncompounded f fraction']
Gf_4 = []
r_tbill = merged_4['TBillTotRet'].values
r1 = (merged_4['CRSP SP500 TotRet'] -  merged_4['TBillTotRet']).values #daily return 1
r2 = (merged_4['USGovBonds(VUSTX)'] -  merged_4['TBillTotRet']).values #daily return 2
r3 = (merged_4['CorpBond'] -  merged_4['TBillTotRet']).values #daily return 3
f1 = sp500_weights_4 # weight of Sp500
f2 = gov_weights_4 # weight of gov (usgov)
f3 = corpbond_weights_4 # weight of corpbond
# port_ret
port_ret = []
for i in range(len(f1)):
    port_ret_i = r_tbill + (f1[i] * r1 + f2[i] * r2 + f3[i] * r3)
    port_ret.append(port_ret_i)

Gf_4 = [] #log return for all 3 assets
# log return
for i in range(len(port_ret)):
    Gf_4_i = np.mean(np.log(1 + port_ret[i]))
    Gf_4.append(Gf_4_i)

loc = np.argmax(Gf_4) 
FullKelly_4 = Ftmp[loc] 
maxGF = Gf_4[loc]

# plot FullKelly 
plt.title("3 Asset Portfolio Expected Log Growth", fontsize=14, weight='bold') 
plt.plot(Ftmp, Gf_4, color = 'r')
plt.vlines(FullKelly_4, ymin=min(Gf_4), ymax=maxGF, color='blue', linestyle='--', label='FullKelly')
plt.xlabel("f fraction")
plt.ylabel("Mean Compounded Log Return")
plt.legend()
plt.grid()

print("Full Kelly: ",FullKelly_4)
print("loc:",loc)
uncom_results_df.iloc[loc]
fullkelly_sp500_weights = f1[loc] 
fullkelly_gov_weights = f2[loc]
fullkelly_corpbond_weights = f3[loc]
fullkelly_tbill_weights = 1-(fullkelly_sp500_weights + fullkelly_gov_weights + fullkelly_corpbond_weights)
header_point_problem_cvar = list(['crspsp500totret', 'tbilltotret', 'usgov','corpbond'])
point_problem_cvar_body = np.array([fullkelly_sp500_weights, fullkelly_tbill_weights, fullkelly_gov_weights, fullkelly_corpbond_weights],order='C')
point_problem_cvar = [header_point_problem_cvar,point_problem_cvar_body ]
print(point_problem_cvar)


# matrix_scenarios
daily_sp500 = merged_4['CRSP SP500 TotRet'].dropna().tolist()
daily_tbill = merged_4['TBillTotRet'].dropna().tolist()
daily_gov = merged_4['USGovBonds(VUSTX)'].dropna().tolist()
daily_corpbond = merged_4['CorpBond'].dropna().tolist()
header_matrix_scenarios = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGov','CorpBond'])
matrix_scenarios_body = np.array(np.column_stack((daily_sp500, daily_tbill,daily_gov,daily_corpbond)),order='C')
matrix_scenarios = [header_matrix_scenarios, matrix_scenarios_body]
print(matrix_scenarios)

# find the maxdd: 
problem_name = f'problem_maxdd_fullkelly'
problem_statement = f"""
calculate
Point: point_problem_cvar
drawdown_dev_max(matrix_scenarios)
cvar_risk(0.95, matrix_scenarios)

"""

problem_dictionary_maxdd = {
    'problem_name': problem_name,
    'problem_statement': problem_statement,
    'matrix_scenarios': matrix_scenarios,
    'point_problem_cvar':point_problem_cvar
        
    }

# Solve the optimal solution
result = psg.psg_solver(problem_dictionary_maxdd, allowExternal, suppressMessages)

# find the fullkelly maxDD
result_maxdd = float(result['output'][4].split()[3])
result_CVaR = float(result['output'][5].split()[3])
print("fullkelly MaxDD:",result_maxdd)
print("fullkelly CVaR:",result_CVaR)
# find the cumulative return 
#uncompounded 
optWeight = sp500_weights_4[2], tbill_weights_4[2], gov_weights_4[2],  corpbond_weights_4[2]
FullKellyReturns = merged_4[['CRSP SP500 TotRet','TBillTotRet','USGovBonds(VUSTX)','CorpBond']] @ optWeight
FullKellySeries = 100 + FullKellyReturns.cumsum() # 100 capital

optWeight = sp500_weights_4[loc], gov_weights_4[loc],  corpbond_weights_4[loc]
OptimalReturns = pd.DataFrame({'SPAdj': merged_4['CRSP SP500 TotRet'] - merged_4['TBillTotRet'], 'GovAdj': merged_4['USGovBonds(VUSTX)'] - merged_4['TBillTotRet'] , 'CorpAdj':merged_4['CorpBond'] - merged_4['TBillTotRet'] })
SeriesOpt3 = OptimalReturns @ optWeight
OptimalReturnsSereisUncom = 100 + SeriesOpt3.cumsum() # 100 capital
OptimalReturnsSereisCom = 100*(1+SeriesOpt3).cumprod()

plt.title("Uncompouned", fontsize=14, weight='bold')
plt.plot(OptimalReturnsSereisUncom, color = 'Blue', label='Optimal 3 Asset Portfolio')
plt.plot(sp500_wealth_uncom, color = 'red', label='sp500_wealth')
plt.legend()
plt.show()
# find the CVaR: 
problem_name = f'problem_maxdd_fullkelly'
problem_statement = f"""
calculate
Point: point_problem_cvar
cvar_risk(0.95, matrix_scenarios)
"""

problem_dictionary_maxdd = {
    'problem_name': problem_name,
    'problem_statement': problem_statement,
    'matrix_scenarios': matrix_scenarios,
    'point_problem_cvar':point_problem_cvar
        
    }

# Solve the optimal solution
result = psg.psg_solver(problem_dictionary_maxdd, allowExternal, suppressMessages)

# find the fullkelly CVaR risk (alpha = 0.05)
result_cvar_risk = float(result['output'][4].split()[3])
print("Fullkelly optimal weight CVaR Risk(alpha = 0.05):",result_cvar_risk) # worest 5% average loss 8.673%


from scipy.optimize import minimize

df_Data3 = pd.DataFrame({
    'SP500': merged_4['RfreeAdjSP500'],

})
meanKC = df_Data3.mean().values *252

tbill = merged_4['TBillTotRet']
def annual_log_return_neg(weights):
    w1 = weights
    daily = (tbill+ w1 * df_Data3['SP500'])
    daily = np.clip(1 + daily, 1e-12, None) # avoid log(<=0)
    ann_log = np.mean(np.log(daily)) * 250
    return -ann_log  # we minimize this

# max dd should be negative 
def max_drawdown(weights):
    w1 = weights
    daily = tbill + w1*df_Data3['SP500'] 
    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()  #same as U28 minimum it in excel

# Set max drawdown requirements:
maxDD_limits = np.arange(-1,-.01, .025) #if set -0.1 (-100%)  #same as no constraint

sp500_weights_com = []
gov_weights_com = []
corpbond_weights_com = []
annual_log_comp = []
achieved_maxDD_comp = [] 
uncom_returnKC = []

for limit in maxDD_limits:
    drawdown_constraint = {
        'type': 'ineq',
        'fun': lambda w, L=limit: max_drawdown(w) - L}  #max_drawdown >= maxDD_limit
        
    # bound & budget
    bounds = [(0, 20)]  #Opt Restrict weight
    initial_guess = [1] # initial weight for each 
    
    
    # optimizing compounded 
    result = minimize(
        annual_log_return_neg,
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, #bounds constraint
        constraints=[drawdown_constraint] #drawdown constraint
    )
    
    #optimizing result compounded
    optimal_weights = result.x
    SPW = optimal_weights[0]

    r_tmp = meanKC @ optimal_weights

    uncom_returnKC.append(r_tmp)
    sp500_weights_com.append(SPW)

    

    #compute log return Compounded
    daily_return = tbill + SPW*df_Data3['SP500']
    annual_log = np.mean(np.log(1 + daily_return)) * 250
    annual_log_comp.append(annual_log)
    maxDD_final = max_drawdown(optimal_weights)
    achieved_maxDD_comp.append(maxDD_final)

results_df_c = pd.DataFrame({"Drawdown_Limit":maxDD_limits,
                             "Compounded Return": annual_log_comp, 
                             "Compounded SP500 Weights": sp500_weights_com, 
                             "Achieved_MaxDD": achieved_maxDD_comp
                            })

results_df_c['Achieved_MaxDD'] = results_df_c['Achieved_MaxDD'] * -1
results_df_c


plt.plot(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.scatter(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.title("Efficient Frontier Kelly SP500", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Compounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()


df_Data3 = pd.DataFrame({
    'SP500': merged_4['RfreeAdjSP500'],
    'GovBond': merged_4['USGovBonds(VUSTX)'] - merged_4['TBillTotRet']
})
meanKC = df_Data3.mean().values *252

tbill = merged_4['TBillTotRet']
def annual_log_return_neg(weights):
    w1, w2 = weights
    daily = (tbill+ w1 * df_Data3['SP500']+ w2 * df_Data3['GovBond'])
    daily = np.clip(1 + daily, 1e-12, None) # avoid log(<=0)
    ann_log = np.mean(np.log(daily)) * 250
    return -ann_log  # we minimize this

# max dd should be negative 
def max_drawdown(weights):
    w1, w2 = weights
    daily = tbill + w1*df_Data3['SP500'] + w2*df_Data3['GovBond'] 
    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()  #same as U28 minimum it in excel

# Set max drawdown requirements:
maxDD_limits = np.arange(-1,-.01, .025) #if set -0.1 (-100%)  #same as no constraint

sp500_weights_com = []
gov_weights_com = []
corpbond_weights_com = []
annual_log_comp = []
achieved_maxDD_comp = [] 
uncom_returnKC = []

for limit in maxDD_limits:
    drawdown_constraint = {
        'type': 'ineq',
        'fun': lambda w, L=limit: max_drawdown(w) - L}  #max_drawdown >= maxDD_limit
        
    # bound & budget
    bounds = [(0, 20), (0, 20)]  #Opt Restrict weight
    initial_guess = [1, 1] # initial weight for each 
    
    
    # optimizing compounded 
    result = minimize(
        annual_log_return_neg,
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, #bounds constraint
        constraints=[drawdown_constraint] #drawdown constraint
    )
    
    #optimizing result compounded
    optimal_weights = result.x
    SPW, GovW = optimal_weights

    r_tmp = meanKC @ optimal_weights

    uncom_returnKC.append(r_tmp)
    sp500_weights_com.append(SPW)
    gov_weights_com.append(GovW)
   
    

    #compute log return Compounded
    daily_return = tbill + SPW*df_Data3['SP500'] + GovW*df_Data3['GovBond'] 
    annual_log = np.mean(np.log(1 + daily_return)) * 250
    annual_log_comp.append(annual_log)
    maxDD_final = max_drawdown(optimal_weights)
    achieved_maxDD_comp.append(maxDD_final)

results_df_c = pd.DataFrame({"Drawdown_Limit":maxDD_limits,
                             "Compounded Return": annual_log_comp, 
                             "Compounded SP500 Weights": sp500_weights_com, 
                             "Compounded USGov Weights": gov_weights_com,
                             "Achieved_MaxDD": achieved_maxDD_comp
                            })
results_df_c
results_df_c['Achieved_MaxDD'] = results_df_c['Achieved_MaxDD'] * -1
results_df_c


plt.plot(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.scatter(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.title("Efficient Frontier Kelly SP500 and Gov", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Compounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()

'''Full Kelly EF all 3'''


df_Data3 = pd.DataFrame({
    'SP500': merged_4['RfreeAdjSP500'],
    'GovBond': merged_4['USGovBonds(VUSTX)'] - merged_4['TBillTotRet'],
    'CorpBond': merged_4['CorpBond'] - merged_4['TBillTotRet']
})
meanKC = df_Data3.mean().values *252

tbill = merged_4['TBillTotRet']
def annual_log_return_neg(weights):
    w1, w2, w3 = weights
    daily = (tbill+ w1 * df_Data3['SP500']+ w2 * df_Data3['GovBond']+ w3 * df_Data3['CorpBond'])
    daily = np.clip(1 + daily, 1e-12, None) # avoid log(<=0)
    ann_log = np.mean(np.log(daily)) * 250
    return -ann_log  # we minimize this

# max dd should be negative 
def max_drawdown(weights):
    w1, w2, w3 = weights
    daily = tbill + w1*df_Data3['SP500'] + w2*df_Data3['GovBond'] + w3*df_Data3['CorpBond']
    cum = (1 + daily).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()  #same as U28 minimum it in excel

# Set max drawdown requirements:
maxDD_limits = np.arange(-1,-.01, .025) #if set -0.1 (-100%)  #same as no constraint

sp500_weights_com = []
gov_weights_com = []
corpbond_weights_com = []
annual_log_comp = []
achieved_maxDD_comp = [] 
uncom_returnKC = []

for limit in maxDD_limits:
    drawdown_constraint = {
        'type': 'ineq',
        'fun': lambda w, L=limit: max_drawdown(w) - L}  #max_drawdown >= maxDD_limit
        
    # bound & budget
    bounds = [(0, 20), (0, 20), (0, 20)]  #Opt Restrict weight
    initial_guess = [1, 1, 1] # initial weight for each 
    
    
    # optimizing compounded 
    result = minimize(
        annual_log_return_neg,
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, #bounds constraint
        constraints=[drawdown_constraint] #drawdown constraint
    )
    
    #optimizing result compounded
    optimal_weights = result.x
    SPW, GovW, CorpW = optimal_weights

    r_tmp = meanKC @ optimal_weights

    uncom_returnKC.append(r_tmp)
    sp500_weights_com.append(SPW)
    gov_weights_com.append(GovW)
    corpbond_weights_com.append(CorpW)
    

    #compute log return Compounded
    daily_return = tbill + SPW*df_Data3['SP500'] + GovW*df_Data3['GovBond'] + CorpW*df_Data3['CorpBond']
    annual_log = np.mean(np.log(1 + daily_return)) * 250
    annual_log_comp.append(annual_log)
    maxDD_final = max_drawdown(optimal_weights)
    achieved_maxDD_comp.append(maxDD_final)

results_df_c = pd.DataFrame({"Drawdown_Limit":maxDD_limits,
                             "Compounded Return": annual_log_comp, 
                             "Compounded SP500 Weights": sp500_weights_com, 
                             "Compounded USGov Weights": gov_weights_com,
                             "Compounded Corpbound Weights": corpbond_weights_com,
                             "Achieved_MaxDD": achieved_maxDD_comp
                            })
results_df_c
results_df_c['Achieved_MaxDD'] = results_df_c['Achieved_MaxDD'] * -1
results_df_c


plt.figure(figsize=(8,5))
plt.plot(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'], color = 'blue')
plt.plot(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'], color = 'blue')
plt.scatter(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.title("Efficient Frontier Kelly", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Expected Log Return", fontsize=12, weight='bold')
plt.grid(True)
plt.show()

W_FK3 = [results_df_c.iloc[0][2],results_df_c.iloc[0][3],results_df_c.iloc[0][4]]
FK3_return = OptimalReturns @ W_FK3
FK3_returnSeriesUncom = 100 + FK3_return.cumsum()
FK3_returnSeriesCom = 100 * (1+FK3_return).cumprod()

plt.title("Uncompounded", fontsize=14, weight='bold')
plt.plot(merged_4['Date'],OptimalReturnsSereisUncom, color = 'Orange', label='3 Asset Optimized Portfolio')
plt.plot(merged['Date'],sp500_wealth_uncom, color = 'red', label='SP500')
plt.plot(merged_4['Date'],FK3_returnSeriesUncom, color = 'Blue', label='Full Kelly 3 Asset')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return Starting at $100')
plt.show()


plt.title("Compounded", fontsize=14, weight='bold')
plt.plot(merged_4['Date'],OptimalReturnsSereisCom, color = 'Orange', label='3 Asset Optimized Portfolio')
plt.plot(df['caldt'],sp500_wealth_com, color = 'red', label='SP500')
plt.plot(merged_4['Date'],FK3_returnSeriesCom, color = 'Blue', label='Full Kelly 3 Asset')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return Starting at $100 (Log scale)')
plt.yscale('log')
plt.show()


'''Uncompooudned drawdown comparison'''
FK3_DD_Cum = pd.DataFrame(FK3_return).cumsum()
FK3D_DD = FK3_DD_Cum.cummax() - FK3_DD_Cum

Opt_ret3_Cum = pd.DataFrame(SeriesOpt3).cumsum()
Opt_ret3_DD = Opt_ret3_Cum.cummax() - Opt_ret3_Cum

plt.figure(figsize=(14,8))
plt.plot(merged_4['Date'], FK3D_DD, label = 'Full Kelly DD')
plt.plot(merged_4['Date'], Opt_ret3_DD, label = 'Optimal portfolio DD', alpha = .65)
plt.legend()
plt.title('Uncompouded Drawdowns of Return Series')
plt.xlabel('Dates')
plt.ylabel('Drawdowns')
plt.show()


plt.figure(figsize=(8,5))
plt.plot(uncom_results_df['Drawdown_Limit'], uncom_results_df['Uncompounded Return'], 'Blue', linewidth=2, label="Uncompounded Efficient Frontier")
plt.scatter(uncom_results_df['Drawdown_Limit'], uncom_results_df['Uncompounded Return'])
plt.title("Efficient Frontier Optimization", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Uncompounded Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(uncom_results_df['Drawdown_Limit'], uncom_results_df['Uncompounded Return'], color = 'red', linewidth=2, label="Uncompounded Efficient Frontier")
plt.scatter(uncom_results_df['Drawdown_Limit'], uncom_results_df['Uncompounded Return'], color = 'red')
plt.vlines(max(uncom_results_df['Drawdown_Limit']), ymin=min(uncom_results_df['Uncompounded Return']), ymax=max(uncom_results_df['Uncompounded Return']), color='red', linestyle='--', label='Optimal Uncompounded')


plt.plot(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'], 'Blue', linewidth=2, label="Compounded Efficient Frontier")
plt.scatter(results_df_c['Achieved_MaxDD'], results_df_c['Compounded Return'])
plt.vlines(max(results_df_c['Achieved_MaxDD']), ymin=min(results_df_c['Compounded Return']), ymax=max(results_df_c['Compounded Return']), color='blue', linestyle='--', label='Optimal Compounded')


plt.title("Efficient Frontier", fontsize=14, weight='bold')
plt.xlabel("MaxDD", fontsize=12, weight='bold')
plt.ylabel("Annual Return", fontsize=12, weight='bold')
plt.grid(True)
plt.legend()
plt.show()



'''Out of sample Testing MaxDD .50'''


dates = ['1997-01-03','1998-01-03','1999-01-03','2000-01-03','2001-01-03','2002-01-03','2003-01-03','2004-01-03','2005-01-03','2006-01-03','2007-01-03','2008-01-03','2009-01-03','2010-01-03','2011-01-03','2012-01-03','2013-01-03','2014-01-03','2015-01-03','2016-01-03','2017-01-03','2018-01-03','2019-01-03','2020-01-03','2021-01-03','2022-01-03','2023-01-03','2024-01-03','2025-01-01']
ReturnTest = []
SPtrack = []
DateOut = []
for i in range(len(dates) - 6):

    TrainingUP = merged_4[merged_4['Date'] <= dates[i + 5]]
    Training = TrainingUP[TrainingUP['Date'] >= dates[i]]
    # Low boounds
    header_lowerbounds = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    point_lowerbounds_body = np.array([0.0,-10e5, 0.0, 0], order='C')
    point_lowerbounds_4 = [header_lowerbounds, point_lowerbounds_body]
    print(point_lowerbounds_4)

    # matrix_annualized_returns
    dailymean_sp500 = Training['CRSP SP500 TotRet'].mean()
    dailymean_tbill = Training['TBillTotRet'].mean() 
    dailymean_gov = Training['USGovBonds(VUSTX)'].mean() 
    dailymean_corpbond = Training['CorpBond'].mean()

    #Uncompounded
    uncom_ann_mean_sp500 = dailymean_sp500*252
    uncom_ann_mean_tbill = dailymean_tbill*252
    uncom_ann_mean_gov = dailymean_gov*252
    uncom_ann_mean_corpbond = dailymean_corpbond*252
    header_matrix_annualized_returns = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    uncom_matrix_annualized_returns_body = np.array(np.column_stack((uncom_ann_mean_sp500, uncom_ann_mean_tbill,uncom_ann_mean_gov,uncom_ann_mean_corpbond)),order='C')
    uncom_matrix_annualized_returns_4 = [header_matrix_annualized_returns, uncom_matrix_annualized_returns_body]
    #print(uncom_matrix_annualized_returns_4)

    # matrix_scenarios
    daily_sp500 = Training['CRSP SP500 TotRet'].dropna().tolist()
    daily_tbill = Training['TBillTotRet'].dropna().tolist()
    daily_gov = Training['USGovBonds(VUSTX)'].dropna().tolist()
    daily_corpbond = Training['CorpBond'].dropna().tolist()

    header_matrix_scenarios = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    matrix_scenarios_body = np.array(np.column_stack((daily_sp500, daily_tbill,daily_gov,daily_corpbond)),order='C')
    matrix_scenarios = [header_matrix_scenarios, matrix_scenarios_body]
    print(matrix_scenarios)

    # budget
    header_budget = ['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond']
    matrix_budget_body = np.array([1.0, 1.0, 1.0, 1.0])  # each column corresponds to a weight coefficient = 1
    matrix_budget = [header_budget, matrix_budget_body]



    allowExternal = True
    suppressMessages = False
    np.random.seed(42)

    problem_name_4 = 'problem_maxdd_4_Testing'

  
    problem_name = 'problem_maxdd_50'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= .24
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary_4 = {
        'problem_name': problem_name_4,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': uncom_matrix_annualized_returns_4,
        'matrix_scenarios': matrix_scenarios,
        'point_lowerbounds': point_lowerbounds_4
        
    }

    # Solve the optimal solution
    resultT = psg.psg_solver(problem_dictionary_4, allowExternal, suppressMessages)
    
    # Get the objective values
    objective_textT = resultT['output'][3]
    objectiveT = resultT['point_problem_1'][1] @ uncom_matrix_annualized_returns_4[1].flatten() #error there  weight * daily return
    

    # Get the optimal weights
    opt_xT = resultT['point_problem_1'][1]
    
    
    sp500_weightT = opt_xT[0]
    
    tbill_weightT = opt_xT[1]
    
    gov_weightT = opt_xT[2]
    
    corpbond_weightT = opt_xT[3]
    

    FtmpT = (sp500_weightT + gov_weightT + corpbond_weightT) -1 
    

    '''Getting Weights from solver and computing next year return'''


    TestingUp = merged_4[merged_4['Date'] >= dates[i+5]]
    Testing =  TestingUp[TestingUp['Date'] <= dates[i+6]]
    Series = pd.concat([
        Testing['CRSP SP500 TotRet']- Testing['TBillTotRet'],
        Testing['USGovBonds(VUSTX)']- Testing['TBillTotRet'],
        Testing['CorpBond']- Testing['TBillTotRet']
    ], axis=1)
    TestingWeights = [sp500_weightT, gov_weightT, corpbond_weightT]
    PSGreturn = Series @ TestingWeights
    
    DateOut.append(Testing['Date'])
    SPtrack.append(Testing['CRSP SP500 TotRet'])

    ReturnTest.append(PSGreturn)

DateOut = pd.concat(DateOut)
SPout = pd.concat(SPtrack, ignore_index=True)
flat_series = pd.concat(ReturnTest, ignore_index=True)    
    
'''Plot MaxDD 0.15 Optimization'''

plt.plot(DateOut,flat_series.cumsum(), label = 'Out Of Sample Optimization')
plt.plot(DateOut,SPout.cumsum(), label = 'SP500')
plt.xlabel('Time')
plt.ylabel('Uncompounded Returns')
plt.legend()

dates = ['1997-01-03','1998-01-03','1999-01-03','2000-01-03','2001-01-03','2002-01-03','2003-01-03','2004-01-03','2005-01-03','2006-01-03','2007-01-03','2008-01-03','2009-01-03','2010-01-03','2011-01-03','2012-01-03','2013-01-03','2014-01-03','2015-01-03','2016-01-03','2017-01-03','2018-01-03','2019-01-03','2020-01-03','2021-01-03','2022-01-03','2023-01-03','2024-01-03','2025-01-01']
ReturnTestFK = []
SPtrackFK = []
DateOutFK = []
drawdownFK= np.arange(.05, .55, .05)

for i in range(len(dates) - 6):
    df_Data3 = pd.DataFrame({
    'SP500Adj': merged_4['RfreeAdjSP500'],
    'GovBondAdj': merged_4['USGovBonds(VUSTX)'] - merged_4['TBillTotRet'],
    'CorpBondAdj': merged_4['CorpBond'] - merged_4['TBillTotRet'],
    'Dates': merged_4['Date'],
    'Tbill': merged_4['TBillTotRet'],
    'SPIndex': merged_4['CRSP SP500 TotRet']
    })

    
    TrainingUP = df_Data3[df_Data3['Dates'] <= dates[i + 5]]
    Training = df_Data3[df_Data3['Dates'] >= dates[i]]

    tbill = Training['Tbill']
    def annual_log_return_neg(weights):
        w1, w2, w3 = weights
        daily = (tbill+ w1 * Training['SP500Adj']+ w2 * Training['GovBondAdj']+ w3 * Training['CorpBondAdj'])
        daily = np.clip(1 + daily, 1e-12, None) # avoid log(<=0)
        ann_log = np.mean(np.log(daily)) * 250
        return -ann_log  # we minimize this

    # max dd should be negative 
    def max_drawdown(weights):
        w1, w2, w3 = weights
        daily = tbill + w1*Training['SP500Adj'] + w2*Training['GovBondAdj'] + w3*Training['CorpBondAdj']
        cum = (1 + daily).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()  #same as U28 minimum it in excel

    maxDD_limit = -.22# (-100%)  #same as no constraint

    drawdown_constraint = {
    'type': 'ineq',
    'fun': lambda w: max_drawdown(w) - maxDD_limit} #max_drawdown >= maxDD_limit
        
    # bound & budget
    bounds = [(0, 20), (0, 20), (0, 20)]  #Opt Restrict weight
    initial_guess = [1, 1, 1] # initial weight for each 
    
    
    # optimizing compounded 
    result = minimize(
        annual_log_return_neg,
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, #bounds constraint
        constraints=[drawdown_constraint] #drawdown constraint
    )
    
    #optimizing result compounded
    optimal_weights = result.x
    
    TestingUp = df_Data3[df_Data3['Dates'] >= dates[i+5]]
    Testing =  TestingUp[TestingUp['Dates'] <= dates[i+6]]
    Series = pd.concat([
        Testing['SP500Adj'],
        Testing['GovBondAdj'],
        Testing['CorpBondAdj']
    ], axis=1)
    seriestmp = Series @ optimal_weights
    
    ReturnTestFK.append(seriestmp)
    
    DateOutFK.append(Testing['Dates'])
    SPtrackFK.append(Testing['SPIndex'])
    
DateFK = pd.concat(DateOutFK)
FK_out = pd.concat(ReturnTestFK)
SPtrackFK = pd.concat(SPtrackFK)


'''Full Kelly optimized by drawdown'''
plt.plot(DateFK, FK_out.cumsum(), label = 'Full Kelly Out Of Sample Optimization')
plt.plot(DateFK, SPtrackFK.cumsum(), label = 'SP500')
plt.plot(DateOut,flat_series.cumsum(), label = 'Drawdown Out Of Sample Optimization')
plt.xlabel('Time')
plt.ylabel('Uncompounded Returns')
plt.legend()

start = pd.to_datetime("1996-12-03")
end   = pd.to_datetime("2025-01-01")

# Generate month starts
month_starts = pd.date_range(start=start, end=end, freq='MS')

# Shift to the 3rd of each month
monthly_dates = (month_starts + pd.offsets.Day(2)).strftime('%Y-%m-%d').tolist()



All_OP = []
All_FK = []

ReturnTest = []
SPtrack = []
DateOut = []
for i in range(len(monthly_dates) - 157):

    TrainingUP = merged_4[merged_4['Date'] <= monthly_dates[i + 156]]
    Training = TrainingUP[TrainingUP['Date'] >= monthly_dates[i]]
    # Low boounds
    header_lowerbounds = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    point_lowerbounds_body = np.array([0.0,-10e5, 0.0, 0], order='C')
    point_lowerbounds_4 = [header_lowerbounds, point_lowerbounds_body]
    print(point_lowerbounds_4)

    # matrix_annualized_returns
    dailymean_sp500 = Training['CRSP SP500 TotRet'].mean()
    dailymean_tbill = Training['TBillTotRet'].mean() 
    dailymean_gov = Training['USGovBonds(VUSTX)'].mean() 
    dailymean_corpbond = Training['CorpBond'].mean()

    #Uncompounded
    uncom_ann_mean_sp500 = dailymean_sp500*252
    uncom_ann_mean_tbill = dailymean_tbill*252
    uncom_ann_mean_gov = dailymean_gov*252
    uncom_ann_mean_corpbond = dailymean_corpbond*252
    header_matrix_annualized_returns = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    uncom_matrix_annualized_returns_body = np.array(np.column_stack((uncom_ann_mean_sp500, uncom_ann_mean_tbill,uncom_ann_mean_gov,uncom_ann_mean_corpbond)),order='C')
    uncom_matrix_annualized_returns_4 = [header_matrix_annualized_returns, uncom_matrix_annualized_returns_body]
    #print(uncom_matrix_annualized_returns_4)

    # matrix_scenarios
    daily_sp500 = Training['CRSP SP500 TotRet'].dropna().tolist()
    daily_tbill = Training['TBillTotRet'].dropna().tolist()
    daily_gov = Training['USGovBonds(VUSTX)'].dropna().tolist()
    daily_corpbond = Training['CorpBond'].dropna().tolist()

    header_matrix_scenarios = list(['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond'])
    matrix_scenarios_body = np.array(np.column_stack((daily_sp500, daily_tbill,daily_gov,daily_corpbond)),order='C')
    matrix_scenarios = [header_matrix_scenarios, matrix_scenarios_body]
    print(matrix_scenarios)

    # budget
    header_budget = ['CRSPSP500TotRet', 'TBillTotRet', 'USGovBonds(VUSTX)','CorpBond']
    matrix_budget_body = np.array([1.0, 1.0, 1.0, 1.0])  # each column corresponds to a weight coefficient = 1
    matrix_budget = [header_budget, matrix_budget_body]



    allowExternal = True
    suppressMessages = False
    np.random.seed(42)

    problem_name_4 = 'problem_maxdd_4_Testing'

  
    problem_name = 'problem_maxdd_50'
    problem_statement = f"""maximize
    linear(matrix_annualized_returns)
    Constraint: <= .35
    drawdown_dev_max(matrix_scenarios)
    Constraint: = 1
    linear(matrix_budget)
    Box: >= point_lowerbounds
    """

    problem_dictionary_4 = {
        'problem_name': problem_name_4,
        'problem_statement': problem_statement,
        'matrix_annualized_returns': uncom_matrix_annualized_returns_4,
        'matrix_scenarios': matrix_scenarios,
        'point_lowerbounds': point_lowerbounds_4
        
    }

    # Solve the optimal solution
    resultT = psg.psg_solver(problem_dictionary_4, allowExternal, suppressMessages)
    
    # Get the objective values
    objective_textT = resultT['output'][3]
    objectiveT = resultT['point_problem_1'][1] @ uncom_matrix_annualized_returns_4[1].flatten() #error there  weight * daily return
    

    # Get the optimal weights
    opt_xT = resultT['point_problem_1'][1]
    
    
    sp500_weightT = opt_xT[0]
    
    tbill_weightT = opt_xT[1]
    
    gov_weightT = opt_xT[2]
    
    corpbond_weightT = opt_xT[3]
    

    FtmpT = (sp500_weightT + gov_weightT + corpbond_weightT) -1 
    

    '''Getting Weights from solver and computing next year return'''


    TestingUp = merged_4[merged_4['Date'] >= monthly_dates[i+156]]
    Testing =  TestingUp[TestingUp['Date'] <= monthly_dates[i+157]]
    Series = pd.concat([
        Testing['CRSP SP500 TotRet']- Testing['TBillTotRet'],
        Testing['USGovBonds(VUSTX)']- Testing['TBillTotRet'],
        Testing['CorpBond']- Testing['TBillTotRet']
    ], axis=1)
    TestingWeights = [sp500_weightT, gov_weightT, corpbond_weightT]
    PSGreturn = Series @ TestingWeights
    
    DateOut.append(Testing['Date'])
    SPtrack.append(Testing['CRSP SP500 TotRet'])

    ReturnTest.append(PSGreturn)

DateOut = pd.concat(DateOut)
SPout = pd.concat(SPtrack, ignore_index=True)
flat_series = pd.concat(ReturnTest, ignore_index=True)    
All_OP.append(flat_series)


'''Monthly Full Kelly Solution'''

ReturnTestFK = []
SPtrackFK = []
DateOutFK = []
drawdownFK= np.arange(.05, .55, .05)

for i in range(len(monthly_dates) - 157):
    df_Data3 = pd.DataFrame({
    'SP500Adj': merged_4['RfreeAdjSP500'],
    'GovBondAdj': merged_4['USGovBonds(VUSTX)'] - merged_4['TBillTotRet'],
    'CorpBondAdj': merged_4['CorpBond'] - merged_4['TBillTotRet'],
    'Dates': merged_4['Date'],
    'Tbill': merged_4['TBillTotRet'],
    'SPIndex': merged_4['CRSP SP500 TotRet']
    })

    
    TrainingUP = df_Data3[df_Data3['Dates'] <= monthly_dates[i + 156]]
    Training = df_Data3[df_Data3['Dates'] >= monthly_dates[i]]

    tbill = Training['Tbill']
    def annual_log_return_neg(weights):
        w1, w2, w3 = weights
        daily = (tbill+ w1 * Training['SP500Adj']+ w2 * Training['GovBondAdj']+ w3 * Training['CorpBondAdj'])
        daily = np.clip(1 + daily, 1e-12, None) # avoid log(<=0)
        ann_log = np.mean(np.log(daily)) * 250
        return -ann_log  # we minimize this

    # max dd should be negative 
    def max_drawdown(weights):
        w1, w2, w3 = weights
        daily = tbill + w1*Training['SP500Adj'] + w2*Training['GovBondAdj'] + w3*Training['CorpBondAdj']
        cum = (1 + daily).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()  #same as U28 minimum it in excel

    maxDD_limit = -.47# (-100%)  #same as no constraint

    drawdown_constraint = {
    'type': 'ineq',
    'fun': lambda w: max_drawdown(w) - maxDD_limit} #max_drawdown >= maxDD_limit
        
    # bound & budget
    bounds = [(0, 20), (0, 20), (0, 20)]  #Opt Restrict weight
    initial_guess = [1, 1, 1] # initial weight for each 
    
    
    # optimizing compounded 
    result = minimize(
        annual_log_return_neg,
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, #bounds constraint
        constraints=[drawdown_constraint] #drawdown constraint
    )
    
    #optimizing result compounded
    optimal_weights = result.x
    
    TestingUp = df_Data3[df_Data3['Dates'] >= monthly_dates[i+156]]
    Testing =  TestingUp[TestingUp['Dates'] <= monthly_dates[i+157]]
    Series = pd.concat([
        Testing['SP500Adj'],
        Testing['GovBondAdj'],
        Testing['CorpBondAdj']
    ], axis=1)
    seriestmp = Series @ optimal_weights
    
    ReturnTestFK.append(seriestmp)
    
    DateOutFK.append(Testing['Dates'])
    SPtrackFK.append(Testing['SPIndex'])
    


DateFK = pd.concat(DateOutFK, ignore_index= True)
FK_out = pd.concat(ReturnTestFK, ignore_index= True)
SPtrackFK = pd.concat(SPtrackFK, ignore_index= True)
All_FK.append(FK_out)


plt.plot(DateFK,FK_out.cumsum(), label = 'Full Kelly')
plt.plot(DateOut, flat_series.cumsum(), label = 'Optimal Portfolio')
plt.xlabel('Dates')
plt.ylabel('Return')
plt.title('Uncompounded Return')
plt.grid(True)
plt.legend()

plt.plot(DateFK, 100* (1 + FK_out).cumprod(), label = 'Full Kelly')
plt.plot(DateOut, 100 *(1 +flat_series).cumprod(), label = 'Optimal Portfolio')
plt.xlabel('Dates')
plt.ylabel('Return (log scale)')
plt.title('Compounded Return')
plt.grid(True)
plt.yscale('log')
plt.legend()


plt.plot(DateFK, FK_out.cumsum(), label = 'Full Kelly Monthly Rebalance')
#plt.plot(DateOutFKMonthly, SPtrackFKMonthly.cumsum(), label = 'SP500')
#plt.plot(DateFK, FK_out.cumsum(), label = 'Full Kelly Annual')

plt.plot(DateOut,flat_series.cumsum(), label = 'DD Optimization Monthly Rebalance')
plt.title('Uncompounded Returns: 10% DD Constraint')
plt.xlabel('Dates')
plt.ylabel('Return')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(All_OP)):
    plt.plot(DateOut, All_OP[i].cumsum(), label=f"{drawdownFK[i]*100:.0f}% drawdown")

plt.plot(DateOut,SPtrackFK.cumsum(), color = 'red', label = 'SP500')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return')
plt.title('Uncompounded Return: Optimized Portfolio')

plt.grid(True)

plt.figure(figsize=(10, 6))
for i in range(len(All_OP)):
    plt.plot(DateOut, 100*(1+All_OP[i]).cumprod(), label=f"{drawdownFK[i]*100:.0f}% drawdown")

plt.plot(DateOut,100*(1+SPtrackFK).cumprod(), color = 'red', label = 'SP500')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return (log scale)')
plt.title('Compounded Return: Optimized Portfolio')
plt.yscale('log')
plt.grid(True)


plt.figure(figsize=(10, 6))
for i in range(len(All_FK)):
    plt.plot(DateOut, All_FK[i].cumsum(), label=f"{drawdownFK[i]*100:.0f}% drawdown")

plt.plot(DateOut, SPtrackFK.cumsum(), color = 'red', label = 'SP500')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return')
plt.title('Uncompounded Return: Full Kelly Portfolio')

plt.grid(True)

plt.figure(figsize=(10, 6))
for i in range(len(All_FK)):
    plt.plot(DateOut, (1+All_FK[i]).cumprod(), label=f"{drawdownFK[i]*100:.0f}% drawdown")

plt.plot(DateOut, (1+SPtrackFK).cumprod(), color = 'red', label = 'SP500')
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Return (log scale)')
plt.title('Compounded Return: Full Kelly Portfolio')
plt.yscale('log')
plt.grid(True)

'''Drawdown'''
plt.figure(figsize = (15,6))
DDFK = (All_FK[8].cumsum()).cummax() -All_FK[8].cumsum()
DDOP = (All_OP[6].cumsum()).cummax() -All_OP[6].cumsum()
SPDD = (SPtrackFK.cumsum()).cummax() -SPtrackFK.cumsum()
plt.plot(DateOut, DDFK, label ='Full Kelly: 45% MaxDD')
plt.plot(DateOut, DDOP, label = 'Optimize MaxDD Port: 35%', alpha = .8)
plt.plot(DateOut, SPDD, color = 'Green', label = 'SP500 DD', alpha = .6)
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Drawdown')
plt.title('Uncompounded Drawdowns: Full Kelly vs Optimal Drawdown Portfolio')
plt.legend()


