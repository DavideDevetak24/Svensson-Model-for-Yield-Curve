#libraries
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pathlib import Path

#data download
path = Path("C:/Users/Mio/OneDrive/Code_files/Uni_Quant/Derivatives/Data")
xlsx_path = (path / "Bond_prices.xlsx").as_posix()

raw_data = pd.read_excel(xlsx_path)
df = raw_data[['Coupon', 'Maturity', 'Price']]
df = df.dropna()
df = df[~((df["Coupon"] > 0) & (df["Maturity"] < 2))]

#bond parameters
principal = 100
div_freq = 4

#function fo calculation of zero_coupon YTM
def spot_no_coupon(price, maturity):
    return -np.log(price / principal) / maturity

for index, row in df.iterrows():
    if row['Coupon'] == 0:
        df.at[index, 'Spot_Rate'] = spot_no_coupon(row['Price'], row['Maturity'])
    else:
        df.at[index, 'Spot_Rate'] = np.nan

#function for calculation of coupon bonds YTM
def bond_ytm(ytm, coupon, maturity, price):
    cash_flows = np.array([coupon / div_freq] * int(maturity * div_freq))
    cash_flows[-1] += principal
    discount_factors = np.exp(-ytm * np.arange(1, len(cash_flows) + 1) / div_freq)
    return np.sum(cash_flows * discount_factors) - price

df['YTM'] = df.apply(lambda row: opt.root_scalar(bond_ytm, args=(row['Coupon'], row['Maturity'], row['Price']), 
                                                        bracket=[-0.1, 0.1], method='brentq').root if row['Coupon'] > 0 else np.nan, axis=1)

#function for the Svensson curve
def svensson(TTM, beta0, beta1, beta2, beta3, tau1, tau2):
    term_1 = (1 - np.exp(-TTM / tau1)) / (TTM / tau1)
    term_2 = term_1 - np.exp(-TTM / tau1)
    term_3 = (1 - np.exp(-TTM / tau2)) / (TTM / tau2) - np.exp(-TTM / tau2)
    

    return beta0 + beta1 * term_1 + beta2 * term_2 + beta3 * term_3

#optimization problem
def objective(params, maturities, zero_rates):
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    return np.sum((svensson(maturities, beta0, beta1, beta2, beta3, tau1, tau2) - zero_rates) ** 2)

initial_guess = [0.03, -0.01, -0.01, 0.01, 2.0, 2.0]
maturities = df['Maturity'].values
spot_rates = np.where(df['Coupon'] == 0, df['Spot_Rate'], df['YTM'])
optimal_param = opt.minimize(objective, initial_guess, args=(maturities, spot_rates)).x

T_range = np.linspace(0.01, max(maturities), 100)
spot_curve = svensson(T_range, *optimal_param)

#plotting
plt.figure(figsize=(12,6))
plt.scatter(maturities, spot_rates, color='red', label='Zero Bonds YTM')
plt.scatter(maturities, df['YTM'], color='orange', label='Bonds >2y YTM')
plt.plot(T_range, spot_curve, label='ECB/Svensson curve', linestyle='--', linewidth=3)
plt.xlabel('Years')
plt.ylabel('YTM')
plt.legend(loc='best')
plt.grid()
plt.show()

#Svensson parameters
print(optimal_param)
