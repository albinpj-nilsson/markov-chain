"""
Anm�rkningar:
    
OBS! I just detta fall blev �verg�ngsmatrisen ganska oanv�ndbar. M�jligen
f�r simpel modell, kanske fel p� kod/logik?
    
1) �ndringar av sj�lva csv-filen:
   - F�rsta raden borttagen
   - "Closingreturn" �ndrad till "Closing return"

2) F�ljande motivering till funktionen power_iteration var on�dig men l�ter den st� kvar

Power iteration is a numerical method to find the dominant eigen vector of a matrix.
THM 10.5.4 says that there exists an unique q such that P*q = q,
where P is the transition matrix and q is the steady-state vector.
This is equivalent with fidning the eigenvector of P with eigenvalue 1 (A*x = lambda*x).

3) Att g�ra?
    - Generalisera dimensionen av �verg�ngsmatrisen
    - Kolla p� flera dagar samtidigt
    - Kolla p� flera faktorer (h�gdimensionell matris? se punkt 4)
    
    - G�ra investeringsstrategi (generaliserad?) och applicera
    
4) Ifall h�gdimentionell matris skulle användas kan det bli sv�rt att l�sa ekvationer mm.
    Metod som g�r att till�mpa �r MCMC (Monte Carlo Markov Chain).
    
5) Valde att anv�nda funktionell programmering, men objektorienterad g�r ocks�.
"""

# Import statements
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import yfinance as yf

# Constants
regimes = ['Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5', 'Regime6', 'Regime7', 'Regime8', 'Regime9', 'Regime10']

# Functions
def create_transition_matrix(df, regimes):
    """
    Creates a transition matrix
    :param df: DataFrame containing 'Regime'-column
    :param regimes: List of regimes
    :return: DataFrame representing transition matrix 
    """
    # Initialize transition matrix
    transition_matrix = pd.DataFrame(0, columns=regimes, index=regimes)

    # Count transitions between regimes
    for i in range(2, len(df)): #first value is NaN
        prev_regime = df.at[i - 1, 'Regime']
        current_regime = df.at[i, 'Regime']
        transition_matrix.at[prev_regime, current_regime] += 1

    # Normalize to obtain transition probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    
    return transition_matrix

def investment_strategy(trans_matrix):
        
    ticker = "AAPL"
    stock_data = yf.download(ticker, start = "2023-01-01", end = "2023-12-01")
    stock_data.reset_index(inplace=True)
    
    # Add column for daily return
    stock_data['Daily return'] = stock_data['Close'].pct_change()
    
    # Define conditions and choices for the 'Regime' column
    bins = np.arange(-0.02,0.025,0.005)
    bins = np.insert(bins, 0, -1)
    bins = np.insert(bins, 10, 1)

    # Use pd.cut to create the 'Regime' column based on bins and labels
    stock_data['Regime'] = pd.cut(stock_data['Daily return'], bins=bins, labels=regimes, right=False)
    
    invest_harshness = 1 
    divest_harshness = 0.2
    stock_data.at[1, 'bank'] = 1000
    stock_data.at[1, 'invested'] = 0
    
    for i in range(2,len(stock_data)):
        regime_today = stock_data.iloc[i]['Regime']
        prob_up_tomorrow = trans_matrix.loc[regime_today]['sump6to10']
        prob_down_tomorrow = 1 - prob_up_tomorrow
        
        stock_data.at[i, 'bank'] = stock_data.at[i-1, 'bank'] - (prob_up_tomorrow*invest_harshness*stock_data.at[i-1, 'bank']) + (prob_down_tomorrow*divest_harshness*stock_data.at[i-1,'invested'])
        stock_data.at[i, 'invested'] = stock_data.at[i-1, 'invested']*(1+stock_data.iloc[i]['Daily return'])+(prob_up_tomorrow*invest_harshness*stock_data.at[i-1, 'bank']) - (prob_down_tomorrow*divest_harshness*stock_data.at[i-1,'invested'])

    stock_data['sum'] = stock_data['bank'] + stock_data['invested']
    return stock_data

# Main
if __name__ == "__main__":
    # Read and format
    ticker = "AAPL"
    df = yf.download(ticker, start = "2010-01-01", end = "2022-12-01")
    df.reset_index(inplace=True)
    
    # Add column for daily return
    df['Daily return'] = df['Close'].pct_change()
    
    # Define conditions and choices for the 'Regime' column
    bins = np.arange(-0.02,0.025,0.005)
    bins = np.insert(bins, 0, -1)
    bins = np.insert(bins, 10, 1)

    # Use pd.cut to create the 'Regime' column based on bins and labels
    df['Regime'] = pd.cut(df['Daily return'], bins=bins, labels=regimes, right=False)
    
    # Create transition matrix
    transition_matrix = create_transition_matrix(df, regimes)
    transition_matrix.to_csv('transition_matrix.csv',index=True)
    
    # Find steady-state vector
    q = power_iteration(transition_matrix)
    
    transition_matrix2 = np.linalg.matrix_power(transition_matrix, 1)
    transition_matrix2 = pd.DataFrame(transition_matrix2, columns=regimes, index=regimes)
    
    regimes6to10 = ['Regime6', 'Regime7', 'Regime8', 'Regime9', 'Regime10']
    regimes1to5 = ['Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5']
    
    transition_matrix2['sump1to5'] = transition_matrix2[regimes1to5].sum(axis=1, numeric_only=True)
    transition_matrix2['sump6to10'] = transition_matrix2[regimes6to10].sum(axis=1, numeric_only=True)    
    
    transition_matrix2.to_csv('transition_matrix.csv',index=True)
    
    #Implement Investment Strategy
    stock_results = investment_strategy(transition_matrix2)


        
        