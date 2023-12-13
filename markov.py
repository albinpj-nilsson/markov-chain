"""
Anmärkningar:
    
OBS! I just detta fall blev övergångsmatrisen ganska oanvändbar. Möjligen
för simpel modell, kanske fel på kod/logik?
    
1) Ändringar av själva csv-filen:
   - Första raden borttagen
   - "Closingreturn" ändrad till "Closing return"

2) Följande motivering till funktionen power_iteration var onödig men låter den stå kvar

Power iteration is a numerical method to find the dominant eigen vector of a matrix.
THM 10.5.4 says that there exists an unique q such that P*q = q,
where P is the transition matrix and q is the steady-state vector.
This is equivalent with fidning the eigenvector of P with eigenvalue 1 (A*x = lambda*x).

3) Att göra?
    - Generalisera dimensionen av övergångsmatrisen
    - Kolla på flera dagar samtidigt
    - Kolla på flera faktorer (högdimensionell matris? se punkt 4)
    
    - Göra investeringsstrategi (generaliserad?) och applicera
    
4) Ifall högdimentionell matris skulle anvÃ¤ndas kan det bli svårt att lösa ekvationer mm.
    Metod som går att tillämpa är MCMC (Monte Carlo Markov Chain).
    
5) Valde att använda funktionell programmering, men objektorienterad går också.
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
    
    #Create Training Data
    ticker = "AAPL"
    stock_data = yf.download(ticker, start = "2023-01-01", end = "2023-12-01")
    stock_data.reset_index(inplace=True)
    
    # Add column for daily return
    stock_data['Daily return'] = stock_data['Close'].pct_change()
    
    # Define conditions and choices for the 'Regime' column
    bins = np.arange(-0.02,0.025,0.005) # Create following numpy array: (-0.02, -0.015, -0.01, -0.05, 0, 0.05, 0.01, 0.015, 0.02)
    bins = np.insert(bins, 0, -1) # Add -1 at index 0
    bins = np.insert(bins, 10, 1) # Add 1 at index 10, the array becomes:(-1, -0.02, -0.015, -0.01, -0.05, 0, 0.05, 0.01, 0.015, 0.02, 1)

    # Use pd.cut to create the 'Regime' column based on bins and labels
    stock_data['Regime'] = pd.cut(stock_data['Daily return'], bins=bins, labels=regimes, right=False) # Daily return between -1 and -0.02 --> Regime 1 etc.
    
    invest_harshness = 1 # Factor for investing
    divest_harshness = 0.2 # Factor for divesting
    stock_data.at[1, 'bank'] = 1000 # Start capital in bank
    stock_data.at[1, 'invested'] = 0 # Start capital invested in stock
    
    # loop over all days
    for i in range(2,len(stock_data)):
        regime_today = stock_data.iloc[i]['Regime'] # extract the regime today
        prob_up_tomorrow = trans_matrix.loc[regime_today]['sump6to10'] # extract probability of the stock going up tomorrow based on todays regime
        prob_down_tomorrow = 1 - prob_up_tomorrow # extract probability of the stock going down tomorrow based on todays regime
        
        # Calculate how much to be divested from the stock into the bank
        stock_data.at[i, 'bank'] = stock_data.at[i-1, 'bank'] - (prob_up_tomorrow*invest_harshness*stock_data.at[i-1, 'bank']) + (prob_down_tomorrow*divest_harshness*stock_data.at[i-1,'invested'])
        
        # Calculate how much to be invested in the stock
        stock_data.at[i, 'invested'] = stock_data.at[i-1, 'invested']*(1+stock_data.iloc[i]['Daily return'])+(prob_up_tomorrow*invest_harshness*stock_data.at[i-1, 'bank']) - (prob_down_tomorrow*divest_harshness*stock_data.at[i-1,'invested'])

    # Calculate the sum of bank and invested = performance
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
    
    #Disregard
    transition_matrix2 = np.linalg.matrix_power(transition_matrix, 1)
    transition_matrix2 = pd.DataFrame(transition_matrix2, columns=regimes, index=regimes)
    
    regimes6to10 = ['Regime6', 'Regime7', 'Regime8', 'Regime9', 'Regime10']
    regimes1to5 = ['Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5']
    
    # Calculate sum probabilities of going up and down based on which regime you are in right now
    transition_matrix2['sump1to5'] = transition_matrix2[regimes1to5].sum(axis=1, numeric_only=True)
    transition_matrix2['sump6to10'] = transition_matrix2[regimes6to10].sum(axis=1, numeric_only=True)    
    
    transition_matrix2.to_csv('transition_matrix.csv',index=True)
    
    #Implement Investment Strategy
    stock_results = investment_strategy(transition_matrix2)


        
        