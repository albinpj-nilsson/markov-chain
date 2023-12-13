import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 



# Constants
regimes = ['Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5', 'Regime6', 'Regime7', 'Regime8', 'Regime9', 'Regime10']
regimes6to10 = ['Regime6', 'Regime7', 'Regime8', 'Regime9', 'Regime10']
regimes1to5 = ['Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5']


#Import transition matrix from markov_chain.py
def import_matrix():
    transition_matrix = pd.read_csv("transition_matrix.csv", index_col=0)
    transition_matrix = np.linalg.matrix_power(transition_matrix, 1)
    transition_matrix = pd.DataFrame(transition_matrix, columns=regimes, index=regimes)

    #Add two columns for sum of regimes for positive and negative return values
    transition_matrix['sump1to5'] = transition_matrix[regimes1to5].sum(axis=1, numeric_only=True)
    transition_matrix['sump6to10'] = transition_matrix[regimes6to10].sum(axis=1, numeric_only=True)

    return transition_matrix

# Place and Hold strategy
def basic_strategy (cap):

    #Set starting value to cap
    df.at[df.index[0], "P&H"] = cap

    # loop over all days    
    for i in range(1, len(df)):
        df.at[df.index[i], "P&H"] = df.at[df.index[i - 1], "P&H"] * (1 + df.at[df.index[i], "Daily Return"]) #Calculate value Place and Hold
    return



def investment_strategy(cap, trans_matrix):
        
    df.reset_index(inplace=True)
    
    # Add column for daily return
    df['Daily Return'] = df['Close'].pct_change()
 

    
    # Define conditions and choices for the 'Regime' column
    bins = np.arange(-0.02,0.025,0.005)
    bins = np.insert(bins, 0, -1)
    bins = np.insert(bins, 10, 1)

    # Use pd.cut to create the 'Regime' column based on bins and labels
    df['Regime'] = pd.cut(df['Daily Return'], bins=bins, labels=regimes, right=False)
    
    invest_harshness = 1 # Factor for investing
    divest_harshness = 0.2 # Factor for divesting
    df.at[0, 'bank'] = cap # Start capital in bank
    df.at[0, 'invested'] = 0 # Start capital invested in stock
    
    # loop over all days
    for i in range(1,len(df)):
        regime_today = df.iloc[i]['Regime'] # extract the regime today
        prob_up_tomorrow = trans_matrix.loc[regime_today]['sump6to10'] # extract probability of the stock going up tomorrow based on todays regime
        prob_down_tomorrow = 1 - prob_up_tomorrow # extract probability of the stock going down tomorrow based on todays regime

        # Calculate how much to be divested from the stock into the bank
        df.at[i, 'bank'] = df.at[i-1, 'bank'] - (prob_up_tomorrow*invest_harshness*df.at[i-1, 'bank']) + (prob_down_tomorrow*divest_harshness*df.at[i-1,'invested'])

        # Calculate how much to be invested in the stock
        df.at[i, 'invested'] = df.at[i-1, 'invested']*(1+df.iloc[i]['Daily Return'])+(prob_up_tomorrow*invest_harshness*df.at[i-1, 'bank']) - (prob_down_tomorrow*divest_harshness*df.at[i-1,'invested'])

    # Calculate the sum of bank and invested = performance
    df['MC Strategy'] = df['bank'] + df['invested']
    return df

    
#Plot the graph
def plot ():

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["P&H"], label="Place and Hold", color='blue')
    plt.plot(df["Date"], df["MC Strategy"], label="MC Strategy", color="red")
    plt.title("OMXS30 (2023-01-01 to 2023-12-01)")
    plt.xlabel("Date")
    plt.ylabel("Value (SEK)")
    plt.legend()
    plt.show()
    return





if __name__ == "__main__":

    #Starting capital
    cap = 100000

    transition_matrix = import_matrix()

    #Download data
    df = yf.download("^OMX", start="2023-01-01",end="2023-12-01") 
    df["Daily Return"] = df["Close"].pct_change()
    
    basic_strategy(cap)
    investment_strategy(cap, transition_matrix)

    plot()
    


    


