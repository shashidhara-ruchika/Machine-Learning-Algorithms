# Apriori Association Rule

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# The dataset is imported without heading for each column
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Preparing the input of lists of lists
transactions = [ [ str(dataset.values[i,j]) for j in range(len(dataset.columns)) if str(dataset.values[i,j]) != 'nan']
                 for i in range(dataset[0].count()) ]

# Training Apriori on the dataset
from apyori import apriori
rules = apriori( transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
results_list = [ [str(results[i][0]),str(results[i][1])] for i in range(len(results)) ]

