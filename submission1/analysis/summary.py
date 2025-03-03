import pandas as pd
import numpy as np
import pickle


taxBurden_data = pd.read_pickle("submission1/data/output/TaxBurden_Data.pkl")

print(taxBurden_data.head())