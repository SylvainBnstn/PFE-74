import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
###############################################################################
# Get data from file
def get_data(file_name = "airbnb_data.csv", train_proportion=0.9):
    share_data = pd.read_csv(file_name) 
    data = share_data[["date", "price", "booked", "room_type"]]

    data_mask = data["room_type"] == "Entire home/apt"
    data = data[data_mask]
    data = data[:3000]
 
###############################################################################
# separe data for train & test
    data_train = data[:int(train_proportion*len(data))]
    data_test = data[int(train_proportion*len(data)):]

    #data_train=data_train[ (data_train["price"] >=100) & (data_train["price"] <=200)]

    #range of price
    a = data_train["price"]
    """
    b=[]
    for i in range(len(a)):
        b.append([(a.iloc[i,0],a.iloc[i,1])])
    """
    Price_range = np.array(a)
    #Price_range = b
    #Price_range = Price_range.tolist()

    return Price_range, data_train, data_test

get_data()



