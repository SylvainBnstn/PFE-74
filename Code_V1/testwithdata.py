import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Get data from file


def get_data(filename='data/airbnb_data2.csv',train_proportion=0.9):
    
    share_data=pd.read_csv(filename,sep=";")
    data = share_data[["date","price", "booked","room_type"]]
    data2= data['room_type']=='Entire home/apt'
    data3= data[data2]
    data=data3[ (data3["price"] >=100) & (data3["price"] <=200)]
    data = data[:1000]
    ###############################################################################
    # split data for training & test
    data_train = data[:int(train_proportion*len(data))]
    data_test = data[int(train_proportion*len(data)):]
    
    #range of price
    a = sorted(set(data_train["price"]))
    Price_range = list(a)

    return Price_range,data_train,data_test

get_data()





