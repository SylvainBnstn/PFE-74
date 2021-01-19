# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:24:03 2021

@author: cecil
"""
import pandas as pd
import matplotlib.pyplot as plt

def get_data(file_name = "airbnb_data.csv", train_proportion=0.9):
    result = pd.read_csv(file_name,sep=",") 
        
    nb_id = result[result["room_type"] == "Entire home/apt"].groupby(["id"], as_index = False).size().reset_index(name="size")
    print("hello1", nb_id)
    nb_id.sort_values(by=["size"], ascending= False, inplace= True)
    print("hello2", nb_id)
    nb_id = nb_id[nb_id["size"] >=70 ]
    
    print("hello3", nb_id)
    df = pd.merge(result[result["room_type"] == "Entire home/apt"], nb_id["id"] ,how = "right" ,on=["id"] )
    
    print("hello4", nb_id)
    df =  df[df["room_type"] == "Entire home/apt"]
    df.duplicated().sum()
    df.drop_duplicates(inplace= True)
    

    #Each row is an id, and each column is a date
    data_with_all_date = df.pivot(index="id", columns="date", values = "price")
    
    #data_with_all_date
    
    
    # Plot of the first 20 id with price < 500$
    list_id = list(df[df["price"] <500].id.unique())[:20] #select the first 20 id n the list
#    list_date =list(df.date.unique())
    plt.figure(figsize=(30, 10))
    for i in range(len(list_id)):
      plt.plot( "date", "price" ,data = df[df["id"] == list_id[i] ] , label = list_id[i])
    plt.xticks(rotation=45)
    plt.ylabel("price")
    plt.xlabel("date")
    plt.legend(title = "ID")
    plt.grid()

    
    return df
    
result = pd.read_csv("airbnb_data.csv") 

d = get_data()