import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_data(file_name = "airbnb_data_nyc.csv"):
    # Read the csv file
    result = pd.read_csv(file_name) 
    
    #
    result.duplicated().sum()
    result.drop_duplicates(inplace= True)
    
    # Get apt id on Entire apt and those which have a price over 69 months ~ 6 years (2015 to 2020)
    nb_id = result[result["room_type"] == "Entire home/apt"].groupby(["id"], as_index = False).size().reset_index(name="size")
    nb_id.sort_values(by=["size","id"], ascending= False, inplace= True)
    nb_id = nb_id[nb_id["size"] >=69 ]
    
    df = pd.merge(result[result["room_type"] == "Entire home/apt"], nb_id["id"] ,how = "right" ,on=["id"] )
    df =  df[df["room_type"] == "Entire home/apt"]
    df.duplicated().sum()
    df.drop_duplicates(inplace= True)
    
    # Each row is an id, and each column is a date
    data_with_all_date = df.pivot(index="id", columns="date", values = "price")
    booked_with_all_date = df.pivot(index="id", columns="date", values = "booked")
    
    # Compute the mean price of each apt over time and keep apt with a mean price between 70 and 350
    mean_price_all_date = df.groupby("id", as_index = False).agg({"price" : ["mean"]})
    mean_price_all_date.columns = ["id","mean_price"]
    mean_price_all_date = mean_price_all_date[(mean_price_all_date["mean_price"] >= 70) & (mean_price_all_date["mean_price"]<=350)]
    
    list_id = (mean_price_all_date["id"]).unique()
    data_with_all_date = data_with_all_date.loc[data_with_all_date.index.isin(list_id)]
    
    return data_with_all_date, booked_with_all_date

#df_p, df_b = get_data()

def training_data(df_price, df_booked):
    # Get train data on 2015 to 2018
    data_train = df_price[df_price.columns[0:45]]
    
    price_grid_total=[]
    # Read through the k apt
    for k in range(len(data_train)):
        price = list(data_train.iloc[k])
        diff_prix=[False]
        demand=[False]
        # Compute for each apt, the variation of price and collect the demand (booked) in the data
        for i in range(1,len(data_train.iloc[k])):
            diff_prix.append( data_train.iloc[k][i] - data_train.iloc[k][i-1] )
            demand.append(df_booked.iloc[k][i])
        
        # Concatenate the columns (price, variation_price, demand) for each apt
        price_grid_temp = np.c_[price,diff_prix,demand]
        price_grid_total.append(price_grid_temp)
        
    # Concatenate the price grid of all apt and get the unique in row
    all_price_grid = np.concatenate(price_grid_total)
    all_unique_price_grid = np.unique(all_price_grid, axis=0)
    
    return all_unique_price_grid
    
#df = training_data(df_p, df_b)

def testing_data_2019(df_price, df_booked):
    # Get test data of 2019
    data_test = df_price[df_price.columns[45:57]]
    data_test_booked = df_booked[df_booked.columns[45:57]]
    return data_test, data_test_booked

def testing_data_2020(df_price, df_booked):
    # Get test data of 2020
    data_test = df_price[df_price.columns[57:]]
    data_test_booked = df_booked[df_booked.columns[57:]]
    return data_test, data_test_booked

#dt19 , dtb19 = testing_data_2019(df_p, df_b)
#dt20 , dtb20 =testing_data_2020(df_p, df_b)





    