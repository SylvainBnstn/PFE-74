import pandas as pd
import numpy as np
import Market_V1 as mkt
import random as rd

import data_analysis as da
import airbnb_processing as ap

# Get the train_proportion corresponding to a column index (= train_proportion * columns total)
def get_proportion(b,proportion):
    return int(b*proportion)

# Clean the data
def get_data(train_proportion,file_name = "airbnb_data.csv"):
    # Read the csv file
    result = pd.read_csv(file_name) 
    
    result.duplicated().sum()
    result.drop_duplicates(inplace= True)
    
    # Number of unique date
    b=len(list(result.date.unique()))
    
    # Get apt id on Entire apt and those which have a price on all the date in the data ~ 6 years (2015 to 2020)
    nb_id = result[result["room_type"] == "Entire home/apt"].groupby(["id"], as_index = False).size().reset_index(name="size")
    nb_id.sort_values(by=["size","id"], ascending= False, inplace= True)
    nb_id = nb_id.loc[nb_id["size"] >=b]
    
    df = pd.merge(result[result["room_type"] == "Entire home/apt"], nb_id["id"] ,how = "right" ,on=["id"] )
    df =  df[df["room_type"] == "Entire home/apt"]
    df.duplicated().sum()
    df.drop_duplicates(inplace= True)
    
    # Each row is an id, and each column is a date
    data_with_all_date = df.pivot(index="id", columns="date", values = "price")
    booked_with_all_date = df.pivot(index="id", columns="date", values = "booked")
    
    # Compute the mean price of each apt over time and keep apt with a mean price between 70 and 230
    mean_price_all_date = df.groupby("id", as_index = False).agg({"price" : ["mean"]})
    mean_price_all_date.columns = ["id","mean_price"]
    mean_price_all_date = mean_price_all_date[(mean_price_all_date["mean_price"] >= 70) & (mean_price_all_date["mean_price"]<=230)]
    
    # Filter on unique id which have mean price between 70 and 230
    list_id = (mean_price_all_date["id"]).unique()
    data_with_all_date = data_with_all_date.loc[data_with_all_date.index.isin(list_id)]
    booked_with_all_date=booked_with_all_date.loc[booked_with_all_date.index.isin(list_id)]
    
    # train_proportion corresponding to a column index (= train_proportion * columns total)
    proportion = get_proportion(b,train_proportion)
    
    return data_with_all_date, booked_with_all_date, proportion

#df_p, df_b,proportion = get_data(0.83)

    
def training_data(df_price, df_booked,proportion):
    # Get train data for us we choose from 2015 to 2019
    data_train = df_price[df_price.columns[0:proportion]]
    
    price_grid_total=[]
    # Read through the k apt
    for k in range(len(data_train)):
        price = list(data_train.iloc[k])
        demand=[0]
        date=[0]
        
        # Compute for each apt, the demand (booked) in the data
        for i in range(1,len(data_train.iloc[k])):
            demand.append(df_booked.iloc[k][i])
            # Date
            date.append(i)
        
        price_grid_total.append(np.c_[price,demand,date])
        
    # Concatenate the price grid of all apt and get the unique in row
    all_price_grid = np.concatenate(price_grid_total)    
    return all_price_grid
    
#df = training_data(df_p, df_b)
    
def create_data(nb_mois_test):
    # Load data from the file airbnb_processing
    df_start=ap.load_data("airbnb_data.csv")
    
    # Filter on "Entire home/apt" and where price is between 100 and 200
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    nombre_de_mois_test = nb_mois_test

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-nombre_de_mois_test:df_final.shape[0]-1]
    
    # Creat the global file of data that we want, price, demand, data and strategic client proportion
    df_glob_sales = pd.DataFrame(columns= ["Prix","Achat_Tot","Date","Part_Strat"])
    
    # Loop on price
    for n in range (160):
        
        print(n, "% effectués")
        
        # Loop on proportion
        for i in range(100):
            
            # The market is created
            market=mkt.Market(100,200,30,1-(i/100),0.85,0.15,df_start,df_final)
            
            # Initialize the price
            p_trace=[rd.randrange(70,230)]
            
            #parcours de date
            for k in range(df_final.shape[0]):    
                
                p = rd.randrange(70,230)
        
                p_trace.append(p)
                
                achat_naiv, achat_strat = market.updates(p,p_trace,k)
                
                # Save data in the empty dataframe
                df_glob_sales = df_glob_sales.append({"Prix":p,"Achat_Tot":(achat_naiv+achat_strat),"Date":k+1,"Part_Strat":(i/100)},ignore_index=True)
                    
    print(df_glob_sales)
    
    # Save into a csv file
    df_glob_sales.to_csv("Data_Model.csv",index=False)
    
"""
def create_data_2(nb_mois_test):
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    nombre_de_mois_test = nb_mois_test

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-nombre_de_mois_test:df_final.shape[0]-1]
    
    df_glob_sales = pd.DataFrame(columns= ["Prix","Achat_Tot","Date","Part_Strat"])
    
     #parcours de prix 
    for n in range (300):
        
        print(n, "% effectués")
        
        #parcours de proportion
        for i in range(100):
            
            #on créé le marché
            market=mkt.Market(100,200,30,1-(i/100),0.85,0.15,df_start,df_final)
            
            p_trace=[rd.randrange(100,200)]
            
            p = rd.randrange(100,200)
            
            #parcours de date
            for k in range(df_final.shape[0]):   
                
                choice = rd.random()
                
                #prix random
                if choice <0.25 :
                    
                    p = rd.randrange(100,200)
        
                if choice >= 0.25 and choice < 0.5 :
                    
                    up = rd.randrange(1,10)
                    p = p + up
                    
                if choice >= 0.5 and choice < 0.75 :
                    
                    down = rd.randrange(1,10)
                    p = p + down
                
                if choice >= 0.75 :
        
                    p = p
                
                
                p_trace.append(p)
                
                achat_naiv, achat_strat = market.updates(p,p_trace,k)
                
                df_glob_sales = df_glob_sales.append({"Prix":p,"Achat_Tot":(achat_naiv+achat_strat),"Date":k+1,"Part_Strat":(i/100)},ignore_index=True)
    print(df_glob_sales)
    
    df_glob_sales.to_csv("Data_Model_2.csv",index=False)
"""
    
def load_data(path, train_proportion ,strat_min_prop, step_prop):
    
    df= pd.read_csv(path)
    df = df.loc[(df["Part_Strat"]>=strat_min_prop) & (df["Part_Strat"]<strat_min_prop+step_prop)]
    df=df.drop(columns=["Part_Strat"])
    
    # Convert to numpy
    price_grid_temp = df.to_numpy()
    price_grid_temp = price_grid_temp[:,:].astype(int)
    
    # Get the train_proportion corresponding to a column index (= train_proportion * columns total)
    proportion = get_proportion(len(price_grid_temp),train_proportion)
    
    # Get the data for test part
    price_grid_test = price_grid_temp[proportion:len(price_grid_temp),:]
    
    # Get the data for train part
    price_grid = price_grid_temp[0:proportion,:]
    
    return price_grid, price_grid_test, proportion
