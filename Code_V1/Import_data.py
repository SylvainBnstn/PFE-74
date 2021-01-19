import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



###############################################################################
# Get data from file
def get_data_01(file_name = "airbnb_data.csv", train_proportion=0.9):
    share_data = pd.read_csv(file_name) 
    data = share_data[["date", "price", "booked", "room_type"]]
    
    data_mask = data["room_type"] == "Entire home/apt"
    data = data[data_mask]
    data = data[:1000]
    """
    nb_id = data.groupby(["id"], as_index = False).size()
    nb_id.sort_values(by=["size"], ascending = False, inplace = True)
    test = nb_id
    """
###############################################################################
# split data for train & test
    data_train = data[:int(train_proportion*len(data))][["price", "booked"]]
    data_test = data[int(train_proportion*len(data)):][["price", "booked"]]
        
    # Calcul 
    prix = list(data_train["price"])
    list_prix_unique_ordonnee = sorted(set(prix))
    counter = [prix.count(p) for p in list_prix_unique_ordonnee]
    moy = np.mean(prix)
    med = np.median(prix)
    ecart = np.std(prix)
    
    pmin = med-2*ecart
    pmax = med+2*ecart
    
    list_param = [moy, med, pmin, pmax]
    
    if pmin < 0:
        pmin = min(prix)
    if pmax > max(prix):
        pmax = max(prix)
          
    # Range of price
    Price_range = data_train[(pmin <= data_train["price"]) & (data_train["price"] <= pmax)]
    Price_range = Price_range.reset_index(drop=True)
    
    diff_prix=[0]
    demand=[0]
    for i in range(1,len(Price_range)):
        diff_prix.append( Price_range["price"][i] - Price_range["price"][i-1])
        demand.append(Price_range["booked"][i])
        
    Price_range["Difference de prix"] = diff_prix
    Price_range["Demand"] = demand
    
    
    
    return Price_range, data_test, list_prix_unique_ordonnee, counter, list_param 
    
#pr, dt, list_prix, count, list_param = get_data()


def plot(list_prix, count, list_param):
    plt.figure(figsize=(16, 12))
    plt.plot(list_prix, count)
    plt.axvline(x=list_param[0] , c='red')
    plt.axvline(x=list_param[2] , c='yellow')
    plt.axvline(x=list_param[3] , c='cyan')
    plt.axvline(x=list_param[4] , c='cyan')
    plt.grid()
    
    

def get_data(file_name = "airbnb_data.csv", train_proportion=0.9):
    result = pd.read_csv(file_name) 
    #result = share_data[["id", "date", "price", "booked", "room_type"]]
    
    #result.id.unique()
    
    nb_id = result[result["room_type"] == "Entire home/apt"].groupby(["id"], as_index = False).size().reset_index(name="size")
    
    #nb_id = pd.DataFrame(nb_id)
    #nb_id = nb_id.rename(columns={0:"size"})
    
    nb_id.sort_values(by=["size"], ascending= False, inplace= True)
    nb_id = nb_id[nb_id["size"] >=70 ]
    nb_id
    
    
    df = pd.merge(result[result["room_type"] == "Entire home/apt"], nb_id["id"] ,how = "right" ,on=["id"] )
    df =  df[df["room_type"] == "Entire home/apt"]
    df.duplicated().sum()
    df.drop_duplicates(inplace= True)
    df=df[::-1]#pr inverser les lignes
    
    #Create a dataframe full of NaN value 
    #Each row is an id, and each column is a date
    data_with_all_date = df.pivot(index="id", columns="date", values = "price")
    data_with_all_date
    
    booked_with_all_date = df.pivot(index="id", columns="date", values = "booked")
    
    #data = df.pivot(index="id", columns="date", values = ["price" ,"booked"]) 
    """
    
    # Plot of the first 20 id with price < 500$
    list_id = list(df[df["price"] <500].id.unique())[:20] #select the first 20 id n the list
    list_date =list(df.date.unique())
    plt.figure(figsize=(20, 10))
    for i in range(len(list_id)):
      plt.plot( "date", "price" ,data = df[df["id"] == list_id[i] ] , label = list_id[i])
    plt.xticks(rotation=45)
    plt.ylabel("price")
    plt.xlabel("date")
    plt.legend(title = "ID")
    plt.grid()

    """
    
    return data_with_all_date,booked_with_all_date


df_p, df_b = get_data()


def training_data(df_price, df_booked):
    
    price_grid_total=[]
    data_train = df_price[df_price.columns[0:45]]
    
    
    # Parcours les k apt
    for k in range(len(data_train)):
        price = list(data_train.iloc[k])
        diff_prix=[False]
        demand=[False]
        # Calcule pour chaque apt la variation de prix et associe à la demande
        for i in range(1,len(data_train.iloc[k])):
            diff_prix.append( data_train.iloc[k][i] - data_train.iloc[k][i-1] )
            demand.append(df_booked.iloc[k][i])
        
        price_grid_temp = np.c_[price,diff_prix,demand]
    
        price_grid_total.append(price_grid_temp)
        
    all_price_grid=np.concatenate(price_grid_total)
    all_price_grid=np.unique(all_price_grid,axis=0)
    
    
    return all_price_grid

#df = training_data(df_p, df_b)
    

### get the index of each aprtment
def index(df_p):
    
    index_=list(df_p.index)
    return index_


indice= index(df_p)

def testing_data_2019(df_price,df_booked):
    
    data_test=df_price[df_price.columns[45:57]]
    data_test_booked = df_booked[df_booked.columns[45:57]]
    return data_test,data_test_booked 

#data_test_2019,data_test_booked_2019=testing_data_2019(df_p,df_b)



def testing_data_2020(df_price,df_booked):
    
    data_test=df_price[df_price.columns[57:]]
    data_test_booked = df_booked[df_booked.columns[57:]]
    return data_test,data_test_booked 
#data_test_2020,data_test_booked_2020=testing_data_2020(df_p,df_b)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


