# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:34:32 2021

@author: huyan
"""

# -*- coding: utf-8 -*-


import pandas as pd

# URLS = ["http://data.insideairbnb.com/germany/bv/munich/2020-10-26/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-06-20/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-05-24/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-04-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-03-19/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-02-27/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2020-01-22/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-12-26/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-11-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-10-20/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-09-24/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-08-24/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-07-16/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-06-24/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-05-22/data/listings.csv.gz",
# "http://data.insideairbnb.com/germany/bv/munich/2019-03-15/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-11-06/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-10-19/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-09-17/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-08-27/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-07-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-06-22/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-05-27/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-04-23/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-03-21/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-02-27/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2020-01-27/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-12-30/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-11-26/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-10-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-09-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-08-29/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-07-21/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-06-25/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-05-24/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-04-20/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-03-19/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-02-13/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2019-01-22/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-12-16/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-11-18/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-10-16/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-09-16/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-08-19/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-07-28/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-05-22/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2018-04-17/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2017-02-18/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2016-08-07/data/listings.csv.gz",
# "http://data.insideairbnb.com/ireland/leinster/dublin/2016-01-06/data/listings.csv.gz"]

def get_url_list(path):
    url_file=open("urls.txt","r")
    lines = url_file.readlines()
    list_urls = []
    for line in lines:
        list_urls.append(line.strip())
    return list_urls

def isolate_expected_urls(url_list,list_city):
    final_url=[]
    for k in range(len(list_city)):
        for i in range(len(url_list)):
            url_split = url_list[i].split("/")
            if url_split[5] == list_city[k]:
                final_url.append(url_list[i])
    return final_url
    
def get_data(url_list,expected_path, list_city):
    
    #creation de la liste des df
    frame = []
    # liste regroupant les noms de colonnes a recuperer
    column_names = [ "id","property_type", "room_type", "accommodates", "bedrooms", "beds", "price","availability_30", "number_of_reviews","review_scores_rating",	"review_scores_accuracy" ]
    
    url_list = isolate_expected_urls(url_list, list_city)
    
    #boucle de parcours de chacun des liens
    for i in range(len(url_list)):
        
        #on read directement sur internet
        df = pd.read_csv(url_list[i])
        #on va splitter l'url pour recuperer le pays la ville et la date
        url_split = url_list[i].split("/")
        #on ne conserve que les colonnes interessantes
        df = df[column_names]
        #on cree le contenu de ces colonnes
        df["counrty"] = url_split[3]
        df["city"] = url_split[5]
        df["date"] = url_split[6]
        #on ajoute la df dans la liste de df
        frame.append(df)
    
    #on concat toute les df dans la list en une seule
    result = pd.concat(frame, ignore_index=True)
    
    
    #affichage de vérification
    print("Dimensions: ",result.shape)
    print("Les NaN: \n",result.isna().sum())

    #Ajustement de types    
    result["price"] = result["price"].str.replace("$","")
    result["price"] = result["price"].str.replace(",","").astype(float)
    result["date"] = pd.to_datetime(result["date"])
    
    #Ajout de deux autres colonnes
    result["revenue_30"] = result["price"] *(30-result["availability_30"])
    # nombre de jours révservés
    result["booked"] = 30-result["availability_30"] 
    
    result.to_csv(expected_path, index=False)
    
#fonction de chargement simplifié
def load_data(path):
    df=pd.read_csv(path)
    print("\n°°°°° Chargement reussi! °°°°°")
    return df

URLS=get_url_list("urls.txt")
get_data(URLS,"data/airbnb_data.csv",["new-york-city","amsterdam"])
load_data("data/airbnb_data3.csv")