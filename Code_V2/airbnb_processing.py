# -*- coding: utf-8 -*-

import pandas as pd
from math import nan

#fonction qui récupère les urls dans le .txt correspondant
def get_url_list(path):
    url_file=open(path,"r")
    lines = url_file.readlines()
    list_urls = []
    for line in lines:
        list_urls.append(line.strip())
    return list_urls

#fonction qui isole les urls des villes contenus dans la list_city
def isolate_expected_urls(url_list,list_city):
    final_url=[]
    for k in range(len(list_city)):
        for i in range(len(url_list)):
            url_split = url_list[i].split("/")
            if url_split[5] == list_city[k]:
                final_url.append(url_list[i])
    return final_url
    
# fonction qui récupère les données des urls de la url_list
def get_data(url_list,expected_path, list_city):
    
    #creation de la liste des df
    frame = []
    # liste regroupant les noms de colonnes a recuperer
    column_names = [ "id","property_type", "room_type", "accommodates", "bedrooms", "beds", "price","availability_30", "number_of_reviews","review_scores_rating",	"review_scores_accuracy" ]

    missed_col =  ['square_feet','cleaning_fee']
    
    #on charge la liste d'url que l'on souhaite
    url_list = isolate_expected_urls(url_list, list_city)
    
    #boucle de parcours de chacun des liens
    for i in range(len(url_list)):
        
        #on read directement sur internet
        df_init = pd.read_csv(url_list[i])
        #on va splitter l'url pour recuperer le pays la ville et la date
        url_split = url_list[i].split("/")
        #on ne conserve que les colonnes interessantes
        df = df_init[column_names]
        
        #mise au point sur les colonnes qui posent ds problème de nan
        for j in range(len(missed_col)):
            if (missed_col[j] in list(df_init.columns)):
                df[missed_col[j]] = df_init[missed_col[j]]
            else:
                df[missed_col[j]] = nan
      
        
        #on cree le contenu de ces colonnes
        df["country"] = url_split[3]
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
    result["cleaning_fee"].fillna("$0.00", inplace = True)
    result["cleaning_fee"] = result["cleaning_fee"].str.replace("$","")
    result["cleaning_fee"] = result["cleaning_fee"].str.replace(",","").astype(float)

    
    #Ajout de deux autres colonnes
    result["revenue_30"] = result["price"] *(30-result["availability_30"])
    # nombre de jours révservés
    result["booked"] = 30-result["availability_30"] 
    result.sort_values(by=["id","date"], inplace = True)
    result.reset_index(drop=True, inplace = True)
    result.drop_duplicates(inplace= True)
    result.to_csv(expected_path, index=False)
    
#fonction de chargement simplifié
def load_data(path):
    df=pd.read_csv(path)
    print("\n°°°°° Chargement reussi! °°°°°")
    return df


#lignes à décommenter pour charger un nouveau set de données

# URLS=get_url_list("urls.txt")
# get_data(URLS,"airbnb_data.csv",["new-york-city"])
# load_data("airbnb_data.csv")