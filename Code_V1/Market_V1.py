 # -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Naiv_Clients_v2 as cnt
import data_analysis as da
import airbnb_processing as ap


import random as rd
import pandas as pd

# pour rappel 
# Class list_naiv_clients est l'objet qui contient tous les clients
# on l'appel donc naiv_clients
# Cette objet contient la list des objets clients qui contiennent eux-mêmes leurs prix min max et WTP

class Market:
    
    def __init__(self,clients):
        self.clients = clients
        
        data=[0,0,0,0,0,0,0,0]
        self.df_naiv_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_naiv_sales.loc[0]=data

        
    #fonction d'execution des ventes
    def check_sales(self, price, clients):
        
        temp_envi = temp_real = temp_aban = temp_inst = temp_repou = 0
        
        #list des index des acheteurs ayant conclu un acht
        list_sales=[]
        
        for i in range(len(self.clients.clients_list)):
            
            #cas ou le prix est compris dans la cible
            if (price >= self.clients.clients_list[i].prix_min and price <= self.clients.clients_list[i].prix_max):
                
                #on montre que l'achat est envisagé
                #temp_str =str("Achat envisagé -> ")
                temp_envi+=1
                proba_temp = rd.random()
                
                #a revoir pour mise en place de WTP exacte
                if proba_temp <= self.clients.clients_list[i].will_to_pay :
                    
                    #temp_str+=str("Achat Réalisé")
                    temp_real+=1
                    #l'acheteur quitte le marché
                    list_sales.append(i)
                
                else:
                    #temp_str+=str("Achat Abandonnée")
                    temp_aban+=1
                
                #print(temp_str)
            
            #cas ou cest moins chère que le prix min
            elif (price <= self.clients.clients_list[i].prix_min):
                #print("Instant achat")
                temp_inst+=1
                list_sales.append(i)
            
            #sinon
            else:
                #print("Achat repoussé")
                temp_repou+=1
        
        nb_row=self.df_naiv_sales.shape[0]
        data_temp=[nb_row,price,temp_envi,temp_real,temp_aban,temp_inst,temp_repou,temp_inst+temp_real]
        self.df_naiv_sales.loc[nb_row]=data_temp
        return list_sales    
        


def test():
    
    ###################################
    
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-12:df_final.shape[0]-1]
    
    #définit une list de clients naif avec prix min, prix max et nb clients
    naiv_clients= cnt.list_naiv_clients(10,20,100,0.85,0.15)
    
    #on créé le marché
    market=Market(naiv_clients)
    
    for k in range(df_final.shape[0]):    
        avail_rate=df_final.loc[df_final.index[k],"mean_availability_30"] / 30 
        print("B",avail_rate)
        final_rate=avail_rate+0.05
        print("B",avail_rate)
        list_del=market.check_sales(14+(k/5),naiv_clients)
        naiv_clients.update_client(10, 20, list_del,0.85,final_rate)
        
    print(df_final)

    
    print(market.df_naiv_sales)
    


        

        
        
test()        
        