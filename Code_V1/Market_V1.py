# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Clients_V1 as cnt
import random as rd
import pandas as pd

# pour rappel 
# Class list_naiv_clients est l'objet qui contient tous les clients
# on l'appel donc naiv_clients
# Cette objet contient la list des objets clients qui contiennent eux-mêmes leurs prix min max et WTP

class Market:
    
    def __init__(self,clients):
        self.clients = clients
        data=[0,0,0,0,0,0,0]
        self.df_naiv_sales = pd.DataFrame(columns= ["Ite","Achat_Envi","Achat_Rea","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_naiv_sales.loc[0]=data
        print(self.df_naiv_sales)
        
    #fonction d'execution des ventes
    def check_sales(self, price, clients):
        
        #list des index des acheteurs ayant conclu un acht
        list_sales=[]
        
        for i in range(len(self.clients.clients_list)):
            
            #cas ou le prix est compris dans la cible
            if (price >= self.clients.clients_list[i].prix_min and price <= self.clients.clients_list[i].prix_max):
                
                #on montre que l'achat est envisagé
                temp_str =str("Achat envisagé -> ")
                proba_temp = rd.random()
                
                #a revoir pour mise en place de WTP exacte
                if proba_temp <= self.clients.clients_list[i].will_to_pay :
                    
                    temp_str+=str("Achat Réalisé")
                    #l'acheteur quitte le marché
                    list_sales.append(i)
                
                else:
                    temp_str+=str("Achat Abandonnée")
                
                print(temp_str)
            
            #cas ou cest moins chère que le prix min
            elif (price <= self.clients.clients_list[i].prix_min):
                print("Instant achat")
                list_sales.append(i)
            
            #sinon
            else:
                print("Achat repoussé")
        return list_sales    
        


def test():
    #définit une list de clients naif avec prix min, prix max et nb clients
    naiv_clients= cnt.list_naiv_clients(10,20,20)
    
    #on créé le marché
    market=Market(naiv_clients)
    
    #première vérification
    list_del=market.check_sales(16,naiv_clients)
    naiv_clients.del_client(list_del)

        

        
        
test()        
        