# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Clients_V1 as cnt
import random as rd

# pour rappel 
# Class list_naiv_clients est l'objet qui contient tous les clients
# on l'appel donc naiv_clients
# Cette objet contient la list des objets clients qui contiennent eux-mêmes leurs prix min max et WTP

class Market:
    
    def __init__(self,clients):
        self.clients = clients
    
    #fonction d'execution des ventes
    def check_sales(self, price, clients):
        for i in range(len(self.clients.clients_list)):
            
            list_sales=[]
            
            #cas ou le prix est compris dans la cible
            if (price >= self.clients.clients_list[i].prix_min and price <= self.clients.clients_list[i].prix_max):
                temp_str =str("Achat envisagé -> ")
                proba_temp = rd.random()
                if proba_temp <= self.clients.clients_list[i].will_to_pay :
                    temp_str+=str("Achat Réalisé")
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
    naiv_clients= cnt.list_naiv_clients(10,20,10)
    
    #on créé le marché
    market=Market(naiv_clients)
    
    print(len(naiv_clients.clients_list))
    
    #première vérification
    list_del=market.check_sales(13,naiv_clients)
    naiv_clients.del_client(list_del)


    print(len(naiv_clients.clients_list))
        

        
        
test()        
        