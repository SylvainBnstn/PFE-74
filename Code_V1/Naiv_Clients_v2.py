# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:48:13 2021

@author: Sylvain
"""


import random
import data_analysis as da
import airbnb_processing as ap

class Naiv_client:
    
    def __init__(self, prix_min,prix_max,will_to_pay):
        self.prix_max=prix_max
        self.prix_min=prix_min
        self.will_to_pay=will_to_pay

    def __str__(self):
        r="Prix achat min: " + str(self.prix_min)+"\n"
        r+="Prix achat max: " + str(self.prix_max)+"\n"
        r+="WTP : " + str(self.will_to_pay)+"\n"
        return r
    
    def __repr__(self):
        r=(self.prix_min,self.prix_max,self.will_to_pay)
        return str(r)
    
class list_naiv_clients:
    
    def __init__(self,prix_min,prix_max,nb_client,wtp,rate_to_assure):
        self.nb_client = nb_client
        self.clients_list=[]
        aleat = int((1-rate_to_assure)*nb_client)
        print (aleat)
        for _ in range(aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp))
            
        for _ in range(nb_client-aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp))
    
    def __str__(self):
        return str(self.clients_list)
    
    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]
            
    def update_client(self,prix_min,prix_max,list_to_del,wtp,rate_to_assure):
        self.del_client(list_to_del)
        
        print("CList avant",len(self.clients_list))
        print("DelList",len(list_to_del))
        
        aleat = int((1-rate_to_assure)*(len(list_to_del)))
        
        print ("Aleat",aleat)
        
        for _ in range(aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp))
            
        print("Chiffre sorti du Q",(len(list_to_del) -aleat))
        
        for _ in range(len(list_to_del) -aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp))
            
        print("CList après",len(self.clients_list))
            
        