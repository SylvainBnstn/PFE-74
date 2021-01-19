# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:48:13 2021

@author: Sylvain
"""


import random 
import data_analysis as da
import airbnb_processing as ap

class Naiv_client:
    
    def __init__(self, prix_min,prix_max,will_to_pay, echeance):
        self.prix_max=prix_max
        self.prix_min=prix_min
        self.will_to_pay=will_to_pay
        self.echeance = echeance

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
            temp_ech = random.randint(0,11)
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp,temp_ech))
            
        for _ in range(nb_client-aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_ech = random.randint(0,11)
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp,temp_ech))
    
    def __str__(self):
        return str(self.clients_list)
    
    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]
            
    def check_sales(self,price, list_resa):
        temp_envi = temp_real = temp_aban = temp_inst = temp_repou = temp_obli = temp_comp = 0
        
        #list des index des acheteurs ayant conclu un acht
        list_sales=[]
        
        
        for i in range(len(self.clients_list)):
            
            # print(i,",",self.clients_list[i].echeance)
            
            #cas normal ou l'écheance est différ
            if self.clients_list[i].echeance >= 1  and list_resa[self.clients_list[i].echeance-1] < 30 :
            
                #cas ou le prix est compris dans la cible
                if (price >= self.clients_list[i].prix_min and price <= self.clients_list[i].prix_max):
                    
                    #on montre que l'achat est envisagé
                    #temp_str =str("Achat envisagé -> ")
                    temp_envi+=1
                    proba_temp = random.random()
                    
                    #a revoir pour mise en place de WTP exacte
                    if proba_temp <= self.clients_list[i].will_to_pay :
                        
                        #temp_str+=str("Achat Réalisé")
                        temp_real+=1
                        #l'acheteur quitte le marché
                        list_sales.append(i)
                        
                        list_resa[self.clients_list[i].echeance-1] += 1
                    
                    else:
                        #temp_str+=str("Achat Abandonnée")
                        temp_aban+=1
                    
                    #print(temp_str)
                
                #cas ou cest moins chère que le prix min
                elif (price <= self.clients_list[i].prix_min):
                    #print("Instant achat")
                    temp_inst+=1
                    list_sales.append(i)
                    
                    list_resa[self.clients_list[i].echeance-1] += 1
                
                #sinon
                else:
                    #print("Achat repoussé")
                    temp_repou+=1
                    
            #cas normal ou l'écheance est différ
            if self.clients_list[i].echeance >= 1  and list_resa[self.clients_list[i].echeance-1] == 30 :
                temp_comp+=1
                
            
            #cas ou il reste de la place pour le mois qui vient
            if self.clients_list[i].echeance == 0 :
                
                if list_resa[self.clients_list[i].echeance-1] < 30 :
                    
                    temp_obli+=1
                    #l'acheteur quitte le marché
                    list_sales.append(i)
                    list_resa[self.clients_list[i].echeance-1] += 1
                else :
                    temp_comp+=1
                    #l'acheteur quitte le marché bredouille
                    list_sales.append(i)
                    
            
            self.clients_list[i].echeance -= 1
        
        return list_sales, temp_envi, temp_real, temp_aban, temp_inst, temp_repou,temp_obli, temp_comp, list_resa
            
    def update_client(self,prix_min,prix_max,list_to_del,wtp,rate_to_assure,resting_time):
        self.del_client(list_to_del)
        
        # print("CList avant",len(self.clients_list))
        # print("DelList",len(list_to_del))
        
        #on assure un nombre "rate to assure" de wtp élevé
        aleat = int((1-rate_to_assure)*(len(list_to_del)))
        
        # print ("Aleat",aleat)
        
        #on fait les WTP random
        for _ in range(aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            temp_ech = random.randint(0,11-resting_time)
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp,temp_ech))
            
        # print("Chiffre sorti du Q",(len(list_to_del) -aleat))
        
        #les wtp forcés
        for _ in range(len(list_to_del) -aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_ech = random.randint(0,11-resting_time)
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp,temp_ech))
            
        # print("CList après",len(self.clients_list))
            
        