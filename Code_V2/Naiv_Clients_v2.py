# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:48:13 2021

@author: Sylvain
"""


import random 
import data_analysis as da
import airbnb_processing as ap

#class client naif
class Naiv_client:
    
    def __init__(self, prix_min,prix_max,will_to_pay, echeance):
        #caractérisé par son prix min, max, wtp, echeance
        self.prix_max=prix_max
        self.prix_min=prix_min
        self.will_to_pay=will_to_pay
        self.echeance = echeance
        
    #fonction de print
    def __str__(self):
        r="Prix achat min: " + str(self.prix_min)+"\n"
        r+="Prix achat max: " + str(self.prix_max)+"\n"
        r+="WTP : " + str(self.will_to_pay)+"\n"
        return r
    #fonction d'affichage dans une liste
    def __repr__(self):
        # r=(self.prix_min,self.prix_max,self.will_to_pay,self.echeance)
        r=(self.echeance)
        return str(r)
    
#classe de liste des clients naifs
class list_naiv_clients:
    
    #init avec prix min et max le nombre de client, le WTP de fixe et le taux à assurer
    def __init__(self,prix_min,prix_max,nb_client,wtp,rate_to_assure):
        self.nb_client = nb_client
        self.clients_list=[]
        aleat = int((1-rate_to_assure)*nb_client)
        
        #les clients wtp aléatoire
        for _ in range(aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            temp_ech = random.randint(0,11)
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp,temp_ech))
            
        #les clients wtp fixe
        for _ in range(nb_client-aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_ech = random.randint(0,11)
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp,temp_ech))
    
    def __str__(self):
        return str(self.clients_list)
    
    #fonction de suppression d'un client
    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]
            
    #vérification des ventes
    def check_sales(self,price, list_resa, list_resa_scnd):
        # print(self.clients_list)
        
        #les statistiques de ventes 
        temp_envi = temp_real = temp_aban = temp_inst = temp_repou = temp_obli = temp_comp = 0
        
        #list des index des acheteurs ayant conclu un acht
        list_sales=[]
        
        #on parcours la liste des clients
        for i in range(len(self.clients_list)):
            
            #cas normal ou l'écheance est différ
            if self.clients_list[i].echeance >= 1  and list_resa[self.clients_list[i].echeance] < 30 :
            
                    #cas ou le prix est compris dans la cible
                    if (price >= self.clients_list[i].prix_min and price <= self.clients_list[i].prix_max):
                        
                        #l'achat est envisagé
                        temp_envi+=1
                        proba_temp = random.random()
                        
                        #si le wtp > à la proba
                        if proba_temp <= self.clients_list[i].will_to_pay :
                            
                            #Achat Réalisé
                            temp_real+=1
                            #l'acheteur quitte le marché
                            list_sales.append(i)
                            #on enregistre la résa
                            list_resa[self.clients_list[i].echeance] += 1
                            list_resa_scnd[self.clients_list[i].echeance] += 1
                        
                        else:
                            #Achat Abandonnée
                            temp_aban+=1
                    
                    #cas ou cest moins chère que le prix min
                    elif (price <= self.clients_list[i].prix_min):
                        #Instant achat
                        temp_inst+=1
                        list_sales.append(i)
                        #on enregistre la résa
                        list_resa[self.clients_list[i].echeance] += 1
                        list_resa_scnd[self.clients_list[i].echeance] += 1
                    
                    #sinon
                    else:
                        #Achat repoussé
                        temp_repou+=1
                        
            #cas normal ou l'écheance est différ mais plus de place
            if self.clients_list[i].echeance >= 1  and list_resa[self.clients_list[i].echeance] == 30 :
                temp_comp+=1
                
            
            #cas ou il reste de la place pour le mois qui vient
            if self.clients_list[i].echeance == 0 :
                
                #si il reste des places
                if list_resa[self.clients_list[i].echeance] < 30 :
                    
                    #instant buy
                    temp_inst+=1
                    #l'acheteur quitte le marché
                    list_sales.append(i)
                    #on enregistre la résa
                    list_resa[self.clients_list[i].echeance] += 1
                    list_resa_scnd[self.clients_list[i].echeance] += 1
                else :
                    #sinon repoussé 
                    temp_repou+=1
                    #l'acheteur quitte le marché bredouille
                    list_sales.append(i)
                    
            #on diminue l'écheance
            self.clients_list[i].echeance -= 1
        #on retourne le tout
        return list_sales, temp_envi, temp_real, temp_aban, temp_inst, temp_repou, list_resa, list_resa_scnd
       
    #on update la liste des clients
    def update_client(self,prix_min,prix_max,list_to_del,wtp,rate_to_assure,max_time,resting_time):
        #en supprimant ceux ayant achete ou sorti
        self.del_client(list_to_del)
        
        #on assure un nombre "rate to assure" de wtp élevé
        aleat = int((1-rate_to_assure)*(len(list_to_del)))
        
        #on fait les WTP random
        for _ in range(aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            if max_time-resting_time-1 <= 0 :
                temp_ech = 0
            else:
                temp_ech = random.randint(0,max_time-resting_time-1)
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp,temp_ech))
        
        #les wtp forcés
        for _ in range(len(list_to_del) -aleat):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            if max_time-resting_time-1 <= 0 :
                temp_ech = 0
            else:
                temp_ech = random.randint(0,max_time-resting_time-1)
            self.clients_list.append(Naiv_client(temp_min,temp_max,wtp,temp_ech))

            
        