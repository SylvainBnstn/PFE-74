 # -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Naiv_Clients as cnt
import data_analysis as da
import airbnb_processing as ap
import strategic_clients as nsc

import matplotlib.pyplot as plt

import statistics as st

import copy

import random as rd
import pandas as pd

# pour rappel 
# Class list_naiv_clients est l'objet qui contient tous les clients
# on l'appel donc naiv_clients
# Cette objet contient la list des objets clients qui contiennent eux-mêmes leurs prix min max et WTP
class Market:
    
    def __init__(self, prix_min, prix_max, nb_clients, taux_naiv, wtp_to_assure, rate_to_assure, df_price, df_mean):        
        data=[0,0,0,0,0,0,0,0]
        
        #on définit les prix min et max des clients
        self.prix_min = prix_min
        self.prix_max = prix_max
        
        #le nombre de naif et de strat
        nb_naiv = int(nb_clients * taux_naiv)
        nb_strat = nb_clients - nb_naiv
        
        #les dataframe
        self.df_price = df_price
        self.df_mean = df_mean
        
        #le WTP à assurer chez les naifs
        self.wtp_to_assure = wtp_to_assure
        
        #on définit les objet list clients
        self.naiv_clients = cnt.list_naiv_clients(prix_min,prix_max,nb_naiv,wtp_to_assure,rate_to_assure)
        self.strat_clients = nsc.list_strat_clients (prix_min,prix_max,nb_strat,self.df_price)
        
        #les listes qui serviront au réservations par type de clients
        self.list_resa_naiv = [0]* self.df_mean.shape[0]
        self.list_resa_naiv_final = []
        self.list_resa_strat = [0]* self.df_mean.shape[0]
        self.list_resa_strat_final = [] 
        
        #celle pour les résa globales
        self.list_resa_glob = [0]* self.df_mean.shape[0]
        self.list_resa_glob_final = [] 
        
        #les dataframes qui enregistrent les données par types de clients
        self.df_naiv_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_naiv_sales.loc[0]=data
        
        self.df_strat_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_strat_sales.loc[0]= data
        
        #et les totales
        self.df_glob_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_glob_sales.loc[0]= data

    #fonction d'execution des ventes
    def check_sales_v2(self, price, echeance, price_trace):
        
        #on commence par les clients naifs
        self.list_del_naiv, temp_envi, temp_real, temp_aban, temp_inst, temp_repou, self.list_resa_glob, self.list_resa_naiv = self.naiv_clients.check_sales(price, self.list_resa_glob, self.list_resa_naiv)
        #puis les stratégiques
        self.list_del_strat ,buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed,self.list_resa_glob, self.list_resa_strat = self.strat_clients.check_sales(price, echeance, price_trace, self.list_resa_glob, self.list_resa_strat )
        
        #on enregistre le tout dans les DF de la class
        nb_row_naiv=self.df_naiv_sales.shape[0]
        self.df_naiv_sales.loc[nb_row_naiv]=[nb_row_naiv,price,temp_envi,temp_real,temp_aban,temp_inst,temp_repou,temp_inst+temp_real]
        
        nb_row_strat=self.df_strat_sales.shape[0]
        self.df_strat_sales.loc[nb_row_strat]=[nb_row_strat, price, buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, instant_buy + buy_done]
        
        #puis dans celle globales
        nb_row_glob=self.df_glob_sales.shape[0]
        self.df_glob_sales.loc[nb_row_glob]=[nb_row_glob, price,temp_envi+buy_considered, temp_real+buy_done, temp_aban+buy_dropped, temp_inst+instant_buy, temp_repou+buy_postponed, temp_inst + temp_real + instant_buy + buy_done]
        
        #et on l'ajoute dans la final
        self.list_resa_naiv_final.append(self.list_resa_naiv[0])
        self.list_resa_strat_final.append(self.list_resa_strat[0])
        self.list_resa_glob_final.append(self.list_resa_glob[0])
        
        del self.list_resa_naiv[0]
        del self.list_resa_strat[0]
        del self.list_resa_glob[0]
        
        #on retourne ensuite les achats effectués dans le tour pour chaque types de clients.
        achat_tot_naiv=temp_inst+temp_real
        achat_tot_strat=instant_buy + buy_done
        
        return achat_tot_naiv, achat_tot_strat
    
    #fonction d'updates qui gère la tout
    def updates(self,price,p_trace,ite):
        
        #calcule du taux de vente a assurer
        avail_rate=self.df_mean.loc[self.df_mean.index[ite],"mean_booked_30"] / 30 
        final_rate=avail_rate+0.05
        
        #verification des ventes
        achat_naiv, achat_strat = self.check_sales_v2( price, (self.df_mean.shape[0]-1), p_trace)
        
        #update des clients (pour conserver un niveau constant de client dispo)
        self.naiv_clients.update_client(self.prix_min, self.prix_min, self.list_del_naiv,self.wtp_to_assure,final_rate,self.df_mean.shape[0]-1,ite)
        self.strat_clients.update_client(self.prix_min, self.prix_max, self.list_del_strat,self.df_mean.shape[0]-1,ite)
        
        return achat_naiv, achat_strat
        

