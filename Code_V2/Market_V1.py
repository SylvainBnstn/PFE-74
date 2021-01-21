 # -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Naiv_Clients_v2 as cnt
import data_analysis as da
import airbnb_processing as ap
import Model_DQN_VF as mdqn
import naiv_strategic_clients as nsc


import random as rd
import pandas as pd

# pour rappel 
# Class list_naiv_clients est l'objet qui contient tous les clients
# on l'appel donc naiv_clients
# Cette objet contient la list des objets clients qui contiennent eux-mêmes leurs prix min max et WTP

class Market:
    
    def __init__(self, prix_min, prix_max, nb_clients, taux_naiv, wtp_to_assure, rate_to_assure, df):        
        data=[0,0,0,0,0,0,0,0]
        
        nb_naiv = int(nb_clients * taux_naiv)
        nb_strat = nb_clients - nb_naiv
        
        self.naiv_clients = cnt.list_naiv_clients(prix_min,prix_max,nb_naiv,wtp_to_assure,rate_to_assure)
        self.strat_clients = nsc.list_smart_clients (prix_min,prix_max,nb_strat,df)
        
        self.df_naiv_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_naiv_sales.loc[0]=data
        
        self.df_strat_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_strat_sales.loc[0]= data

        
    #fonction d'execution des ventes
    def check_sales(self, price, naiv_clients, list_resa_naiv, strat_clients, list_resa_strat, echeance, price_trace):
        
        list_del_naiv, temp_envi, temp_real, temp_aban, temp_inst, temp_repou, list_resa_naiv = naiv_clients.check_sales(price, list_resa_naiv)
        
        list_del_strat ,buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, list_resa_strat = strat_clients.check_sales(price, echeance, price_trace, list_resa_strat )
        
        nb_row_naiv=self.df_naiv_sales.shape[0]
        data_temp_naiv=[nb_row_naiv,price,temp_envi,temp_real,temp_aban,temp_inst,temp_repou,temp_inst+temp_real]
        achat_tot_naiv=temp_inst+temp_real
        self.df_naiv_sales.loc[nb_row_naiv]=data_temp_naiv
        
        nb_row_strat=self.df_strat_sales.shape[0]
        data_temp_strat=[nb_row_strat, price, buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, instant_buy + buy_done]
        achat_tot_strat=instant_buy + buy_done
        self.df_strat_sales.loc[nb_row_strat]=data_temp_strat
        
        return list_del_naiv, list_resa_naiv, list_del_strat, list_resa_strat , achat_tot_naiv,achat_tot_strat
        


def test():
    
    ###################################
    
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    print(df_final.columns)


    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-12:df_final.shape[0]-1]
    
    
    #définit une list de clients naif avec prix min, prix max et nb clients
    naiv_clients= cnt.list_naiv_clients(100,200,30,0.85,0.15)
    strat_clients = nsc.list_smart_clients (100,200,30,df_start)
    
    #on init le DQN
    dqn=mdqn.DQN()
    dqn.dqn_training(40)
    
    #on créé le marché
    market=Market(100,200,30,0.8,0.85,0.15,df_start)
    
    #TEEEEEEEEST
    
    ########
    state = dqn.env_initial_test_state(150,0)#init price 
    reward_trace = []
    p_trace = [state[0,0]]
    booked=[state[0,1]]
    #############
    
    
    list_resa_naiv = [0]* df_final.shape[0]
    list_resa_naiv_final = []
    list_resa_strat = list_resa_naiv.copy()
    list_resa_strat_final = []    
    
    for k in range(df_final.shape[0]):    
        
        #arrivée d'un unique prix 
        
        ########## Remplacer par un aleat compris entre 50 et 250
        p, state= dqn.dqn_interaction(state)

        p_trace.append(p)
        
        #########
        
        print(p)
        
        avail_rate=df_final.loc[df_final.index[k],"mean_availability_30"] / 30 
        final_rate=avail_rate+0.05
        
        list_del_naiv, list_resa_naiv, list_del_strat, list_resa_strat, achat_naiv, achat_strat = market.check_sales(p,naiv_clients,list_resa_naiv,strat_clients,list_resa_strat,(df_final.shape[0]-1),p_trace)
        naiv_clients.update_client(100, 200, list_del_naiv,0.85,final_rate,k)
        strat_clients.update_client(100, 200, list_del_strat,k)
        
        print (list_resa_naiv)
        print (list_resa_strat)
        
        list_resa_naiv_final.append(list_resa_naiv[0])
        list_resa_strat_final.append(list_resa_strat[0])
        
        state[1,0]=achat_strat
        reward=dqn.profit_t_d(state[0,0],state[1,0])
        reward_trace.append(reward)
        
        del list_resa_naiv[0]
        del list_resa_strat[0]
        
        #retour de la demande 
        
    print(df_final)
    print(list_resa_naiv_final)    
    print(market.df_naiv_sales)
    
    print(list_resa_strat_final)    
    print(market.df_strat_sales)
    
    print(reward_trace)

        

        
        
test()        
        