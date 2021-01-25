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
    
    def __init__(self, prix_min, prix_max, nb_clients, taux_naiv, wtp_to_assure, rate_to_assure, df_price, df_mean):        
        data=[0,0,0,0,0,0,0,0]
        
        self.prix_min = prix_min
        self.prix_max = prix_max
        
        nb_naiv = int(nb_clients * taux_naiv)
        nb_strat = nb_clients - nb_naiv
        
        self.df_price = df_price
        self.df_mean = df_mean
        
        self.wtp_to_assure = wtp_to_assure
        
        self.naiv_clients = cnt.list_naiv_clients(prix_min,prix_max,nb_naiv,wtp_to_assure,rate_to_assure)
        self.strat_clients = nsc.list_smart_clients (prix_min,prix_max,nb_strat,self.df_price)
        
        self.list_resa_naiv = [0]* self.df_mean.shape[0]
        self.list_resa_naiv_final = []
        self.list_resa_strat = [0]* self.df_mean.shape[0]
        self.list_resa_strat_final = [] 
        
        self.list_resa_glob = [0]* self.df_mean.shape[0]
        self.list_resa_glob_final = [] 
        
        self.df_naiv_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_naiv_sales.loc[0]=data
        
        self.df_strat_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_strat_sales.loc[0]= data
        
        self.df_glob_sales = pd.DataFrame(columns= ["Ite","Prix","Achat_Envi","Achat_Real","Achat_Aban","Achat_Inst","Achat_Repou","Achat_Tot"])
        self.df_glob_sales.loc[0]= data

        
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
    
    def check_sales_v2(self, price, echeance, price_trace):
        
        self.list_del_naiv, temp_envi, temp_real, temp_aban, temp_inst, temp_repou, self.list_resa_glob, self.list_resa_naiv = self.naiv_clients.check_sales(price, self.list_resa_glob, self.list_resa_naiv)

        self.list_del_strat ,buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed,self.list_resa_glob, self.list_resa_strat = self.strat_clients.check_sales(price, echeance, price_trace, self.list_resa_glob, self.list_resa_strat )
        
        nb_row_naiv=self.df_naiv_sales.shape[0]
        self.df_naiv_sales.loc[nb_row_naiv]=[nb_row_naiv,price,temp_envi,temp_real,temp_aban,temp_inst,temp_repou,temp_inst+temp_real]
        
        nb_row_strat=self.df_strat_sales.shape[0]
        self.df_strat_sales.loc[nb_row_strat]=[nb_row_strat, price, buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, instant_buy + buy_done]
        
        nb_row_glob=self.df_glob_sales.shape[0]
        self.df_glob_sales.loc[nb_row_glob]=[nb_row_glob, price,temp_envi+buy_considered, temp_real+buy_done, temp_aban+buy_dropped, temp_inst+instant_buy, temp_repou+buy_postponed, temp_inst + temp_real + instant_buy + buy_done]
        
        self.list_resa_naiv_final.append(self.list_resa_naiv[0])
        self.list_resa_strat_final.append(self.list_resa_strat[0])
        self.list_resa_glob_final.append(self.list_resa_glob[0])
        
        del self.list_resa_naiv[0]
        del self.list_resa_strat[0]
        del self.list_resa_glob[0]
        
        achat_tot_naiv=temp_inst+temp_real
        achat_tot_strat=instant_buy + buy_done
        
        return achat_tot_naiv, achat_tot_strat
    
    def updates(self,price,p_trace,ite):
        
        avail_rate=self.df_mean.loc[self.df_mean.index[ite],"mean_availability_30"] / 30 
        final_rate=avail_rate+0.05
        
        # self.list_del_naiv, self.list_resa_naiv, self.list_del_strat, self.list_resa_strat, achat_naiv, achat_strat = self.check_sales(price,self.naiv_clients,self.list_resa_naiv,self.strat_clients,self.list_resa_strat,(self.df_mean.shape[0]-1),p_trace)
        
        achat_naiv, achat_strat = self.check_sales_v2( price, (self.df_mean.shape[0]-1), p_trace)
        
        self.naiv_clients.update_client(self.prix_min, self.prix_min, self.list_del_naiv,self.wtp_to_assure,final_rate,self.df_mean.shape[0]-1,ite)
        self.strat_clients.update_client(self.prix_min, self.prix_max, self.list_del_strat,self.df_mean.shape[0]-1,ite)
        
        return achat_naiv, achat_strat
        


def test():
    
    ###################################
    
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    print(df_final.columns)
    
    nombre_de_mois_test = 12

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-nombre_de_mois_test:df_final.shape[0]-1]
    
    
    #on init le DQN
    dqn=mdqn.DQN("airbnb_data.csv",0.95,0.015,0.83)
    return_trace, p_trace =dqn.dqn_training(80)
    
    dqn.plot_result(return_trace, p_trace)
    
    #on créé le marché
    market=Market(100,200,30,0.8,0.85,0.15,df_start,df_final)
    
    #TEEEEEEEEST
    
    ########
    state = dqn.env_initial_test_state(150,0,1)#init price 
    print("C",state)
    reward_trace = []
    p_trace = [state[0,0]]
    booked=[state[0,1]]
    #############
    
    for k in range(df_final.shape[0]):    
        
        #arrivée d'un unique prix 
        
        ########## Remplacer par un aleat compris entre 50 et 250
        p, state= dqn.dqn_interaction(state)

        p_trace.append(p)
        
        #########
        
        print(p)
        
        achat_naiv, achat_strat = market.updates(p,p_trace,k)
        
        #a corriger
        state[0,1]=achat_strat + achat_naiv
        reward=dqn.profit_t_d(state[0,0],state[0,1])
        reward_trace.append(reward)
           
        #retour de la demande 
    
    print(df_final)
    print(market.list_resa_naiv_final)    
    print(market.df_naiv_sales)
    
    print(market.list_resa_strat_final)    
    print(market.df_strat_sales)
    
    print(market.list_resa_glob_final)    
    print(market.df_glob_sales)
    
    print(reward_trace)

    
    
     
       
test()        
        