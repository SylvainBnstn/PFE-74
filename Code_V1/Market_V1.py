 # -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:29:46 2020

@author: Sylvain
"""

import Naiv_Clients_v2 as cnt
import data_analysis as da
import airbnb_processing as ap
import Test_Class as tst


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
    def check_sales(self, price, clients, list_resa):
        
        list_sales, temp_envi, temp_real, temp_aban, temp_inst, temp_repou, list_resa = clients.check_sales_naiv(price, list_resa)
        
        nb_row=self.df_naiv_sales.shape[0]
        data_temp=[nb_row,price,temp_envi,temp_real,temp_aban,temp_inst,temp_repou,temp_inst+temp_real]
        self.df_naiv_sales.loc[nb_row]=data_temp
        return list_sales, list_resa
        


def test():
    
    ###################################
    
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-12:df_final.shape[0]-1]
    
    #définit une list de clients naif avec prix min, prix max et nb clients
    naiv_clients= cnt.list_naiv_clients(100,200,30,0.85,0.15)
    
    #on init le DQN
    dqn=tst.DQN()
    
    #on créé le marché
    market=Market(naiv_clients)
    
    #TEEEEEEEEST
    
    ########
    #state = dqn.env_initial_test_state(150)#init price 
    #reward_trace = []
    #p_trace = [state[0]]
    #############
    
    
    list_resa = [0]* df_final.shape[0]
    list_resa_final = []    
    
    for k in range(df_final.shape[0]):    
        
        #arrivée d'un unique prix 
        
        ########## Remplacer par un aleat compris entre 50 et 250
        #reward,p, state= dqn.dqn_test(state)
        #reward_trace.append(reward)
        p = random.randint(50, 250)
        p_trace.append(p)
        
        #########
        
        print(p)
        
        avail_rate=df_final.loc[df_final.index[k],"mean_availability_30"] / 30 
        final_rate=avail_rate+0.05
        
        list_del, list_resa = market.check_sales(p,naiv_clients,list_resa)
        naiv_clients.update_client(100, 200, list_del,0.85,final_rate,k)
        
        print (list_resa)
        
        list_resa_final.append(list_resa[0])
        del list_resa[0]
        
        #retour de la demande 
        
    print(df_final)
    print(list_resa_final)    
    print(market.df_naiv_sales)
    


        

        
        
test()        
        