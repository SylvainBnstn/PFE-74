# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:11:19 2021

@author: Sylvain
"""


import Naiv_Clients_v2 as cnt
import data_analysis as da
import airbnb_processing as ap
import Model_DQN_VF as mdqn
import naiv_strategic_clients as nsc
import Market_V1 as mkt

import numpy as np

import matplotlib.pyplot as plt

import statistics as st

import copy

import random as rd
import pandas as pd


def find_cst_p_better_than_dqn(vente_tot, CA_dqn):
    list_CA_p_cst = []
    for cst_price in range(70,230):
        CA_p_cst = sum([cst_price*x for x in vente_tot])
        list_CA_p_cst.append(CA_p_cst)
        
        if CA_p_cst > CA_dqn:
            return cst_price
            #print("CA avec prix fixe > CA avec prix variable (dqn), pour le prix cst : ", cst_price)


def test(nb_episode, learn_rate,nb_part,batch_size):
    
    ###################################
    
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    nombre_de_mois_test = 12

    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-nombre_de_mois_test:df_final.shape[0]-1]
    
    number_of_part= nb_part
    steps=100/number_of_part
    
    
    
    df_result= pd.DataFrame(columns= ["Ite","Prix Moyen DQN","Prix_Min","Prix_Max","Liste_Prix","Nb_V_Naif","Nb_V_Strat","CA_Tot","Rev_Tot","Rev_Naif","Rev_Strat","CA_tampon","Prix_tampon", 'Rev_Naif_tampon', 'Rev_Strat_tampon',"CA_Market_DQN","CA_Market_Temoin"])
    
   
        
    for n in range(number_of_part):
        print("Part ",n) 
        #on init le DQN
        dqn=mdqn.DQN("Data_Model_2.csv",0.9,learn_rate,0.83,(n*steps)/100,((n+1)*steps)/100,batch_size)
        return_trace, p_trace = dqn.dqn_training(nb_episode)

        dqn.plot_result(return_trace, p_trace)
        
        list_p_trace=[]
        
        for j in range(1):
    
            df_temp= pd.DataFrame(columns= ["Ite","Prix Moyen DQN","Prix_Min","Prix_Max","Liste_Prix","Nb_V_Naif","Nb_V_Strat","CA_Tot","Rev_Tot","Rev_Naif","Rev_Strat","CA_tampon","Prix_tampon", 'Rev_Naif_tampon', 'Rev_Strat_tampon',"CA_Market_DQN","CA_Market_Temoin"])
            
            print("Ite",j)
        
            for i in range(int(n*steps),int((n+1)*steps)):
            
                #on créé le marché
                market=mkt.Market(100,200,30,1-(i/100),0.85,0.15,df_start,df_final)
                market_tampon = copy.deepcopy(market)
                
                #TEEEEEEEEST
                
                ########
                state = dqn.env_initial_test_state(150,0,1)#init price 
    
                reward_trace = []
                p_trace = [state[0,0]]
                booked=[state[0,1]]
                #############
                
                list_achat_naiv=[]
                list_achat_strat=[]
                vente_tot=[]
                
                for k in range(df_final.shape[0]):    
                    
                    #arrivée d'un unique prix 
                    
                    ########## Remplacer par un aleat compris entre 50 et 250
                    p, state= dqn.dqn_interaction(state)
            
                    p_trace.append(p)
                    
                    #########
                    
                    # print(p)
                    
                    achat_naiv, achat_strat = market.updates(p,p_trace,k)
                    
                    list_achat_naiv.append(achat_naiv)
                    list_achat_strat.append(achat_strat)
                    vente_tot.append(achat_naiv+achat_strat)
                    
                    #a corriger
                    state[0,1]=achat_strat + achat_naiv
                    state[0,2]=k+1
                    reward=dqn.profit_t_d(state[0,0],state[0,1])
                    reward_trace.append(reward)
                       
                    #retour de la demande 
                
                # print(df_final)
                # print(market.list_resa_naiv_final)    
                # print(market.df_naiv_sales)
                
                # print(market.list_resa_strat_final)    
                # print(market.df_strat_sales)
                
                # print(market.list_resa_glob_final)    
                # print(market.df_glob_sales)
                
                # print(reward_trace)
                list_p_trace.append(p_trace)
                # print(list_achat_naiv)
                # print(list_achat_strat)
                # print(vente_tot)
                
                # print("CA: ", sum([a*b for a,b in zip( p_trace, vente_tot)]))
                # print("Revenus générés", sum(reward_trace))
                
                # print("Revenus générés par les strats",sum([a*b for a,b in zip( p_trace, list_achat_strat)]))
                # print("Revenus générés par les naif",sum([a*b for a,b in zip( p_trace, list_achat_naiv)]))
                
                # print("CA avec prix fixe",sum([150*x for x in vente_tot])) 
                
                # price_tampon = find_cst_p_better_than_dqn(vente_tot, sum([a*b for a,b in zip( p_trace, vente_tot)]))
                
                price_tampon = int(sum(p_trace)/len(p_trace))
                
                p_trace_tampon = [price_tampon]
            
                
                list_achat_naiv_tampon=[]
                list_achat_strat_tampon=[]
                vente_tot_tampon=[]
                
                for k in range(df_final.shape[0]):    
                    
                    price_tampon =rd.randrange(130,170)
                    p_trace_tampon.append(price_tampon)
                    
                    #########
                    
                    # print(p)
                    
                    achat_naiv_tampon, achat_strat_tampon = market_tampon.updates(price_tampon,p_trace_tampon,k)
                    list_achat_naiv_tampon.append(achat_naiv_tampon)
                    list_achat_strat_tampon.append(achat_strat_tampon)
                    
                    vente_tot_tampon.append(achat_naiv_tampon+achat_strat_tampon)
                    
                
                df_temp.loc[i] = [i,
                                    int(sum(p_trace)/len(p_trace)),
                                    min(p_trace),
                                    max(p_trace),
                                    p_trace,
                                    sum(list_achat_naiv),
                                    sum(list_achat_strat),
                                    sum([a*b for a,b in zip( p_trace, vente_tot)]),
                                    sum(reward_trace),
                                    sum([a*b for a,b in zip( p_trace, list_achat_naiv)]),
                                    sum([a*b for a,b in zip( p_trace, list_achat_strat)]),
                                    sum([a*b for a,b in zip( p_trace_tampon, vente_tot_tampon)]),
                                    price_tampon,
                                    sum([a*b for a,b in zip( p_trace_tampon, list_achat_naiv_tampon)]),
                                    sum([a*b for a,b in zip( p_trace_tampon, list_achat_strat_tampon)]),
                                    sum([a*b for a,b in zip( p_trace, df_final["mean_booked_30"].tolist())]),
                                    sum([a*b for a,b in zip( p_trace_tampon, df_final["mean_booked_30"].tolist())])]
            
        plot_ptrace(list_p_trace)
        df_result = pd.concat([df_result,df_temp])
    
    print(df_result)
    df_result.to_excel("Evol_Strat.xlsx",index=False)

def revenue_plot(df):
    plt.figure(figsize=(14, 8))
    Stat_columns = ['CA_Tot', 'Rev_Naif', 'Rev_Strat','CA_tampon', 'Rev_Naif_tampon', 'Rev_Strat_tampon']
    #parcours
    color_list=["r","g","b"]
    
    label_list=['CA Total Pricer DQN', 'CA Naif Pricer DQN', 'CA Strat Pricer DQN']
    label_list2 = ['CA Total Prix Alea', 'CA Naif Prix Alea', 'CA Strat Prix Alea']
    for i in range(0,3):
        #plot
        plt.plot("Ite",Stat_columns[i],str(color_list[i]+"-"), data = df, label = label_list[i])
        
    for i in range(3,6):
        #plot
        plt.plot("Ite",Stat_columns[i],str(color_list[i-3]+":"), data = df, label = label_list2[i-3])

    #label et legende
    plt.xlabel("Parts de Client Strategiques dans le marché (%)")
    plt.ylabel("CA ($)")
    plt.title("Comparaison CA DQN/Aleat avec la demande simulee ")
    plt.legend()
    plt.grid()
    
def plot_ptrace(list_toplot):
    plt.figure(figsize=(14, 8))
    mois = np.arange(0, 13, 1)

    for i in range(len(list_toplot)):
        #plot
        plt.plot(mois,list_toplot[i], color="k" ,alpha = 0.1)
    #label et legende
    plt.xlabel("Mois")
    plt.ylabel("Prix (USD)")
    plt.legend()
    plt.grid()
    
def revenue_plot_fixe(df):
    plt.figure(figsize=(12, 8))
    Stat_columns = ["CA_Market_DQN","CA_Market_Temoin"]
    #parcours
    color_list=["r","g","b"]
    label_list = ["CA_Market_DQN","CA_Market_fixe"]
    for i in range(0,2):
        #plot
        plt.plot("Ite",Stat_columns[i],str(color_list[i]+"--"), data = df, label = label_list[i])
        
    #label et legende
    plt.xlabel("Parts de Strategiques")
    plt.ylabel("CA ($)")
    plt.title("Comparaison Prix DQN/Fixe avec la demande observee")
    plt.legend()
    plt.grid()
    
def plot_diff(df):
    plt.figure(figsize=(14, 8))
    plt.plot("Ite","Diff_CA","b-",data = df)
    plt.ylabel("Différence de CA ($)")
    plt.xlabel("Parts de Strategiques")
    plt.title("Différence CA tot DQN/Aleat : Moyenne = " + str(df["Diff_CA"].mean()) + "$")
    plt.legend()
    plt.grid()
    
def revenue_plot2():
    df = pd.read_excel("Evol_Strat.xlsx")
    # df['Rev_Tot'] = df['Rev_Tot'].str.replace(',','.').astype(float)
    df= df.groupby(["Ite"], as_index = False).agg({
        'Prix Moyen DQN': ["mean"], 
        'Prix_Min': ["mean"], 
        'Prix_Max': ["mean"], 
        'Nb_V_Naif': ["mean"],
        'Nb_V_Strat': ["mean"], 
        'CA_Tot': ["mean"], 
        'Rev_Tot': ["mean"], 
        'Rev_Naif': ["mean"], 
        'Rev_Strat': ["mean"],
        'CA_tampon': ["mean"],
        'Prix_tampon': ["mean"],
        'Rev_Naif_tampon': ["mean"], 
        'Rev_Strat_tampon': ["mean"],
        "CA_Market_DQN": ["mean"],
        "CA_Market_Temoin": ["mean"]})
    df.columns = ['Ite', 'Prix Moyen DQN', 'Prix_Min', 'Prix_Max', 'Nb_V_Naif','Nb_V_Strat', 'CA_Tot', 'Rev_Tot', 'Rev_Naif', 'Rev_Strat','CA_tampon','Prix_tampon', 'Rev_Naif_tampon', 'Rev_Strat_tampon',"CA_Market_DQN","CA_Market_Temoin"]
    df["Diff_CA"] = df["CA_Tot"] - df["CA_tampon"]

    df.to_excel("Evol_Strat_Mean.xlsx",index=False)
    print(df.tail(10))
    revenue_plot(df)
    # revenue_plot_fixe(df)
    plot_diff(df)
    return df["Diff_CA"].mean()


# test(500,0.01,5,128)        
mean_perf = revenue_plot2()
print(mean_perf)
