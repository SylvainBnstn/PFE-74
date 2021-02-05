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

#fonction de test global avec le nombre d'épisode désiré
#le learning rate 
#le nombre de subdivision de DQN
#et le batch size
def test(nb_episode, learn_rate,nb_part,batch_size):
    
    ###################################
    #chargement initiale des données qui serviront pour les clients
    df_start=ap.load_data("airbnb_data.csv")
    df_start=df_start.loc[(df_start["room_type"]=="Entire home/apt") & (df_start["price"]>=100) & (df_start["price"]<=200)]
    df_final=da.review_data(df_start,"new-york-city")
    
    #choix du nombre de mois de test
    nombre_de_mois_test = 12
    
    #on sélectionne alors la bonne partie de la df (qui servira pour appliquer la demande observé aux client naifs)
    df_final=df_final.reset_index(drop=True)
    df_final=df_final.loc[df_final.shape[0]-nombre_de_mois_test:df_final.shape[0]-1]
    
    #calcul des subdivision de DQN
    number_of_part= nb_part
    steps=100/number_of_part
    
    #initialisation de la df de résultat
    df_result= pd.DataFrame(columns= ["Ite","Prix Moyen DQN","Prix_Min","Prix_Max","Liste_Prix","Nb_V_Naif","Nb_V_Strat","CA_Tot","Rev_Tot","Rev_Naif","Rev_Strat","CA_tampon","Prix_tampon", 'Rev_Naif_tampon', 'Rev_Strat_tampon',"CA_Market_DQN","CA_Market_Temoin"])
    
    
    #pour chauqe partie
    for n in range(number_of_part):
        print("Part ",n) 
        #on init le DQN
        dqn=mdqn.DQN("Data_Model_2.csv",0.9,learn_rate,0.83,(n*steps)/100,((n+1)*steps)/100,batch_size)
        #on l'entraine
        return_trace, p_trace = dqn.dqn_training(nb_episode)
        #plot de surveillance
        dqn.plot_result(return_trace, p_trace)
        #liste qui va retenir les séquence de prix du DQN
        list_p_trace=[]
        
        #boucle pour peut-etre multiplier les test avec un entrainement
        for j in range(1):
            
            #enregistrement temporaire des données
            df_temp= pd.DataFrame(columns= ["Ite","Prix Moyen DQN","Prix_Min","Prix_Max","Liste_Prix","Nb_V_Naif","Nb_V_Strat","CA_Tot","Rev_Tot","Rev_Naif","Rev_Strat","CA_tampon","Prix_tampon", 'Rev_Naif_tampon', 'Rev_Strat_tampon',"CA_Market_DQN","CA_Market_Temoin"])
            
            # print("Ite",j)
        
            #pour chaque proportions comprise dans la part
            for i in range(int(n*steps),int((n+1)*steps)):
            
                #on créé le marché
                market=mkt.Market(100,200,30,1-(i/100),0.85,0.15,df_start,df_final)
                #on créé un marché temoin
                market_temoin = copy.deepcopy(market)
                
                #etat init
                state = dqn.env_initial_test_state(150,0,1)#init price 
                
                #différent enregistrement
                reward_trace = []
                p_trace = [state[0,0]]
                list_achat_naiv=[]
                list_achat_strat=[]
                vente_tot=[]
                
                #on parcours le nombre de mois
                for k in range(df_final.shape[0]):    
                    
                    #arrivée d'un unique prix 
                    p, state= dqn.dqn_interaction(state)
                    #on enregistre le prix
                    p_trace.append(p)

                    #on update le marché                    
                    achat_naiv, achat_strat = market.updates(p,p_trace,k)
                    
                    #on enregistre
                    list_achat_naiv.append(achat_naiv)
                    list_achat_strat.append(achat_strat)
                    vente_tot.append(achat_naiv+achat_strat)
                    
                    #on modifie le state
                    state[0,1]=achat_strat + achat_naiv
                    state[0,2]=k+1
                    reward=dqn.profit_t_d(state[0,0],state[0,1])
                    reward_trace.append(reward)
                
                #on ajoute la séquence de prix dans son enregistreur
                list_p_trace.append(p_trace)
                
                #on set le price temoin init
                price_temoin = 150
                #2on l'enregistre
                p_trace_temoin = [price_temoin]
            
                #on créé les support d'enregistrement du témoin 
                list_achat_naiv_temoin=[]
                list_achat_strat_temoin=[]
                vente_tot_temoin=[]
                
                #on réalise le marché temoin
                for k in range(df_final.shape[0]):    
                    
                    #on set le prix aléatoire et on l'enregistre
                    price_temoin =rd.randrange(130,170)
                    p_trace_temoin.append(price_temoin)
                    
                    #on update et on enregistre
                    achat_naiv_temoin, achat_strat_temoin = market_temoin.updates(price_temoin,p_trace_temoin,k)
                    list_achat_naiv_temoin.append(achat_naiv_temoin)
                    list_achat_strat_temoin.append(achat_strat_temoin)
                    vente_tot_temoin.append(achat_naiv_temoin+achat_strat_temoin)
                    
                #on enregistre les données de cette propostion dans la dataframe
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
                                    sum([a*b for a,b in zip( p_trace_temoin, vente_tot_temoin)]),
                                    price_temoin,
                                    sum([a*b for a,b in zip( p_trace_temoin, list_achat_naiv_temoin)]),
                                    sum([a*b for a,b in zip( p_trace_temoin, list_achat_strat_temoin)]),
                                    sum([a*b for a,b in zip( p_trace, df_final["mean_booked_30"].tolist())]),
                                    sum([a*b for a,b in zip( p_trace_temoin, df_final["mean_booked_30"].tolist())])]
        
        #on affiche les séquence de prix
        plot_ptrace(list_p_trace)
        #on ajoute dans la df de résultat
        df_result = pd.concat([df_result,df_temp])
    
    print(df_result)
    #on sauvegarde le tout dans un excel
    df_result.to_excel("Evol_Strat.xlsx",index=False)

#fonction de plot des résultats principaux
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
    
#fonction de plot superposé d'une séquence de prix contenu dans une liste
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
    

#plot de la différence de CA entre DQn et l'alea
def plot_diff(df):
    plt.figure(figsize=(14, 8))
    plt.plot("Ite","Diff_CA","b-",data = df)
    plt.ylabel("Différence de CA ($)")
    plt.xlabel("Parts de Strategiques")
    plt.title("Différence CA tot DQN/Aleat : Moyenne = " + str(df["Diff_CA"].mean()) + "$")
    plt.legend()
    plt.grid()
    
def moyenne_plot():
    #on charge les données enregistrées
    df = pd.read_excel("Evol_Strat.xlsx")
    #on les agrège en moyenne
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
    #on sauvegarde
    df.to_excel("Evol_Strat_Mean.xlsx",index=False)
    #on fait les plots
    revenue_plot(df)
    plot_diff(df)
    return df["Diff_CA"].mean()

# à décommenter pour le test 
# test(500,0.01,5,128)        
# mean_perf = moyenne_plot()
# print(mean_perf)
