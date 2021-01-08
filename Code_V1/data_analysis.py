# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 15:28:54 2021

@author: Sylvain
"""


import airbnb_processing as ap
import pandas as pd
import matplotlib.pyplot as plt
    
#fonction de multiplot par type de chambre
def multiplot_by_roomtype(df,y_data_name,y_label):
    plt.figure(figsize=(20, 15))
    #liste des types de chambres dont on dispose
    room_type_list = list(df["room_type"].unique())
    #parcours
    for i in range(len(room_type_list)):
        #plot
        plt.plot("date",y_data_name, data = df[df["room_type"] == room_type_list[i]], label = room_type_list[i])
    #label et legende
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()

#fonction d'ajustement des données
def review_data(df_all,city):
    #première df avec les moyennes de prix et de dispo
    result_1 = df_all[df_all["city"] == city].groupby(["date","room_type"], as_index = False).agg({"price" : ["mean"], "availability_30" : ["mean"]})
    result_1.columns = ["date","room_type","mean_price", "mean_availability_30"]
    #nombre de room par types disponible à l'instant 
    room_count =  df_all[df_all["city"] == city].groupby(["date","room_type"], as_index = False).size()
    room_count = room_count.to_frame()
    room_count.columns = ["size"]
    #fusion de room
    df_result = pd.merge(room_count, result_1, on=["date","room_type"])
    df_result["date"]=pd.to_datetime(df_result["date"])
    return df_result

df_start=ap.load_data("airbnb_data.csv")
df_final=review_data(df_start,"new-york-city")

multiplot_by_roomtype(df_final,"mean_price","Prix Moyen")
multiplot_by_roomtype(df_final,"mean_availability_30","Dispo Moyen")
multiplot_by_roomtype(df_final,"size","Nombre de dispo")