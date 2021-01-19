# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:30:18 2020

@author: Sylvain
"""


import random as rnd
import data_analysis as da
import airbnb_processing as ap

'''

    WILLINGNESS TO PAY: must be a function, taking the price as a parameter
                        the output is a probability of buying the current price
    Wtp is a decreasing function, as the bracket [min, max] represents the acceptable prices

    If wtp>x we buy, if not we do not (x needs to be adjusted)

    Use moving average do determine min and max, and then adjust the client's min and max accordingly.

'''


class Smart_client:

    def __init__(self, prix_min, prix_max, will_to_pay, echeance):
        self.prix_max = prix_max
        self.prix_min = prix_min
        self.will_to_pay = will_to_pay
        self.echeance = echeance

    def __str__(self):
        r = "Minimum Purchase Price: " + str(self.prix_min) + "\n"
        r += "Maximum Purchase Price: " + str(self.prix_max) + "\n"
        r += "Willing To Pay: " + str(self.will_to_pay) + "\n"
        r += "Echeance: " + str(self.echeance) + "\n"
        return r

    def __repr__(self):
        r = (self.prix_min, self.prix_max, self.will_to_pay)
        return str(r)


class list_smart_clients:

    '''
        Smart Clients are defined by:
            - a well-thought minimum and maximum purchase price
            - a willing to pay price which depends on:
                - client's income
                - period of the year impacting the market (e.g. week-ends or holidays for plane tickets)
                -

        Disposable Information:
            - Price
            - Availability
            - Number of available type of rent (whole property, part, shared...)

    '''

    def __init__(self, prix_min, prix_max, nb_client, df):
        self.df = df
        self.nb_client =nb_client
        self.clients_list = []
        for _ in range(nb_client):
            temp_min = rnd.uniform(
                prix_min, prix_min + (prix_max - prix_min) / 2)
            temp_max = rnd.uniform(
                prix_min + (prix_max - prix_min) / 2, prix_max)
            temp_wtp = rnd.random()  # Willing To Pay
            self.clients_list.append(
                Smart_client(
                    temp_min,
                    temp_max,
                    temp_wtp,
                    echeance=rnd.randint(0,11)))
            # DATE RANDOM

    def __str__(self):
        return str(self.clients_list)

    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]

    # Fonction d'actualisation du paramètre WTP ( Willingness To Pay ) en
    # fonction du prix et de l'échéance
    def wtp_actualisation(self, price, max_echeance, current_client):

        # TRES IMPORTANT AU NIVEAU DU POIDS : Le total des poids des paramètres price et echeance doivent être égal à 2
        # Poids du paramètre price dans le wtp
        poids_price = 1
        # Poids du paramètre echeance dans le wtp
        poids_echeance = 1
        # Petit message d'alerte des familles
        if(poids_price + poids_echeance != 2):
            print('Attention : Le total des poids des paramètres price et echeance doivent être égal à 2, vos calculs peuvent être faussés')

        # Pas des paramètres price et echeance dans les boucles pour le calcul
        # du wtp
        range_price = (self.clients_list[current_client].prix_max-self.clients_list[current_client].prix_min)
        pas_price = round(0.5/range_price, 3)       #0.005
        pas_echeance = round(0.5/max_echeance, 3) #0.045

        # Boucle de détermination wtp du paramètre price
        for i in range(range_price):
            if(price == (self.clients_list[current_client].prix_max - i)):
                wtp_price = pas_price * i
        # Limite de détermination wtp du paramètre price
        if(price <= self.clients_list[current_client].prix_min):
            wtp_price = 0.5

        # Boucle de détermination wtp du paramètre echeance
        for i in range(max_echeance):
            if(self.clients_list[current_client].echeance == (max_echeance - i)):
                wtp_echeance = pas_echeance * i
        # Limite de détermination wtp du paramètre echeance
        if(echeance == 0):
            wtp_echeance = 0.5

        # Calcul final du wtp en fonction du prix et de l'échéance avec leur
        # poids respectif
        wtp = wtp_price * poids_price + wtp_echeance * poids_echeance

        # Retourne la valeur du wtp actualisée en fonction du prix et de
        # l'échéance
        return wtp

    # Fonction de détermination si le client stratégique achète ou pas en fonction des prix qu'il rencontre au fur et à mesure au moment présent sans connaissance des prix futurs
    # Si le dernier prix est inférieur aux deux d'avant
    def strategic_price(self, price_trace, current_client):
        length = len(price_trace)
        poids = 0
        buy = False
        # Applique l'achat si deux baisses consécutives de prix
        for i in range(2):
            if (price_trace[length-1]<price_trace[length-(i+1)-1]):
                poids = poids + 1
        if (poids >= 2):
            if (price_trace[length-1]<=self.clients_list[current_client].prix_max or price_trace[length-1]>=self.clients_list[current_client].prix_min):
                buy = True
        return buy

    #Deux baisses consécutives uniquement
    def strategic_price_v2(self, price_trace, current_client):
        length = len(price_trace)
        poids = 0
        buy = False
        # Applique l'achat si deux baisses consécutives de prix
        for i in range(2):
            if (price_trace[length-i-1]<price_trace[length-(i+1)-1]):
                poids = poids + 1
        if (poids >= 2):
            if (price_trace[length-1]<=self.clients_list[current_client].prix_max or price_trace[length-1]>=self.clients_list[current_client].prix_min):
                buy = True
        return buy

    def get_min_x_percent(self, x):
        return self.df['price'].head(int(x * self.df.size)).mean()

    def get_max_x_percent(self, x):
        return self.df['price'].tail(int(x * self.df.size)).mean()

    def update_min_max(self, price, max_echeance):
        # UPDATE ECHEANCE
        #0.2 pour 20% des max et des min
        new_min = self.get_min_x_percent(0.2)
        new_max = self.get_max_x_percent(0.2)
        # Boucle de parcours pour l'actualisation
        for i in range(len(self.clients_list)):
            wtp = self.wtp_actualisation(price, max_echeance,i)
            self.clients_list[i].prix_min = new_min
            self.clients_list[i].prix_max = new_max
            self.clients_list[i].will_to_pay = wtp

        # Sales verification and sold items deletion

    def check_sales(self, price, max_echeance, price_trace, list_resa):
        
        ####" Mettre update
    
        buy_considered = buy_done = buy_dropped = instant_buy = buy_postponed = 0

        # list des index des acheteurs ayant conclu un acht
        list_sales = []

        for current_client in range(len(self.clients_list)):
            # cas ou le prix est compris dans la cible
            if (price >= self.clients_list[current_client].prix_min and price <=
                    self.clients_list[current_client].prix_max):

                # on montre que l'achat est envisagé
                #temp_str =str("Achat envisagé -> ")
                buy_considered += 1

                actual_willingness = rnd.random()/3 #From 0 to 1/3
                # Adapt value according to the time left
                poids_price, poids_echeance, poids_rnd = 0.5, 2, 0.5
                
                
                range_price = int(self.clients_list[current_client].prix_max-self.clients_list[current_client].prix_min)
                
                pas_price = round(1/3/range_price, 3)      
                pas_echeance = round(1/3/max_echeance, 3)
                for i in range(range_price):
                    if(price ==int (self.clients_list[current_client].prix_max - i)):
                        wtp_price = pas_price * i
                if(price <= self.clients_list[current_client].prix_min):
                    wtp_price = 1/3 #pas_price * 100
                for i in range(max_echeance):
                    if( self.clients_list[current_client].echeance == max_echeance - i):
                        wtp_echeance = pas_echeance * i
                if(self.clients_list[current_client].echeance == 0):
                    wtp_echeance = 1/3

                wtp = wtp_price * poids_price + wtp_echeance * poids_echeance + actual_willingness * poids_rnd

                # a revoir pour mise en place de WTP exacte
                if (1-wtp) <= self.clients_list[current_client].will_to_pay:
                    
                    condition_prix = self.strategic_price(price_trace, current_client)
                    #condition_prix = strategic_price_v2(p_trace)
                    if( condition_prix == True):
                        #temp_str+=str("Achat Réalisé")
                        buy_done += 1
                        
                        list_resa[self.clients_list[current_client].echeance-1] += 1
                        
                        # l'acheteur quitte le marché
                        list_sales.append(current_client)

                else:
                    #temp_str+=str("Achat Abandonné")
                    buy_dropped += 1

                # print(temp_str)

            # Lower price than minimum
            elif (price <= self.clients_list[current_client].prix_min):
                #print("Instant achat")
                instant_buy += 1
                list_resa[self.clients_list[current_client].echeance-1] += 1
                list_sales.append(current_client)

            # Higher price than maximum
            else:
                #print("Achat repoussé")
                buy_postponed += 1
                
            self.clients_list[current_client].echeance -= 1

        
        return list_sales ,buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, list_resa
    
    def update_client(self,prix_min,prix_max,list_to_del,resting_time):
        self.del_client(list_to_del)
        
        #on fait les WTP random
        for _ in range(self.nb_client - len (list_to_del)):
            temp_min = self.get_min_x_percent(0.2)
            temp_max = self.get_max_x_percent(0.2)
            temp_wtp = rnd.random()
            temp_ech = rnd.randint(0,11-resting_time)
            self.clients_list.append(Smart_client(temp_min,temp_max,temp_wtp,temp_ech))
            



