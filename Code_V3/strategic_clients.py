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


class strat_client:

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
        # r = (self.prix_min, self.prix_max, self.will_to_pay,self.echeance)
        r = (self.echeance)
        return str(r)


class list_strat_clients:

    '''
        strat Clients are defined by:
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
        self.df = df.sort_values(by=["price"],ascending = True)
        self.nb_client =nb_client
        self.clients_list = []
        for _ in range(nb_client):
            temp_min = self.get_min_x_percent(0.2)
            temp_max = self.get_max_x_percent(0.2)
            temp_wtp = rnd.random()  # Willing To Pay
            self.clients_list.append(
                strat_client(
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

        # TRES IMPORTANT AU NIVEAU DU POIDS : Le total des poids des paramètres price et echeance doivent être égal à 2 ( ici 1 + 1 = 2, mais cela est modulable)
        # Poids du paramètre price dans le wtp
        poids_price = 1
        # Poids du paramètre echeance dans le wtp
        poids_echeance = 1
        # Petit message d'alerte, que l'utilisateur soit en connaissance de causes
        if(poids_price + poids_echeance != 2):
            print('Attention : Le total des poids des paramètres price et echeance doivent être égal à 2, vos calculs peuvent être faussés')

        #Tranche de prix entre le minimum et le maximum du client, pour permettre aux calculs suivants
        range_price = int(self.clients_list[current_client].prix_max-self.clients_list[current_client].prix_min)
        # Pas des paramètres price et echeance dans les boucles pour le calcul du wtp
        pas_price = round(0.5/range_price, 3)       
        pas_echeance = round(0.5/max_echeance, 3)

        # Limite de détermination wtp du paramètre price
        if(price <= self.clients_list[current_client].prix_min):
            wtp_price = 0.5
        # Si le prix est supérieur au prix maximum du client
        elif(price >= self.clients_list[current_client].prix_max):
            # WTP nul, donc aucune volonté d'achat
            wtp_price = 0 #pas_price * 100
        
        #Quand le prix est donc compris entre le maximum et le minimum du client
        else :
            # Boucle de détermination wtp du paramètre price
            for i in range(range_price+1):
                #Calcul du wtp_price qui dépend du prix
                if(price == int(self.clients_list[current_client].prix_max - i)+1):

                    wtp_price = (pas_price * i)

        # Boucle de détermination wtp du paramètre echeance
        for i in range(max_echeance):
            #Calcul du wtp_echeance qui dépend de l'échéance
            if(self.clients_list[current_client].echeance == (max_echeance - i)):
                wtp_echeance = pas_echeance * i
        # Limite de détermination wtp du paramètre echeance
        if(self.clients_list[current_client].echeance == 0):
            wtp_echeance = 0.5

        # Calcul final du wtp en fonction du prix et de l'échéance avec leur poids respectif
        wtp = wtp_price * poids_price + wtp_echeance * poids_echeance

        # Retourne la valeur du wtp actualisée en fonction du prix et de l'échéance
        return wtp

    # Fonction de détermination si le client stratégique achète ou pas en fonction des prix qu'il rencontre au fur et à mesure au moment présent sans connaissance des prix futurs
    # Si le dernier prix est inférieur aux deux d'avant
    def strategic_price(self, price_trace, current_client):
        length = len(price_trace)
        poids = 0
        buy = False
        # Applique l'achat si le dernier prix rencontré par le client stratégique est inféieur aux deux précédents
        for i in range(2):
            if (price_trace[length-1]<=price_trace[length-(i+1)-1]):
                poids = poids + 1
        #Si le dernier prix rencontré par le client stratégique est bien inféieur aux deux précédents, le booléean prend True
        if (poids >= 2):
            if (price_trace[length-1]<=self.clients_list[current_client].prix_max or price_trace[length-1]>=self.clients_list[current_client].prix_min):
                buy = True
        # Retourne le booléen, déterminant s'il y a poursuite de réservation ( True ) ou non ( False )
        return buy

    def get_min_x_percent(self, x):
        row = self.df.shape[0]        
        return self.df['price'].head(int(x * row)).mean()

    def get_max_x_percent(self, x):
        row = self.df.shape[0]        
        return self.df['price'].tail(int(x * row)).mean()

    def update_min_max(self, price, max_echeance):
        # UPDATE ECHEANCE
        #0.2 pour 20% des max et des min
        new_min = self.get_min_x_percent(0.2)
        new_max = self.get_max_x_percent(0.2)
        # Boucle de parcours pour l'actualisation
        for i in range(len(self.clients_list)):
           
            self.clients_list[i].prix_min = int(new_min)
            self.clients_list[i].prix_max = int(new_max) 
            wtp = self.wtp_actualisation(price, max_echeance,i)
            self.clients_list[i].will_to_pay = wtp

        # Sales verification and sold items deletion

    def check_sales(self, price, max_echeance, price_trace, list_resa, list_resa_scnd):

        self.update_min_max(price,max_echeance)
        
        ####" Mettre update

        # Définition des variables représentant le nombre d'achats considérés, effectués, abandonnées, et les achats instantanés, et les achats repoussés
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

                # Ajout d'une variable aléatoire dans le calcul de la variable de conditionnement d'achat, pour mieux représenter la réalité
                actual_willingness = rnd.random()/3 #From 0 to 1/3
                # Adapt value according to the time left

                # Poids de chaque paramètre, le poids de l'échéance ici sera bien plus important que celui du prix et de l'aléatoire
                # Le poids total doit être stricement égal à 3, ici 0.5 + 2 + 0.5 = 3
                poids_price, poids_echeance, poids_rnd = 0.5, 2, 0.5
                
                # Tranche de prix entre le minimum et le maximum du client, pour permettre aux calculs suivants
                range_price = int(self.clients_list[current_client].prix_max-self.clients_list[current_client].prix_min)
                
                # Pas des paramètres price et echeance dans les boucles pour le calcul du wtp
                pas_price = round(1/3/range_price, 3)      
                pas_echeance = round(1/3/max_echeance, 3)
                
                # Boucle de détermination pour le paramètre des prix pour la variable de conditionnement
                # Parcourant la tranche de prix
                for i in range(range_price):
                    # Calcul en fonction du prix
                    if(price ==int (self.clients_list[current_client].prix_max - i)):
                        wtp_price = pas_price * i
                # Limite lorsque le prix est inférieur au prix minimum du client
                if(price <= self.clients_list[current_client].prix_min):
                    wtp_price = 1/3 #pas_price * 100
                # Limite lorsque le prix est supérieur au prix max du client
                elif(price > self.clients_list[current_client].prix_max):
                    # Paramètre nul, aucune volonté d'achat du côté du prix
                    wtp_price = 0 #pas_price * 100
                
                # Boucle de détermination pour le paramètre de l'échéance pour la variable de condtionnement
                for i in range(max_echeance):
                    # Calcul du paramètre de l'échéance
                    if( self.clients_list[current_client].echeance == max_echeance - i):
                        wtp_echeance = pas_echeance * i
                #Limite lorsque l'échéance client est à 0, démontrant une volonté certaine d'achat du côté de l'échéance
                if(self.clients_list[current_client].echeance == 0):
                    wtp_echeance = 1/3
                # Calcul de la variable de conditionnement appelée ici wtp, avec les paramètres du prix, de l'échéance ,et de l'aléatoire avec leur poids respectif
                wtp = wtp_price * poids_price + wtp_echeance * poids_echeance + actual_willingness * poids_rnd

                # Comparaison pour voir si le client peut poursuivre la démarche de réservation ou non
                if ((1-wtp) <= self.clients_list[current_client].will_to_pay) and list_resa[self.clients_list[current_client].echeance] < 30:
                    
                    # Appel à la fonction strategic_price, pour vois si l'achat peut se poursuivre ou non
                    condition_prix = self.strategic_price(price_trace, current_client)
                    # Si le dernier prix rencontré est inférieur aux deux précédents
                    if( condition_prix == True):
                        # Achat Réalisé et incrémentation des variables concernées
                        buy_done += 1
                        list_resa[self.clients_list[current_client].echeance] += 1
                        list_resa_scnd[self.clients_list[current_client].echeance] += 1
                        
                        # L'acheteur quitte le marché
                        list_sales.append(current_client)
                    # Si l'échéance du client est à 0, il est obligé de réserver, car il ne peut plus attendre
                    elif self.clients_list[current_client].echeance == 0: 
                        # Achat Réalisé et incrémentation des variables concernées
                        buy_done += 1
                        list_resa[self.clients_list[current_client].echeance] += 1
                        list_resa_scnd[self.clients_list[current_client].echeance] += 1
                        # l'acheteur quitte le marché
                        list_sales.append(current_client)
                    # Sinon
                    else:
                        # Achat Abandonné
                        buy_dropped += 1
                # Si la variable de conditionnement ne permet pas de poursuivre jusqu'à la réservation
                else:
                    # Achat Abandonné
                    buy_dropped += 1
                    # Si l'échéance du client est à 0, et que son achat est abandonné
                    if  self.clients_list[current_client].echeance == 0: 

                        # L'acheteur quitte le marché, sans avoir réserver
                        list_sales.append(current_client)


            # Lower price than minimum
            elif (price <= self.clients_list[current_client].prix_min) and list_resa[self.clients_list[current_client].echeance] < 30:
                # Achat instantané et incrémentation des variables concernées
                instant_buy += 1
                list_resa[self.clients_list[current_client].echeance] += 1
                list_resa_scnd[self.clients_list[current_client].echeance] += 1
                # L'acheteur quitte le marché
                list_sales.append(current_client)

            # Higher price than maximum
            else:
                # Achat repoussé incrémentation de la variable concernée
                buy_postponed += 1
                # Si l'échéance du client est à 0, et que son achat est abandonné
                if  self.clients_list[current_client].echeance == 0: 

                    # L'acheteur quitte le marché, sans avoir réserver
                    list_sales.append(current_client)
            # Décrémentation de l'échéance client    
            self.clients_list[current_client].echeance -= 1

        
        return list_sales ,buy_considered, buy_done, buy_dropped, instant_buy, buy_postponed, list_resa, list_resa_scnd
    
    def update_client(self,prix_min,prix_max,list_to_del,max_time,resting_time):
        self.del_client(list_to_del)
        
        # Affectation des WTP random
        for _ in range(len (list_to_del)):
            temp_min = self.get_min_x_percent(0.2)
            temp_max = self.get_max_x_percent(0.2)
            temp_wtp = rnd.random()
            if max_time-resting_time-1<= 0 :
                temp_ech = 0
            else:
                temp_ech = rnd.randint(0,max_time-resting_time-1)
            self.clients_list.append(strat_client(temp_min,temp_max,temp_wtp,temp_ech))