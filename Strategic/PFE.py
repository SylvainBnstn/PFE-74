# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:30:18 2020

@author: Sylvain
"""
import random as rnd


class Naiv_client:

    def __init__(self, prix_min, prix_max, will_to_pay):
        self.prix_max = prix_max
        self.prix_min = prix_min
        self.will_to_pay = will_to_pay

    def __str__(self):
        r = "Prix achat min: " + str(self.prix_min) + "\n"
        r += "Prix achat max: " + str(self.prix_max) + "\n"
        r += "WTP : " + str(self.will_to_pay) + "\n"
        return r

    def __repr__(self):
        r = (self.prix_min, self.prix_max, self.will_to_pay)
        return str(r)


class list_naiv_clients:

    def __init__(self, prix_min, prix_max, nb_client):
        self.clients_list = []
        for _ in range(nb_client):
            temp_min = rnd.uniform(
                prix_min, prix_min + (prix_max - prix_min) / 2)
            temp_max = rnd.uniform(
                prix_min + (prix_max - prix_min) / 2, prix_max)
            temp_wtp = rnd.random()
            self.clients_list.append(Naiv_client(temp_min, temp_max, temp_wtp))

    def __str__(self):
        return str(self.clients_list)

    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]

    def get_min_x_percent(self, df, x):
        return df['price'].head(int(x * df.size)).mean()

    def get_max_x_percent(self, df, x):
        return df['price'].tail(int(x * df.size)).mean()

    # fonction d'execution des ventes
    def check_sales(self, price, clients):

        buy_considered = buy_done = buy_dropped = instant_buy = buy_postponed = 0

        # list des index des acheteurs ayant conclu un acht
        list_sales = []

        for current_client in range(len(self.clients.clients_list)):

            # cas ou le prix est compris dans la cible
            if (price >= self.clients.clients_list[current_client].prix_min and price <=
                    self.clients.clients_list[current_client].prix_max):

                # on montre que l'achat est envisagé
                #temp_str =str("Achat envisagé -> ")
                buy_considered += 1
                rnd_willingness = rnd.random()

                # a revoir pour mise en place de WTP exacte
                if rnd_willingness <= self.clients.clients_list[current_client].will_to_pay:
                    #temp_str+=str("Achat Réalisé")
                    buy_done += 1
                    # l'acheteur quitte le marché
                    list_sales.append(current_client)

                else:
                    #temp_str+=str("Achat Abandonné")
                    buy_dropped += 1

                # print(temp_str)

            # Lower price than minimum
            elif (price <= self.clients.clients_list[current_client].prix_min):
                #print("Instant achat")
                instant_buy += 1
                list_sales.append(current_client)

            # Higher price than maximum
            else:
                #print("Achat repoussé")
                buy_postponed += 1

        nb_row = self.df_naiv_sales.shape[0]
        data_temp = [
            nb_row,
            price,
            buy_considered,
            buy_done,
            buy_dropped,
            instant_buy,
            buy_postponed,
            instant_buy +
            buy_done]
        self.df_naiv_sales.loc[nb_row] = data_temp
        return list_sales


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

    def __init__(self, prix_min, prix_max, nb_client):
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
                    echeance=12))  # DATE RANDOM

    def __str__(self):
        return str(self.clients_list)

    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]

    # Fonction d'actualisation du paramètre WTP ( Willingness To Pay ) en
    # fonction du prix et de l'échéance
    def wtp_actualisation(self, price, echeance):

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
        range_price = (self.prix_max-self.prix_min)
        range_echeance = 12
        pas_price = round(0.5/range_price, 3)       #0.005
        pas_echeance = round(0.5/range_echeance, 3) #0.045

        # Boucle de détermination wtp du paramètre price
        for i in range(range_price):
            if(price == (self.prix_max - i)):
                wtp_price = pas_price * i
        # Limite de détermination wtp du paramètre price
        if(price <= self.prix_min):
            wtp_price = pas_price * 100

        # Boucle de détermination wtp du paramètre echeance
        for i in range(range_echeance):
            if(echeance == (self.echeance - i)):
                wtp_echeance = pas_echeance * i
        # Limite de détermination wtp du paramètre echeance
        if(echeance == 1):
            wtp_echeance = 0.5

        # Calcul final du wtp en fonction du prix et de l'échéance avec leur
        # poids respectif
        wtp = wtp_price * poids_price + wtp_echeance * poids_echeance

        # Retourne la valeur du wtp actualisée en fonction du prix et de
        # l'échéance
        return wtp

    def get_min_x_percent(self, df, x):
        return df['price'].head(int(x * df.size)).mean()

    def get_max_x_percent(self, df, x):
        return df['price'].tail(int(x * df.size)).mean()

    def update_min_max(self, df, price):
        # UPDATE ECHEANCE
        new_min = self.get_min_x_percent(df, 0.2)
        new_max = self.get_max_x_percent(df, 0.2)
        # Boucle de parcours pour l'actualisation
        for i in range(len(self.clients_list)):
            # Verify whether echeance is zero before -1
            self.clients_list[i].echeance -= 1
            wtp = self.wtp_actualisation()

            self.clients_list[i].prix_min = new_min
            self.clients_list[i].prix_max = new_max
            self.clients_list[i].will_to_pay = wtp

        # Sales verification and sold items deletion

    def check_sales(self, price, clients, echeance):
    
        buy_considered = buy_done = buy_dropped = instant_buy = buy_postponed = 0

        # list des index des acheteurs ayant conclu un acht
        list_sales = []

        for current_client in range(len(self.clients.clients_list)):
            # cas ou le prix est compris dans la cible
            if (price >= self.clients.clients_list[current_client].prix_min and price <=
                    self.clients.clients_list[current_client].prix_max):

                # on montre que l'achat est envisagé
                #temp_str =str("Achat envisagé -> ")
                buy_considered += 1

                actual_willingness = rnd.random()/3 #From 0 to 1/3
                # Adapt value according to the time left
                poids_price, poids_echeance, poids_rnd = 0.5, 2, 0.5
                range_price = (self.prix_max-self.prix_min)
                range_echeance = 12
                pas_price = round(1/3/range_price, 3)      
                pas_echeance = round(1/3/range_echeance, 3)
                for i in range(range_price):
                    if(price == (self.prix_max - i)):
                        wtp_price = pas_price * i
                if(price <= self.prix_min):
                    wtp_price = 1/3 #pas_price * 100
                for i in range(range_echeance):
                    if(echeance == (self.echeance - i)):
                        wtp_echeance = pas_echeance * i
                if(echeance == 1):
                    wtp_echeance = 1/3

                wtp = wtp_price * poids_price + wtp_echeance * poids_echeance + actual_willingness * poids_rnd

                # a revoir pour mise en place de WTP exacte
                if (1-wtp) <= self.clients.clients_list[current_client].will_to_pay:

                    #temp_str+=str("Achat Réalisé")
                    buy_done += 1
                    # l'acheteur quitte le marché
                    list_sales.append(current_client)

                else:
                    #temp_str+=str("Achat Abandonné")
                    buy_dropped += 1

                # print(temp_str)

            # Lower price than minimum
            elif (price <= self.clients.clients_list[current_client].prix_min):
                #print("Instant achat")
                instant_buy += 1
                list_sales.append(current_client)

            # Higher price than maximum
            else:
                #print("Achat repoussé")
                buy_postponed += 1

        nb_row = self.df_naiv_sales.shape[0]
        data_temp = [
            nb_row,
            price,
            buy_considered,
            buy_done,
            buy_dropped,
            instant_buy,
            buy_postponed,
            instant_buy +
            buy_done]
        self.df_naiv_sales.loc[nb_row] = data_temp
        return list_sales


def test():
    a = Naiv_client(10, 15, 0.2)
    a2 = Naiv_client(12, 15, 0.7)
    list2 = [a, a2]
    print(list2)
    list1 = list_naiv_clients(10, 20, 10)
    print(list1)