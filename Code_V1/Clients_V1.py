# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:30:18 2020

@author: Sylvain
"""
import random

class Naiv_client:
    
    def __init__(self, prix_min,prix_max,will_to_pay):
        self.prix_max=prix_max
        self.prix_min=prix_min
        self.will_to_pay=will_to_pay

    def __str__(self):
        r="Prix achat min: " + str(self.prix_min)+"\n"
        r+="Prix achat max: " + str(self.prix_max)+"\n"
        r+="WTP : " + str(self.will_to_pay)+"\n"
        return r
    
    def __repr__(self):
        r=(self.prix_min,self.prix_max,self.will_to_pay)
        return str(r)
    
class list_naiv_clients:
    
    def __init__(self,prix_min,prix_max,nb_client):
        self.clients_list=[]
        for _ in range(nb_client):
            temp_min = random.uniform(prix_min,prix_min+(prix_max-prix_min)/2)
            temp_max = random.uniform(prix_min+(prix_max-prix_min)/2,prix_max)
            temp_wtp = random.random()
            self.clients_list.append(Naiv_client(temp_min,temp_max,temp_wtp))
    
    def __str__(self):
        return str(self.clients_list)
    
    def del_client(self, list_i):
        for index in sorted(list_i, reverse=True):
            del self.clients_list[index]
            
        


def test():    
    a=Naiv_client(10,15,0.2)
    a2=Naiv_client(12,15,0.7)
    list2=[a,a2]
    print(list2)
    list1=list_naiv_clients(10,20,10)
    print(list1)


