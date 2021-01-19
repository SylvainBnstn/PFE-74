# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 19:10:38 2021

@author: LOIC
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import nan 

URLS = ["http://data.insideairbnb.com/united-states/ny/new-york-city/2020-12-10/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-11-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-10-05/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-09-07/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-08-15/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-07-07/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-06-08/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-05-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-04-08/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-03-13/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-02-12/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2020-01-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-12-04/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-11-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-10-14/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-08-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-07-08/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-06-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-05-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-04-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-03-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-02-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2019-01-09/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-12-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-11-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-10-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-09-08/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-08-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-07-05/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-06-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-05-09/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-04-06/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-03-04/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-02-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-01-10/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2018-01-10/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-12-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-11-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-10-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-09-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-08-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-07-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-06-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-05-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-04-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-03-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-02-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2017-01-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-12-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-11-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-10-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-09-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-08-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-07-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-06-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-05-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-04-03/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-02-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2016-01-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-12-02/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-11-20/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-11-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-10-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-09-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-08-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-06-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-05-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-03-01/data/listings.csv.gz",
"http://data.insideairbnb.com/united-states/ny/new-york-city/2015-01-01/data/listings.csv.gz"]



frame = []
column_names = [ "id","property_type", "room_type", "accommodates", "bedrooms", "beds", "price","availability_30", "number_of_reviews","review_scores_rating",	"review_scores_accuracy" ]
missed_col =  ['square_feet','cleaning_fee']

for i in range(len(URLS)):
  df = pd.read_csv(URLS[i])
  boolean  = missed_col[0] in list(df.columns)
  url_split = URLS[i].split("/")
  df2 = df[column_names]
  for j in range(len(missed_col)):
    if (missed_col[j] in list(df.columns)):
      df2[missed_col[j]] = df[missed_col[j]]
    else:
      df2[missed_col[j]] = nan
      
  df2["country"] = url_split[3]
  df2["city"] = url_split[5]
  df2["date"] = url_split[6]
  frame.append(df2)

result = pd.concat(frame, ignore_index=True)

result["price"] = result["price"].str.replace("$","")
result["price"] = result["price"].str.replace(",","").astype(float)
# result["date"] = pd.to_datetime(result["date"])
result["cleaning_fee"].fillna("$0.00", inplace = True)
result["cleaning_fee"] = result["cleaning_fee"].str.replace("$","")
result["cleaning_fee"] = result["cleaning_fee"].str.replace(",","").astype(float)
result["revenue_30"] = (result["price"] + result["cleaning_fee"] )*(30-result["availability_30"]) 
result["booked"] = 30-result["availability_30"] # nombre de jours révservés
result.sort_values(by=["id","date"], inplace = True)
result.reset_index(drop=True, inplace = True)
result

###########################################################################################################"


#/!\ la date a été mis en string au lieu d'un datetime (voir ligne 109)

nb_id = result[result["room_type"] == "Entire home/apt"].groupby(["id"], as_index = False).size()
nb_id.sort_values(by=["size"], ascending= False, inplace= True)
nb_id = nb_id[nb_id["size"] >=70 ]
nb_id 




df = pd.merge(result[result["room_type"] == "Entire home/apt"], nb_id["id"] ,how = "right" ,on=["id"] )
df =  df[df["room_type"] == "Entire home/apt"]
df.duplicated().sum()
df.drop_duplicates(inplace= True)

#Create a dataframe full of NaN value 
#Each row is an id, and each column is a date
data_with_all_date = df.pivot(index="id", columns="date", values = "price")
data_with_all_date


# Plot of the first 20 id with price < 500$
list_id = list(df[df["price"] <500].id.unique())[:20] #select the first 20 id n the list
list_date =list(df.date.unique())
plt.figure(figsize=(30, 10))

for i in range(len(list_id)):
  plt.plot( "date", "price" ,data = df[df["id"] == list_id[i] ] , label = list_id[i])
plt.xticks(rotation=45)
plt.ylabel("price")
plt.xlabel("date")
plt.legend(title = "ID")
plt.grid()





