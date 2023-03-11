#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy
import datetime
datetime.datetime.strptime
numpy.set_printoptions(threshold=numpy.inf)
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


# In[2]:


def summary_df(temp):
    summary = []
    for i in range(0, len(temp.axes[1])):
        summary.append(
        {
            'Name': temp.axes[1][i],
            'Missing': temp.iloc[:,i].isna().sum(),
            'Uniques': len(temp.iloc[:,i].unique())
        }
        )
    print(pd.DataFrame(summary))


# In[3]:


input_df = pd.read_csv(r"C:\Sai_Aishwarya\ML\Market Segmentation\data.csv", encoding = 'unicode_escape')
input_df


# In[4]:


len(input_df.index)


# In[5]:


#keep - to keep some rows
#inplace - same df or diff df
#ignore_index - ignore old index
input_df.drop_duplicates(keep=False, inplace=True, ignore_index=True)
input_df


# In[6]:


input_df["InvoiceDate"] = pd.to_datetime(input_df["InvoiceDate"])
input_df = input_df.astype({'CustomerID':'string', 'InvoiceNo':'string', 'StockCode':'string', 'Description':'string', 'Country':'string'})


# In[7]:


input_df["BillingPrice"] = input_df["Quantity"] * input_df["UnitPrice"]
input_df


# In[8]:


input_df = input_df.query("InvoiceDate >= '2010-12-01' and InvoiceDate <'2011-12-09'")
input_df


# In[9]:


input_df.info()


# In[10]:


summary_df(input_df)


# In[11]:


input_df = input_df.dropna(subset=['CustomerID'])
input_df = input_df.reset_index(drop=True)
summary_df(input_df)


# In[12]:


input_df['StockCode'] = input_df['StockCode'].str.lower().str.strip()
input_df['Description'] = input_df['Description'].str.lower().str.strip()
input_df['Description'] = input_df['Description'].str.replace('set of', 'set of ')
input_df['Description'] = input_df['Description'].str.replace('set/', 'set of ')
input_df['Description'] = input_df['Description'].str.replace('s/', 'set of ')
input_df['Description'] = input_df['Description'].str.replace('+', ' and ')
input_df['Description'] = input_df['Description'].str.rstrip('.')


# In[13]:


input_df.drop_duplicates(keep=False, inplace=True, ignore_index=True)
input_df


# ## Geographic Segementation

# In[14]:


# Get the number of customers per country
customer_counts = input_df.groupby('Country')['CustomerID'].nunique().reset_index()
print('Total no. of Customers in the United Kingdom is ' + str(customer_counts[customer_counts['Country']=='United Kingdom']['CustomerID'].item()))
customer_counts = customer_counts[customer_counts['Country'] != 'United Kingdom']
# Create a choropleth map
fig = px.choropleth(customer_counts, 
                    locations='Country', 
                    locationmode='country names',
                    color='CustomerID',
                    hover_name='Country',
                    color_continuous_scale=px.colors.sequential.Plasma)

# Add title and show the map
fig.update_layout(title='Number of Customers per Country')
fig.show()

customer_counts_top15 = customer_counts.sort_values('CustomerID', ascending=False).head(15).reset_index()
fig = px.bar(customer_counts_top15, x = "Country", y = "CustomerID", title = "Customer By Country")
fig.show()


# In[15]:


order_country = input_df.groupby('Country')['InvoiceNo'].nunique().reset_index()
print('Total no. of Orders in the United Kingdom is ' + str(order_country[order_country['Country'] == 'United Kingdom']['InvoiceNo'].item()))
order_country = order_country[order_country['Country'] != 'United Kingdom']
# Create a choropleth map
fig = px.choropleth(order_country, 
                    locations='Country', 
                    locationmode='country names',
                    color='InvoiceNo',
                    hover_name='Country',
                    color_continuous_scale=px.colors.sequential.Plasma)

# Add title and show the map
fig.update_layout(title='Number of Orders per Country')
fig.show()

order_country_top15 = order_country.sort_values('InvoiceNo', ascending=False).head(15).reset_index()
fig = px.bar(order_country_top15, x = "Country", y = "InvoiceNo", title = "Orders By Country")
fig.show()


# # RFM Segementation

# In[16]:


pd.options.display.float_format = '{:.2f}'.format
monetary_df = pd.DataFrame({"Total_revenue": input_df.groupby("CustomerID")["BillingPrice"].sum()})
monetary_df["NoOfOrders"]= input_df.groupby("CustomerID")["InvoiceNo"].nunique()
monetary_df["AverageOrderValue"]= monetary_df["Total_revenue"]/monetary_df["NoOfOrders"]
monetary_df["NoOfItems"] = input_df.groupby("CustomerID")["Quantity"].sum()
monetary_df["Recency"] = input_df.groupby("CustomerID")["InvoiceDate"].max()
monetary_df["Recency"] = (numpy.datetime64('2011-12-11') - monetary_df["Recency"]).apply(lambda l: l.days)
monetary_df = monetary_df[monetary_df.Total_revenue>=0]
monetary_df.reset_index(inplace=True)
monetary_df


# In[17]:


temp = input_df.groupby(["CustomerID","InvoiceNo"]).nth(0)
#temp = temp.reset_index()
temp = temp.drop(['StockCode', 'Description','Quantity', 'UnitPrice', 'Country', 'BillingPrice' ], axis=1)
temp

time_diff_df=pd.DataFrame(columns=['CustomerID','AvgTimeDiff'])
#customer_id takes value of customer and group is the group of values for that particular customerID
for customer_id, group in temp.groupby("CustomerID"):
    #to get InvoiceNo s
    invoices = group.index.get_level_values(1)
    invoice_dates = group["InvoiceDate"]
    avg_time_diff = invoice_dates.diff().dt.days.mean()
    if(numpy.isnan(avg_time_diff)):
        avg_time_diff = -1
        #append is deprecated
        #time_diff_df = time_diff_df.append({'CustomerID': customer_id, 'AvgTimeDiff': avg_time_diff}, ignore_index=True)
    row = {"CustomerID": customer_id, "AvgTimeDiff": avg_time_diff}
    time_diff_df = pd.concat([time_diff_df, pd.DataFrame([row])], ignore_index=True)
print(time_diff_df)


# In[18]:


monetaryFinal_df = pd.merge(monetary_df, time_diff_df, on='CustomerID', how='inner')
monetaryFinal_df


# In[19]:


fig = px.histogram(monetaryFinal_df, x="Recency")
fig.show()


# In[20]:


fig = px.histogram(monetaryFinal_df, x="Total_revenue")
fig.show()


# In[21]:


fig = px.histogram(monetaryFinal_df, x="NoOfOrders")
fig.show()


# In[22]:


fig = px.scatter(monetaryFinal_df, x="Recency", y="Total_revenue")
fig.show()


# In[23]:


fig = px.scatter(monetaryFinal_df, x="NoOfOrders", y="Total_revenue")
fig.show()


# In[24]:


fig = px.scatter(monetaryFinal_df, x="Recency", y="Total_revenue", color="NoOfOrders")
fig.show()


# In[25]:


pd.qcut(monetaryFinal_df['Recency'], q=3)
pd.qcut(monetaryFinal_df['Total_revenue'], q=3)
pd.qcut(monetaryFinal_df['NoOfOrders'], q=3)


# In[26]:


def cal_R(x):
    #0-40 41-200 201-300 300<
    #[(0.999, 25.0] < (25.0, 90.0] < (90.0, 374.0]]
    if (x['Recency']<=25):
        r = 1
    elif (x['Recency'] > 25 and x['Recency'] <=90):
        r = 2
    elif (x['Recency'] > 90):
        r = 3
    return r
        
def cal_F(x):
    # 0-10 11-30 30-80 80<
    #(0.999, 2.0] < (2.0, 4.0] < (4.0, 248.0]
    if (x['NoOfOrders']>4):
        f = 1
    elif (x['NoOfOrders'] > 2 and x['NoOfOrders'] <=4):
        f = 2
    elif (x['NoOfOrders'] <=2):
        f = 3
    return f
        
def cal_M(x):
    #100< 50-99 11-49 0-10
    #[(-4287.631, 370.8] < (370.8, 1133.25] < (1133.25, 279489.02]]
    if (x['Total_revenue']>1133.25):
        m = 1
    elif (x['Total_revenue'] > 370.8 and x['Total_revenue'] <=1133.25):
        m = 2
    elif (x['Total_revenue'] <=370.8):
        m = 3
    return m

def segment_cust(x):
    if x['R']==1 and x['F']==1 and x['M']==1:
        return 'Plantinum'
    elif x['R']==2 and x['F']==2 and x['M']==2:
        return 'Gold'
    elif x['R']==3 and x['F']==3 and x['M']==3:
        return 'Silver'


# In[27]:


monetaryFinal_df['R'] = monetaryFinal_df.apply(cal_R,axis=1)
monetaryFinal_df['F'] = monetaryFinal_df.apply(cal_R,axis=1)
monetaryFinal_df['M'] = monetaryFinal_df.apply(cal_R,axis=1)
pd.set_option('display.max_rows', None)
# monetary_df.loc[monetary_df['M']!=monetary_df['F']]
monetaryFinal_df["Segment"] = monetaryFinal_df.apply(segment_cust, axis=1)
monetaryFinal_df


# In[30]:


fig = px.bar(monetaryFinal_df, x = 'Segment')
fig.show()


# ## Time Series

# In[31]:


timeseries_df = input_df
timeseries_df['Hour'] = timeseries_df['InvoiceDate'].dt.hour
timeseries_df['Month'] = timeseries_df['InvoiceDate'].dt.month
timeseries_df['Day'] = timeseries_df['InvoiceDate'].dt.dayofweek


# In[32]:


day_ts = timeseries_df.groupby("Day")["InvoiceNo"].nunique().reset_index()
fig = px.line(day_ts, x="Day", y = "InvoiceNo")
fig.show()


# In[33]:


hour_ts = timeseries_df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
fig = px.line(hour_ts, x="Hour", y = "InvoiceNo")
fig.show()


# In[34]:


month_ts = timeseries_df.groupby("Month")["InvoiceNo"].nunique().reset_index()
fig = px.line(month_ts, x="Month", y = "InvoiceNo")
fig.show()


# In[37]:


pop_items = input_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
pop_items
fig = px.bar(pop_items, x= "Description", y = "Quantity")
fig.show()


# ## Association Rules

# In[40]:


txn_matrix_df = (input_df.groupby(["InvoiceNo","Description"])["Quantity"]
                 .sum().unstack().reset_index().fillna(0)
                 .set_index("InvoiceNo"))
txn_matrix_df.drop('postage', inplace=True, axis=1)
txn_matrix_df = txn_matrix_df.astype(int)

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1
  
# Encoding the datasets
txn_matrix_df = txn_matrix_df.applymap(hot_encode)
txn_matrix_df


# In[44]:


frequent_itemsets = apriori(txn_matrix_df, min_support=0.02, use_colnames=True)
pd.set_option('display.max_rows', None)
frequent_itemsets


# In[45]:


# Collecting the inferred rules in a dataframe
rules = association_rules(frequent_itemsets, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules


# ##### UK had a bulk of the orders of the total dataset, around 89%. So, the data is divided into baskets for each country so we can get rules for each country. Which made more sense in the output.

# In[47]:


basket_UK = (input_df[input_df['Country'] == 'United Kingdom'].groupby(["InvoiceNo","Description"])["Quantity"]
                 .sum().unstack().reset_index().fillna(0)
                 .set_index("InvoiceNo"))
basket_UK.drop('postage', inplace=True, axis=1)
basket_UK = basket_UK.applymap(hot_encode)
display("No. of orders in UK " + str(len(basket_UK)))
frequent_itemsets_UK = apriori(basket_UK, min_support=0.02, use_colnames=True)
pd.set_option('display.max_rows', None)
frequent_itemsets_UK
rules = association_rules(frequent_itemsets_UK, metric ="confidence", min_threshold = 0.5)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules = rules[rules['lift'] > 5].reset_index()
display("No. of rules in UK " + str(len(rules)))
rules


# In[48]:


# basket_UK = (input_df[input_df['Country'] == 'United Kingdom'].groupby(["InvoiceNo","Description"])["Quantity"]
#                  .sum().unstack().reset_index().fillna(0)
#                  .set_index("InvoiceNo"))
# basket_UK.drop('postage', inplace=True, axis=1)
# basket_UK = basket_UK.applymap(hot_encode)
# display("No. of orders in UK " + str(len(basket_UK)))
# frequent_itemsets_UK = fpgrowth(basket_UK, min_support=0.02, use_colnames=True)
# pd.set_option('display.max_rows', None)
# frequent_itemsets_UK
# rules = association_rules(frequent_itemsets_UK, metric ="confidence", min_threshold = 0.5)
# rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
# rules = rules[rules['lift'] > 5].reset_index()
# display("No. of rules in UK " + str(len(rules)))
# rules


# In[49]:


basket_France = (input_df[input_df['Country'] == 'France'].groupby(["InvoiceNo","Description"])["Quantity"]
                 .sum().unstack().reset_index().fillna(0)
                 .set_index("InvoiceNo"))
basket_France.drop('postage', inplace=True, axis=1)
basket_France = basket_France.applymap(hot_encode)
display("No. of orders in France " + str(len(basket_France)))
frequent_itemsets_France = apriori(basket_France, min_support=0.04, use_colnames=True)
pd.set_option('display.max_rows', None)
frequent_itemsets_UK
rules = association_rules(frequent_itemsets_France, metric ="confidence", min_threshold = 0.5)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules = rules[rules['lift'] > 9].reset_index()
display("No. of rules in France " + str(len(rules)))
rules


# In[50]:


basket_Germany = (input_df[input_df['Country'] == 'Germany'].groupby(["InvoiceNo","Description"])["Quantity"]
                 .sum().unstack().reset_index().fillna(0)
                 .set_index("InvoiceNo"))
basket_Germany.drop('postage', inplace=True, axis=1)
basket_Germany = basket_Germany.applymap(hot_encode)
display("No. of orders in Germany " + str(len(basket_Germany)))
frequent_itemsets_Germany = apriori(basket_Germany, min_support=0.03, use_colnames=True)
pd.set_option('display.max_rows', None)
rules = association_rules(frequent_itemsets_Germany, metric ="confidence", min_threshold = 0.4)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules = rules[rules['lift'] > 6].reset_index()
display("No. of rules in Germany " + str(len(rules)))
rules


# In[51]:


basket_EIRE = (input_df[input_df['Country'] == 'EIRE'].groupby(["InvoiceNo","Description"])["Quantity"]
                 .sum().unstack().reset_index().fillna(0)
                 .set_index("InvoiceNo"))
# basket_EIRE.drop('postage', inplace=True, axis=1)
basket_EIRE = basket_EIRE.applymap(hot_encode)
display("No. of orders in EIRE " + str(len(basket_EIRE)))
frequent_itemsets_EIRE = apriori(basket_EIRE, min_support=0.055, use_colnames=True)
pd.set_option('display.max_rows', None)
rules = association_rules(frequent_itemsets_EIRE, metric ="confidence", min_threshold = 0.7)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
rules = rules[rules['lift'] > 9].reset_index()
display("No. of rules in EIRE " + str(len(rules)))
rules

