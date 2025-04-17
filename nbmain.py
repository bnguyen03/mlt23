
import pandas as pd
    # import sklearn
from sklearn.naive_bayes import GaussianNB    
    # Dictionary for Non ML prediction

InvFeatureDict = {
    "1111" : "Invest ESG",
    "1110" : "Monitor ESG",
    "0001" : "Prof Non ESG",
    "0010" : "No Prof No ESG",
    "0011" : "Prof Non ESG",
    "0101" : "Prof Non ESG",
    "0111" : "Prof Non ESG",
    "1001" : "Prof Non ESG",
    "1011" : "Prof Non ESG",
    "1101" : "Prof Non ESG",
    "0000" : "No Prof No ESG",
    "0100" : "No Prof No ESG",
    "0110" : "No Prof No ESG",
    "1000" : "No Prof No ESG",
    "1100" : "No Prof No ESG",
    "0000" : "No Prof No ESG",
    "1010" : "No Prof No ESG"
    }

    # Dictionary for Prediction answer
InvType = {
    1 : "Invest_ESG",
    2 : "Monitor_ESG",
    3 : "Prof_Non_ESG",
    4 : "No_Prof_No_ESG"
    }

df = pd.read_csv("ESG-less-data.csv")
print(df.head())

# separate feature and label
x = df[df.columns[0:4]] # Feature
y = df.Type # Label

# Training model
gb = GaussianNB()
gb.fit(x,y)

for i in range(0,16):
    int_to_binary=f"{i:b}"
    inp = f"{int_to_binary:0>4}"

    data = {
        "EP": inp[0],
        "EI": inp[1],
        "DB": inp[2],
        "PR": inp[3]
    }

    pdt = pd.DataFrame(data, columns=["EP","EI","DB","PR"], index=[0])
    g = gb.predict(pdt)

    print(f"{inp}: {InvType[g[0]]}")