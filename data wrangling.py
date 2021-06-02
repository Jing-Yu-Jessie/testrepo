import pandas as pd
import matplotlib.pylab as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type",\
           "num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ration","horsepower","peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers

print(df.head(n = 5))

# Preserve progress anytime by saving modified dataset
path=r"D:\python code\coursera\data analyze with python\importing datasets\automobile.csv"
df.to_csv(path)

# Data types
print(df.dtypes)

# Returns a statistical summary
print(df.describe())
# Provides full summary statistics
print(df.describe(include="all"))

# Provides a concise summary of your DataFrame
print(df.info())


##################################  DATA  WRANGLING  ##############


# Simple dataframe operations
df["symboling"] = df["symboling"] + 1

print(df.head(n = 5))

# deal with missing data
df.replace("?", np.nan, inplace = True)  # replace "?" to NaN
missing_data = df.isnull() # use python's built_in functions to identify missing values
missing_data.head(5) # ture stands for missing value, false stands for not missing value
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
# axis=0 drops the entire row, axis=1 drops the entire column
# inplace=True writes the result back into the data frame, equivalent to df = df.dropna(subset=["price"], axis=0)
df.dropna(subset=["price"], axis=0, inplace = True)

# Data formatting
# applying calculations to an entire column
df["city-mpg"] = 235/df["city-mpg"]  # convert "mpg" to "L/100km" in Car dataset
df.rename(columns={"city_mpg":"city-L/100km"}, inplace=True) # rename the column "city_mpg" to "city-L/100km"
# correcting data types
# dataframe.dtypes() to identify data type.
# dataframe.astype() to convert data type.
df["price"] = df["price"].astype("int") # convert fata type to integer in column "price"

# Data normalization
# Simple feature scaling
df["length"] = df["length"]/df["length"].max()
# Min-max
df["length"] = (df["length"]-df["length"].min())/(df["length"].max()-df["length"].min())
# Z-score
df["length"] = (df["length"]-df["length"].mean())/df["length"].std()

# Binning
bins = np.linspace(min(df["price"]),max(df["price"]),4)
group_names = ["Low","Medium","High"]
df["price-binned"] = pd.cur(df["price"],bins,labels=group_names,include_lowest=True)

# Dummy variables
pd.get_dummies(df['fuel'])

###########################  Exploratory Data  ##########################
# Descriptive Statistics
df.describe() #Summarize statistics using pandas describe
drive_wheels_counts = df["drive-wheels"].value_counts() #summarize the categorical data
drive_wheels_counts.rename(colunms={'drive-wheels':'value_counts'} inplace=True)
drive_wheels_counts.index.name = 'drive_wheels'
                           
