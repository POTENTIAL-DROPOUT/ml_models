import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
import sklearn.tree 

# Load the data
oecd_bli = pd.read_csv("datasets/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("datasets/gdp_per_capita.csv",thousands=',',delimiter='\t',
encoding='latin1', na_values="n/a")

def prepare_country_stats(oecd_bli, gdp_per_capita):
    # Keep only the columns we need from the wellbeing dataset
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

    # Rename the GDP column and set the index correctly
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    # Join the dataframes
    full_country_stats = pd.merge(oecd_bli, gdp_per_capita,
                                  left_index=True, right_index=True)

    # Select only the required columns
    full_country_stats = full_country_stats[["GDP per capita", "Life satisfaction"]]

    # Remove missing values and sort
    full_country_stats = full_country_stats.dropna()
    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    return full_country_stats

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model1 = sklearn.linear_model.LinearRegression()
model3 = sklearn.tree.DecisionTreeRegressor()
model2=  sklearn.neighbors.KNeighborsRegressor(
n_neighbors=3)

# Train the model
model1.fit(X, y)
model2.fit(X, y)
model3.fit(X, y)


# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus's GDP per capita
print(model1.predict(X_new)) 
print(model2.predict(X_new)) 
print(model3.predict(X_new))