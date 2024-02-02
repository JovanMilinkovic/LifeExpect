import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

life_expectancy = pd.read_csv('Life-expectancy-years.csv')

#pd.options.display.max_columns = None
print(life_expectancy.info())
print(life_expectancy.describe())

print(life_expectancy['2013'].unique())
#sns.distplot(life_expectancy["2013"].unique(),bins=10)
sns.histplot(life_expectancy["2013"].unique(),color="red")
plt.show()

sns.histplot(data=life_expectancy, x='1920', color='green')
plt.show()

life_expectancy=life_expectancy.dropna()
life_expectancy['1800-1810'] = life_expectancy[['1800', '1801', '1802', '1803','1804', '1805', '1806', '1807','1808', '1809', '1810']].mean(axis=1)
print(life_expectancy['1800-1810'].head(15))

life_expectancy['2016'] = life_expectancy['2016'].max()
print(life_expectancy['2016'].head(15))

life_expectancy['1997'] = life_expectancy[life_expectancy.columns.difference(['Life expectancy'])].mean(axis=1) #??????
print(life_expectancy['1997'].head(15))

sns.histplot(data=life_expectancy, x=life_expectancy['2000'], y=life_expectancy['2002'], color="red")
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

life_expectancy = life_expectancy.drop(['Life expectancy'], axis=1)
y = life_expectancy['2016']
X = life_expectancy.drop('2016', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression()
model = model.fit(X_train, y_train)
predict = model.predict(X_test)

print("Srednje kvadratna greska je:", mean_squared_error(predict, y_test))
print("Srednje apsolutna greska je:", mean_absolute_error(predict, y_test))
print("Koren srednje kvadratna greska je:", np.sqrt(mean_absolute_error(predict, y_test)))