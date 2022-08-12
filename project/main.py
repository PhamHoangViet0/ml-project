import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm, linear_model
from sklearn.model_selection import train_test_split

df = quandl.get('WIKI/GOOGL', api_key='zxNJD8pnTVUfBj5JegFT')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#clf = linear_model.LinearRegression()
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

# print(len(df), forecast_out)
# print(df.head())
# print(df.tail())
print(len(X), len(y))
print('Prediction for {} forward have {:.6f} accuracy'.format(forecast_out, accuracy))
