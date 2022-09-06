import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from datetime import datetime

# select dataframe
print ("\nPredict temperature with scikit-learn SplineTransformer\n")
set = ['min', 'avg', 'max']
print ("Select data set: \n1 = "+set[0]+"\n2 = "+set[1]+"\n3 = "+set[2])
id = int(input ("Your choice: "))

# read temp log csv
f = "templog-"+set[id-1]+".csv"
df = pd.read_csv(f)

# train model
X = df[['day']]
y = df['temp']
model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model.fit(X.values, y)

# get user input date
date_format = "%d/%m"
target_date = input("Insert target date (dd/mm): ")
target_date = datetime.strptime(target_date, date_format)
day = target_date - datetime.strptime("1/1", date_format)

# predict target temp
X_test = [[day.days]]
y_pred = model.predict(X_test)
print("Target day = ", day.days)
print("Predicted "+set[id-1]+" temp = %.1f°C" % y_pred)

# plot diagram
df.plot(x ='day', y='temp', kind = 'scatter', label='temp')
plt.plot(X, model.predict(X.values), label = "model")
plt.plot(X_test, y_pred, marker="o", markerfacecolor="red", markeredgecolor="none", label='predict')
plt.text(day.days, y_pred+1.2, str(round(float(y_pred[0]),1)),backgroundcolor='1',alpha=.5)
plt.title("Predicted "+set[id-1]+" temperature")
plt.legend()
plt.draw()
plt.pause(1)
input ("\n[Press 'Enter' to quit]")
plt.close()
exit()
