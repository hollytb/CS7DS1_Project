import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def main():
    df = pd.read_csv("Unified_EST_FIELD_data.csv")
    print(df.head())
    X1 = df.iloc[: ,0]
    X2 = df.iloc[: ,1]


    X = np.column_stack((X1, X2))
    y = df.iloc[: ,2]
    print(len(y))
    poly = PolynomialFeatures(5)
    XPoly=poly.fit_transform(X)

    mean_error=[];
    std_error=[]
    Ci_range = [0.0001, 0.001, 0.01, 0.1, 1]
    for Ci in Ci_range:
        model = Ridge(alpha=1/(2*Ci))
        temp=[]
        kf = KFold(n_splits=5)
        for train, test in kf.split(XPoly):
            model.fit(XPoly[train], y[train])
            ypred = model.predict(XPoly[test])
            temp.append(mean_absolute_error(y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.xlim((0.0001, 1))
    plt.errorbar(Ci_range,mean_error,yerr=std_error)
    plt.xlabel('Ci')
    plt.xticks((0.01, 0.1, 1))
    plt.ylabel('Mean absolute error')
    plt.title('Mean absolute error and Standard deviance for C values')
    plt.show()
    print(ypred)
if __name__ == "__main__":
    main()