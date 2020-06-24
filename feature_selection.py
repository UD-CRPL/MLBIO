import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import sklearn
import xgboost

def rare_allele_enrichment(dataset, balanceRatio, plot):
    negative_input = dataset[:int(len(dataset) * balanceRatio)]
    positive_input = dataset[int(len(dataset)*balanceRatio):len(dataset)]
    negative_input = negative_input.T
    positive_input = positive_input.T

    negative_input = pd.DataFrame(negative_input)
    positive_input = pd.DataFrame(positive_input)

    negative_input['sum'] = negative_input.sum(axis=1)
    positive_input['sum'] = positive_input.sum(axis=1)

    negative_input['ratio'] = (negative_input['sum']).div(2 * len(positive_input))
    positive_input['ratio'] = (positive_input['sum']).div(2 * len(negative_input))

    distribution = negative_input['ratio'] - positive_input['ratio']

    q3 = np.quantile(distribution, .75)
    q1 = np.quantile(distribution, .25)
    iqr = q3 - q1
    minimum = q1 - 1.5 * (iqr)
    maximum = q3 + 1.5 * (iqr)

    if(plot):
        plot_rae(distribution)

    input = pd.concat([dataset.T, distribution], axis=1)

    input = input.loc[(input['ratio'] < minimum) | (input['ratio'] > maximum)]

    top_15 = input.nlargest(15, "ratio")
    bottom_15 = input.nsmallest(15, "ratio")
    input = pd.concat([top_15, bottom_15])


    input = input.drop(['ratio'], axis=1)

    input = input.T

    return input

def shapley_values(dataset, labels, plot):
    reductionSize = len(dataset)/2
    if(reductionSize >= len(dataset.columns)):
        reductionSize = len(dataset.columns)/2
    model = xgboost.XGBClassifier(silence=1)
    model.fit(dataset, labels)
    shap.initjs()
    shap_values = shap.TreeExplainer(model).shap_values(dataset)
    distribution = np.absolute(shap_values)
    distribution = distribution.sum(axis=0)
    if(plot):
        plot_shap(shap_values, dataset, distribution)
    index = np.argpartition(distribution, -30)[-30:]
    slice = dataset.iloc[:,index]
    return slice

def principal_component_analysis(dataset, labels, plot):
    X_pca = sklearn.decomposition.PCA().fit_transform(dataset)
    X_selected = X_pca[:,:250]
    if(plot):
        plot_pca(X_pca, labels)
    return X_selected

def plot_pca(X_pca, labels):
        plt.figure(figsize=(10,5))
        plt.title('PCA - Genetic Variant Dataset')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.scatter(X_pca[:,1],X_pca[:,2],c=labels)
        plt.show()
        return

def plot_rae(distribution):
        plt.title('Rare Allele Enrichment Approach')
        plt.xlabel('Difference Value')
        plt.ylabel('Frequency')
        plt.boxplot(distribution, 0,'rD', 0)
        hist = distribution.hist(bins=100)
        plt.show()
        return

def plot_shap(shap_values, dataset, distribution):
        shap.summary_plot(shap_values, features=dataset, feature_names=dataset.columns)
        plt.title('Summed Shap Values Plot')
        plt.ylabel('Shap Values')
        plt.xlabel('Feature')
        plot = plt.plot(distribution)
        plt.show()
        return
