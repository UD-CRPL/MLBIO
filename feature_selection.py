import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import shap
import sklearn
import xgboost
import data_generator

def testing(dataset, labels):

    numpy_labels = np.asarray(labels)
    columns = dataset.columns
    dataset_temp = dataset.copy()
    dataset_temp['labels'] = numpy_labels

    print(np.sum(numpy_labels == 0))
    variant_control_yes = dataset[(dataset_temp[columns[0]] == 0) & (dataset_temp['labels'] == 0)].count()[0]
    variant_control_no = dataset[(dataset_temp[columns[0]] != 0) & (dataset_temp['labels'] == 0)].count()[0]
    print(variant_control_yes)
    print(variant_control_no)
    print(dataset[(dataset_temp['labels'] == 0)].apply(pd.value_counts))

    return variant_control_yes

#dataset_test, labels_test, spike_indexes, spiked_array = data_generator.createDataset(500, 12, 0.5, 10, 10, [], [])
#feature_set = testing(dataset_test, labels_test)


def rare_allele_enrichment_or(dataset, labels, chi, rand, betaglobin, pvalue_threshold):

    if rand:
            selected_features = data_generator.generate_spike_indexes(dataset.shape[1], 30, betaglobin, [])
            dataset = dataset.T
            dataset = dataset.loc[selected_features]
            dataset = dataset.T
            return dataset
    else:

        numpy_labels = np.asarray(labels)
        columns = dataset.columns
        dataset_temp = dataset.copy()
        dataset_temp['labels'] = numpy_labels
        tables = []
        #number = 0
        #control_size = np.sum(numpy_labels == 0)
        #disease_size = np.sum(numpy_labels == 1)
        control_size = dataset[(dataset_temp['labels'] == 0)].apply(pd.value_counts)
        disease_size = dataset[(dataset_temp['labels'] == 1)].apply(pd.value_counts)
        control_size.fillna(0, inplace=True)
        disease_size.fillna(0, inplace=True)
        for feature in columns:
            variant_control_yes = control_size[feature].iloc[0]
            variant_control_no = control_size[feature].iloc[1] # + control_size[feature].iloc[2]
            variant_disease_yes = disease_size[feature].iloc[1]
            variant_disease_no = disease_size[feature].iloc[0]  # + disease_size[feature].iloc[1]
            #variant_control_yes = dataset[(dataset_temp[feature] == 0) & (dataset_temp['labels'] == 0)].count()[0]
            #variant_control_no = control_size - variant_control_yes
            #variant_disease_yes = dataset[(dataset_temp[feature] == 2) & (dataset_temp['labels'] == 1)].count()[0]
            #variant_disease_no = disease_size - variant_control_no
            control = [variant_control_no, variant_control_yes]
            disease = [variant_disease_yes, variant_disease_no]
            contigency_table = [control, disease]
            contigency_table = np.array([control, disease])
            tables.append(contigency_table)
            #number = number + 1
            #if number % 1000 == 0:
            #    print(number)

        if chi:
            chi2s = []
            pvalues = []
            dofs = []
            expected = []
            chi2, p, d, e = chi2_contingency(table)
            chi2s.append(chi2)
            pvalues.append(p)
            dofs.append(d)
            expected.append(e)

        else:
            odd_ratios = []
            pvalues = []
            for table in tables:
                #print(table)
                oddsratio, pvalue = stats.fisher_exact(table)
                odd_ratios.append(odd_ratios)
                pvalues.append(pvalue)

            dataset = dataset.T
            #print(len(pvalues))
            #print(dataset.shape)
            dataset['pvalues'] = pvalues
            new_dataset = dataset.drop(dataset[dataset['pvalues'] > pvalue_threshold].index)
            selected_features = new_dataset.nsmallest(30, 'pvalues')
            if(dataset.shape[1] == 0):
                new_dataset = dataset.iloc[0]
            #print(dataset.shape)
            if(pvalue_threshold == 0.05 or pvalue_threshold == 0.03):
                pvalue_threshold = selected_features['pvalues'].iloc[0]
            selected_features.drop('pvalues', axis=1, inplace=True)
            selected_features = selected_features.T
            # print(selected_features)
            return selected_features, pvalue_threshold




#dataset_test, labels_test, spike_indexes, spiked_array = data_generator.createDataset(500, 1000, 0.5, 10, 10, [], [])
#features = rare_allele_enrichment_or(dataset_test, labels_test, 0, 0, 10)

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

    if len(input) > 30:
        top_15 = input.nlargest(15, "ratio")
    #    print(top_15['ratio'])
        bottom_15 = input.nsmallest(15, "ratio")
    #    print(bottom_15['ratio'])
        input = pd.concat([top_15, bottom_15])

    #else:
    #    print(input['ratio'])

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
