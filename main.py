import numpy as np
import sklearn
import parser
import feature_selection
import classification
import validation
import data_generator
import statistics

def main():

    opt = parser.get_parser()
    sample_number = opt.sample
    feature_number = opt.feature
    balance_ratio = opt.balance
    betaglobin_index = 10
    number_of_spikes = opt.spikes
    combination = opt.combination
    tuning = opt.tuning
    test_type = opt.test_type
    threshold = opt.threshold
    iterations = opt.iterations
    my_functions = opt.my_functions

    plot_rae = 0
    plot_shap = 0
    plot_pca = 0
    seed = np.random.randint(0, 1000000)

    if(test_type == 1):
        number_of_spikes = [0, 1, 5, 10]
        label_array = ['0 Spikes', '1 Spike', '5 Spikes', '10 Spikes']
        iterations = len(label_array)
    elif(test_type == 2):
        feature_number = [200, 2000, 20000, 200000]
        label_array = ['200 features', '2000 features', '20000 features', '200000 features']
        iterations = len(label_array)
    elif(test_type == 3):
        sample_number = [50, 100, 500, 1000]
        label_array = ['50 samples', '100 samples', '500 samples', '1000 samples']
        iterations = len(label_array)
    elif(test_type == 4):
        balance_ratio = [0.25, 0.50, 0.75]
        label_array = ['Unbalanced control-25/disease-75', 'Balanced conytol-50/disease-50', 'Unbalanced control-75/disease-25']
        iterations = len(label_array)

    if(test_type == 0):

        acc_1 = []
        acc_2 = []
        sens_1 = []
        sens_2 = []
        spec_1 = []
        spec_2 = []

        for i in range(0, iterations):
            # Generates new synthetic dataset based on the dataset parameters listed above
            print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
            dataset, labels, spike_indexes = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes, betaglobin_index)
            print("Dataset Created")

            print("Feature Selection Module Begins")

            print("Feature Selection: RAE")
            dataset_rae = feature_selection.rare_allele_enrichment(dataset, balance_ratio, plot_rae)
            print("RAE Dataset shape: (" + str(dataset_rae.shape[0]) + ", " + str(dataset_rae.shape[1]) + ")")

            #print("Feature Selection: PCA")
            #dataset_pca = feature_selection.principal_component_analysis(dataset, labels, plot_pca)
            #print("PCA Dataset shape: (" + str(dataset_pca.shape[0]) + ", " + str(dataset_pca.shape[1]) + ")")

            #print("Feature Selection: SHAP")
            #dataset_shap = feature_selection.shapley_values(dataset, labels, plot_shap)
            #print("RAE Dataset shape: (" + str(dataset_shap.shape[0]) + ", " + str(dataset_shap.shape[1]) + ")")

            print("Feature Selection Finished")

            print("Splitting Dataset Into Train and Test Sets")
            x_train_rae, x_test_rae, y_train_rae, y_test_rae = sklearn.model_selection.train_test_split(dataset_rae, labels, test_size = .25, random_state = seed, shuffle = True)
            #x_train_pca, x_test_pca, y_train_pca, y_test_pca = sklearn.model_selection.train_test_split(dataset_pca, labels, test_size = .25, random_state = seed, shuffle = True)
            #x_train_shap, x_test_shap, y_train_shap, y_test_shap = sklearn.model_selection.train_test_split(dataset_shap, labels, test_size = .25, random_state = seed, shuffle = True)
            print("Dataset Splitting Finished")

            print("Classification Module Begin")
            model_1_rae, model_2_rae = classification.train_model(x_train_rae, y_train_rae, combination, tuning)
            #model_1_pca, model_2_pca = classification.train_model(x_train_pca, y_train_pca, combination, tuning)
            #model_1_shap, model_2_shap = classification.train_model(x_train_shap, y_train_shap, combination, tuning)
            print("Classification Module Finished")

            print("Validation Module Begins")
            fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae = validation.validate_model(model_1_rae, x_test_rae, y_test_rae, threshold, my_functions)
            fpr_2_rae, tpr_2_rae, roc_score_2_rae, accuracy_2_rae, sensitivity_2_rae, specificity_2_rae = validation.validate_model(model_2_rae, x_test_rae, y_test_rae, threshold, my_functions)
            print("Validation Module Finished")

            acc_1.append(accuracy_1_rae)
            acc_2.append(accuracy_2_rae)
            sens_1.append(sensitivity_1_rae)
            sens_2.append(sensitivity_2_rae)
            spec_1.append(specificity_1_rae)
            spec_2.append(specificity_2_rae)

        print()

    else:
        tpr_1 = []
        fpr_1 = []
        roc_1 = []
        tpr_2 = []
        fpr_2 = []
        roc_2 = []

        for i in range(0, iterations):

            if(test_type == 1):
                # Generates new synthetic dataset based on the dataset parameters listed above
                print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes[i]) + " Spikes")
                dataset, labels, spike_indexes = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes[i], betaglobin_index)
                print("Dataset Created")
            elif(test_type == 2):
                # Generates new synthetic dataset based on the dataset parameters listed above
                print("Generating Dataset: " + str(sample_number[i]) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
                dataset, labels, spike_indexes = data_generator.createDataset(sample_number[i], feature_number, balance_ratio, number_of_spikes, betaglobin_index)
                print("Dataset Created")
            elif(test_type == 3):
                # Generates new synthetic dataset based on the dataset parameters listed above
                print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number[i]) + " Features, " + str(number_of_spikes) + " Spikes")
                dataset, labels, spike_indexes = data_generator.createDataset(sample_number, feature_number[i], balance_ratio, number_of_spikes, betaglobin_index)
                print("Dataset Created")
            elif(test_type == 4):
                # Generates new synthetic dataset based on the dataset parameters listed above
                print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
                dataset, labels, spike_indexes = data_generator.createDataset(sample_number, feature_number, balance_ratio[i], number_of_spikes, betaglobin_index)
                print("Dataset Created")

            print("Feature Selection Module Begins")

            print("Feature Selection: RAE")
            dataset_rae = feature_selection.rare_allele_enrichment(dataset, balance_ratio, plot_rae)
            print("RAE Dataset shape: (" + str(dataset_rae.shape[0]) + ", " + str(dataset_rae.shape[1]) + ")")

            #print("Feature Selection: PCA")
            #dataset_pca = feature_selection.principal_component_analysis(dataset, labels, plot_pca)
            #print("PCA Dataset shape: (" + str(dataset_pca.shape[0]) + ", " + str(dataset_pca.shape[1]) + ")")

            #print("Feature Selection: SHAP")
            #dataset_shap = feature_selection.shapley_values(dataset, labels, plot_shap)
            #print("RAE Dataset shape: (" + str(dataset_shap.shape[0]) + ", " + str(dataset_shap.shape[1]) + ")")

            print("Feature Selection Finished")

            print("Splitting Dataset Into Train and Test Sets")
            x_train_rae, x_test_rae, y_train_rae, y_test_rae = sklearn.model_selection.train_test_split(dataset_rae, labels, test_size = .25, random_state = seed, shuffle = True)
            #x_train_pca, x_test_pca, y_train_pca, y_test_pca = sklearn.model_selection.train_test_split(dataset_pca, labels, test_size = .25, random_state = seed, shuffle = True)
            #x_train_shap, x_test_shap, y_train_shap, y_test_shap = sklearn.model_selection.train_test_split(dataset_shap, labels, test_size = .25, random_state = seed, shuffle = True)
            print("Dataset Splitting Finished")

            print("Classification Module Begin")
            model_1_rae, model_2_rae = classification.train_model(x_train_rae, y_train_rae, combination, tuning)
            #model_1_pca, model_2_pca = classification.train_model(x_train_pca, y_train_pca, combination, tuning)
            #model_1_shap, model_2_shap = classification.train_model(x_train_shap, y_train_shap, combination, tuning)
            print("Classification Module Finished")

            print("Validation Module Begins")
            fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae = validation.validate_model(model_1_rae, x_test_rae, y_test_rae, threshold, my_functions)
            fpr_2_rae, tpr_2_rae, roc_score_2_rae, accuracy_2_rae, sensitivity_2_rae, specificity_2_rae = validation.validate_model(model_2_rae, x_test_rae, y_test_rae, threshold, my_functions)
            print("Validation Module Finished")

            fpr_1.append(fpr_1_rae)
            tpr_1.append(tpr_1_rae)
            roc_1.append(roc_score_1_rae)
            fpr_2.append(fpr_2_rae)
            tpr_2.append(tpr_2_rae)
            roc_2.append(roc_score_2_rae)

        validation.plot_roc_curve(fpr_1, tpr_1, roc_1, label_array, 'ROC-AUC Curve for RAE & Random Forest', 'rf_rae_spike')
        validation.plot_roc_curve(fpr_2, tpr_2, roc_2, label_array, 'ROC-AUC Curve for RAE & Gradient Boosting', 'gdb_rae_spike')

    print("Run Finished")

if __name__ == '__main__':
    main()
