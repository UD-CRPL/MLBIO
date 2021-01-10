import numpy as np
import sklearn
import parser
import feature_selection
import classification
import validation
import data_generator
import statistics


def pvalue_distribution(sampleNumber, featureNumber, balanceRatio, betaglobinIndex, iterations):
    mylist = []
    for i in range (0, iterations):
        if (i % 10 == 0):
            print(i)
        dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sampleNumber, featureNumber, balanceRatio, 0, betaglobinIndex, [], [])
        dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobinIndex, 0.03)
        mylist.append(pvalue_threshold)
    import csv
    with open('pvalue_distribution.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(mylist)
     return

def cross_validation_pipeline(sampleNumber, featureNumber, balanceRatio, betaglobinIndex, k):
    from sklearn.model_selection import KFold
    print("Creating dataset")
    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sampleNumber, featureNumber, balanceRatio, 0, betaglobinIndex, [], [])
    labels = np.array(labels)
    print("Feature selection")
    dataset, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobinIndex, 0.05)
    kf = KFold(n_splits=k, shuffle=True)
    fpr_array = []
    tpr_array = []
    roc_array = []
    label_array = ['0 Spikes', '1 Spike', '5 Spikes', '10 Spikes']
    for train, test in kf.split(dataset):
        print("Kfold")
        print("%s %s" % (train, test))
        x_train, x_test = dataset.iloc[train], dataset.iloc[test]
        y_train, y_test = labels[train], labels[test]
        model_1_rae, model_2_rae = classification.train_model(x_train, y_train, 0, 0)
        fpr_1_rae, tpr_1_rae, roc_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae, rf_pred = validation.validate_model(model_1_rae, x_test, y_test, 0.7, 1)
        fpr_array.append(fpr_1_rae)
        tpr_array.append(tpr_1_rae)
        roc_array.append(roc_1_rae)
        validation.plot_roc_curve(fpr_1_rae, tpr_1_rae, roc_1_rae, label_array, 'ROC-AUC Curve for Random Forest', 'rf_rae_spike', 0)

    fpr_array = np.array(fpr_array)
    tpr_array = np.array(tpr_array)
    roc_array = np.array(roc_array)

    final_fpr = np.average(fpr_array, axis=0)
    final_tpr = np.average(tpr_array, axis=0)
    final_roc = np.average(roc_array, axis=0)
    validation.plot_roc_curve(final_fpr, final_tpr, final_roc, label_array, 'ROC-AUC Curve for Random Forest', 'rf_rae_spike', 0)



def final_test(sampleNumber, featureNumber, balanceRatio, betaglobinIndex, seed, threshold, my_functions, subtract, combination, tuning):
    tpr_1 = []
    fpr_1 = []
    roc_1 = []
    tpr_2 = []
    fpr_2 = []
    roc_2 = []
    label_array = ['0 Spikes', '1 Spike', '5 Spikes', '10 Spikes']
    pvalue_threshold = 0
    spike_list = []
    # First dataset
    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sampleNumber, featureNumber, balanceRatio, 0, betaglobinIndex, [], [])
    print("first dataset created")
    for i in range(0, 4):
        # Inject dataset depending on
        print('dataset injecting:' + str(i))
        dataset, new_spikes  = inject_dataset(dataset, i, sampleNumber, featureNumber, balanceRatio, betaglobinIndex)
        spike_list.append(new_spikes)
        print("Feature selection")
        if(i == 0):
            #dataset_rae = feature_selection.rare_allele_enrichment(dataset, balanceRatio, 0)
            dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobinIndex, 0.03)
        else:
            #dataset_rae = feature_selection.rare_allele_enrichment(dataset, balanceRatio, 0)
            dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobinIndex, pvalue_threshold)
        print("Feature selection done")

        print("Splitting Dataset Into Train and Test Sets")
        x_train_rae, x_test_rae, y_train_rae, y_test_rae = sklearn.model_selection.train_test_split(dataset_rae, labels, test_size = .25, random_state = seed, shuffle = True)
        print("Dataset Splitting Finished")

        print("Classification Module Begin")
        model_1_rae, model_2_rae = classification.train_model(x_train_rae, y_train_rae, combination, tuning)
        print("Classification Module Finished")

        print("Validation Module Begins")
        fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae, prediction = validation.validate_model(model_1_rae, x_test_rae, y_test_rae, threshold, my_functions)
        fpr_2_rae, tpr_2_rae, roc_score_2_rae, accuracy_2_rae, sensitivity_2_rae, specificity_2_rae, prediction = validation.validate_model(model_2_rae, x_test_rae, y_test_rae, threshold, my_functions)
        print("Validation Module Finished")

        fpr_1.append(fpr_1_rae)
        tpr_1.append(tpr_1_rae)
        roc_1.append(roc_score_1_rae)
        fpr_2.append(fpr_2_rae)
        tpr_2.append(tpr_2_rae)
        roc_2.append(roc_score_2_rae)

    print(spike_list)
    print("FINAL PVALUE THRESHOLD: " + str(pvalue_threshold))
    validation.plot_roc_curve(fpr_1, tpr_1, roc_1, label_array, 'ROC-AUC Curve for Random Forest', './results_0826/new_rf_rae_spike' + str(sampleNumber) + '_' + str(balanceRatio), subtract)
    validation.plot_roc_curve(fpr_2, tpr_2, roc_2, label_array, 'ROC-AUC Curve for Gradient Boosting', './results_0826/new_gdb_rae_spike' + str(sampleNumber) + '_' + str(balanceRatio), subtract)

    return

def inject_dataset(dataset, iteration, sampleNumber, featureNumber, balanceRatio, betaglobinIndex):
    betaglobin = [2] * sampleNumber
    if(iteration == 0):
        return dataset, []
    elif(iteration == 1):
        spikedArray = data_generator.generateSpikes(sampleNumber, balanceRatio, 1)
        dataset, spikeIndexes = data_generator.insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, [])
        return dataset, spikeIndexes
    elif(iteration == 2):
        spikedArray = data_generator.generateSpikes(sampleNumber, balanceRatio, 4)
        dataset, spikeIndexes = data_generator.insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, [])
        return dataset, spikeIndexes
    elif(iteration == 3):
        spikedArray = data_generator.generateSpikes(sampleNumber, balanceRatio, 5)
        dataset, spikeIndexes = data_generator.insertSpike(dataset, spikedArray, betaglobin, featureNumber, betaglobinIndex, [])
        return dataset, spikeIndexes

def main():

    final_fpr_1 = []
    final_tpr_1 = []
    final_roc_1 = []

    final_fpr_nofs = []
    final_tpr_nofs = []
    final_roc_nofs = []

    for i in range(0, 5):

        print("ITERATION NUMBER: " + str(i))
        fpr_1, tpr_1, roc_1, fpr_nofs, tpr_nofs, roc_nofs = real_main()

        final_fpr_1.append(fpr_1)
        final_tpr_1.append(tpr_1)
        final_roc_1.append(roc_1)

        final_fpr_nofs.append(fpr_nofs)
        final_tpr_nofs.append(tpr_nofs)
        final_roc_nofs.append(roc_nofs)


    new_final_fpr_1 = np.average(final_fpr_1, axis=0)
    new_final_tpr_1 = np.average(final_tpr_1, axis=0)
    new_final_roc_1 = np.average(final_roc_1, axis=0)

    new_final_fpr_nofs = np.average(final_fpr_nofs, axis=0)
    new_final_tpr_nofs = np.average(final_tpr_nofs, axis=0)
    new_final_roc_nofs = np.average(final_roc_nofs, axis=0)

    np.savetxt("nofs_fpr.csv", new_final_fpr_nofs, delimiter=",")
    np.savetxt("nofs_tpr.csv", new_final_tpr_nofs, delimiter=",")
    np.savetxt("nofs_roc.csv", new_final_roc_nofs, delimiter=",")

    label_array = ['0 Spikes', '1 Spike', '5 Spikes', '10 Spikes']
    validation.plot_roc_curve(new_final_fpr_1, new_final_tpr_1, new_final_roc_1, label_array, 'ROC-AUC Curve for Random Forest', '.', 0)
    validation.plot_roc_curve(new_final_fpr_nofs, new_final_tpr_nofs, new_final_roc_nofs, label_array, 'No Feature Selection: ROC-AUC Curve for Random Forest', '.', 0)


def real_main():

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
    robustness = opt.robustness
    contingency = opt.contingency
    subtract = opt.subtract
    test_random_data = opt.test_random_data
    final_testing = opt.final_testing

    plot_rae = 0
    plot_shap = 0
    plot_pca = 0
    seed = np.random.randint(0, 1000000)

    if(final_testing):
        #pvalue_distribution(sample_number, feature_number, balance_ratio, betaglobin_index, 1000)
        #final_test(sample_number, feature_number, balance_ratio, betaglobin_index, seed, threshold, my_functions, subtract, combination, tuning)
        cross_validation_pipeline(sample_number, feature_number, balance_ratio, betaglobin_index, 5)
    else:
        if(test_type == 1):
            number_of_spikes = [0, 1, 5, 10]
            label_array = ['0 Spikes', '1 Spike', '5 Spikes', '10 Spikes']
            #number_of_spikes = [0, 5]
            #label_array = ['0 Spikes', '5 Spikes']
            iterations = len(label_array)
            subtract = opt.subtract
        elif(test_type == 2):
            sample_number = [50, 100, 500, 1000]
            label_array = ['50 samples', '100 samples', '500 samples', '1000 samples']
            iterations = len(label_array)
        elif(test_type == 3):
            feature_number = [100, 1000, 5000, 10000]
            label_array = ['100 features', '1000 features', '5000 features', '10000 features']
            iterations = len(label_array)
        elif(test_type == 4):
            balance_ratio = [0.25, 0.50, 0.75]
            label_array = ['Unbalanced control-25/disease-75', 'Balanced control-50/disease-50', 'Unbalanced control-75/disease-25']
            iterations = len(label_array)

        if(test_type == 0):

            acc_1 = []
            acc_2 = []
            sens_1 = []
            sens_2 = []
            spec_1 = []
            spec_2 = []
            spiked_array = []
            spike_indexes = []

            for i in range(0, iterations):
                # Generates new synthetic dataset based on the dataset parameters listed above
                print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
                # Test robustness of model, saves the set of spikes and then changes the randomness (other values) of the data and validates
                if robustness:
                    if i == 0:
                        dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes, betaglobin_index, [], [])
                    else:
                        dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes, betaglobin_index, spiked_array, spike_indexes)
                    print(spike_indexes)
                # Test a random dataset with nothing added to it
                elif test_random_data:
                    dataset, labels = data_generator.random_data((sample_number, feature_number), balance_ratio)
                # Regular run
                else:
                    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes, betaglobin_index, [], [])
                print("Dataset Created")

                print("Feature Selection Module Begins")

                print("Feature Selection: RAE")
                dataset_nofs = dataset
                dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobin_index, 0.05)
                #dataset_rae = feature_selection.rare_allele_enrichment(dataset, balance_ratio, plot_rae)
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
                x_train_nofs, x_test_nofs, y_train_nofs, y_test_nofs = sklearn.model_selection.train_test_split(dataset_nofs, labels, test_size = .25, random_state = seed, shuffle = True)
                #x_train_pca, x_test_pca, y_train_pca, y_test_pca = sklearn.model_selection.train_test_split(dataset_pca, labels, test_size = .25, random_state = seed, shuffle = True)
                #x_train_shap, x_test_shap, y_train_shap, y_test_shap = sklearn.model_selection.train_test_split(dataset_shap, labels, test_size = .25, random_state = seed, shuffle = True)
                print("Dataset Splitting Finished")

                print("Classification Module Begin")
                model_1_rae, model_2_rae = classification.train_model(x_train_rae, y_train_rae, combination, tuning)
                model_1_nofs, model_2_nofs = classification.train_model(x_train_nofs, y_train_nofs, combination, tuning)
                #model_1_pca, model_2_pca = classification.train_model(x_train_pca, y_train_pca, combination, tuning)
                #model_1_shap, model_2_shap = classification.train_model(x_train_shap, y_train_shap, combination, tuning)
                print("Classification Module Finished")

                print("Validation Module Begins")
                fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae, rf_pred = validation.validate_model(model_1_rae, x_test_rae, y_test_rae, threshold, my_functions)
                #fpr_2_rae, tpr_2_rae, roc_score_2_rae, accuracy_2_rae, sensitivity_2_rae, specificity_2_rae, xgb_pred = validation.validate_model(model_2_rae, x_test_rae, y_test_rae, threshold, my_functions)
                fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae, rf_pred = validation.validate_model(model_1_nofs, x_test_nofs, y_test_nofs, threshold, my_functions)
                if(contingency):
                    if(i == 0):
                        contingency_tables = validation.contingency_matrix_test(x_test_rae, rf_pred)
                        print(contingency_tables[0])
                        validation.plot_matrix(contingency_tables)
                print("Validation Module Finished")

                if test_random_data == 1:
                    if(i == 0):
                        validation.plot_roc_curve_random(fpr_1_rae, tpr_1_rae, roc_score_1_rae, 'ROC-AUC Curve for Random Forest', 'rf_rae_rand')
                        validation.plot_roc_curve_random(fpr_2_rae, tpr_2_rae, roc_score_2_rae, 'ROC-AUC Curve for Gradient Boosting', 'xgb_rae_rand')

                acc_1.append(accuracy_1_rae)
                acc_2.append(accuracy_2_rae)
                sens_1.append(sensitivity_1_rae)
                sens_2.append(sensitivity_2_rae)
                spec_1.append(specificity_1_rae)
                spec_2.append(specificity_2_rae)
                acc_nofs.append(accuracy_1_nofs)
                sens_nofs.append(sensitivity_1_nofs)
                spec_nofs.append(specificity_1_nofs)


                validation.plot_roc_curve_random(fpr_1_rae, tpr_1_rae, roc_score_1_rae, 'ROC-AUC Curve for Random Forest', 'rf_rae_rand')
                #validation.plot_roc_curve_random(fpr_2_rae, tpr_2_rae, roc_score_2_rae, 'ROC-AUC Curve for Gradient Boosting', 'xgb_rae_rand')

            if(robustness):
                print(acc_1)
                print(sens_1)
                print(spec_1)

        else:
            tpr_1 = []
            fpr_1 = []
            roc_1 = []
            tpr_2 = []
            fpr_2 = []
            roc_2 = []
            fpr_nofs = []
            tpr_nofs = []
            roc_nofs = []

            pvalue_threshold = 0

            for i in range(0, iterations):

                if(test_type == 1):
                    # Generates new synthetic dataset based on the dataset parameters listed above
                    print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes[i]) + " Spikes")
                    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio, number_of_spikes[i], betaglobin_index, [], [])
                    #dataset, labels, spike_indexes = data_generator.create_dataset(sample_number, feature_number, balance_ratio, number_of_spikes[i], betaglobin_index)
                    print("Dataset Created")
                elif(test_type == 2):
                    # Generates new synthetic dataset based on the dataset parameters listed above
                    print("Generating Dataset: " + str(sample_number[i]) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
                    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number[i], feature_number, balance_ratio, number_of_spikes, betaglobin_index, [], [])
                    # Dataset used to find the pvalue threshold
                    random_dataset, random_labels, random_spike_indexes, random_spiked_array = data_generator.createDataset(sample_number[i], feature_number, balance_ratio, 0, betaglobin_index, [], [])
                    print("Dataset Created")
                elif(test_type == 3):
                    # Generates new synthetic dataset based on the dataset parameters listed above
                    print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number[i]) + " Features, " + str(number_of_spikes) + " Spikes")
                    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number[i], balance_ratio, number_of_spikes, betaglobin_index, [], [])
                    # Dataset used to find the pvalue threshold
                    random_dataset, random_labels, random_spike_indexes, random_spiked_array = data_generator.createDataset(sample_number, feature_number[i], balance_ratio, 0, betaglobin_index, [], [])
                    print("Dataset Created")
                elif(test_type == 4):
                    # Generates new synthetic dataset based on the dataset parameters listed above
                    print("Generating Dataset: " + str(sample_number) + " Samples, " + str(feature_number) + " Features, " + str(number_of_spikes) + " Spikes")
                    dataset, labels, spike_indexes, spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio[i], number_of_spikes, betaglobin_index, [], [])
                    # Dataset used to find the pvalue threshold
                    random_dataset, random_labels, random_spike_indexes, random_spiked_array = data_generator.createDataset(sample_number, feature_number, balance_ratio[i], 0, betaglobin_index, [], [])
                    print("Dataset Created")

                print("Feature Selection Module Begins")

                print("Feature Selection: RAE")
                dataset_nofs = dataset
                #random_dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(random_dataset, random_labels, 0, 0, betaglobin_index, 0.03)
                print("PVALUE THRESHOLD: " + str(pvalue_threshold))
                #if(i > 0):
                dataset_rae, pvalue_threshold = feature_selection.rare_allele_enrichment_or(dataset, labels, 0, 0, betaglobin_index, 0.05)
                #    dataset_rae = feature_selection.rare_allele_enrichment(dataset, balance_ratio[i], plot_rae)
                #else:
                #    dataset_rae = dataset.sample(n = 30, axis = 1)
                #print(spike_indexes)
                print("RAE Dataset shape: (" + str(dataset_rae.shape[0]) + ", " + str(dataset_rae.shape[1]) + ")")

                #print(spike_indexes)
                #print(dataset_rae.columns)

                #print("Feature Selection: PCA")
                #ataset_pca = feature_selection.principal_component_analysis(dataset, labels, plot_pca)
                #print("PCA Dataset shape: (" + str(dataset_pca.shape[0]) + ", " + str(dataset_pca.shape[1]) + ")")

                #print("Feature Selection: SHAP")
                #dataset_shap = feature_selection.shapley_values(dataset, labels, plot_shap)
                #print("RAE Dataset shape: (" + str(dataset_shap.shape[0]) + ", " + str(dataset_shap.shape[1]) + ")")

                print("Feature Selection Finished")

                print("Splitting Dataset Into Train and Test Sets")
                x_train_rae, x_test_rae, y_train_rae, y_test_rae = sklearn.model_selection.train_test_split(dataset_rae, labels, test_size = .25, random_state = seed, shuffle = True)
                x_train_nofs, x_test_nofs, y_train_nofs, y_test_nofs = sklearn.model_selection.train_test_split(dataset_nofs, labels, test_size = .25, random_state = seed, shuffle = True)
                #x_train_pca, x_test_pca, y_train_pca, y_test_pca = sklearn.model_selection.train_test_split(dataset_pca, labels, test_size = .25, random_state = seed, shuffle = True)
                #x_train_shap, x_test_shap, y_train_shap, y_test_shap = sklearn.model_selection.train_test_split(dataset_shap, labels, test_size = .25, random_state = seed, shuffle = True)
                print("Dataset Splitting Finished")

                print("Classification Module Begin")
                model_1_rae, model_2_rae = classification.train_model(x_train_rae, y_train_rae, combination, tuning)
                model_1_nofs, model_2_nofs = classification.train_model(x_train_nofs, y_train_nofs, combination, tuning)
                #model_1_pca, model_2_pca = classification.train_model(x_train_pca, y_train_pca, combination, tuning)
                #model_1_shap, model_2_shap = classification.train_model(x_train_shap, y_train_shap, combination, tuning)
                print("Classification Module Finished")

                print("Validation Module Begins")
                fpr_1_rae, tpr_1_rae, roc_score_1_rae, accuracy_1_rae, sensitivity_1_rae, specificity_1_rae, prediction = validation.validate_model(model_1_rae, x_test_rae, y_test_rae, threshold, my_functions)
                fpr_2_rae, tpr_2_rae, roc_score_2_rae, accuracy_2_rae, sensitivity_2_rae, specificity_2_rae, prediction = validation.validate_model(model_2_rae, x_test_rae, y_test_rae, threshold, my_functions)
                fpr_1_nofs, tpr_1_nofs, roc_score_1_nofs, accuracy_1_nofs, sensitivity_1_nofs, specificity_1_nofs, prediction = validation.validate_model(model_1_nofs, x_test_nofs, y_test_nofs, threshold, my_functions)
                #fpr_1_pca, tpr_1_pca, roc_score_1_pca, accuracy_1_pca, sensitivity_1_pca, specificity_1_pca, prediction = validation.validate_model(model_1_pca, x_test_pca, y_test_pca, threshold, my_functions)
                #fpr_2_pca, tpr_2_pca, roc_score_2_pca, accuracy_2_pca, sensitivity_2_pca, specificity_2_pca, prediction = validation.validate_model(model_2_pca, x_test_pca, y_test_pca, threshold, my_functions)
                #fpr_1_shap, tpr_1_shap, roc_score_1_shap, accuracy_1_shap, sensitivity_1_shap, specificity_1_shap, prediction = validation.validate_model(model_1_shap, x_test_shap, y_test_shap, threshold, my_functions)
                #fpr_2_shap, tpr_2_shap, roc_score_2_shap, accuracy_2_shap, sensitivity_2_shap, specificity_2_shap, prediction = validation.validate_model(model_2_shap, x_test_shap, y_test_shap, threshold, my_functions)
                print("Validation Module Finished")

                fpr_1.append(fpr_1_rae)
                tpr_1.append(tpr_1_rae)
                roc_1.append(roc_score_1_rae)
                fpr_2.append(fpr_2_rae)
                tpr_2.append(tpr_2_rae)
                roc_2.append(roc_score_2_rae)

                fpr_nofs.append(fpr_1_nofs)
                tpr_nofs.append(tpr_1_nofs)
                roc_nofs.append(roc_score_1_nofs)

                #fpr_1.append(fpr_1_pca)
                #tpr_1.append(tpr_1_pca)
                #roc_1.append(roc_score_1_pca)
                #fpr_2.append(fpr_2_pca)
                #tpr_2.append(tpr_2_pca)
                #roc_2.append(roc_score_2_pca)
                #fpr_1.append(fpr_1_shap)
                #tpr_1.append(tpr_1_shap)
                #roc_1.append(roc_score_1_shap)
                #fpr_2.append(fpr_2_shap)
                #tpr_2.append(tpr_2_shap)
                #roc_2.append(roc_score_2_shap)

            #validation.plot_roc_curve(fpr_1, tpr_1, roc_1, label_array, 'RAE: ROC-AUC Curve for Random Forest', 'rf_rae_spike', subtract)
            #validation.plot_roc_curve(fpr_2, tpr_2, roc_2, label_array, 'RAE: ROC-AUC Curve for Gradient Boosting', 'gdb_rae_spike', subtract)
            #validation.plot_roc_curve(fpr_nofs, tpr_nofs, roc_nofs, label_array, 'NO FEATURE SELECTION: ROC-AUC Curve for Random Forest', 'rf_nos_spike', subtract)

            return fpr_1, tpr_1, roc_1, fpr_nofs, tpr_nofs, roc_nofs

    print("Run Finished")

def contingency_table_test():
    true = [0, 1, 0, 1, 1, 0, 1, 0]
    pred = [0, 1, 0, 0, 0, 1, 1, 0]
    table = validation.my_contingency_matrix(true, pred, .70)

if __name__ == '__main__':
    main()
#contingency_table_test()
