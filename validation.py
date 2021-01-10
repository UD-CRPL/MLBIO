import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import sklearn

def contingency_matrix_test(dataset, labels):
    tables = []
    for column in dataset.columns:

        zero_disease = 0
        one_disease = 0
        two_disease = 0

        zero_control = 0
        one_control = 0
        two_control = 0

        label_index = 0

        for item in dataset[column].iteritems():
            if labels[label_index] == 1:
                if item[1] == 0:
                    zero_disease = zero_disease + 1
                elif item[1] == 1:
                    one_disease = one_disease + 1
                elif item[1] == 2:
                    two_disease = two_disease + 1
                else:
                    print("Weird value")
            if labels[label_index] == 0:
                if item[1] == 0:
                    zero_control = zero_control + 1
                elif item[1] == 1:
                    one_control = one_control + 1
                elif item[1] == 2:
                    two_control = two_control + 1
                else:
                    print("Weird value")

            label_index = label_index + 1
        disease_cohort = [zero_disease, one_disease, two_disease]
        control_cohort = [zero_control, one_control, one_control]
        contigency_table = [control_cohort, disease_cohort]
        tables.append(contigency_table)

        print(len(tables))
    return tables

def majority_voting(votes):
    # Chooses the classifier that has been voted the most
    bestClassifier = most_frequent(votes)
    return bestClassifier

def most_frequent(arr):
    # Insert all elements in Hash.
    hash = dict()
    for i in range(len(arr)):
        if arr[i] in hash.keys():
            hash[arr[i]] += 1
        else:
            hash[arr[i]] = 1
    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < hash[i]):
            res = i
            max_count = hash[i]
    return res

## PREDICTIONS ##
def assign_class(pred, threshold):
    temp = []
    # Iterates through each test observation
    for prediction in pred:
        # If the observation probability is higher than the threshold, assign label 1 to the observation, otherwise assign 0
        if prediction > threshold:
            temp.append(1)
        else:
            temp.append(0)
    return temp

def my_roc_curve(true, pred, thresholds):
  tpr = []
  fpr = []
  # Iterates through list of thresholds
  for threshold in thresholds:
    # Assigns labels to each test case
    predictions = assign_class(pred, threshold)
    # Finds the true positive, true negative, false negative, and false positive values
    tn, tp, fn, fp = my_confusion_matrix(true, predictions)
    cp = tp + fn
    cn = fp + tn
    # Calculates True Positive Rate and False Positive Rate, then adds to a list. The list will have the calculated TPR and FPR points for each threshold iterated
    tpr.append(tp/cp)
    fpr.append(1 - tn/cn)
  return tpr, fpr

def my_confusion_matrix(true, pred):
  tn = 0
  tp = 0
  fn = 0
  fp = 0
  # Iterates through all test observations
  for i in range(len(pred)):
      # the predicted label equals the ground truth label and they are both 0, adds to true negative
    if pred[i] == 0 and true[i] == 0:
      tn = tn + 1
      # the predicted label equals the ground truth label and they are both 1, adds to true positive
    elif pred[i] == 1 and true[i] == 1:
      tp = tp + 1
      # the predicted label does not equals the ground truth label and they are 0 and 1 respectively, adds to false negative
    elif pred[i] == 0 and true[i] == 1:
      fn = fn + 1
      # the predicted label does not equals the ground truth label and they are 1 and 0 respectively, adds to false positive
    elif pred[i] == 1 and true[i] == 0:
      fp = fp + 1
    else:
      print("Invalid combination")
  return tn, tp, fn, fp

def make_thresholds(n):
  thresholds = []
  # Creates threshold list from 0/n, 1/n, 2/n, .... to n/n
  for i in range(0, n + 1):
    thresholds.append(i/n)
  return thresholds

def validate_model(model, x_test, y_test, threshold, my_functions):

    # Assigns probability for each test case in the form of (Probability for class 0, probability for class 1)
    prediction_proba = model.predict_proba(x_test)
    # Grabs the probability for class 1
    prediction_proba = prediction_proba[:, 1]
    # Assigns class label to each test case based on a threshold
    prediction = assign_class(prediction_proba, threshold) # TABLES IN SLIDES
    # Calculates the False Positive Rate (FPR) and True Positive Rate (TPR) of the test case probability predictions
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, prediction_proba) # What's used to plot the ROC Curves
    # Calculates the Area Under the Curve score
    roc_score = sklearn.metrics.roc_auc_score(y_test, prediction_proba)
    # Calculates accuracy based on ground truth (y_test) and the assigned labels above
    accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
    # Creates a confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, prediction)

    sensitivity = cm[0][0]/ (cm[0][0] + cm[1][0]) # TP / (TP + FN)
    specificity = cm[1][1]/ (cm[1][1] + cm[0][1]) # TN / (TN + FP)

    # Uses my functions if given as an option
    if(my_functions):
        # Generates a List of thresholds that will be used for the ROC calculations
        thresholds = make_thresholds(10000)
        # Calculates the False Positive Rate (FPR) and True Positive Rate (TPR) of the test case probability predictions using the threshold list above
        tpr, fpr = my_roc_curve(y_test, prediction_proba, thresholds)
        # Gets the True Negative (TN), True Positive (TP), False Negative (FN), False Positive (FP) Values
        tn, tp, fn, fp = my_confusion_matrix(y_test, prediction)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

    return fpr, tpr, roc_score, accuracy, sensitivity, specificity, prediction



def plot_baseline():

    fpr = np.array([0., 0., 0., 0.015625, 0.015625, 0.03125,  0.03125,  0.046875,
 0.046875, 0.0625, 0.0625, 0.125, 0.125, 0.171875, 0.171875, 0.1875,
 0.1875,  0.203125, 0.203125, 0.234375, 0.234375, 0.25, 0.25, 0.28125,
 0.28125,  0.296875, 0.296875, 0.34375,  0.34375,  0.40625,  0.40625,  0.46875,
 0.46875,  0.515625, 0.515625, 0.53125,  0.53125,  0.5625,   0.5625,   0.59375,
 0.59375,  0.65625,  0.65625,  0.6875,   0.6875,   0.734375, 0.734375, 0.765625,
 0.765625, 0.78125,  0.78125,  0.921875, 0.921875, 1.]) # X in the 0 Spike Curves
    tpr = np.array([0., 0.01639344, 0.14754098, 0.14754098, 0.18032787, 0.18032787,
 0.27868852, 0.27868852, 0.32786885, 0.32786885, 0.39344262, 0.39344262,
 0.42622951, 0.42622951, 0.47540984, 0.47540984, 0.50819672, 0.50819672,
 0.52459016, 0.52459016, 0.54098361, 0.54098361, 0.59016393, 0.59016393,
 0.63934426, 0.63934426, 0.68852459, 0.68852459, 0.70491803, 0.70491803,
 0.75409836, 0.75409836, 0.7704918,  0.7704918,  0.78688525, 0.78688525,
 0.81967213, 0.81967213, 0.83606557, 0.83606557, 0.85245902, 0.85245902,
 0.8852459, 0.8852459,  0.90163934, 0.90163934, 0.91803279, 0.91803279,
 0.93442623, 0.93442623, 0.96721311, 0.96721311, 1., 1.]) # y in the 0 Spike Curve
    array = make_thresholds(len(fpr) - 1) # X and Y in the baseline (Array where points are i/len(fpr))

    distance_fpr = []
    distance_tpr = []
    for i in range(0, len(fpr)):
        #d_fpr = np.sqrt(abs(fpr[i] ** 2 - array[i] ** 2)
        #d_tpr = np.sqrt(abs(tpr[i]**2 - array[i]**2)) # y
        d_fpr = fpr[i] - array[i] # x1 - x2 Horizontal Distance
        d_tpr = tpr[i] - array[i] # y1 - y2 Vertical Distance
        c = np.sqrt(d_fpr ** 2 + d_tpr ** 2) # c = sqrt( (x1 - x2) ^ 2 + (y1 - y2) ^2)
        deltaxy = np.sqrt((c ** 2) // 2)
        distance_fpr.append(deltaxy)
        distance_tpr.append(deltaxy)

    #print(distance_fpr)
    fpr = fpr - distance_fpr # Xi - Horizontal Distance
    tpr = tpr - distance_tpr # Yi - Vertical Distance
    plt.plot(fpr, tpr, color='red') # Plots roc_1
    plt.plot(array, array, color='darkblue', linestyle='--') # Plots baseline
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('baseline')
    plt.legend()
    plt.show()

#plot_baseline()

def correction(array):
  prev_val = 100000
  for i in range(0, len(array)):
    if (array[i] < prev_val):
      prev_val = array[i]
    else:
      array[i] = prev_val
  return array

def plot_roc_curve(fpr_array, tpr_array, roc_score_array, label_array, title, fig_name, roc_subtract):
    colors = ['orange', 'red', 'green', 'purple', 'yellow']
    fpr_array = np.array(fpr_array)
    tpr_array = np.array(tpr_array)
    roc_score_array = np.array(roc_score_array)
    # Iterates through the dataset scenarios (different number of spikes, different number of features, etc)
    print(fpr_array[0])
    print(tpr_array[0])

    if roc_subtract:
        for i in range(0, len(tpr_array)):
            # plots the corresponding FPR and TPR values found for the dataset scenario, shifting the difference of the spiked 0 curve to the other ones
            corrected = correction(fpr_array[i] + (tpr_array[0] - fpr_array[0]))
            from sklearn.metrics import auc
            area = auc(tpr_array[i], corrected)
            plt.plot(corrected, tpr_array[i], color=colors[i], label=label_array[i] +', auc=' + str(round(1 - area, 3)))
    else:
        for i in range(0, len(tpr_array)):
            # plots the corresponding FPR and TPR values found for the dataset scenario
            plt.plot(fpr_array[i], tpr_array[i], color=colors[i], label=label_array[i] +', auc=' + str(round(roc_score_array[i], 3)))
            #plt.plot(fpr_array, tpr_array, color='orange', label='0 Spikes' +', auc=' + str(roc_score_array))
            #continue
    # (0.5 AUC Score, Random Guess) Line
    #plt.plot(fpr_array, tpr_array, color='orange', label='0 Spikes' +', auc=' + str(roc_score_array.round(3)))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    # Sets graph axis labels, title, legend table
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend()
    # Saves the graph as a png file
    plt.savefig(fig_name + '.png')
    plt.show()
    plt.clf()
    if(roc_subtract):
        return
    else:
        plot_roc_curve(fpr_array, tpr_array, roc_score_array, label_array, title, fig_name, 1)
    return

def plot_roc_curve_random(fpr_array, tpr_array, roc_score_array, title, fig_name):
    colors = ['orange', 'red', 'green', 'purple', 'yellow']
    plt.plot(fpr_array, tpr_array, color='orange', label= '0 Spikes, auc=' + str(round(roc_score_array, 3)))
    # (0.5 AUC Score, Random Guess) Line
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    # Sets graph axis labels, title, legend table
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend()
    plt.show()
    # Saves the graph as a png file
    plt.savefig(fig_name + '.png')
    plt.clf()
    return

def plot_matrix(tables):
    from statsmodels.graphics.mosaicplot import mosaic
    for table in tables:
        table = np.array(table)
        mosaic(data=table.T, gap=0.01, title='contingency table')
        plt.show()
        plt.clf()

def plot_prec_recall(model, x_test, y_test, pred, fig_name):
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, pred)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import plot_precision_recall_curve

    disp = plot_precision_recall_curve(classifier, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
    disp.savefig(fig_name + '.png')
    return
