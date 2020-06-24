import matplotlib.pyplot as plt
import sklearn

def majority_voting(vote_1, vote_2, vote_3):
    votes = [vote_1, vote_2, vote_3]
    bestClassifier = most_frequent(votes, len(votes))
    return bestClassifier

def most_frequent(arr, n):

    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]

    return res

## PREDICTIONS ##
def assign_class(pred, threshold):
    temp = []
    for prediction in pred:
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
    predictions = assign_class(pred, threshold)
    tn, tp, fn, fp = my_confusion_matrix(true, predictions)
    cp = tp + fn
    cn = fp + tn
    tpr.append(tp/cp)
    fpr.append(1 - tn/cn)
  return tpr, fpr

def my_confusion_matrix(true, pred):
  tn = 0
  tp = 0
  fn = 0
  fp = 0
  for i in range(len(pred)):
    if pred[i] == 0 and true[i] == 0:
      tn = tn + 1
    elif pred[i] == 1 and true[i] == 1:
      tp = tp + 1
    elif pred[i] == 0 and true[i] == 1:
      fn = fn + 1
    elif pred[i] == 1 and true[i] == 0:
      fp = fp + 1
    else:
      print("Invalid combination")
  return tn, tp, fn, fp

def make_thresholds(n):
  thresholds = []
  # Creates threshold list from 0/n, 1/n, 2/n, .... to n/n
  for i in range(n + 1):
    thresholds.append(i/n)
  return thresholds

def validate_model(model, x_test, y_test, threshold, my_functions):

    prediction_proba = model.predict_proba(x_test)
    prediction_proba = prediction_proba[:, 1]

    prediction = assign_class(prediction_proba, threshold)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, prediction_proba)

    roc_score = sklearn.metrics.roc_auc_score(y_test, prediction_proba)

    accuracy = sklearn.metrics.accuracy_score(y_test, prediction)
    cm = sklearn.metrics.confusion_matrix(y_test, prediction)

    sensitivity = cm[0][0]/ (cm[0][0] + cm[1][0])
    specificity = cm[1][1]/ (cm[1][1] + cm[0][1])

    if(my_functions):
        thresholds = make_thresholds(1000)
        tpr, fpr = my_roc_curve(y_test, prediction_proba, thresholds)
        tn, tp, fn, fp = my_confusion_matrix(y_test, prediction)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

    return fpr, tpr, roc_score, accuracy, sensitivity, specificity

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

def calculateBestScore(model_array, x_test, y_test, threshold, my_functions):
        fpr_array = []
        tpr_array = []
        roc_array = []
        accuracy_array = []
        sensitivity_array = []
        specificity_array = []

        for i in range(0, len(model_array)):
            fpr, tpr, roc_score, accuracy, sensitivity, specificity = validate_model(model_array[i], x_test[i], y_test[i], threshold, my_functions)
            fpr_array.append(fpr)
            tpr_array.append(tpr)
            roc_array.append(roc_score)
            accuracy_array.append(accuracy)
            sensitivity_array.append(sensitivity)
            specificity_array.append(specificity)

        return fpr_array, tpr_array, roc_array, accuracy_array, sensitivity_array, specificity_array

def plot_roc_curve(fpr_array, tpr_array, roc_score_array, label_array, title, fig_name):
    colors = ['orange', 'red', 'green', 'purple', 'yellow']
    for i in range(0, len(tpr_array)):
        plt.plot(fpr_array[i], tpr_array[i], color=colors[i], label=label_array[i] +', auc=' + str(round(roc_score_array[i], 3)))
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
