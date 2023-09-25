import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
import scipy as sp
import scipy.stats
import matplotlib


def binary_AUC_plot(true_labels, pred_labels):
    # roc curve for classes
    fpr, tpr, thresh = roc_curve(true_labels, pred_labels)
    auc_plt = auc(fpr, tpr)

    # plotting
    #plt.figure(figsize=(5, 5))
    #plt.plot(fpr, tpr, linestyle='--', color='orange',
    #         label='IPF vs non-IPF ILDs : AUC = {0:0.2f}'.format(auc(fpr, tpr)))
    #plt.title(' ROC curve')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive rate')
    #plt.legend(loc='best')
    #plt.show()

    return auc_plt


def variance_corr_rfe(train_feat, val_feat, train_labels, val_labels, feature_names):
    # Mean Max Normalization
    norm = StandardScaler().fit(train_feat)
    train_feat = norm.transform(train_feat)
    val_feat = norm.transform(val_feat)

    # Constant thresholds
    variance_const = 0
    correlation_threshold = 0.95

    # Step one: Remove Constant Features
    v_threshold = VarianceThreshold(threshold=variance_const)
    v_threshold.fit(train_feat)
    selected = v_threshold.get_support()

    train_feat = train_feat[:, selected]
    val_feat = val_feat[:, selected]
    feature_names = feature_names[selected]

    # Step two: remove highly correlated features
    df = pd.DataFrame(data=train_feat)
    df = df.astype(float)
    val_df = pd.DataFrame(data=val_feat)
    val_df = val_df.astype(float)

    ## Create correlation matrix
    corr_matrix = df.corr(method='spearman').abs()
    ## Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    ## Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    ## Drop features
    df.drop(to_drop, axis=1, inplace=True)
    train_feat = df.to_numpy()
    val_df.drop(to_drop, axis=1, inplace=True)
    val_feat = val_df.to_numpy()
    sel_ind = [False if x in to_drop else True for x in range(len(feature_names))]
    feature_names = feature_names[sel_ind]

    ############## Step three################

    ## define random forest classifier
    rfe = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=1)

    ## define Boruta feature selection method
    feat_selector = RFE(estimator=rfe, n_features_to_select=12)

    ## find all relevant features
    feat_selector.fit(train_feat, train_labels)

    ## check selected features
    feat_selector.support_

    ## check ranking of features
    feat_selector.ranking_

    ## call transform() on X to filter it down to selected features
    train_feat = feat_selector.transform(train_feat)
    val_feat = feat_selector.transform(val_feat)
    feature_names = feature_names[feat_selector.support_]

    print(list(feature_names))
    return train_feat, val_feat, feature_names, norm


df_o = pd.read_excel(r'DataAll.xlsx')

# Column names
feature_names = df_o.columns[3:]  # First three are id, outcome, diagnosis

# Feature Values
data_matrix = df_o.to_numpy()

normal = {}
ipf = {}  # ID 0 87 ipf cases
ild = {}  # ID 1 279 ild cases
emphysema = {}

# test set 108 cases - 54 ipf 62 ild
ild_id = 0
ipf_id = 1

for lungs_feature in data_matrix:
    if lungs_feature[1] == 0:
        if lungs_feature[0] not in normal:
            normal[lungs_feature[0]] = []
        normal[lungs_feature[0]].append(lungs_feature[3:])
    elif lungs_feature[1] == 1:
        if lungs_feature[0] not in ipf:
            ipf[lungs_feature[0]] = []
        ipf[lungs_feature[0]].append(lungs_feature[3:])
    elif lungs_feature[1] == 2:
        if lungs_feature[0] not in ild:
            ild[lungs_feature[0]] = []
        ild[lungs_feature[0]].append(lungs_feature[3:])
    elif lungs_feature[1] == 3:
        if lungs_feature[0] not in emphysema:
            emphysema[lungs_feature[0]] = []
        emphysema[lungs_feature[0]].append(lungs_feature[3:])

## Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Train-Test Split
normal_kf = [x for x in kf.split(normal)]
ipf_kf = [x for x in kf.split(ipf)]
ild_kf = [x for x in kf.split(ild)]
emphysema_kf = [x for x in kf.split(emphysema)]


def get_array_from_dict(dict_array, fold_idx):
    temp = []
    count = 0
    for key, val in dict_array.items():
        if count in fold_idx:
            if len(dict_array[key]) == 2:
                temp.append(np.concatenate((np.expand_dims(dict_array[key][0], axis=0),
                                            np.expand_dims(dict_array[key][1], axis=0)), axis=1))
            else:
                temp.append(np.concatenate((np.expand_dims(dict_array[key][0], axis=0),
                                            np.zeros((1, len(dict_array[key][0])))), axis=1))
        count += 1
    temp = np.vstack(temp)
    return temp


################## SELECT THE FOLD HERE ###########################################
FOLD_NUM = 4

# train arrays
normal_train_l = get_array_from_dict(normal, normal_kf[FOLD_NUM][0])
ipf_train_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][0])
ild_train_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][0])
emphysema_train_l = get_array_from_dict(emphysema, emphysema_kf[FOLD_NUM][0])

# validation array
normal_val_l = get_array_from_dict(normal, normal_kf[FOLD_NUM][1])
ipf_val_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][1])
ild_val_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][1])
emphysema_val_l = get_array_from_dict(emphysema, emphysema_kf[FOLD_NUM][1])

################## SELECT THE FOLD HERE ###########################################
for FOLD_NUM in range(5):
    # train arrays
    normal_train_l = get_array_from_dict(normal, normal_kf[FOLD_NUM][0])
    ipf_train_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][0])
    ild_train_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][0])
    emphysema_train_l = get_array_from_dict(emphysema, emphysema_kf[FOLD_NUM][0])

    # validation array
    normal_val_l = get_array_from_dict(normal, normal_kf[FOLD_NUM][1])
    ipf_val_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][1])
    ild_val_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][1])
    emphysema_val_l = get_array_from_dict(emphysema, emphysema_kf[FOLD_NUM][1])

    ## train features, train labels
    # features
    args = (ipf_train_l, ild_train_l)
    train_feat_l = np.concatenate(args, axis=0)

    # labels
    args = (np.ones((len(ipf_train_l), 1)) * ipf_id,
            np.ones((len(ild_train_l), 1)) * ild_id)
    train_labels_l = np.squeeze(np.concatenate(args, axis=0))

    ## validation features, validation labels
    # features
    args = (ipf_val_l, ild_val_l)
    val_feat_l = np.concatenate(args, axis=0)

    # labels
    args = (np.ones((len(ipf_val_l), 1)) * ipf_id,
            np.ones((len(ild_val_l), 1)) * ild_id)
    val_labels_l = np.squeeze(np.concatenate(args, axis=0))

    train = np.concatenate((train_feat_l[:, :147], train_feat_l[:, 147:]), axis=0)
    val = np.concatenate((val_feat_l[:, :147], val_feat_l[:, 147:]), axis=0)
    train_labels = np.concatenate((train_labels_l, train_labels_l), axis=0)
    val_labels = np.concatenate((val_labels_l, val_labels_l), axis=0)

    a = variance_corr_rfe(train, val, train_labels, val_labels, feature_names)

## train features, train labels
# features
args = (ipf_train_l, ild_train_l)
train_feat_l = np.concatenate(args, axis=0)

# labels
args = (np.ones((len(ipf_train_l), 1)) * ipf_id,
        np.ones((len(ild_train_l), 1)) * ild_id)
train_labels_l = np.squeeze(np.concatenate(args, axis=0))

## validation features, validation labels
# features
args = (ipf_val_l, ild_val_l)
val_feat_l = np.concatenate(args, axis=0)

# labels
args = (np.ones((len(ipf_val_l), 1)) * ipf_id,
        np.ones((len(ild_val_l), 1)) * ild_id)
val_labels_l = np.squeeze(np.concatenate(args, axis=0))

all_train = np.concatenate((train_feat_l, val_feat_l), axis=0)
all_labels = np.concatenate((train_labels_l, val_labels_l), axis=0)

# sel_features = ['Fractal_average', 'Fractal_lacunarity', 'GLCM_autocorr',
#       'GLCM_clusTend', 'GLCM_contrast', 'GLCM_correl1', 'GLCM_infoCorr2',
#       'GLDZM_DZN', 'GLDZM_HISDE', 'GLDZM_IN', 'GLDZM_IV', 'GLDZM_SDE',
#       'GLRLM_RE', 'GLSZM_HILAE', 'GLSZM_SAE', 'IH_max', 'IH_qcod',
#       'LocInt_peakGlobal', 'NGTDM_busyness', 'NGTDM_contrast',
#       'NGTDM_strength', 'Stats_mean', 'Stats_min', 'Stats_p10']


all_features = []
all_features.extend(feature_names)
all_features.extend(feature_names)

sel_features = ['Fractal_average', 'Fractal_lacunarity', 'GLCM_autocorr',
                'GLCM_clusTend', 'GLCM_contrast', 'GLCM_correl1', 'GLCM_infoCorr2',
                'GLDZM_DZN', 'GLDZM_HISDE', 'GLDZM_IN', 'GLDZM_IV', 'GLDZM_SDE',
                'GLRLM_RE', 'GLSZM_HILAE', 'GLSZM_SAE', 'IH_max', 'IH_qcod',
                'LocInt_peakGlobal', 'NGTDM_busyness', 'NGTDM_contrast',
                'NGTDM_strength', 'Stats_mean', 'Stats_min', 'Stats_p10']

sel_features = ['GLCM_clusTend', 'GLCM_correl1', 'GLDZM_DZN', 'GLDZM_HISDE', 'GLDZM_INN', 'GLRLM_LRHGE', 'GLRLM_RE',
                'GLSZM_HILAE', 'GLSZM_SAE', 'IH_qcod', 'NGLDM_DE', 'Stats_min']
## Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Train-Test Split
ipf_kf = [x for x in kf.split(ipf)]
ild_kf = [x for x in kf.split(ild)]


def get_array_from_dict(dict_array, fold_idx):
    temp = []
    count = 0
    for key, val in dict_array.items():
        if count in fold_idx:
            if len(dict_array[key]) == 2:
                temp.append(np.concatenate((np.expand_dims(dict_array[key][0], axis=0),
                                            np.expand_dims(dict_array[key][1], axis=0)), axis=1))
            else:
                temp.append(np.concatenate((np.expand_dims(dict_array[key][0], axis=0),
                                            np.zeros((1, len(dict_array[key][0])))), axis=1))
        count += 1
    temp = np.vstack(temp)
    return temp


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., 100 - 1)
    return m, m - h, m + h


font = {'family': 'arial',
        'color': 'black',
        'weight': 'normal',
        'size': 18,
        }

font_prop = matplotlib.font_manager.FontProperties(size=12)

# ROC curve
train_roc = []
test_roc = []
onehot_encoder = OneHotEncoder(sparse=False)

labels = ["ILDs", "IPF"]

matrices = []
for fold_no_i in range(0, 5):
    ################## SELECT THE FOLD HERE ###########################################
    FOLD_NUM = fold_no_i

    # train arrays
    ipf_train_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][0])
    ild_train_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][0])

    # validation array
    ipf_val_l = get_array_from_dict(ipf, ipf_kf[FOLD_NUM][1])
    ild_val_l = get_array_from_dict(ild, ild_kf[FOLD_NUM][1])

    ## train features, train labels
    # features
    args = (ipf_train_l, ild_train_l)
    train_feat_l = np.concatenate(args, axis=0)

    # labels
    args = (np.ones((len(ipf_train_l), 1)) * ipf_id,
            np.ones((len(ild_train_l), 1)) * ild_id)
    train_labels_l = np.squeeze(np.concatenate(args, axis=0))

    ## validation features, validation labels
    # features
    args = (ipf_val_l, ild_val_l)
    val_feat_l = np.concatenate(args, axis=0)

    # labels
    args = (np.ones((len(ipf_val_l), 1)) * ipf_id,
            np.ones((len(ild_val_l), 1)) * ild_id)
    val_labels_l = np.squeeze(np.concatenate(args, axis=0))

    # Normalization Left
    norm = StandardScaler().fit(train_feat_l)
    train_feat_l = norm.transform(train_feat_l)
    val_feat_l = norm.transform(val_feat_l)

    # Feature Selection Left
    select_features = [True if x in sel_features else False for x in all_features]
    sel_train_feat_l = train_feat_l[:, select_features]
    sel_val_feat_l = val_feat_l[:, select_features]

    # clf = SVC(C=0.1, gamma=0.01, random_state = 42, kernel = 'linear', probability=True, class_weight ={0:1, 1:4, 2:1, 3:1})
    clf = RandomForestClassifier(n_estimators=1000, max_depth=5, max_features=2,
                                 min_samples_leaf=25, min_samples_split=5,
                                 class_weight={0: 1, 1: 3})  #
    # clf = RandomForestClassifier(n_estimators = 1000, class_weight ={0:1, 1:3}, min_samples_leaf=50, min_samples_split=50)
    # clf = RandomForestClassifier(n_estimators=100, max_depth=4)#
    result = clf.fit(sel_train_feat_l, train_labels_l)

    # Predicting Training Results
    yhat = result.predict_proba(sel_train_feat_l)
    yhat_p = result.predict(sel_train_feat_l)

    # Predicting Validation Results
    yhat = result.predict_proba(sel_val_feat_l)
    yhat_p = result.predict(sel_val_feat_l)

    # Validation plot
    binary_AUC_plot(val_labels_l, yhat[:, 1])

    temp = []
    y_test = np.array(val_labels_l)
    y_test = y_test.reshape(len(yhat), 1)

    temp.append(yhat)
    temp.append(y_test)
    test_roc.append(temp)

    print(multilabel_confusion_matrix(val_labels_l, yhat_p, labels=[0, 1]))
    print(classification_report(val_labels_l, yhat_p))
    matrices.append(multilabel_confusion_matrix(val_labels_l, yhat_p, labels=[0, 1]))
    df = pd.read_excel(r'NIH.xlsx')

    # Column names
    feature_names = df.columns[2:]  # First three are id, outcome, diagnosis

    # Feature Values
    data_matrix = df.to_numpy()

    ipf_t = {}  # ID 1
    ild_t = {}  # ID 2

    for lungs_feature in data_matrix:
        if lungs_feature[1] == 1:
            if lungs_feature[0] not in ipf_t:
                ipf_t[lungs_feature[0]] = []
            ipf_t[lungs_feature[0]].append(lungs_feature[2:])
        elif lungs_feature[1] == 0:
            if lungs_feature[0] not in ild_t:
                ild_t[lungs_feature[0]] = []
            ild_t[lungs_feature[0]].append(lungs_feature[2:])

    ipf_test = get_array_from_dict(ipf_t, np.array([x for x in range(len(ipf))]))
    ild_test = get_array_from_dict(ild_t, np.array([x for x in range(len(ild))]))

    ipf_test = norm.transform(ipf_test)
    ild_test = norm.transform(ild_test)

    ## train features, train labels
    # features
    args = (ipf_test, ild_test)
    test_feat = np.concatenate(args, axis=0)

    # labels
    args = (np.ones((len(ipf_test), 1)) * ipf_id,
            np.ones((len(ild_test), 1)) * ild_id)
    test_labels = np.squeeze(np.concatenate(args, axis=0))

    # using the selected features
    select_features = [True if x in sel_features else False for x in all_features]
    test_feat = test_feat[:, select_features]

    yhat = result.predict_proba(test_feat)
    yhat_p = result.predict(test_feat)

    conf_matrices = multilabel_confusion_matrix(test_labels, yhat_p, labels=[0, 1])
    print("testing")
    binary_AUC_plot(test_labels, yhat[:, 1])
    print(multilabel_confusion_matrix(test_labels, yhat_p, labels=[0, 1]))
    print(classification_report(test_labels, yhat_p))

# structures
fpr = dict()
tpr = dict()
roc_auc = dict()

#for folds_n in range(5):
#    preds = test_roc[folds_n][0]
#    y_test = test_roc[folds_n][1]
#    n_classes = 1

#    mean_fpr = np.linspace(0, 1, 100)

#    fpr_t, tpr_t, _ = metrics.roc_curve(y_test, preds[:, 1])
#    interp_tpr = np.interp(mean_fpr, fpr_t, tpr_t)
#    interp_tpr[0] = 0.0
#    fpr[folds_n] = mean_fpr
#    tpr[folds_n] = interp_tpr

# roc for each class
#fig, ax = plt.subplots(figsize=(10, 10))
#ax.plot([0, 1], [0, 1], 'k--')
#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([0.0, 1.05])
#ax.set_xlabel('False Positive Rate', fontdict=font)
#ax.set_ylabel('True Positive Rate', fontdict=font)
#ax.set_title('Receiver Operating Characteristic Curve', fontdict=font)
#for j in range(n_classes):
#    fpr_t = []
#    fpr_t.append(fpr[0])
#    fpr_t.append(fpr[1])
#    fpr_t.append(fpr[2])
#    fpr_t.append(fpr[3])
#    fpr_t.append(fpr[4])
#    fpr_t = np.vstack(fpr_t)

#    tpr_t = []
#    tpr_t.append(tpr[0])
#    tpr_t.append(tpr[1])
#    tpr_t.append(tpr[2])
#    tpr_t.append(tpr[3])
#    tpr_t.append(tpr[4])
#    tpr_t = np.vstack(tpr_t)

#    fpr_m = np.mean(fpr_t, axis=0)
#    tpr_m = np.mean(tpr_t, axis=0)

#    roc_auc_m = round(metrics.auc(fpr_m, tpr_m), 2)
#    mean_tpr, low_tpr, high_tpr = mean_confidence_interval(tpr_t, confidence=0.95)

#    roc_auc_l = round(metrics.auc(fpr_m, low_tpr), 3)
#    roc_auc_h = round(metrics.auc(fpr_m, high_tpr), 3)

#    ax.plot(fpr_m, tpr_m, label="ROC curve AUC : " + str(roc_auc_m)
#                                + " (95 % CI : " + str(roc_auc_l) + " - " + str(roc_auc_h)
#                                + " )", c="red")

#    ax.fill_between(fpr_m, high_tpr, low_tpr, alpha=0.05, linewidth=2, color="red")

#plt.xticks(fontname="Arial", fontsize=16)
#plt.yticks(fontname="Arial", fontsize=16)
#ax.legend(loc="best", prop=font_prop)
#ax.grid(alpha=.4)
#plt.savefig("cross_val_results.png", dpi=600)
#plt.show()

cross_val_matrix = np.vstack(matrices)
labels = ["non-ILDs", "IPF"]
for i in range(2):
    precision = []
    recall = []
    for j in range(5):
        temp_m = cross_val_matrix[i + 2 * j]
        temp_precision = temp_m[1][1] / (temp_m[1][1] + temp_m[0][1])
        temp_recall = temp_m[1][1] / (temp_m[1][1] + temp_m[1][0])
        precision.append(temp_precision)
        recall.append(temp_recall)
    precision = np.array(precision)
    recall = np.array(recall)
    print("Precision Class ", labels[i], " : ", np.mean(precision), " ± ", np.std(precision))
    print("Recall Class ", labels[i], " : ", np.mean(recall), " ± ", np.std(recall))

import pickle

## train features, train labels
# features
args = (ipf_train_l, ild_train_l)
train_feat_l = np.concatenate(args, axis=0)

# labels
args = (np.ones((len(ipf_train_l), 1)) * ipf_id,
        np.ones((len(ild_train_l), 1)) * ild_id)
train_labels_l = np.squeeze(np.concatenate(args, axis=0))

## validation features, validation labels
# features
args = (ipf_val_l, ild_val_l)
val_feat_l = np.concatenate(args, axis=0)

# labels
args = (np.ones((len(ipf_val_l), 1)) * ipf_id,
        np.ones((len(ild_val_l), 1)) * ild_id)
val_labels_l = np.squeeze(np.concatenate(args, axis=0))

all_train = np.concatenate((train_feat_l, val_feat_l), axis=0)
all_labels = np.concatenate((train_labels_l, val_labels_l), axis=0)

sel_features = ['Fractal_average', 'Fractal_lacunarity', 'GLCM_autocorr',
                'GLCM_clusTend', 'GLCM_contrast', 'GLCM_correl1', 'GLCM_infoCorr2',
                'GLDZM_DZN', 'GLDZM_HISDE', 'GLDZM_IN', 'GLDZM_IV', 'GLDZM_SDE',
                'GLRLM_RE', 'GLSZM_HILAE', 'GLSZM_SAE', 'IH_max', 'IH_qcod',
                'LocInt_peakGlobal', 'NGTDM_busyness', 'NGTDM_contrast',
                'NGTDM_strength', 'Stats_mean', 'Stats_min', 'Stats_p10']

sel_features = ['GLCM_clusTend', 'GLCM_correl1', 'GLDZM_DZN', 'GLDZM_HISDE', 'GLDZM_INN', 'GLRLM_LRHGE', 'GLRLM_RE',
                'GLSZM_HILAE', 'GLSZM_SAE', 'IH_qcod', 'NGLDM_DE', 'Stats_min']
# Normalization Left
norm = StandardScaler().fit(all_train)
all_train = norm.transform(all_train)

# Feature Selection Left
select_features = [True if x in sel_features else False for x in all_features]
all_train = all_train[:, select_features]

# Training
n_class = 2

# clf = SVC(C=0.1, gamma=0.01, random_state = 42, kernel = 'linear', probability=True)

clf = RandomForestClassifier(n_estimators=1000, max_depth=5, max_features=2,
                             min_samples_leaf=25, min_samples_split=5,
                             class_weight={0: 1, 1: 4})  #
result = clf.fit(all_train, all_labels)

filename_svm = 'test_model.sav'
pickle.dump(result, open(filename_svm, 'wb'))

# Predicting Training Results
yhat = result.predict_proba(all_train)
yhat_p = result.predict(all_train)

# train plot
binary_AUC_plot(all_labels, yhat[:, 1])

# classification report training
print(multilabel_confusion_matrix(all_labels, yhat_p, labels=[0, 1]))
df = pd.read_excel(r'NIH.xlsx')

# Column names
feature_names = df.columns[2:]  # First three are id, outcome, diagnosis

# Feature Values
data_matrix = df.to_numpy()

ipf_t = {}  # ID 1
ild_t = {}  # ID 2

for lungs_feature in data_matrix:
    if lungs_feature[1] == 1:
        if lungs_feature[0] not in ipf_t:
            ipf_t[lungs_feature[0]] = []
        ipf_t[lungs_feature[0]].append(lungs_feature[2:])
    elif lungs_feature[1] == 0:
        if lungs_feature[0] not in ild_t:
            ild_t[lungs_feature[0]] = []
        ild_t[lungs_feature[0]].append(lungs_feature[2:])

ipf_test = get_array_from_dict(ipf_t, np.array([x for x in range(len(ipf))]))
ild_test = get_array_from_dict(ild_t, np.array([x for x in range(len(ild))]))

ipf_test = norm.transform(ipf_test)
ild_test = norm.transform(ild_test)

## train features, train labels
# features
args = (ipf_test, ild_test)
test_feat = np.concatenate(args, axis=0)

# labels
args = (np.ones((len(ipf_test), 1)) * ipf_id,
        np.ones((len(ild_test), 1)) * ild_id)
test_labels = np.squeeze(np.concatenate(args, axis=0))

# using the selected features
select_features = [True if x in sel_features else False for x in all_features]
test_feat = test_feat[:, select_features]
test_l = np.array(test_labels)#["non-IPF ILDs" if x == 0 else 'IPF' for x in test_labels ])
feat_save = np.concatenate((test_feat, test_l[..., None]), axis=1)

np.save("test_file.npy", feat_save)
yhat = result.predict_proba(test_feat)
yhat_p = result.predict(test_feat)

conf_matrices = multilabel_confusion_matrix(test_labels, yhat_p, labels=[0, 1])
print("testing")
binary_AUC_plot(test_labels, yhat[:, 1])
print(multilabel_confusion_matrix(test_labels, yhat_p, labels=[0, 1]))
print(classification_report(test_labels, yhat_p))
