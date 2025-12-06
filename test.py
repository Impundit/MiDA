import inspect
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import sys
from tensorflow.keras.models import load_model

acc_list = []
f1_list = []
rec_list = []
pre_list = []

def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list

def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)

eventlog = sys.argv[1]

with open("fold/" + eventlog + "/" + eventlog + '_num_cols.pickle', 'rb') as pickle_file:
    num_view = pickle.load(pickle_file)
with open("fold/" + eventlog + "/" + eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
    cat_view = pickle.load(pickle_file)
with open("fold/" + eventlog + "/" + eventlog + '_seq_length.pickle', 'rb') as pickle_file:
    seq_length = pickle.load(pickle_file)

for f in range(3):
    # train views
    outfile2 = open(eventlog+".txt", 'a')


    model = load_model("models/" + eventlog + "model_hpopt_" + str(f) + "_layer.h5")

    list_cat_view_train = []
    for col in cat_view:
        list_cat_view_train.append(np.load("fold/" + eventlog + "/" + eventlog + "_" + col + "_" + str(f) + "_train.npy"))

    list_cat_view_test = []
    for col in cat_view:
        list_cat_view_test.append(np.load("fold/" + eventlog + "/" + eventlog + "_" + col + "_" + str(f) + "_test.npy"))

    list_num_view_train = []
    for col in num_view:
        list_num_view_train.append(np.load("fold/" + eventlog + "/" + eventlog + "_" + col + "_" + str(f) + "_train.npy",allow_pickle=True))

    list_num_view_test = []
    for col in num_view:
        list_num_view_test.append(np.load("fold/" + eventlog + "/" + eventlog + "_" + col + "_" + str(f) + "_test.npy",allow_pickle=True))

    y_train = np.load("fold/" + eventlog + "/" + eventlog + "_y_" + str(f) + "_train.npy")
    y_test = np.load("fold/" + eventlog + "/" + eventlog + "_y_" + str(f) + "_test.npy")

    df_labels = np.unique(list(y_train) + list(y_test))

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    encoder_kwargs = {}
    if 'sparse_output' in inspect.signature(preprocessing.OneHotEncoder).parameters:
        encoder_kwargs['sparse_output'] = False
    else:
        encoder_kwargs['sparse'] = False
    onehot_encoder = preprocessing.OneHotEncoder(**encoder_kwargs)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(y_train).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    Y_train = np.asarray(train_onehot_encoded)

    test_integer_encoded = label_encoder.transform(y_test).reshape(-1, 1)
    test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
    Y_test = np.asarray(test_onehot_encoded)
    Y_test_int = np.asarray(test_integer_encoded)

    n_classes = len(df_labels)
    list_view = Union(list_cat_view_test, list_num_view_test)
    preds_a = model.predict(list_view)

    y_a_test = np.argmax(Y_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        Y_test_int, preds_a, average='macro', pos_label=None)
    auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
    prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

    labels = list(range(n_classes))
    target_names = [str(label_mapping[idx]) for idx in labels]
    report = classification_report(
        Y_test_int,
        preds_a,
        labels=labels,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )

    mapping_text = "Label mapping (index -> original label):\n" + "\n".join(
        f"{idx}: {label}" for idx, label in label_mapping.items()
    ) + "\n"

    print(mapping_text)
    print(report)
    outfile2.write(mapping_text)
    outfile2.write(report)
    outfile2.write('\nAUC: ' + str(auc_score_macro))
    outfile2.write('\nPRAUC: '+ str(prauc_score_macro))
    outfile2.write('\n')


    outfile2.flush()

outfile2.close()
