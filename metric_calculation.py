import pandas as pd
import numpy as np
import glob
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, confusion_matrix


def pretty_per_num(num):
    return round(num * 100, 2)


def calc_metrics(pred, target):
    auc_score = roc_auc_score(target, pred)
    fpr, tpr, thr = roc_curve(target, pred)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    bestg = gmeans[ix]
    bestthr = thr[ix]
    val = "Best Threshold=%f, G-Mean=%.3f" % (bestthr, pretty_per_num(bestg))
    pred_thr = [True if p > bestthr else False for p in pred]
    cm1 = confusion_matrix(pred_thr, target)
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print(
        "Following are the metrics: AUC_Score:",
        pretty_per_num(auc_score),
        val,
        "Sensitivity:",
        pretty_per_num(sensitivity),
        "Specificity:",
        pretty_per_num(specificity),
    )


def calculate_metric_ensemble(path):

    files = glob.glob(path)

    targets = {}
    preds = {}
    samessies = {}
    for f in files:
        data = pd.read_csv(f)
        targ = data["target"] == data["mel_index"].iloc[0]
        targ = list(targ.values)
        nm = f.split("/")[-1].replace("melData_", "").replace("_.csv", "")
        targets[nm] = targ
        if tuple(targ) in samessies.keys():
            samessies[tuple(targ)].append(nm)
        else:
            samessies[tuple(targ)] = [nm]
        preds[nm] = list(data["pred"].values)
        calc_metrics(
            list(data["pred"]), list((data["target"] == data["mel_index"].iloc[0]))
        )

    targets = pd.DataFrame(targets)
    preds = pd.DataFrame(preds)

    col_nms = list(samessies.values())[0]
    target = list(samessies.keys())[0]

    print("Following Models were selected for the ensemble: ", col_nms)

    calc_metrics(list(preds[col_nms].mean(axis=1).values), target)


def main():
    calculate_metric_ensemble("./results/*.csv")


if __name__ == "__main__":
    main()
