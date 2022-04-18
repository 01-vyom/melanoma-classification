import pandas as pd
import glob
import warnings
from src.metric_utility import calc_metrics


# Remove this to see the warning.
warnings.filterwarnings("ignore")


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
            list(data["pred"]), list((data["target"] == data["mel_index"].iloc[0])), nm
        )

    targets = pd.DataFrame(targets)
    preds = pd.DataFrame(preds)
    col_nms = list(samessies.values())[0]
    target = list(samessies.keys())[0]
    # Only include models with no-meta data used and which have same fold data.
    cols_no_meta = [
        "9c_b5ns_1.5e_640_ext_15ep",
        "9c_b4ns_768_768_ext_15ep",
        "9c_nest101_2e_640_ext_15ep",
        "9c_b4ns_2e_896_ext_15ep",
        "9c_b6ns_640_ext_15ep",
        "9c_b7ns_1e_640_ext_15ep",
        "9c_b4ns_768_640_ext_15ep",
        "4c_b5ns_1.5e_640_ext_15ep",
        "9c_se_x101_640_ext_15ep",
    ]
    col_nms = list(set(col_nms) & set(cols_no_meta))
    # Calculate Ensemble Score
    calc_metrics(list(preds[col_nms].mean(axis=1).values), target, "Ensemble")


def main():
    calculate_metric_ensemble("../../results/CNN-Ha/*.csv")


if __name__ == "__main__":
    main()

