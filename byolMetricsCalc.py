import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

nb_classes = 10
confusion_matrix = np.zeros((nb_classes, nb_classes))
with torch.no_grad():
    for i, (inputs, classes) in enumerate(val_loader):
        inputs = inputs.cuda()
        classes = classes.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

plt.figure(figsize=(15,10))
#class_names = list(label2class.values())
cm1 = pd.DataFrame(confusion_matrix).astype(int)
FP = cm1.sum(axis=0) - np.diag(cm1) 
FN = cm1.sum(axis=1) - np.diag(cm1)
TP = np.diag(cm1)
TN = cm1.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("Sensitivity:", TPR)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print("Specificity:", TNR)
