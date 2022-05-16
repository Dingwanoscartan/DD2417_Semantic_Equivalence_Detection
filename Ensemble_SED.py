import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


dir_name = 'representations'
# data from concatGRU
train_concatGRU = pd.read_csv(dir_name + '/concatGRU_training.csv')
test_concatGRU = pd.read_csv(dir_name + '/concatGRU_test.csv')
# data from SimGRU
train_simGRU = pd.read_csv(dir_name + '/SimGRU_train.csv')
test_simGRU = pd.read_csv(dir_name + '/SimGRU_test.csv')
# concatenation
train = pd.concat([train_concatGRU, train_simGRU], axis=1)
test = pd.concat([test_concatGRU, test_simGRU], axis=1)
# attributes / label split
train_x = train[['concatGRU_1', 'concatGRU_2', 'SimGRU']]
train_y = train['label']
test_x = test[['concatGRU_1', 'concatGRU_2', 'SimGRU']]
test_y = test['label']
model = RandomForestClassifier(n_estimators=20)
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
print('Accuracy: ' + str(accuracy_score(test_y, pred_y)))
print(confusion_matrix(test_y, pred_y))
while True:
    print("concatGRU_1: ")
    text = input("> ")
    if text == 'q':
        break
    if text == "":
        continue
    try:
        c1 = float(text)
    except KeyError:
        print("Erroneous input string")
        continue
    print("concatGRU_2: ")
    text = input("> ")
    if text == 'q':
        break
    if text == "":
        continue
    try:
        c2 = float(text)
    except KeyError:
        print("Erroneous input string")
        continue
    print("SimGRU: ")
    text = input("> ")
    if text == 'q':
        break
    if text == "":
        continue
    try:
        s = float(text)
    except KeyError:
        print("Erroneous input string")
        continue

    if model.predict([[c1, c2, s]])[0] == 1:
        print("Is duplicate")
    else:
        print("Is not duplicate")