from sklearn.metrics import accuracy_score
import sys
import os

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[1]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep

labeled_samples = dict()

with open(VALIDATION_DATASET_PATH+'annotations.csv') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        if index == 0:
            continue
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '')
        cols[1] = cols[1].replace('\r', '')
        labeled_samples[str(cols[0])] = cols[1]


results = dict()

with open('result.csv') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '')
        cols[1] = cols[1].replace('\r', '')
        results[cols[0]] = cols[1]

truth = []
predicted = []
for image in results:
    truth.append(labeled_samples[image])
    predicted.append(results[image])

percentage = accuracy_score(truth, predicted)*100

print(percentage)
