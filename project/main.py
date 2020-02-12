from process import extract_club_name_from_image, train_or_load_club_logo_recognition_model, get_info_about_club
import glob
import sys
import os


if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep

if len(sys.argv) > 1:
    CLUBS_PATH = sys.argv[3]
else:
    CLUBS_PATH = '.'+os.path.sep+'clubs'

label_dict = dict()
with open(TRAIN_DATASET_PATH+'annotations.csv', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index > 0:
            cols = line.replace('\n', '').split(',')
            label_dict[cols[0]] = cols[1]

train_image_paths = []
train_image_labels = []
for image_name in os.listdir(TRAIN_DATASET_PATH):
    if '.jpg' in image_name:
        train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, image_name))
        train_image_labels.append(label_dict[image_name])



model = train_or_load_club_logo_recognition_model(train_image_paths, train_image_labels)

processed_image_names = []
extracted_club_name = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    extracted_club_name.append(extract_club_name_from_image(model, image_path))

result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, extracted_club_name[image_index])
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

for name in extracted_club_name:
    get_info_about_club(CLUBS_PATH, name)
