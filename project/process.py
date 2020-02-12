from builtins import print
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import numpy as np
import cv2
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump, load


def train_or_load_club_logo_recognition_model(train_image_paths, train_image_labels):
    arsenal_features = []                   #0
    astonvilla_features = []                #1
    bournemouth_features = []               #2
    brightonandhovealbion_features = []     #3
    burnley_features = []                   #4
    chelsea_features = []                   #5
    crystal_palace_features = []            #6
    everton_features = []                   #7
    leicester_city_features = []            #8
    liverpool_features = []                 #9
    manchester_city_features = []           #10
    manchester_united_features = []         #11
    newcastle_united_features = []          #12
    norwich_city_features = []              #13
    sheffield_united_features = []          #14
    southampton_features = []               #15
    tottenham_hotspur_features = []         #16
    watford_features = []                   #17
    westham_united_features = []            #18
    wolverhampton_wanderers_features = []   #19

    labels = []

    for i, label in enumerate(train_image_labels):
        if label == 'Arsenal':
            image = load_image(train_image_paths[i])
            h = hog(image)
            arsenal_features.append(h.compute(image))
            labels.append(0)
        elif label == 'Aston_Villa':
            image = load_image(train_image_paths[i])
            h = hog(image)
            astonvilla_features.append(h.compute(image))
            labels.append(1)
        elif label == 'AFC_Bournemouth':
            image = load_image(train_image_paths[i])
            h = hog(image)
            bournemouth_features.append(h.compute(image))
            labels.append(2)
        elif label == 'Brighton_and_Hove_Albion':
            image = load_image(train_image_paths[i])
            h = hog(image)
            brightonandhovealbion_features.append(h.compute(image))
            labels.append(3)
        elif label == 'Burnley':
            image = load_image(train_image_paths[i])
            h = hog(image)
            burnley_features.append(h.compute(image))
            labels.append(4)
        elif label == 'Chelsea':
            image = load_image(train_image_paths[i])
            h = hog(image)
            chelsea_features.append(h.compute(image))
            labels.append(5)
        elif label == 'Crystal_Palace':
            image = load_image(train_image_paths[i])
            h = hog(image)
            crystal_palace_features.append(h.compute(image))
            labels.append(6)
        elif label == 'Everton':
            image = load_image(train_image_paths[i])
            h = hog(image)
            everton_features.append(h.compute(image))
            labels.append(7)
        elif label == 'Leicester_City':
            image = load_image(train_image_paths[i])
            h = hog(image)
            leicester_city_features.append(h.compute(image))
            labels.append(8)
        elif label == 'Liverpool':
            image = load_image(train_image_paths[i])
            h = hog(image)
            liverpool_features.append(h.compute(image))
            labels.append(9)
        elif label == 'Manchester_City':
            image = load_image(train_image_paths[i])
            h = hog(image)
            manchester_city_features.append(h.compute(image))
            labels.append(10)
        elif label == 'Manchester_United':
            image = load_image(train_image_paths[i])
            h = hog(image)
            manchester_united_features.append(h.compute(image))
            labels.append(11)
        elif label == 'Newcastle_United':
            image = load_image(train_image_paths[i])
            h = hog(image)
            newcastle_united_features.append(h.compute(image))
            labels.append(12)
        elif label == 'Norwich_City':
            image = load_image(train_image_paths[i])
            h = hog(image)
            norwich_city_features.append(h.compute(image))
            labels.append(13)
        elif label == 'Sheffield_United':
            image = load_image(train_image_paths[i])
            h = hog(image)
            sheffield_united_features.append(h.compute(image))
            labels.append(14)
        elif label == 'Southampton':
            image = load_image(train_image_paths[i])
            h = hog(image)
            southampton_features.append(h.compute(image))
            labels.append(15)
        elif label == 'Tottenham_Hotspur':
            image = load_image(train_image_paths[i])
            h = hog(image)
            tottenham_hotspur_features.append(h.compute(image))
            labels.append(16)
        elif label == 'Watford':
            image = load_image(train_image_paths[i])
            h = hog(image)
            watford_features.append(h.compute(image))
            labels.append(17)
        elif label == 'West_Ham_United':
            image = load_image(train_image_paths[i])
            h = hog(image)
            westham_united_features.append(h.compute(image))
            labels.append(18)
        elif label == 'Wolverhampton_Wanderers':
            image = load_image(train_image_paths[i])
            h = hog(image)
            wolverhampton_wanderers_features.append(h.compute(image))
            labels.append(19)

    arsenal_features = np.array(arsenal_features)
    astonvilla_features = np.array(astonvilla_features)
    bournemouth_features = np.array(bournemouth_features)
    brightonandhovealbion_features = np.array(brightonandhovealbion_features)
    burnley_features = np.array(burnley_features)
    chelsea_features = np.array(chelsea_features)
    crystal_palace_features = np.array(crystal_palace_features)
    everton_features = np.array(everton_features)
    leicester_city_features = np.array(leicester_city_features)
    liverpool_features = np.array(liverpool_features)
    manchester_city_features = np.array(manchester_city_features)
    manchester_united_features = np.array(manchester_united_features)
    newcastle_united_features = np.array(newcastle_united_features)
    norwich_city_features = np.array(norwich_city_features)
    sheffield_united_features = np.array(sheffield_united_features)
    southampton_features = np.array(southampton_features)
    tottenham_hotspur_features = np.array(tottenham_hotspur_features)
    watford_features = np.array(watford_features)
    westham_united_features = np.array(westham_united_features)
    wolverhampton_wanderers_features = np.array(wolverhampton_wanderers_features)

    x = np.vstack((arsenal_features,
                   astonvilla_features,
                   bournemouth_features,
                   brightonandhovealbion_features,
                   burnley_features,
                   chelsea_features,
                   crystal_palace_features,
                   everton_features,
                   leicester_city_features,
                   liverpool_features,
                   manchester_city_features,
                   manchester_united_features,
                   newcastle_united_features,
                   norwich_city_features,
                   sheffield_united_features,
                   southampton_features,
                   tottenham_hotspur_features,
                   watford_features,
                   westham_united_features,
                   wolverhampton_wanderers_features))
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)

    x_train = reshape_data(x_train)
    x_test = reshape_data(x_test)

    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)
    model = None
    try:
        model = load('model.joblib')
    except Exception as e:
        model = None
    if model == None:
        print("Obucavanje pocelo")
        model = SVC(kernel='linear', probability=True)
        model = model.fit(x_train, y_train)
        print("Obucavanje zavrsilo")
        dump(model, 'model.joblib')
        print("Serijalizovano")

    return model


def extract_club_name_from_image(trained_model, image_path):
    club_name = ""
    image = load_image(image_path)
    h = hog(image)
    image_features = h.compute(image)
    image_features = image_features.reshape(1, -1)
    club_logo = trained_model.predict(image_features)
    if club_logo == 0:
        club_name = "Arsenal"
    elif club_logo == 1:
        club_name = "Aston_Villa"
    elif club_logo == 2:
        club_name = "AFC_Bournemouth"
    elif club_logo == 3:
        club_name = "Brighton_and_Hove_Albion"
    elif club_logo == 4:
        club_name = "Burnley"
    elif club_logo == 5:
        club_name = "Chelsea"
    elif club_logo == 6:
        club_name = "Crystal_Palace"
    elif club_logo == 7:
        club_name = "Everton"
    elif club_logo == 8:
        club_name = "Leicester_City"
    elif club_logo == 9:
        club_name = "Liverpool"
    elif club_logo == 10:
        club_name = "Manchester_City"
    elif club_logo == 11:
        club_name = "Manchester_United"
    elif club_logo == 12:
        club_name = "Newcastle_United"
    elif club_logo == 13:
        club_name = "Norwich_City"
    elif club_logo == 14:
        club_name = "Sheffield_United"
    elif club_logo == 15:
        club_name = "Southampton"
    elif club_logo == 16:
        club_name = "Tottenham_Hotspur"
    elif club_logo == 17:
        club_name = "Watford"
    elif club_logo == 18:
        club_name = "West_Ham_United"
    elif club_logo == 19:
        club_name = "Wolverhampton_Wanderers"
    return club_name


def get_info_about_club(clubs_path,club_name):
    my_url = 'https://www.premierleague.com/tables'
    uClient = uReq(my_url)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, "html.parser")

    currentTable = page_soup.findAll("tr", {"data-compseason": "274"})

    if '_' not in club_name:
        club_name = club_name
    else:
        club_name = club_name.replace('_', ' ')

    with open(clubs_path + os.path.sep + club_name + '.csv', 'w') as f:
        headers = "position, club_name, short_club_name, played, won, drawn, lost, goals_for, goals_against, " \
                  "goal_difference, points, next_opponent, next_date, next_time\n "
        f.write(headers)
        for club in currentTable:
            name = club["data-filtered-table-row-name"]
            if club_name == name:
                # 1
                position = club["data-position"]
                # 2
                name = club["data-filtered-table-row-name"]
                # 3
                shortName = club.find("span", {"class": "short"}).text
                games = club.findAll("td", {"class": None})
                # 4
                played = games[0].text
                # 5
                won = games[1].text
                # 6
                draw = games[2].text
                # 7
                lost = games[3].text
                forAndAgainstGoals = club.findAll("td", {"class": "hideSmall"})
                ForGoals = forAndAgainstGoals[0]
                AgainstGoals = forAndAgainstGoals[1]
                # 8
                ForGoals = ForGoals.text
                # 9
                AgainstGoals = AgainstGoals.text
                # 10
                goalDifference = games[4].text
                goalDifference = goalDifference.strip()

                # 11
                points = club.find("td", {"class": "points"}).text
                nextMatch = club.find("td", {"class": "nextMatchCol hideMed"})
                # 12
                nextMatch_opponent = nextMatch.find("span", {"class": "visuallyHidden"}).text
                # 13
                nextMatch_date = nextMatch.find("span", {"class": "matchInfo"}).text
                # 14
                nextMatch_time = nextMatch.a.span.div.time.text
                f.write(
                    position + "," + name + "," + shortName + "," + played + "," + won + "," + draw + "," + lost + "," + ForGoals + "," + AgainstGoals + "," + goalDifference + "," + points + "," + nextMatch_opponent + "," + nextMatch_date + "," + nextMatch_time + "\n")

        f.close()


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (600, 600))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def hog(img):
    nbins = 9
    cell_size = (8, 8)
    block_size = (3, 3)

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog
