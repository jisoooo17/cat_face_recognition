# 얼굴 주요 포인트를 점으로 찍은 json 파일과 사진을 대조하는 코드입니다.
# 원본사진을 통해 작업된 파일을 확인할 수 있습니다.

import cv2
import json
import matplotlib.pyplot as plt

def show_image_with_landmarks(image_path, landmarks_json_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(landmarks_json_path, 'r') as json_file:
        landmarks_data = json.load(json_file)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for point in landmarks_data['points']:
        x, y = float(point['points']['x']), float(point['points']['y'])
        ax.plot(x, y, 'o', color='red', markersize=4)

    plt.show()

image_path = '12865110_0.jpg'  
json_path = '12865110_0.json' 

show_image_with_landmarks(image_path, json_path)




# 사진을 기반으로 json파일을 만들어주는 코드입니다.
# 얼굴 주요 부위의 점은 수동으로 데이터 처리를 하며 
# 데이터 생성 및 전처리 단계이므로 실제로 업체에서 수동으로 만들어 줍니다
import cv2
import json

def draw_points(image, points):
    for point in points:
        x, y = point['x'], point['y']
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  

# Load the cat image
image_path = 'test.jpg' 
image = cv2.imread(image_path)
image_copy = image.copy()  

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append({'x': x, 'y': y})
        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)  
        cv2.imshow('Cat Image', image_copy)

cv2.imshow('Cat Image', image)
cv2.setMouseCallback('Cat Image', click_event)


while len(points) < 8:
    cv2.waitKey(1)

cv2.destroyAllWindows()

landmarks_data = {
    "points": points,
    "valid": True,
    "url": "https://example.com/cat_image.jpg"
}

with open('cat_landmarks.json', 'w') as json_file:
    json.dump(landmarks_data, json_file, indent=2)

print("JSON file 'cat_landmarks.json' has been created.")




# 준비된 json 파일을 기반으로 새로운 데이터 도착 시 동일 반려동물인지를 확인합니다.
# 훈련시킬 사진 데이터가 3장뿐이라 아쉽지만 3장만으로도 67% 정도의 정확성을 나타내고 있습니다.
# 보통 데이터는 지금보다 더 많이 준비하여 모델을 훈련시킵니다.
import dlib
import json
from sklearn.svm import SVC
import numpy as np

def extract_landmarks_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        face_points = json.load(file)


    return [(point['points']['x'], point['points']['y']) for point in face_points['points']]


def train_face_recognition_model(data):
    X = [np.array(features).flatten() for features in data['features']]
    y = data['labels']
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    return model

json_files = ['12865110_0.json', '12865110_1.json', '12865110_2.json']
all_features = []
all_labels = []

for i, json_file in enumerate(json_files):
    landmarks = extract_landmarks_from_json(json_file)
    all_features.append(landmarks)
    all_labels.append(i)  

data = {'features': all_features, 'labels': all_labels}
face_recognition_model = train_face_recognition_model(data)

new_json_file_path = '12865110_5.json'
new_landmarks = extract_landmarks_from_json(new_json_file_path)
new_features = np.array(new_landmarks).flatten().reshape(1, -1)

probabilities = face_recognition_model.predict_proba(new_features)

confidence_threshold = 0.8

predicted_label = np.argmax(probabilities)
confidence = probabilities[0, predicted_label]
print(confidence)

if confidence > confidence_threshold:
    print(f"Predicted label for the new face: {predicted_label} with confidence: {confidence}")
    print("The new face is recognized as the same face.")
else:
    print("The new face is not recognized as the same face.")
