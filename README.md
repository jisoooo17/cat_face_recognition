### python을 활용한 고양이 안면인식 프로그램


얼굴 주요 포인트를 점으로 찍은 JSON 파일과 원본 사진을 대조하는 코드입니다. 

1. show_image_with_landmarks 함수는 주어진 이미지와 해당 이미지에 표시된 얼굴 주요 포인트를 JSON 파일에서 읽어와서 시각화합니다. 

2. 주어진 사진을 기반으로 사용자가 클릭하여 얼굴 주요 부위의 점을 지정하고, 이를 JSON 파일로 저장합니다. 

3. 사전에 훈련된 모델을 사용하여 새로운 얼굴이 이전 얼굴과 일치하는지 확인합니다. 이를 통해 동일한 고양이인지 아닌지를 식별합니다.


---

#### 필수 요구사항
1. Python 환경

---

### 시연 영상
![stack](https://github.com/jisoooo17/readme_img/blob/main/bbangkkeut_campaign/cat_face_recognition.gif)
