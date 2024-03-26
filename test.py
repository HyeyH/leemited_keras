import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# 저장된 모델 불러오기
loaded_model = tf.keras.models.load_model('leemited.keras')
# 테스트 이미지를 예측하고 결과 출력
test_images = []
test_filenames = []
test_dir = 'milk/test'
for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = image.load_img(img_path, target_size=(640, 640))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # 이미지 정규화
    test_images.append(img_array)
    test_filenames.append(filename)

test_images = np.array(test_images)

predictions = loaded_model.predict(test_images)

# 이미지와 예측 결과를 함께 시각화
for i in range(len(predictions)):
    img_path = os.path.join(test_dir, test_filenames[i])
    img = image.load_img(img_path, target_size=(640, 640))
    
    plt.imshow(img)
    if predictions[i] < 0.5:
        plt.title("Prediction: bad")
    else:
        plt.title("Prediction: good")
    plt.axis('off')
    plt.show()
