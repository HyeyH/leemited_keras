from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf



train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


train_dataset = train.flow_from_directory('milk/train/', 
                                          target_size = (640, 640),
                                          batch_size = 15, 
                                          class_mode = 'binary')

# 각 이미지와 해당 레이블을 출력
for images, labels in train_dataset:
    print("Images shape:", images.shape)
    print("Labels:", labels)
    break  # 한 번만 반복하고 중단

validation_dataset = validation.flow_from_directory('milk/validation/', 
                                          target_size = (640, 640),
                                          batch_size = 4, 
                                          class_mode = 'binary')

model = tf.keras.models.Sequential([  tf.keras.layers.Conv2D(16, (3,3),activation='relu', input_shape=(640, 640, 3)),
                                      tf.keras.layers.MaxPool2D(2, 2), 
                                      tf.keras.layers.Conv2D(32, (3, 3),activation='relu'), 
                                      tf.keras.layers.MaxPool2D(2, 2), 
                                      tf.keras.layers.Conv2D(64, (3, 3),activation='relu'), 
                                      tf.keras.layers.MaxPool2D(2, 2), 
                                      tf.keras.layers.Flatten(),
                                      tf.keras.layers.Dense(512, activation='relu'),
                                      tf.keras.layers.Dense(1, activation='sigmoid')  
                                                                                    ])

model.compile(loss = 'binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
# GPT
# 모델 훈련
model_fit = model.fit(train_dataset, steps_per_epoch=1, epochs=20, validation_data=validation_dataset)
model.save('leemited.keras')
