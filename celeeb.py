import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import classification_report

root_dir = r"D:\deep-learning\celeb\Dataset_Celebrities\cropped"
celebrities=os.listdir(root_dir)

print("--------------------------------------\n")


dataset = []
label = []
img_siz = (128, 128)

for i, celebrity_name in tqdm(enumerate(celebrities), desc="Loading Data"):
    celebrity_path = os.path.join(root_dir, celebrity_name)
    celebrity_images = os.listdir(celebrity_path)
    
    for image_name in celebrity_images:
        if image_name.split('.')[1] == 'png':
            image = cv2.imread(os.path.join(celebrity_path, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize(img_siz)
            dataset.append(np.array(image))
            label.append(i)

dataset = np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("Train-Test Split")

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3, random_state=42)

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_split=0.1)


print("--------------------------------------\n")
print("Model Evalutaion Phase.\n")
loss,accuracy=model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')

print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img, model, celebrities):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = Image.fromarray(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    input_img = tf.keras.utils.normalize(input_img, axis=1) 
    predictions = model.predict(input_img)
    predicted_class = np.argmax(predictions)
    celebrity_name = celebrities[predicted_class]
    print(f"The Predicted Celebrity is: {celebrity_name}")

        
make_prediction(os.path.join(root_dir, "lionel_messi", "lionel_messi6.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "roger_federer", "roger_federer4.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "virat_kohli", "virat_kohli6.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "maria_sharapova", "maria_sharapova4.png"), model, celebrities)
make_prediction(os.path.join(root_dir, "serena_williams", "serena_williams7.png"), model, celebrities)