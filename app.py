## Importing The Dependencies

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 
from cv2 import cv2_imshow
import PIL 
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix


#Loading MINST Data from Keras.datasets

(x_train,y_train),(x_test,y_test) = mnist.load_data()
type(x_train)
# Shape of Numpy Arrays

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# Training Data = 60000
# Testing Data = 10000
# Image Diemention = 28x28
# Grayscale Image = 1 Channel
#Printing 10th images

print(x_train[10])
print(x_train[10].shape)
#Displaying The Imgae
plt.imshow(x_train[25])

#Displaying Labels
print(y_train[25])

## Image Labels
print(y_train.shape,y_test.shape)
#uinque Values in Y_train
print(np.unique(y_train))

#uinque Values in Y_test
print(np.unique(y_test))

# We can use these labels as such or we can also apply OneHOtencoding

# All the images have same diemention in this data set ,if not ,we have to resize all the images to a common dimention
#Scalling the values

x_train = x_train/255
x_test = x_test/255
#Printing 10th images

print(x_train[10])
# Building The Neural Network 
# Setting up the layers of Neural Network

model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28,28)),
      keras.layers.Dense(50,activation='relu'),
      keras.layers.Dense(50,activation='relu'),
      keras.layers.Dense(10,activation='sigmoid')



])
#Compiling the neural network

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
# Training the Neural Network

model.fit(x_train,y_train,epochs=10,)

# Training Data Acurracy is : 98.83%




# ***Accuracy on Test Data***

loss,accuracy = model.evaluate(x_test,y_test)
print(accuracy)
## **Test Data Acurracy is : 96.99%**

print(x_test.shape)
#First test point in x_test

plt.imshow(x_test[0])
plt.show()
print(y_test[0])
Y_pred = model.predict(x_test)
print(Y_pred.shape)
print(Y_pred[0])

# model.predict gives prediction of probability of each class for that data point

# Converting the prediction probability to class label

Label_for_first_image = np.argmax(Y_pred[0])
print(Label_for_first_image)
# Converting the prediction probability to class label for all test data

Y_pred_label = [np.argmax(i) for i in Y_pred]
print(Y_pred_label)


# y_test  - is my true Labels 
# Y_pred labels -  my prdicted labels

## confusion Matrix
conf_max = confusion_matrix(y_test,Y_pred_label)
print(conf_max)
plt.figure(figsize=(15,7))
sns.heatmap(conf_max,annot=True,fmt='d',cmap='Blues')


## Building a Predictive System
input_image_path = '/content/download.png'

input_image = cv2.imread(input_image_path)

type(input_image)
print(input_image)
cv2_imshow(input_image)
input_image.shape
Grayscale = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)
Grayscale.shape
input_image_resize = cv2.resize(Grayscale,(28,28))
input_image_resize.shape
cv2_imshow(input_image_resize)
input_image_resize  = input_image_resize/255
input_reshaped = np.reshape(input_image_resize,[1,28,28])
input_prediction = model.predict(input_reshaped)
print(input_prediction)
input_pred_label = np.argmax(input_prediction)
print(input_pred_label)
# Predictive System
input_image_path = input("Path of the image to be predicted :")

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

Grayscale = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(Grayscale,(28,28))

input_image_resize  = input_image_resize/255

input_reshaped = np.reshape(input_image_resize,[1,28,28])

input_prediction = model.predict(input_reshaped)

input_pred_label = np.argmax(input_prediction)

print("the Handwritten digit recognized as : ",input_pred_label)



import gradio as gr


def predict_image(img):
  img_3d=img.reshape(-1,28,28)
  im_resize=img_3d/255.0
  prediction=model.predict(im_resize)
  pred=np.argmax(input_prediction)
  return pred

iface = gr.Interface(predict_image, inputs="sketchpad", outputs="label")

iface.launch(debug='True')
