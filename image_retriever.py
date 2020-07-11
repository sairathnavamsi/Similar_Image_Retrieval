import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.datasets import mnist
import cv2
from sklearn.metrics import label_ranking_average_precision_score

(x_train, y_train),(x_test, y_test)=mnist.load_data()
n_samples=30
image_id = 3422

x_train=x_train.astype('float32') / 255
x_test=x_test.astype('float32') / 255
x_train=np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test=np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder = load_model('grayscale_model.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
 

trained_data = encoder.predict(x_train)
reshaped_trained_data = trained_data.reshape(trained_data.shape[0], trained_data.shape[1] * trained_data.shape[2] * trained_data.shape[3])

queryimg_data = encoder.predict(np.array([x_test[image_id]]))
reshapes_queryimg_data = queryimg_data.reshape(queryimg_data.shape[1] * queryimg_data.shape[2] * queryimg_data.shape[3])

distances = []

for img in reshaped_trained_data:
    distance = np.linalg.norm(img - reshapes_queryimg_data)
    distances.append(distance)


distances = np.array(distances)
trained_data_index = np.arange(reshaped_trained_data.shape[0])
labels = np.copy(y_train).astype('float32')
labels[labels != y_test[image_id]] = 0
labels[labels == y_test[image_id]] = 1
distance_with_labels = np.stack((distances, labels, trained_data_index), axis=-1)
sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]
sorted_distances = 1-sorted_distance_with_labels[:, 0]
sorted_labels = sorted_distance_with_labels[:, 1]
sorted_indexes = sorted_distance_with_labels[:, 2]
kept_indexes = sorted_indexes[:n_samples]

score = label_ranking_average_precision_score(np.array([sorted_labels[:n_samples]]), np.array([sorted_distances[:n_samples]]))

print("Accuracy: ", score)
cv2.imshow('Original Image', x_test[image_id])
retrieved_images = x_train[int(kept_indexes[0]), :]
for i in range(1, n_samples):
    retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[i]), :]))
cv2.imshow('Retrieved Images', retrieved_images)
cv2.waitKey(0)


