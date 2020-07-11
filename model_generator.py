from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
noise_factor=0.5

(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train=x_train.astype('float32') / 255
x_test=x_test.astype('float32') / 255
x_train=np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test=np.reshape(x_test, (len(x_test), 28, 28, 1))
x_train_noisy=x_train+noise_factor*np.random.normal(loc=0,scale=1,size=x_train.shape)
x_test_noisy=x_test+noise_factor*np.random.normal(loc=0,scale=1,size=x_test.shape)
x_train_noisy=np.clip(x_train_noisy,0,1)
x_test_noisy=np.clip(x_test_noisy,0,1)



ip = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu', padding='same')(ip)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
x = Model(ip, x)
x.compile(optimizer='adadelta', loss='binary_crossentropy')
x.fit(x_train_noisy, x_train,epochs=5,batch_size=128,shuffle=True,validation_data=(x_test_noisy, x_test))
x.save('grayscale_model.h5')

