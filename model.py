from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam

IMGWIDTH = 64

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class CGFACE(Classifier):
    def __init__(self, learning_rate = 0.0001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 1))
        
        x1 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x)
        x1 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x1 = Dropout(0.2)(x1)
        
        x2 = Conv2D(16,(3, 3), padding='same', activation = 'relu')(x1)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        x2 = Dropout(0.2)(x2)
        
        x3 = Conv2D(16, (3, 3), padding='same', activation = 'relu')(x2)
        #x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        x3 = Dropout(0.2)(x3)
        
        x4 = Conv2D(16, (3, 3), padding='same', activation = 'relu')(x3)
        x4 = Dropout(0.2)(x4)
        
        y = Flatten()(x4)
        y = Dense(256)(y)
        y = Dense(1, activation = 'sigmoid')(y)
                  
        return KerasModel(inputs = x, outputs = y)

IMAGE ADD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
path = r"D:\2nd_semester\bigdatascience\project\P1\real_faces\\"
X = []
Y = []

for i in tqdm(os.listdir(path)):
    img = cv2.imread(path + i , cv2.IMREAD_GRAYSCALE)
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    X.append(norm_image)
    Y.append(1)

GENERATOR & DISCRIMINATOR
def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator

def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator



generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')
