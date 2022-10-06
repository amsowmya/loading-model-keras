from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from  keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(32, (3,3), input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model