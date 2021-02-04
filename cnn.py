from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import keras
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint  
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.core import Activation
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

num_classes=3
epochs = 16
batch_size=16
img_width, img_height, channels = 48, 48, 3

def load_dataset(path, shuffle):
    data = load_files(path,shuffle=shuffle)
    condition_files = np.array(data['filenames'])
    condition_targets = np_utils.to_categorical(np.array(data['target']), num_classes)
    return condition_files, condition_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('CaltechTiny/train', True)
valid_files, valid_targets = load_dataset('CaltechTiny/val', False)
test_files, test_targets = load_dataset('CaltechTiny/test', False)

print('\nThere are %s total images.' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d validation images.' % len(valid_files))
print('There are %d test images.'% len(test_files))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (48, 48, 3)
    img = np.float32(img)
    img = img/255
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
    
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/1
valid_tensors = paths_to_tensor(valid_files).astype('float32')/1
test_tensors = paths_to_tensor(test_files).astype('float32')/1

model = Sequential()
inputShape = (img_width,img_height,channels)
model.add(Conv2D(32, (3, 3), padding="same", input_shape= inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second set of CONV => RELU => POOL layers
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# third set of CONV => RELU => POOL layers
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# softmax classifier
model.add(Dense(256))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
#model = multi_gpu_model(model, gpus=8)
#opt = keras.optimizers.Adadelta()
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpointer = ModelCheckpoint(filepath='caltechtiny_model.h5',period=1)

train_tensors, train_targets = shuffle(train_tensors, train_targets)

model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[checkpointer], verbose=1)

model = load_model('caltechtiny_model.h5')

# get index of predicted label for each image in test set
condition_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(condition_predictions)==np.argmax(test_targets, axis=1))/len(condition_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

y_true = np.argmax(test_targets, axis=1)
y_pred = np.array(condition_predictions)
acc = accuracy_score(y_true, y_pred)
print('acc:', acc)

conf = confusion_matrix(y_true, y_pred)
print('confussion matrix')
print(conf)
