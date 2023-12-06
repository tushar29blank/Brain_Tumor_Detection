import shutil
import os
import math
import glob
import matplotlib.pyplot as plt # to plot graphs for viusal differencing between parameters
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras
# to count the total no of images in each folder/classes in a dict with index 0 - Tumor , 1- No_Tumor
Root="/content/drive/MyDrive/Brain_tumor_dataset"
no_of_img= {}

# Q='/content/train'
# D='/content/drive/MyDrive/Brain_tumor_dataset/Tumor'
# for i in os.listdir(Q):
#   if(i[0]=='.'):
#     continue

#   if(i[0]!='1' and i[0]!='2' and i[0]!='3' and i[0]!='4' and i[0]!='5' and i[0]!='6' and i[0]!='7' and i[0]!='8' and i[0]!='i' ):
#     E=os.path.join(Q,i)
#     shutil.copy(E,D)
  # E=os.path.join(Q,i)
# shutil.copy('/content/train/1.jpg',D)

for i in os.listdir(Root):
    # or a method is if else if you are dumb enough not to understand this >> approach

    #no_of_img[i]=len(os.listdir("/content/drive/MyDrive/Brain_tumor_dataset"+"/"+i))

    # other method is
    no_of_img[i]=len(os.listdir(os.path.join(Root,i)))


# shutil.copy(Q,D)

no_of_img.items()
# fxn created to split the data acc to required data% and name

def calculate_for_other(name,split):
  if not os.path.exists('./'+name):
    os.mkdir('./'+name)
    for i in os.listdir(Root):
      os.mkdir(os.path.join('.',name,i))

      #for more randomness in selecting the data we use the random method of numpy
      for j in np.random.choice(a= os.listdir(Root+"/"+i),size = math.floor((split)*no_of_img[i])-5,replace = False):

        O=os.path.join(Root,i,j)
        D=os.path.join('./',name,i)

        shutil.copy(O,D) # copies files from one directory/folder to another OR copying images from one path to other
        # os.remove(O) # removes a file

  else :
    print("The folder already exists")

calculate_for_other('train',0.7)
calculate_for_other('valid',0.15)
calculate_for_other('test',0.15)

# CNN MODEL

model= Sequential()

# convloutional layer
# building the convolutional layers
model.add(Conv2D(filters = 16, kernel_size = (3,3),activation = 'relu', input_shape = (224,224,3)))

model.add(Conv2D(filters = 36, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(rate = 0.25)) # to filter or drop 25% data
model.add(Flatten())
model.add(Dense(units = 64 ,activation = 'relu'))
model.add(Dropout(rate = 0.25))

# if the answer is the opposite check this 1
model.add(Dense(units = 1 , activation = 'sigmoid'))

model.summary()
model.compile(optimizer = 'adam', loss = keras.losses.binary_crossentropy, metrics = ['accuracy'])
def preprocessingImages1(path):
  """
    input : Path
    output : Pre Processed Images
  """
  image_data = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, rescale = 1/255, horizontal_flip = True) # DATA AUGMENTATION

  image = image_data.flow_from_directory(directory = path, target_size = (224,224), batch_size = 32 , class_mode = 'binary')

  return image
path1 = '/content/train'
train_data = preprocessingImages1( path1 )
def preprocessingImages2(path):
  """
    input : Path
    output : Pre Processed Images
  """
  image_data = ImageDataGenerator( rescale = 1/255)
  image = image_data.flow_from_directory(directory = path,target_size = (224,224),batch_size = 32 , class_mode = 'binary')

  return image
path2 = '/content/test'
test_data = preprocessingImages2(path2)
path3 = '/content/valid'
valid_data = preprocessingImages2(path3)
# early stoppping and model checking
 # we use early stopping as sometimes our results can come before executing all other epochs
from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping

es = EarlyStopping( monitor = 'val_accuracy' , min_delta = 0.01 , patience = 5,verbose = 1, mode='auto')

# model checkpoint

mc = ModelCheckpoint( monitor = 'val_accuracy' , filepath = './bestmodel.h5',verbose = 1,save_best_only = True, mode='auto')

cd=[es,mc] # as callbacks takes an array so we made an array to pass with early stopping and  model checkpoint
hs = model.fit_generator(generator = train_data ,
                         steps_per_epoch = 20,
                         epochs = 30,
                         verbose =1,
                         validation_data = valid_data ,
                         validation_steps = 10,
                         callbacks = cd )
h = hs.history

h.keys()
plt.plot(h['accuracy'],c='red')

plt.plot(h['val_accuracy'])

plt.title(' Accuracy VS Valid_Accuracy')

plt.show()
plt.plot(h['loss'],c='red')

plt.plot(h['val_loss'])

plt.title(' Loss VS Valid_Loss')

plt.show()
# Model Accuracy

from keras.models import load_model

model = load_model("/content/bestmodel.h5")
acc = model.evaluate_generator(test_data)[1]

print(f"Our model accuracy is {acc*100} %")
from keras.preprocessing.image import load_img,img_to_array

path = '/content/drive/MyDrive/c0472761-800px-wm.jpg'

img = load_img(path,target_size = (224,224))
input_arr = img_to_array(img)/255 # /255 to normalise the image

input_arr = np.expand_dims(input_arr,axis = 0)

pred = model.predict(input_arr)

train_data.class_indices

if(pred == 1):
  print(" The MRI image has a tumor")
else :
  print(" The MRI image is a Healthy image")
