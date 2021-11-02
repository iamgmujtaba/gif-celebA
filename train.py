import os
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import pandas as pd
import numpy as np
import cv2

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model 
from keras.optimizers import SGD

from keras.applications.inception_v3 import InceptionV3, preprocess_input

from utils.lib_utils import print_config, write_config, prepare_output_dirs
from utils.visdata import save_history, plot_confusion_matrix, visualize_dataAug, data_disribution
from config import parse_opts

####################################################################
####################################################################

config = parse_opts()
config = prepare_output_dirs(config)

print_config(config)
write_config(config, os.path.join(config.save_dir, 'config.json'))


# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join(config.checkpoint_dir,'{epoch:03d}-{val_loss:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Save results.
csv_logger = CSVLogger(os.path.join(config.log_dir,'training.log'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=config.early_stopping_patience)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=config.log_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


images_path = os.path.join(config.dataset_path,'img_align_celeba/')
TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000

#dictionary to name the prediction
gender_target = {0: 'Female'
                , 1: 'Male'}

EXAMPLE_PIC = images_path + '000066.jpg'

####################################################################
####################################################################

# import the data set that include the attribute for each picture
df_attr = pd.read_csv(config.dataset_path + 'list_attr_celeba.csv')
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
print(df_attr.shape)
print(df_attr['Male'].value_counts())
# input()

data_disribution(df_attr, os.path.join(config.save_dir,'data_distribute.png'))
visualize_dataAug(EXAMPLE_PIC, os.path.join(config.save_dir,'data_augmen.png'))


# Recomended partition
df_partition = pd.read_csv(config.dataset_path  + 'list_eval_partition.csv')
df_partition.head()

print(df_partition['partition'].value_counts().sort_index())
df_partition['partition'].value_counts().sort_index()
# input()
# join the partition with the attributes
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr['Male'], how='inner')
df_par_attr.head()

####################################################################
####################################################################

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)
    return x

def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''

    df_ = df_par_attr[(df_par_attr['partition'] == partition) 
                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) 
                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_path + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_path + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (config.spatial_size_width, config.spatial_size_hight)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_

####################################################################
####################################################################
# Generate image generator for data augmentation

# Train - Data Preparation - Data Augmentation with generators
train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=30.,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

# Train data
x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
    x_train, 
    y_train,
    batch_size=config.batch_size,)

# Validation Data
x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)

####################################################################
####################################################################

def get_model(show_summary = True):
    # create the base pre-trained model
    base_model = InceptionV3(weights=None, input_shape=(config.spatial_size_hight,config.spatial_size_width, 3), include_top=False)

    # add a global spatial average pooling layer
    input_layer = base_model.output

    layer1 = GlobalAveragePooling2D()(input_layer)
    layer1 = Dense(1024, activation='relu')(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = Dense(512, activation='relu')(layer1)
    
    predictions = Dense(config.num_classes, activation='softmax')(layer1)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Lock initial layers to do not be trained
    for layer in model.layers[:52]:
        layer.trainable = False

    optimizer = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    if show_summary:
        model.summary()
    return model


####################################################################
####################################################################
if __name__ == '__main__':
    model = get_model()

    hist = model.fit_generator(
            train_generator,
            validation_data=(x_valid, y_valid),
            steps_per_epoch=TRAINING_SAMPLES/config.batch_size,
            epochs=config.num_epochs,
            callbacks=[checkpointer, early_stopper, tensorboard, csv_logger, reduce_lr],
            verbose=1)

    save_history(hist, os.path.join(config.save_dir, 'evaluate.png'))
    
    # Save the confusion Matrix
    preds = np.argmax(model.predict(x_valid), axis = 1)
    y_orig = np.argmax(y_valid, axis = 1)
    conf_matrix = confusion_matrix(preds, y_orig)

    keys = OrderedDict(sorted(gender_target.items(), key=lambda t: t[1])).keys()
    plot_confusion_matrix( os.path.join(config.save_dir,'cm.png'), conf_matrix, keys, normalize=True)