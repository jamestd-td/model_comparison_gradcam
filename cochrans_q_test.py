
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
import glob
import pickle

from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

# =============================================================================
#  Tensorflow setup for using distributed / parallel computing
# =============================================================================


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

strategy=tf.distribute.MirroredStrategy()
print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))



BATCH_SIZE_PER_REPLICA = 16
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

img_height =224
img_width = 224

# =============================================================================
# Pre processing applying to datagenerator
# =============================================================================

datagen_d = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input)

datagen_m =tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=None)

datagen_e = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

datagen_n = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.nasnet.preprocess_input)

datagen_r50 = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

datagen_r101 = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)


datagen_v = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

# =============================================================================
# Set model path
# =============================================================================

path_d121='load model which saved in model directory'
path_d169='load model which saved in model directory'
path_eb3='load model which saved in model directory'
path_eb4='load model which saved in model directory'
path_mv3='load model which saved in model directory'
path_nnet='load model which saved in model directory'
path_r50='load model which saved in model directory'
path_r101='load model which saved in model directory'
path_v16='load model which saved in model directory'

# =============================================================================
# Load all nine models
# =============================================================================

model_d121=tf.keras.models.load_model(path_d121)
model_d169=tf.keras.models.load_model(path_d169)
model_eb3=tf.keras.models.load_model(path_eb3)
model_eb4=tf.keras.models.load_model(path_eb4)
model_mv3=tf.keras.models.load_model(path_mv3)
model_nnet=tf.keras.models.load_model(path_nnet)
model_r50=tf.keras.models.load_model(path_r50)
model_r101=tf.keras.models.load_model(path_r101)
model_v16=tf.keras.models.load_model(path_v16)


# =============================================================================
# Load data from data directory 
# =============================================================================

test_dir='load data from data directory'

# =============================================================================
# Data generation
# =============================================================================

test_dense_net=datagen_d.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )

test_effi=datagen_e.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )


test_mon=datagen_m.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )

test_nas=datagen_n.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )

test_r50=datagen_r50.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )


test_r101=datagen_r101.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )


test_v=datagen_v.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )





# Model Predition
d121_predict=np.argmax(model_d121.predict(test_dense_net),axis=1)
d169_predict=np.argmax(model_d169.predict(test_dense_net),axis=1)
effb3_predict=np.argmax(model_eb3.predict(test_effi),axis=1)
effb4_predict=np.argmax(model_eb4.predict(test_effi),axis=1)
mobv3_predict=np.argmax(model_mv3.predict(test_mon),axis=1)
nnet_predict=np.argmax(model_nnet.predict(test_nas),axis=1)
r50_predict=np.argmax(model_r50.predict(test_r50),axis=1)
r101_predict=np.argmax(model_r101.predict(test_r101),axis=1)
v16_predict=np.argmax(model_v16.predict(test_v),axis=1)


# =============================================================================
# Ground Truth
# =============================================================================

y_true=test_dense_net.labels

# =============================================================================
# Print Cochrain Test results
# =============================================================================

q, p_value = cochrans_q(y_true, 
                        d121_predict, d169_predict, effb3_predict,
                        effb4_predict, mobv3_predict, nnet_predict,
                        r50_predict, r101_predict,v16_predict)
                        
print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)

# =============================================================================
# Pairwise comparison Pair 1
# =============================================================================

pair_m1=[d121_predict,
d121_predict,
d121_predict,
d121_predict,
d121_predict,
d121_predict,
d121_predict,
d121_predict,
d169_predict,
d169_predict,
d169_predict,
d169_predict,
d169_predict,
d169_predict,
d169_predict,
effb3_predict,
effb3_predict,
effb3_predict,
effb3_predict,
effb3_predict,
effb3_predict,
effb4_predict,
effb4_predict,
effb4_predict,
effb4_predict,
effb4_predict,
mobv3_predict,
mobv3_predict,
mobv3_predict,
mobv3_predict,
nnet_predict,
nnet_predict,
nnet_predict,
r50_predict,
r50_predict,
r101_predict]

# =============================================================================
# Pairwise comparison Pair 2
# =============================================================================

pair_m2=[d169_predict,
effb3_predict,
effb4_predict,
mobv3_predict,
nnet_predict,
r50_predict,
r101_predict,
v16_predict,
effb3_predict,
effb4_predict,
mobv3_predict,
nnet_predict,
r50_predict,
r101_predict,
v16_predict,
effb4_predict,
mobv3_predict,
nnet_predict,
r50_predict,
r101_predict,
v16_predict,
mobv3_predict,
nnet_predict,
r50_predict,
r101_predict,
v16_predict,
nnet_predict,
r50_predict,
r101_predict,
v16_predict,
r50_predict,
r101_predict,
v16_predict,
r101_predict,
v16_predict,
v16_predict

]

# =============================================================================
# Pairwise comparison results
# =============================================================================

chi2=dict()
p_value=dict()
num_pair=36

len( pair_m2)

for i in range (num_pair):
    chi2[i], p_value[i] = cochrans_q(y_true, 
                               pair_m1[i], 
                               pair_m2[i])
    print('Cochran\'s Q Chi^2: %.3f Q p-value: %.3f' % (chi2[i] , p_value[i]))

chi2, p_value = cochrans_q(y_true, 
                           d121_predict, 
                           effb3_predict)

print('Cochran\'s Q Chi^2: %.3f Q p-value: %.3f' % (chi2 , p_value))
print('Cochran\'s Q p-value: %.3f' % p_value)
