
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report


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


# =============================================================================
#  setting batch size, input image size
# =============================================================================


BATCH_SIZE_PER_REPLICA = 16
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

img_height =224
img_width = 224


# =============================================================================
#  Set Pre processing function for datagenerator
#  Pre-processing fucnction should choose appropriate function that will train 
#  the revelent model
#  for example training Densenet 121 layered model use 
#  tf.keras.applications.densenet.preprocess_input as pre-processing function
# =============================================================================


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function='appropriate pre-processing function here')


# =============================================================================
#  Load the training, validation and holdout test data from data directory
# =============================================================================

train_dir='training data directory path from data directory'
val_dir='Validation data directory path from data directory'
test_dir='Holdout test data directory path from data directory'



train_ds=datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 224x224
        target_size=(img_height, img_width),
        color_mode='rgb', 
        #classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=True,
        seed=42,
        interpolation='bicubic'
        )

val_ds=datagen.flow_from_directory(
        # This is the target directory
        val_dir,
        # All images will be resized to 224x224
        target_size=(img_height, img_width),
        color_mode='rgb', 
        #classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=True,
        seed=42,
        interpolation='bicubic'
        )

test_ds_ho=datagen.flow_from_directory(
        # This is the target directory
        test_dir,
        target_size=(img_height, img_width),
        color_mode='rgb', 
        #classes=['normal', 'tb'],
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        shuffle=False,
        seed=42,
        interpolation='bicubic'
        )


# =============================================================================
#  Ground Truth
# =============================================================================

ground_trouth=test_ds_ho.class_indices
print(ground_trouth)



# =============================================================================
#  Model Builder
#  Base model sould be load from base model directory with include_top = False
#  for initializing the model, first train the added layers and then save the model
#  for training session 2
#  load saved model from saved model directory and train N1 layers (N1 is varying from 
#  model to model ( readme doc) and save the trained model
#  For training session 3
#  Load model from training session 2 and trained N2 layers (N2 is varying from 
#  model to model ( readme doc) and save the trained model
#  Save final model in model directory
# =============================================================================



base_model=tf.keras.models.load_model('Load appropriate model from directory')

base_model.trainable = False
for layer in base_model.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True

inputs = base_model.input
outputs = base_model.output

model = tf.keras.Model(inputs, outputs)
for layer in model.layers['get appropriate number from readme table']:
   layer.trainable = True
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
              metrics=["accuracy"])

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True,verbose=1)

epochs = 'get appropriate number'
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    batch_size=batch_size,
    shuffle=True,
    callbacks=[callback],
    verbose=1)


# =============================================================================
#  Model testing
# =============================================================================

y_pred=model.predict(test_ds_ho)
y_pred_original=model.predict(test_ds_ho)
y_true=test_ds_ho.labels
y_pred = np.argmax(y_pred, axis = 1)
y_pred_proba=y_pred_original[:,1]

# =============================================================================
#  Delivery report and model performance
# =============================================================================


f1s = [0,0,0]

y_true = tf.cast(y_true, tf.float64)
y_pred = tf.cast(y_pred, tf.float64)

TP = tf.math.count_nonzero(y_pred * y_true)
TN = tf.math.count_nonzero((y_pred -1) * (y_true -1) )
FP = tf.math.count_nonzero(y_pred * (y_true - 1))
FN = tf.math.count_nonzero((y_pred - 1) * y_true)

accuracy=(TP+TN)/(TP+TN+FP+FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
f1 = 2 * precision * recall / (precision + recall )


print("-"*90)
print("Derived Report")
print("-"*90)
print("%s%.2f%s"% ("Accuracy          : ", accuracy*100, "%"))
print("%s%.2f%s"% ("Precision          : ", precision*100, "%"))
print("%s%.2f%s"% ("Sensitivity        : ", recall*100,    "%"))
print("%s%.2f%s"% ("Specificity        : ", specificity*100,    "%"))
print("%s%.2f%s"% ("F1-Score           : ", f1*100,        "%"))
#print("%s%.2f%s"% ("AUC ROC           : ", auc_roc_score_test_set_ho,""))

print("%s%.2f%s"% ("TP                 : ",TP,        ""))
print("%s%.2f%s"% ("TN                 : ",TN,        ""))
print("%s%.2f%s"% ("FP                 : ",FP,        ""))
print("%s%.2f%s"% ("FN                 : ",FN,        ""))

print("-"*90)
print("\n\n")

