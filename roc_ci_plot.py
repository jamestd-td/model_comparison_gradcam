
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  auc
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from sklearn.metrics import roc_curve, precision_recall_curve,confusion_matrix


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
#  Settings for batch size and image size 
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
#  Load the intramural and two extramural data from data directory
# =============================================================================


test_dir='intramural test data path here'
ext_test='extramural test data path here'
ext_test1='extramural test data path here'

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

test_ds_et=datagen.flow_from_directory(
        # This is the target directory
        ext_test,
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


test_ds_et1=datagen.flow_from_directory(
        # This is the target directory
        ext_test1,
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
#  Load model
# =============================================================================

model = 'path to model from model directory'
model=tf.keras.models.load_model(model)

# =============================================================================
#  Holdout test set IMTS1
# =============================================================================

y_pred=model.predict(test_ds_ho)
y_pred_original=model.predict(test_ds_ho)
y_true=test_ds_ho.labels
y_pred = np.argmax(y_pred, axis = 1)
y_pred_proba=y_pred_original[:,1]

# =============================================================================
# external test set1 EMTS1
# =============================================================================

y_pred1=model.predict(test_ds_et)
y_pred_original1=model.predict(test_ds_et)
y_true1=test_ds_et.labels
y_pred1 = np.argmax(y_pred1, axis = 1)
y_pred_proba1=y_pred_original1[:,1]

# =============================================================================
# external test set1 EMTS2
# =============================================================================

y_pred2=model.predict(test_ds_et1)
y_pred_original2=model.predict(test_ds_et1)
y_true2=test_ds_et1.labels
y_pred2 = np.argmax(y_pred2, axis = 1)
y_pred_proba2=y_pred_original2[:,1]

# =============================================================================
# ploting ROC
# =============================================================================

from scipy.stats import sem, t
from numpy import interp


fpr = dict()
tpr = dict()
roc_auc_score = dict()
mean_fpr = np.linspace(0, 1, 100)


fpr1 = dict()
tpr1 = dict()
roc_auc_score1 = dict()
mean_fpr1 = np.linspace(0, 1, 100)


fpr2 = dict()
tpr2 = dict()
roc_auc_score2 = dict()
mean_fpr2 = np.linspace(0, 1, 100)


num_classes=2

# =============================================================================
# ploting ROC IMTS
# =============================================================================

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true, y_pred_proba)
    roc_auc_score[i] = auc(fpr[i], tpr[i])

tprs = []
tprs.append(interp(mean_fpr, fpr[1], tpr[1]))
tprs[-1][0] = 0.0
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_err = sem(tprs, axis=1)

h = std_err * t.ppf(1.95/2, len(mean_tpr) - 1)
tprs_upper = np.minimum(mean_tpr + h, 1)
tprs_lower = np.maximum(mean_tpr - h, 0)
tprs_upper[0]=0.0
tprs_lower[99]=1.0

# =============================================================================
# ploting ROC EMTS1
# =============================================================================

for i in range(num_classes):
    fpr1[i], tpr1[i], _ = roc_curve(y_true1, y_pred_proba1)
    roc_auc_score1[i] = auc(fpr1[i], tpr1[i])

tprs1 = []
tprs1.append(interp(mean_fpr1, fpr1[1], tpr1[1]))
tprs1[-1][0] = 0.0
mean_tpr1 = np.mean(tprs1, axis=0)
mean_tpr1[-1] = 1.0
std_err1 = sem(tprs1, axis=1)

h1 = std_err1 * t.ppf(1.95/2, len(mean_tpr1) - 1)
tprs_upper1 = np.minimum(mean_tpr1 + h1, 1)
tprs_lower1 = np.maximum(mean_tpr1 - h1, 0)
tprs_upper1[0]=0.0
tprs_lower1[99]=1.0

# =============================================================================
# ploting ROC EMTS2
# =============================================================================

for i in range(num_classes):
    fpr2[i], tpr2[i], _ = roc_curve(y_true2, y_pred_proba2)
    roc_auc_score2[i] = auc(fpr2[i], tpr2[i])

tprs2 = []
tprs2.append(interp(mean_fpr2, fpr2[1], tpr2[1]))
tprs2[-1][0] = 0.0
mean_tpr2 = np.mean(tprs2, axis=0)
mean_tpr2[-1] = 1.0
std_err2 = sem(tprs2, axis=1)

h2 = std_err2 * t.ppf(1.95/2, len(mean_tpr2) - 1)
tprs_upper2 = np.minimum(mean_tpr2 + h2, 1)
tprs_lower2 = np.maximum(mean_tpr2 - h2, 0)
tprs_upper2[0]=0.0
tprs_lower2[99]=1.0

# =============================================================================
# ploting ROC 
# =============================================================================

fig=plt.figure(figsize=(15,10), dpi=300)
ax = fig.add_subplot(1, 1, 1)
#ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

major_ticks = np.arange(0.0, 1.10, 0.10)
minor_ticks = np.arange(0.0, 1.10, 0.05)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')
lw = 2.5
legend_properties = {'size':16,'weight':'bold'}

plt.plot(fpr[1], tpr[1], '*-', color='xkcd:royal blue', label=' AUC (IMTS) :  %0.3f CI [%0.3f - %0.3f]' % (roc_auc_score[1], 0.997,1.000),lw = lw,alpha = 1)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'xkcd:royal blue', alpha = 0.3,
                    label=r'95% CI')


plt.plot(fpr1[1], tpr1[1], '*-', color='xkcd:dark maroon', label=' AUC (EMTS1) :  %0.3f CI [%0.3f - %0.3f]' % (roc_auc_score1[1], 0.876,0.912),lw = lw,alpha = 1)
plt.fill_between(mean_fpr1, tprs_lower1, tprs_upper1, color = 'xkcd:dark maroon', alpha = 0.3,
                    label=r'95% CI')


plt.plot(fpr2[1], tpr2[1], '*-', color='xkcd:mulberry', label=' AUC (EMTS2) :  %0.3f CI [%0.3f - %0.3f]' % (roc_auc_score2[1], 0.807,0.924),lw = lw,alpha = 1)
plt.fill_between(mean_fpr2, tprs_lower2, tprs_upper2, color = 'xkcd:mulberry', alpha = 0.3,
                    label=r'95% CI')


plt.plot([0, 1], [0, 1], ':', color='xkcd:red', lw=lw)
plt.xlim([0.0, 1.00])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)

plt.legend(loc="lower right",fontsize=20, prop=legend_properties)
plt.savefig('save fig directory',bbox_inches='tight')
plt.show()
