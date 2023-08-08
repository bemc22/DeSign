# COLAB = False
# if COLAB:
#   from google.colab import drive
#   drive.mount('/content/drive', force_remount=True)
#   local_path = '/content/drive/My Drive/hdsp/binary_project/binary-relu'
# else:
#   local_path = r'./'


# import os 
# os.chdir(local_path)
import os
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from binarynet.main import build_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from models.layers  import Threshold2D, Threshold3D
from data.Dataset import load_cifar, load_stl10, load_cifar100

tf.random.set_seed(2224)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#             # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#             # Memory growth must be set before GPUs have been initialized
#         print(e)


BATCH_SIZE = 180 
DATASET = "cifar"
MODE =  None

DATA_AUGMENTATION = dict(
    augment = True,
)


DATASETS = {
    "cifar": load_cifar,
    "stl10": load_stl10,
    "cifar100": load_cifar100,
}

if DATASET == "cifar":
    NUM_CLASSES = 10
elif DATASET == "stl10":
    NUM_CLASSES = 10
elif DATASET == "cifar100":
    NUM_CLASSES = 100


load_fun = DATASETS[DATASET]

train_ds, test_ds = load_fun(batch_size=BATCH_SIZE, num_classes=NUM_CLASSES, **DATA_AUGMENTATION)


EXPERIMENT_ID = 0
EPOCHS = 3
LR_START = 1e-2
LR_END = 1e-4
OPTIMIZER = Adam

K_decay = np.log(LR_END / LR_START) * (1 / EPOCHS)
M_decay = ( LR_END - LR_START) / EPOCHS
B_decay = LR_START

F1_score = tfa.metrics.F1Score(num_classes=10, threshold=None)
optimizer = OPTIMIZER(LR_START)

def exp_decay(epoch, lr):
    print("actual learning_rate: ", lr)
    return LR_START*np.exp(K_decay*epoch)

def lineal_decay(epoch, lr):
    print("actual learning_rate: ", lr)
    return B_decay + M_decay*epoch

lr_decay = tf.keras.callbacks.LearningRateScheduler(lineal_decay)

PESOS_PATH = f'./pesos/binary_relu_{DATASET}_{MODE}_best.tf'
PESOS_LAST_PATH = f'./pesos/binary_relu_{DATASET}_{MODE}_last.tf'

callbacks = [
             ModelCheckpoint(PESOS_PATH, monitor='val_accuracy', save_best_only=True,
              save_weights_only=True, mode='max', verbose=1), lr_decay
] 


THRESH_MODES = {
    'threshold3d': Threshold3D,
    'threshold2d': Threshold2D,
    None: None,
}


KERNEL_FILENAME = 'threshold_2x2x4_shifting_n1_right_v1.mat'
KERNEL_PATH = f'./thresholds/{KERNEL_FILENAME}' 

KERNEL_FILENAME ="random" 
KERNEL_PATH = KERNEL_FILENAME

threshold_layer = THRESH_MODES[MODE]


model = build_model(size=32, kernel_filename=KERNEL_PATH, threshold_layer=threshold_layer)

model.compile(optimizer=optimizer, loss='squared_hinge', metrics=['accuracy', F1_score])
model.summary()

history = model.fit( train_ds , epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks)
model.save_weights(PESOS_LAST_PATH)

print("LAST WEIGTHS")
print("TESTING PERFORMANCE")
model.evaluate(test_ds)
print('')

model.load_weights(PESOS_PATH)

print("BEST WEIGTHS")
print("TESTING PERFORMANCE")
model.evaluate(test_ds)
print('')