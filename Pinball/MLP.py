# Import libraries
import os
from re import T
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np
import scipy.io
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Read data
data = scipy.io.loadmat('Case 1/Pinball_PIV.mat') 

# PIV data
T_PIV = data['T_PIV']
noise_level = 0 # Noise level
U_PIV = data['U_PIV'] 
np.random.seed(0)
U_PIV = U_PIV + np.random.normal(0, noise_level, size = U_PIV.shape)
V_PIV = data['V_PIV']
np.random.seed(1)
V_PIV = V_PIV + np.random.normal(0, noise_level, size = V_PIV.shape)
# Probe data 
U_probe = data['U_probe']
np.random.seed(2)
U_probe = U_probe + np.random.normal(0, noise_level, size = U_probe.shape)
V_probe = data['V_probe']
np.random.seed(3)
V_probe = V_probe + np.random.normal(0, noise_level, size = V_probe.shape)

# Test data 
U_test_probe = data['U_probe_test'] 
V_test_probe = data['V_probe_test'] 
U_test_PIV = data['U_PIV_test']
V_test_PIV = data['V_PIV_test']
T_test_PIV = data['T_PIV_test']
X_test_PIV = data['X_PIV_test']
Y_test_PIV = data['Y_PIV_test']

# Subsequent probe sequence per PIV snapshot
MLP_PIV = data['MLP_PIV']
np.random.seed(4)
MLP_PIV = MLP_PIV + np.random.normal(0, noise_level, size = MLP_PIV.shape)
MLP_PIV_test = data['MLP_PIV_test']
np.random.seed(5)
MLP_PIV_test = MLP_PIV_test + np.random.normal(0, noise_level, size = MLP_PIV_test.shape)

del data

# Remove NaN from cylinders
u_mean_PIV = np.mean(U_PIV, axis = 1)[:, None]
U_nan_index = np.argwhere(np.isnan(u_mean_PIV))
U_PIV = np.delete(U_PIV, U_nan_index[:, 0], axis = 0)
V_PIV = np.delete(V_PIV, U_nan_index[:, 0], axis = 0)
U_test_PIV = np.delete(U_test_PIV, U_nan_index[:, 0], axis = 0)
V_test_PIV = np.delete(V_test_PIV, U_nan_index[:, 0], axis = 0)
T_test_PIV = np.delete(T_test_PIV, U_nan_index[:, 0], axis = 0)
X_test_PIV = np.delete(X_test_PIV, U_nan_index[:, 0], axis = 0)
Y_test_PIV = np.delete(Y_test_PIV, U_nan_index[:, 0], axis = 0)

# Dimensions
dim_T_PIV = U_PIV.shape[1]
dim_N_PIV = U_PIV.shape[0]
dim_T_probe = U_probe.shape[1]
dim_N_probe = U_probe.shape[0]
dim_T_test_PIV = U_test_PIV.shape[1]
dim_N_test_PIV = U_test_PIV.shape[0]

# Remove NaN from PIV data
for loop_nan in range(0, 2, 1):
    U_nan_index = np.argwhere(np.isnan(U_PIV))
    U_PIV[U_nan_index[:, 0],  U_nan_index[:, 1]] = U_PIV[U_nan_index[:, 0], U_nan_index[:, 1] - 1]
    V_nan_index = np.argwhere(np.isnan(V_PIV))
    V_PIV[V_nan_index[:, 0],  V_nan_index[:, 1]] = V_PIV[V_nan_index[:, 0], V_nan_index[:, 1] - 1]
    
    U_nan_index = np.argwhere(np.isnan(U_test_PIV))
    U_test_PIV[U_nan_index[:, 0],  U_nan_index[:, 1]] = U_test_PIV[U_nan_index[:, 0], U_nan_index[:, 1] - 1]
    V_nan_index = np.argwhere(np.isnan(V_test_PIV))
    V_test_PIV[V_nan_index[:, 0],  V_nan_index[:, 1]] = V_test_PIV[V_nan_index[:, 0], V_nan_index[:, 1] - 1]

print('Double-check for NaN in PIV data', np.sum(np.isnan(U_PIV)))
print('Double-check for NaN in test data', np.sum(np.isnan(U_test_PIV)))

# POD most energetic modes extraction
def energy_modes(sig_PIV, threshold):
    energy = np.zeros((sig_PIV.shape[0], 1))
    for index in range(0, sig_PIV.shape[0]):
        energy[index, 0] = np.sum(sig_PIV[: index] ** 2) / np.sum(sig_PIV ** 2)  
    return np.argwhere(energy > threshold)[0, 0]

## POD of PIV data
u_mean_PIV = np.mean(U_PIV, axis = 1)[:, None]
v_mean_PIV = np.mean(V_PIV, axis = 1)[:, None]

uvw_PIV = tf.concat(((U_PIV - u_mean_PIV).T, (V_PIV - v_mean_PIV).T), axis = 1)
psi_PIV, sig_PIV, phiT_PIV = np.linalg.svd(uvw_PIV, full_matrices = False)

# Eliminate higher energy modes
print('Original modes = ', len(sig_PIV))
N_modes = energy_modes(sig_PIV, 0.9) 
print('Elbow method = ', N_modes)
psi_PIV_red = psi_PIV[ : , 0 : N_modes]
sig_PIV_red = sig_PIV[ : N_modes]
phiT_PIV_red = phiT_PIV[0 : N_modes, : ]
A_PIV_red = psi_PIV_red @ np.diag(sig_PIV_red)

# POD test data
uvw_test_PIV = tf.concat(((U_test_PIV - u_mean_PIV).T, (V_test_PIV - v_mean_PIV).T), axis = 1)
psi_test_PIV = np.array(uvw_test_PIV @ phiT_PIV.T @ np.diag(sig_PIV ** (-1))) # 
A_test_PIV = psi_test_PIV[:, 0 : N_modes] @ np.diag(sig_PIV_red)

del T_PIV, U_PIV, V_PIV

## Model
n_features = 2 # U, V (probes)
n_steps = dim_N_probe * 150
neurons = int(np.ceil(np.sqrt(n_steps * n_features * N_modes)))

inputs = Input(shape = (n_steps * n_features, ))
h = Dense(neurons, activation = 'tanh')(inputs) 
h = Dense(neurons, activation = 'sigmoid')(h) 
h = Dense(neurons, activation = 'relu')(h) 
h = Dense(neurons, activation = 'relu')(h)
outputs = Dense(N_modes)(h)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

# Loss function
mse = tf.keras.losses.MeanSquaredError()
def loss_temporal(model, MLP_PIV_batch, A_PIV_red_batch, training): 
    A_PIV_pred_batch = model(MLP_PIV_batch, training = training)
    corr =  np.zeros((1, MLP_PIV_batch.shape[0]))
    for index in range(0, MLP_PIV_batch.shape[0]):
        corr[0, index] = np.corrcoef(A_PIV_red_batch[index, :], A_PIV_pred_batch[index, :])[0, 1]
    return  mse(y_true = A_PIV_red_batch, y_pred = A_PIV_pred_batch) / np.mean(abs(corr))

def loss_total(model, MLP_PIV_batch, A_PIV_red_batch, training):
    temporal_e = loss_temporal(model, MLP_PIV_batch, A_PIV_red_batch, training = training)
    return temporal_e 

# Optimize model - gradients:
def grad(model, MLP_PIV_batch, A_PIV_red_batch):
    with tf.GradientTape() as tape:
        loss_value = loss_total(model, MLP_PIV_batch, A_PIV_red_batch, training = True)
    gradient_model = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradient_model

# Create an optimizer
model_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

# Keep results for plotting
train_loss_MLP = []

# Training
num_epochs = 1000 # Number of epochs
snap_batch = int(np.floor(MLP_PIV.shape[0] / 10)) # Batch size

epoch_test = np.zeros((1, num_epochs))
temporal =  np.zeros((1, num_epochs))
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    # Randomize
    idx_epoch = np.random.choice(MLP_PIV.shape[0], MLP_PIV.shape[0], replace = False)
    MLP_PIV_epoch = MLP_PIV[idx_epoch, :]
    A_PIV_epoch = A_PIV_red[idx_epoch, :] 

    for snap in range(0, MLP_PIV.shape[0], snap_batch):
        MLP_PIV_batch = tf.convert_to_tensor(MLP_PIV_epoch[snap : snap + snap_batch, :], dtype = 'float32')
        A_PIV_red_batch = tf.convert_to_tensor(A_PIV_epoch[snap : snap + snap_batch, :], dtype = 'float32')
    
        loss_train, grads = grad(model, MLP_PIV_batch, A_PIV_red_batch)
        model_optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss_train)
    
    # End epoch
    train_loss_MLP.append(epoch_loss_avg.result())

    # Update learning rate (adaptive)
    if epoch_loss_avg.result() > 5e-2:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    elif epoch_loss_avg.result() > 5e-4:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    elif epoch_loss_avg.result() > 5e-6:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    elif epoch_loss_avg.result() > 5e-7:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7)
    else:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
 
    # Test
    A_test = np.zeros((dim_T_test_PIV, N_modes))
    corr =  np.zeros((1, dim_T_test_PIV))
    for snap in range(0, dim_T_test_PIV):
        MLP_out = MLP_PIV_test[snap : snap + 1, :]        
        X_out = tf.convert_to_tensor(MLP_out, dtype = 'float32')
        # Prediction
        Y_out = model(X_out, training = False)
        A_test[snap : snap + 1, :] = Y_out

    for index in range(0, A_test.shape[0]):
        corr[0, index] = np.corrcoef(A_test_PIV[index, :], A_test[index, :])[0, 1]
    epoch_test[0, epoch] = np.mean(corr)


    print("Epoch: {:d} Loss_training: {:.3e} Test: {:.3e}" 
        .format(epoch, epoch_loss_avg.result(), epoch_test[0, epoch]))

    if (epoch + 1) % num_epochs == 0:
        # Field reconstruction
        A_TR = np.zeros((dim_T_test_PIV, N_modes))
        for snap in range(0, dim_T_test_PIV, 1):
            MLP_out = MLP_PIV_test[snap : snap + 1, :]         
            X_out = tf.convert_to_tensor(MLP_out, dtype = 'float32')
            del MLP_out

            Y_out = model(X_out, training = False)
            A_TR[snap : snap + 1, :] = Y_out

        uvw_pred = np.array(A_TR @ phiT_PIV_red)

        U_MLP = np.array(uvw_pred[:, 0 : int(uvw_pred.shape[1] / 2)]).T + u_mean_PIV
        V_MLP = np.array(uvw_pred[:, int(uvw_pred.shape[1] / 2) : ]).T + v_mean_PIV

        T_MLP = T_test_PIV
        X_MLP = X_test_PIV
        Y_MLP = Y_test_PIV

        scipy.io.savemat('Pinball_MLP.mat',
                            {'U_MLP': U_MLP, 'V_MLP': V_MLP, 'T_MLP': T_MLP, 'X_MLP': X_MLP, 'Y_MLP': Y_MLP, 
                            'Train_loss_MLP': train_loss_MLP, 'Test': epoch_test})

print('Training completed')