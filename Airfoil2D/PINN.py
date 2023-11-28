# Import libraries
import os
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
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Activation,  Dense
from tensorflow.keras.models import Model

# Read data
data = scipy.io.loadmat('Airfoil2D_PIV.mat') # For test purposes
dataMLP = scipy.io.loadmat('Airfoil2D_MLP.mat') # Output of MLP.py

conv_time = 61 # Snapshots per convective time
# Data coming from MLP
T_MLP = dataMLP['T_MLP'][:, 0 : conv_time]
T_MLP = T_MLP - np.min(T_MLP)
U_MLP = dataMLP['U_MLP'][:, 0 : conv_time]
V_MLP = dataMLP['V_MLP'][:, 0 : conv_time]
X_MLP = dataMLP['X_MLP'][:, 0 : conv_time]
Y_MLP = dataMLP['Y_MLP'][:, 0 : conv_time]

# Test data 
T_test = data['T_PIV_test'][:, 0 : conv_time]
T_test = T_test - np.min(T_test)
U_test = data['U_PIV_test'][:, 0 : conv_time]
V_test = data['V_PIV_test'][:, 0 : conv_time]
X_test = data['X_PIV_test'][:, 0 : conv_time]
Y_test = data['Y_PIV_test'][:, 0 : conv_time]

del data, dataMLP

# Dimensions
dim_T_MLP = T_MLP.shape[1]
dim_N_MLP = T_MLP.shape[0]

# PINN grid (increase in refinement)
X_PINN = np.arange(np.min(X_MLP), np.max(X_MLP), 0.005)
Y_PINN = np.arange(np.min(Y_MLP), np.max(Y_MLP), 0.005)
[X_PINN, Y_PINN] = np.meshgrid(X_PINN, Y_PINN)
X_PINN = X_PINN.flatten()[:, None]
Y_PINN = Y_PINN.flatten()[:, None]
dim_N_PINN = X_PINN.shape[0]

T_PINN = T_MLP[0, :][None, :]
T_PINN = np.tile(T_PINN, (dim_N_PINN, 1))
X_PINN = X_PINN[:, 0][:, None]
X_PINN = np.tile(X_PINN, (1, dim_T_MLP))
Y_PINN = Y_PINN[:, 0][:, None]
Y_PINN = np.tile(Y_PINN, (1, dim_T_MLP))
dim_T_PINN = T_PINN.shape[1]

# Model

# Customized dense layer 
class GammaBiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(GammaBiasLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        
        self.gamma = self.add_weight('gamma',
                                     shape = (self.units,),
                                     initializer = 'ones',
                                     trainable = True)
        
        self.w = tfa.layers.WeightNormalization(Dense(self.units, use_bias = False, 
                                    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None),
                                    trainable = True, activation = None))
        

    def call(self, input_tensor):
        return self.gamma * self.w(input_tensor) + self.bias


num_input_variables = 3 # (T, X, Y)
num_output_variables = 4 # (U, V, P_x, P_y)

layers = [num_input_variables] + 10*[num_output_variables*100] + [num_output_variables]

inputs = Input(shape = (num_input_variables, ))
h = GammaBiasLayer(layers[1])(inputs)
h = Activation('tanh')(h) 
for l in layers[2 : -3]:
    h = GammaBiasLayer(l)(h)
    h = Activation('tanh')(h)  
h = GammaBiasLayer(layers[-3])(h)
h = GammaBiasLayer(layers[-2])(h)
outputs = GammaBiasLayer(layers[-1])(h)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

# Loss function
mse = tf.keras.losses.MeanSquaredError()

# Navier-Stokes compliance
@tf.function
def loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, training):
    mse = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent = True) as tape2:
        tape2.watch((t_eqns_batch, x_eqns_batch, y_eqns_batch))
        with tf.GradientTape(persistent = True) as tape1:
            tape1.watch((t_eqns_batch, x_eqns_batch, y_eqns_batch))
            X_eqns_batch = tf.concat([t_eqns_batch, x_eqns_batch, y_eqns_batch], axis = 1)
            Y_eqns_batch = model(X_eqns_batch, training = training) 
            [u_eqns_pred, v_eqns_pred, px_eqns_pred, py_eqns_pred] = tf.split(Y_eqns_batch, num_or_size_splits=Y_eqns_batch.shape[1], axis=1)

    # Derivatives 

        u_t_eqns_pred = tape1.gradient(u_eqns_pred, t_eqns_batch)
        v_t_eqns_pred = tape1.gradient(v_eqns_pred, t_eqns_batch)

        u_x_eqns_pred = tape1.gradient(u_eqns_pred, x_eqns_batch)
        v_x_eqns_pred = tape1.gradient(v_eqns_pred, x_eqns_batch)
        
        u_y_eqns_pred = tape1.gradient(u_eqns_pred, y_eqns_batch)
        v_y_eqns_pred = tape1.gradient(v_eqns_pred, y_eqns_batch)
        
    u_xx_eqns_pred = tape2.gradient(u_x_eqns_pred, x_eqns_batch)
    v_xx_eqns_pred = tape2.gradient(v_x_eqns_pred, x_eqns_batch)

    u_yy_eqns_pred = tape2.gradient(u_y_eqns_pred, y_eqns_batch)
    v_yy_eqns_pred = tape2.gradient(v_y_eqns_pred, y_eqns_batch)

    # Navier-Stokes residuals
    e1 = u_x_eqns_pred + v_y_eqns_pred
    e2 = u_t_eqns_pred + (u_eqns_pred * u_x_eqns_pred + v_eqns_pred * u_y_eqns_pred) + px_eqns_pred - (1.0/Rey) * (u_xx_eqns_pred + u_yy_eqns_pred) 
    e3 = v_t_eqns_pred + (u_eqns_pred * v_x_eqns_pred + v_eqns_pred * v_y_eqns_pred) + py_eqns_pred - (1.0/Rey) * (v_xx_eqns_pred + v_yy_eqns_pred)

    NS_e = mse(0, e1) + mse(0, e2) + mse(0, e3)

    return NS_e

# Compliance with Navier-Stokes and reference velocity field 
@tf.function
def loss_NS_2D_data(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, Rey, lamb, training):
    mse = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent = True) as tape2:
        tape2.watch((t_data_batch, x_data_batch, y_data_batch))
        with tf.GradientTape(persistent = True) as tape1:
            tape1.watch((t_data_batch, x_data_batch, y_data_batch))
            X_data_batch = tf.concat([t_data_batch, x_data_batch, y_data_batch], axis = 1)
            Y_data_batch = model(X_data_batch, training = training) 
            [u_data_pred, v_data_pred, px_data_pred, py_data_pred] = tf.split(Y_data_batch, num_or_size_splits=Y_data_batch.shape[1], axis=1)

    # Derivatives 

        u_t_data_pred = tape1.gradient(u_data_pred, t_data_batch)
        v_t_data_pred = tape1.gradient(v_data_pred, t_data_batch)

        u_x_data_pred = tape1.gradient(u_data_pred, x_data_batch)
        v_x_data_pred = tape1.gradient(v_data_pred, x_data_batch)

        u_y_data_pred = tape1.gradient(u_data_pred, y_data_batch)
        v_y_data_pred = tape1.gradient(v_data_pred, y_data_batch)

    u_xx_data_pred = tape2.gradient(u_x_data_pred, x_data_batch)
    v_xx_data_pred = tape2.gradient(v_x_data_pred, x_data_batch)

    u_yy_data_pred = tape2.gradient(u_y_data_pred, y_data_batch)
    v_yy_data_pred = tape2.gradient(v_y_data_pred, y_data_batch)

    # Navier-Stokes error

    e1 = u_x_data_pred + v_y_data_pred
    e2 = lamb * (u_t_data_pred + (u_data_pred * u_x_data_pred + v_data_pred * u_y_data_pred) + px_data_pred - (1.0/Rey) * (u_xx_data_pred + u_yy_data_pred))
    e3 = lamb * (v_t_data_pred + (u_data_pred * v_x_data_pred + v_data_pred * v_y_data_pred) + py_data_pred - (1.0/Rey) * (v_xx_data_pred + v_yy_data_pred)) 

    return mse(0, e1) + mse(0, e2) + mse(0, e3) + mse(u_data_batch, u_data_pred) + mse(v_data_batch, v_data_pred)


def loss_total(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, lamb, training):
    NS_e = lamb * loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, training)
    UV_e = loss_NS_2D_data(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, Rey, lamb, training) 

    total_e = NS_e  + UV_e

    return  (NS_e ** 2 + UV_e ** 2) / total_e 

# Optimize model - gradients:
def grad(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, lamb):
    with tf.GradientTape() as tape:
        loss_value = loss_total(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, lamb, training = True)
    gradient_model = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradient_model


# Create an optimizer
model_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# Keep results for plotting
train_loss_results = []
NS_loss_results = []
UV_loss_results = []

# Training
num_epochs = 1000 # Number of epochs
batch_PINN = int(np.floor(dim_N_PINN / 2)) # Batch size for PINN reconstruction
batch_MLP = int(np.ceil(batch_PINN * dim_N_MLP / dim_N_MLP)) # Batch size for MLP reference data
lamb = 1 # Lambda control parameter for NS regularization
Rey = 3670 # Reynolds number
dim_T_data = dim_T_MLP 
dim_N_data = dim_N_MLP
dim_T_eqns = dim_T_PINN
dim_N_eqns = dim_N_PINN

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_NS_loss_avg = tf.keras.metrics.Mean()
    epoch_UV_loss_avg = tf.keras.metrics.Mean()

    # Reference data
    idx_t = np.random.choice(dim_T_MLP, dim_T_data, replace = False)
    idx_x = np.random.choice(dim_N_MLP, dim_N_data, replace = False)
    t_data = T_MLP[:, idx_t][idx_x,:].flatten()[:,None]
    x_data = X_MLP[:, idx_t][idx_x,:].flatten()[:,None]
    y_data = Y_MLP[:, idx_t][idx_x,:].flatten()[:,None]
    u_data = U_MLP[:, idx_t][idx_x,:].flatten()[:,None]
    v_data = V_MLP[:, idx_t][idx_x,:].flatten()[:,None]

    # Final output grid for regularization
    idx_t = np.random.choice(dim_T_PINN, dim_T_eqns, replace = False)
    idx_x = np.random.choice(dim_N_PINN, dim_N_eqns, replace = False)
    t_eqns = T_PINN[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_PINN[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_PINN[:, idx_t][idx_x,:].flatten()[:,None]

    # Randomize
    idx_batch = np.random.choice(dim_N_MLP * dim_T_MLP, dim_N_data * dim_T_data, replace = False)
    t_data = t_data[idx_batch, :]
    x_data = x_data[idx_batch, :]
    y_data = y_data[idx_batch, :]
    u_data = u_data[idx_batch, :]
    v_data = v_data[idx_batch, :]

    idx_batch = np.random.choice(dim_N_PINN * dim_T_PINN, dim_N_eqns * dim_T_eqns, replace = False)
    t_eqns = t_eqns[idx_batch, :]
    x_eqns = x_eqns[idx_batch, :]
    y_eqns = y_eqns[idx_batch, :]

    div_MLP = range(0, len(x_data), batch_MLP)
    div_PINN = range(0, len(x_eqns), batch_PINN)
    min_div = min([len(div_MLP), len(div_PINN)]) 
    for index in range(0, min_div):
        index_MLP = div_MLP[index]
        index_PINN = div_PINN[index]
        # index_probe = div_probe[index]

        t_data_batch = tf.convert_to_tensor(t_data[index_MLP : index_MLP + batch_MLP, :], dtype = 'float32')
        x_data_batch = tf.convert_to_tensor(x_data[index_MLP : index_MLP + batch_MLP, :], dtype = 'float32')
        y_data_batch = tf.convert_to_tensor(y_data[index_MLP : index_MLP + batch_MLP, :], dtype = 'float32')
        u_data_batch = tf.convert_to_tensor(u_data[index_MLP : index_MLP + batch_MLP, :], dtype = 'float32')
        v_data_batch = tf.convert_to_tensor(v_data[index_MLP : index_MLP + batch_MLP, :], dtype = 'float32')

        t_eqns_batch = tf.convert_to_tensor(t_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')
        x_eqns_batch = tf.convert_to_tensor(x_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')
        y_eqns_batch = tf.convert_to_tensor(y_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')

        X_data_batch = tf.concat([t_data_batch, x_data_batch, y_data_batch], 1)
        loss_train, grads = grad(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, lamb)

        NS_loss = loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, Rey, training = True)
        UV_loss = loss_NS_2D_data(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, v_data_batch, Rey, lamb, training = True)
    
        model_optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_train)
        epoch_NS_loss_avg.update_state(NS_loss)
        epoch_UV_loss_avg.update_state(UV_loss)
     
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    NS_loss_results.append(epoch_NS_loss_avg.result())
    UV_loss_results.append(epoch_UV_loss_avg.result())
    
    # Update learning rate (adaptive)
    if epoch_loss_avg.result() > 1e-2:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif epoch_loss_avg.result() > 1e-3:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    print("Epoch: {:d} Loss_training: {:.3e} NS_loss: {:.3e} UV_loss: {:.3e}".format(epoch, epoch_loss_avg.result(), 
    epoch_NS_loss_avg.result(), epoch_UV_loss_avg.result()))
    
    ################# Save Data ###########################
    if (epoch + 1) % num_epochs == 0:
        U_PINN = np.zeros((dim_N_PINN, dim_T_PINN))
        V_PINN = np.zeros((dim_N_PINN, dim_T_PINN))
        Px_PINN = np.zeros((dim_N_PINN, dim_T_PINN))
        Py_PINN = np.zeros((dim_N_PINN, dim_T_PINN))
        for snap in range(0, dim_T_PINN):
            t_out = T_PINN[:, snap : snap + 1]
            x_out = X_PINN[:, snap : snap + 1]
            y_out = Y_PINN[:, snap : snap + 1]

            X_out = tf.concat([t_out, x_out, y_out], 1)

            # Prediction
            Y_out = model(X_out, training = False)
            [u_pred_out, v_pred_out, px_pred_out, py_pred_out] = tf.split(Y_out, num_or_size_splits=Y_out.shape[1], axis=1)

            U_PINN[:,snap : snap + 1] = u_pred_out 
            V_PINN[:,snap : snap + 1] = v_pred_out 
            Px_PINN[:,snap : snap + 1] = px_pred_out
            Py_PINN[:,snap : snap + 1] = py_pred_out

        scipy.io.savemat('Airfoil2D_PINN.mat',
                            {'U_PINN': U_PINN, 'V_PINN': V_PINN, 'Px_PINN': Px_PINN, 'Py_PINN': Py_PINN, 'T_PINN': T_PINN, 'X_PINN': X_PINN, 'Y_PINN': Y_PINN,
                            'Train_loss_PINN' : train_loss_results, 'NS_loss_PINN' : NS_loss_results, 'UV_loss_PINN' : UV_loss_results})

print('Training completed')