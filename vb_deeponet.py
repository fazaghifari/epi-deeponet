import math
import joblib

import tf_keras as keras
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import src.data_processor as dp
import src.deeponet as don

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tf_keras.models import Model
from tf_keras.layers import Input, Lambda, Dense
from matplotlib import pyplot as plt
from timeit import default_timer

def scheduler(epoch):
    if epoch < 200:
        lr = 1e-3
    elif epoch < 1200:
        lr = 5e-4
    elif epoch < 2700:
        lr = 1e-4
    else:
        lr = 5e-5
    
    return lr

def data_preprocess(data_input, data_target, sensor_loc, 
                    n_total=2000, n_test=500, eval_points=50):
    random_num = 42

    u_in_use, g_uy_out_use = data_input[:n_total,...], data_target[:n_total,...]

    idxs = np.arange(u_in_use.shape[0])
    idx_train, idx_test = train_test_split(idxs, test_size=n_test/n_total, 
                                            random_state=random_num)

    u_train, g_train = u_in_use[idx_train,:], g_uy_out_use[idx_train,:]
    u_test, g_test = u_in_use[idx_test,:], g_uy_out_use[idx_test,:]

    u_in, x_t_in, s_in = dp.data_formatter(u_train, g_train, sensor_loc, nsamp=u_train.shape[0], 
                                           eval_points=eval_points)
    
    u_in = dp.normalizer(u_in, u_in)
    x_t_in = dp.normalizer(x_t_in, x_t_in)
    s_in = dp.normalizer(s_in, s_in)
    u_test = dp.normalizer(u_test, u_train)
    s_test = dp.normalizer(g_test, g_train)

    return (u_in, x_t_in, s_in, u_test, s_test)

def train(u_in, s_in, x_t_in, batch_size, network_params, epochs=2000):

    trunk_params = network_params["trunk"]
    branch_params = network_params["branch"]
    
    print("#####  Initializing Model  #####")
    model = don.deeponet(y_in_size=(x_t_in.shape[1]), u_in_size=(u_in.shape[1]), 
                         trunk_params=trunk_params, branch_params=branch_params, batch_size=batch_size, out_feat=1)
    model.summary()
    
    # ELBO loss function
    negloglikelihood = lambda y_true, y_pred: (keras.backend.sum(-y_pred.log_prob(y_true)) + 
                                               (sum(model.losses) / batch_size))
    
    callback = keras.callbacks.LearningRateScheduler(scheduler)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-4),
        loss=negloglikelihood,
        metrics=["mse"])
    
    history = model.fit( x=[x_t_in, u_in], y=s_in, batch_size=batch_size, 
                        epochs=epochs, callbacks=[callback], verbose=1)
    
    return model, history

def mc_preds(y_test, u, model):
    pred_dist = model([y_test,u])
    y_mean = pred_dist.mean()
    y_sigma = np.sqrt(pred_dist.variance())

    return y_mean, y_sigma

def predict(u_test, s_test, sensor_loc, model, plot=True):
    preds_mean = []
    preds_std = []
    y_test = sensor_loc[:,None]
    for i in tqdm(range(100)):
        u = u_test[i][None,:].repeat(100,0)
        pred_mean, pred_std = mc_preds(y_test, u, model)
        preds_mean.append(pred_mean)
        preds_std.append(pred_std)
    
    if plot:
        plotting(preds_mean, preds_std, s_test, y_test)


def plotting(preds_mean, preds_std, s_test, y_test, list_num=[0,7,42,77]):
    
    plt.rcParams['font.family'] = 'Times New Roman' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    colormap = plt.cm.jet  
    colors = [colormap(i) for i in np.linspace(0, 1, 5)]

    fig2 = plt.figure(figsize = (10, 4), dpi=300)
    fig2.suptitle('1D Stochastic PDE - DeepONet')

    index = 0
    for i in list_num:
        plt.plot(y_test[:,0], s_test[i, :], color=colors[index], label='Actual')
        plt.plot(y_test[:,0], preds_mean[i], '--', color=colors[index], label='Prediction')
        plt.fill_between(
                    y_test[:,0],
                    (preds_mean[i] - 2*preds_std[i])[:,0],
                    (preds_mean[i] + 2*preds_std[i])[:,0],
                    alpha=0.2,
                )
        index += 1

    plt.legend(ncol=4, loc=4, labelspacing=0.25, columnspacing=0.25, handletextpad=0.5, handlelength=1)
    plt.grid(True)
    plt.margins(0)
    plt.show()


if __name__ == "__main__":
    
    n_grid = 100

    u_in = np.load("data/poisson1Dinput.npy")
    g_uy_out = np.load("data/poisson1Doutput.npy") * 20
    sensor_data = np.linspace(0,1,n_grid)

    u_in, x_t_in, s_in, u_test, s_test = data_preprocess(u_in, g_uy_out, sensor_data, 
                                                         n_total=2000, n_test=500, eval_points=50)
    
    trunk_params=dict()
    trunk_params["output_features"] = 32
    trunk_params["hidden_features"] = 32
    trunk_params["num_hidden_layers"] = 3

    branch_params=dict()
    branch_params["output_features"] = 32
    branch_params["hidden_features"] = 32
    branch_params["num_hidden_layers"] = 3

    params=dict()
    params["trunk"] = trunk_params
    params["branch"] = branch_params

    model,_ = train(u_in, s_in, x_t_in, batch_size=2048, network_params=params, epochs=3000)

    predict(u_test, s_test, sensor_data, model, plot=True)