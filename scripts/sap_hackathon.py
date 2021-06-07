import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Name of the sensors (sensor directory)
sensors = ['LORD-6305-6000-69503', 'LORD-6305-6000-69504', 'LORD-6305-6000-69505', 'LORD-6305-6000-69506',
           'LORD-6305-6000-69507', 'LORD-6305-6000-69508', 'LORD-6305-6182-92378', 'LORD-6305-6182-92379',
           'LORD-6305-6182-92380', 'LORD-6305-6182-92381', 'LORD-6305-6182-92382', 'LORD-6305-6182-92384',
           'LORD-6305-6182-92385', 'LORD-6305-6182-92386', 'LORD-6305-6182-92387', 'LORD-6305-6182-92388',
           'LORD-6305-6182-92389', 'LORD-6305-6182-92390', 'LORD-6305-6182-92391', 'LORD-6305-6182-92392']

# Training data timings
hour = ['04', '05', '06', '07', '08', '09']

# 50-th quartile of the gravity vector for each sensor listed in sensors
df_grav = {
    0: 9.81979415406838, 1: 9.82009593289354, 2: 9.82009593289354, 3: 9.814867314412359,
    4: 9.818041247913204, 5: 9.815811502542907, 6: 9.812124227663352, 7: 9.814436111134423,
    8: 9.817685170524484, 9: 9.81436124766922, 10: 9.815147201320617, 11: 9.813095176013956,
    12: 9.811811139706977, 13: 9.814886446681331, 14: 9.816697329081773, 15: 9.816717505319415,
    16: 9.815571640047468, 17: 9.816942745154781, 18: 9.813383633528996, 19: 9.816244433680895
}

# 50-th quartile of the gravity vector for each component of sensor listed in sensors
df_quart = {
    0: np.array([9.81771946, 0.15768795, 0.1260014 ]),
    1: np.array([ 9.81933880e+00, -9.21780057e-03,  1.21592447e-01]),
    2: np.array([ 9.81933880e+00, -9.21780057e-03,  1.21592447e-01]),
    3: np.array([ 9.81263828, -0.20169808,  0.05539148]),
    4: np.array([ 9.81706333, -0.13705148, -0.03542068]),
    5: np.array([ 9.81552029, -0.02724112,  0.07053282]),
    6: np.array([ 0.28656024, -9.80789471,  0.02713777]),
    7: np.array([ 0.12891909, -9.81348133, -0.04378248]),
    8: np.array([-0.04740103, -9.81755257,  0.0166657 ]),
    9: np.array([-0.11354903, -9.81324577, -0.09432067]),
    10: np.array([ 0.02930932, -9.81446362, -0.11143518]),
    11: np.array([ 0.11207955, -9.81109524,  0.16171493]),
    12: np.array([ 0.1492956 , -9.81005478,  0.09736438]),
    13: np.array([ 0.25921586, -9.81127262, -0.06109543]),
    14: np.array([ 0.34640458, -9.81041718, -0.05477846]),
    15: np.array([ 0.22514035, -9.81402588,  0.03506773]),
    16: np.array([ 0.06013279, -9.75634289, -1.07483304]),
    17: np.array([ 9.90979224e-02, -9.81643867e+00,  8.72953702e-03]),
    18: np.array([ 0.34735337, -9.80609131,  0.15212294]),
    19: np.array([ 0.21614356, -9.81358624,  0.08078066])
}


def project_back(df, s):
    alpha, beta, gamma = np.arccos(df_quart[s-1] / np.linalg.norm(df_quart[s-1]))
    norms = np.array([df_quart[s-1][0]*np.sin(alpha), df_quart[s-1][1]*np.sin(beta), df_quart[s-1][2]*np.sin(gamma)])
    max_norm_idx = np.argmax(np.abs(norms))
    phi = np.arccos(norms / norms[max_norm_idx])
    df['z_' + str(s)] += df_grav[s-1]
    if max_norm_idx == 0:
        mat = np.array([
            [-np.sin(alpha)*np.cos(phi[1]), np.sin(beta)*np.cos(phi[0]), 0],
            [-np.sin(alpha)*np.cos(phi[2]), 0, np.sin(gamma)*np.cos(phi[0])],
            [np.cos(alpha), np.cos(beta), np.cos(gamma)],
        ])
    if max_norm_idx == 1:    
        mat = np.array([
            [np.sin(alpha)*np.cos(phi[1]), -np.sin(beta)*np.cos(phi[0]), 0],
            [0, -np.sin(beta)*np.cos(phi[2]), np.sin(gamma)*np.cos(phi[1])],
            [np.cos(alpha), np.cos(beta), np.cos(gamma)],
        ])
    if max_norm_idx == 2:
        mat = np.array([
            [np.sin(alpha)*np.cos(phi[2]), 0, -np.sin(gamma)*np.cos(phi[0])],
            [0, np.sin(beta)*np.cos(phi[2]), -np.sin(gamma)*np.cos(phi[1])],
            [np.cos(alpha), np.cos(beta), np.cos(gamma)],
        ])
    X = df[['x_'+str(s), 'y_'+str(s), 'z_'+str(s)]].to_numpy()
    y = np.dot(X, np.linalg.inv(mat.T))
    df_1 = df[['_t']]
    df_1 = pd.concat([df_1, pd.DataFrame(y, columns=['ch1', 'ch2', 'ch3'])], axis=1)
    return df_1


def transform_train_space(df, s):
    df_1 = df[['_t']].copy()    
    df_quart = df.describe().loc['50%', ['ch1', 'ch2', 'ch3']].to_numpy()    
    alpha, beta, gamma = np.arccos(df_quart / np.linalg.norm(df_quart))
    norms = np.array([df_quart[0]*np.sin(alpha), df_quart[1]*np.sin(beta), df_quart[2]*np.sin(gamma)])
    max_norm_idx = np.argmax(np.abs(norms))
    phi = np.arccos(norms / norms[max_norm_idx])
    if max_norm_idx == 0:
        df_1['x_' + str(s)] = df['ch2']*np.sin(beta) - df['ch1']*np.sin(alpha)*np.cos(phi[1])
        df_1['y_' + str(s)] = df['ch3']*np.sin(gamma) - df['ch1']*np.sin(alpha)*np.cos(phi[2])
    if max_norm_idx == 1:
        df_1['x_' + str(s)] = df['ch1']*np.sin(alpha) - df['ch2']*np.sin(beta)*np.cos(phi[0])
        df_1['y_' + str(s)] = df['ch3']*np.sin(gamma) - df['ch2']*np.sin(beta)*np.cos(phi[2])
    if max_norm_idx == 2:
        df_1['x_' + str(s)] = df['ch1']*np.sin(alpha) - df['ch3']*np.sin(gamma)*np.cos(phi[0])
        df_1['y_' + str(s)] = df['ch2']*np.sin(beta) - df['ch3']*np.sin(gamma)*np.cos(phi[1])
    df_1['z_' + str(s)] = (df['ch1']*np.cos(alpha)) + (df['ch2']*np.cos(beta)) + (df['ch3']*np.cos(gamma))
    df_1_quart = df_1.describe().loc['50%', 'z_' + str(s)]
    df_1['z_' + str(s)] = df_1['z_' + str(s)] - df_1_quart
    return df_1, df_quart, df_1_quart 


def transform_test_space(df, df_quart, df_grav, s, plot_space=False):
    df_1 = df[['_t']].copy()
    alpha, beta, gamma = np.arccos(df_quart / np.linalg.norm(df_quart))
    norms = np.array([df_quart[0]*np.sin(alpha), df_quart[1]*np.sin(beta), df_quart[2]*np.sin(gamma)])
    max_norm_idx = np.argmax(np.abs(norms))
    phi = np.arccos(norms / norms[max_norm_idx])
    if max_norm_idx == 0:
        df_1['x_' + str(s)] = df['ch2']*np.sin(beta) - df['ch1']*np.sin(alpha)*np.cos(phi[1])
        df_1['y_' + str(s)] = df['ch3']*np.sin(gamma) - df['ch1']*np.sin(alpha)*np.cos(phi[2])
    if max_norm_idx == 1:
        df_1['x_' + str(s)] = df['ch1']*np.sin(alpha) - df['ch2']*np.sin(beta)*np.cos(phi[0])
        df_1['y_' + str(s)] = df['ch3']*np.sin(gamma) - df['ch2']*np.sin(beta)*np.cos(phi[2])
    if max_norm_idx == 2:
        df_1['x_' + str(s)] = df['ch1']*np.sin(alpha) - df['ch3']*np.sin(gamma)*np.cos(phi[0])
        df_1['y_' + str(s)] = df['ch2']*np.sin(beta) - df['ch3']*np.sin(gamma)*np.cos(phi[1])
    df_1['z_' + str(s)] = (df['ch1']*np.cos(alpha)) + (df['ch2']*np.cos(beta)) + (df['ch3']*np.cos(gamma))
    if not plot_space:
        df_1['z_' + str(s)] = df_1['z_' + str(s)] - df_grav
    return df_1 


def read_file(sensor_name, year, month, day, hours):
    begin = True
    for y in year:
        for m in month:
            for d in day:
                for h in hours:
                    pth = os.path.join(sensor_name, 'year=' + y, 'month=' + m, 'day=' + d, 'hour=' + h)
                    for dirname, _, filenames in os.walk(pth):
                        for i, filename in enumerate(filenames):
                            file_name = os.path.join(dirname, filename)
                            if begin:
                                df = pd.read_csv(file_name)
                                begin = False
                            else:
                                df = pd.concat([df, pd.read_csv(file_name)], ignore_index=True) 
    return df


def read_train_data(sensor_name, year, month, day, hours, s):
    df = read_file(sensor_name, year, month, day, hours)
    df.drop(['date', 'hour', 'day', 'month'], inplace=True, axis=1)
    df.drop_duplicates(ignore_index=True, inplace=True)
    return transform_train_space(df, s+1)


color = ['blue', 'red', 'green']
label = ['channel 1', 'channel 2', 'channel 3']

def plot_before_transformation(df, s):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    fig.suptitle('Before Transformation - Visualization of Data for Sensor ' + sensors[s])
    for i in range(3):
        ax[0].plot(df['ch' + str(i+1)].to_numpy(), color=color[i], label=label[i])
        ax[i+1].plot(df['ch' + str(i+1)].to_numpy(), color=color[i], label=label[i])
        ax[i+1].set_xlabel('Time in seconds')
        ax[i+1].set_ylabel('Acceleration in m/s^2')
        ax[i+1].legend()
    ax[0].set_xlabel('Time in seconds')
    ax[0].set_ylabel('Acceleration in m/s^2')
    ax[0].legend()
    plt.show()


def plot_after_reconstruction(df, s):
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    fig.suptitle('After Reconstruction - Visualization of Data for Sensor ' + sensors[s])
    for i in range(3):
        ax[0].plot(df['ch' + str(i+1)].to_numpy(), color=color[i], label=label[i])
        ax[i+1].plot(df['ch' + str(i+1)].to_numpy(), color=color[i], label=label[i])
        ax[i+1].set_xlabel('Time in seconds')
        ax[i+1].set_ylabel('Acceleration in m/s^2')
        ax[i+1].legend()
    ax[0].set_xlabel('Time in seconds')
    ax[0].set_ylabel('Acceleration in m/s^2')
    ax[0].legend()
    plt.show()


def plot_transformed_axis(df, sensor_idx):
    axis = ['x_' + str(sensor_idx + 1) , 'y_' + str(sensor_idx + 1), 'z_' + str(sensor_idx + 1)]
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
    fig.suptitle('After Transformation - Visualization of Data for Sensor ' + sensors[sensor_idx])
    for i in range(3):
        ax[0].plot(df[axis[i]].to_numpy(), color=color[i], label=label[i])
        ax[i+1].plot(df[axis[i]].to_numpy(), color=color[i], label=label[i])
        ax[i+1].set_xlabel('Time in seconds')
        ax[i+1].set_ylabel('Acceleration in m/s^2')
        ax[i+1].legend()
    ax[0].set_xlabel('Time in seconds')
    ax[0].set_ylabel('Acceleration in m/s^2')
    ax[0].legend()
    plt.show()


def read_test_data(sensor_name, year, month, day, hours, s, print_graph=False):
    df = read_file(sensor_name, year, month, day, hours)
    df.drop(['date', 'hour', 'day', 'month'], inplace=True, axis=1)
    df.drop_duplicates(ignore_index=True, inplace=True)
    if print_graph:
        plot_before_transformation(df, s)
        tr_data = transform_test_space(df, df_quart[s], df_grav[s], s+1, True)
        plot_transformed_axis(tr_data, s)
        tr_data['z_' + str(s+1)] = tr_data['z_' + str(s+1)] - df_grav[s]
        return tr_data
    else:
        return transform_test_space(df, df_quart[s], df_grav[s], s+1)


def estimate_common_time(df_list):
    common_time = None
    for i in range(len(df_list)):
        if i == 0:
            common_time = set(df_list[i]['_t'].tolist())
        else:
            common_time = common_time.intersection(set(df_list[i]['_t'].tolist()))
    common_time = list(common_time)
    return common_time



