import numpy as np

def normalizer(target, source):
    max_s = np.max(source)
    min_s = np.min(source)

    norm_target = (target-min_s)/(max_s-min_s)

    return norm_target


def data_formatter(u_data, g_data, sensor_data, nsamp, eval_points=20, random_eval=True):

    # Assume u_data and g_data are numpy arrays of shape (500, 100)
    all_u_vals = []
    all_g_vals = []
    all_y_coords = []

    for i in range(nsamp):
        u_i = u_data[i]   # shape: (100,)
        g_i = g_data[i]   # shape: (100,)

        # Randomly sample y-locations (interpolate if needed)
        if random_eval:
            y_samples = np.random.uniform(0, 1, eval_points)
        else:
            y_samples = np.linspace(0, 1, eval_points)

        g_samples = np.interp(y_samples, sensor_data, g_i)

        for y, g_y in zip(y_samples, g_samples):
            all_u_vals.append(u_i)               # (100,)
            all_y_coords.append([y])             # (1,)
            all_g_vals.append([g_y])             # (1,)

    # Convert to tensors
    u_tensor = np.array(all_u_vals)     # shape: (500×eval_points, 100)
    y_tensor = np.array(all_y_coords)   # shape: (500×eval_points, 1)
    g_tensor = np.array(all_g_vals)     # shape: (500×eval_points, 1)

    return u_tensor, y_tensor, g_tensor