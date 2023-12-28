import numpy as np
from tqdm import tqdm
import scipy.io as sio
from utils import select_file

def find_range(array, value, window):
    """
        Finds the range of indices in 'array' such that the elements are within a specified 'window' of a given 'value'.
        It uses binary search to identify the initial index close to the 'value' and then linearly searches outwards to find the inclusive range.

        Parameters:
        array (array-like): A sorted array of numerical values.
        value (numeric): The center value to find neighbors around.
        window (numeric): The window size around the 'value' to consider for finding neighbors.

        Returns:
        numpy.ndarray: An array of indices indicating the positions in 'array' that are within the 'value' +/- 'window'.
        If no such indices are found, returns an empty array.
    """
    low, high = 0, len(array)
    while low < high:
        mid = (low + high) // 2
        if array[mid] < value - window:
            low = mid + 1
        elif array[mid] > value + window:
            high = mid
        else:
            # Now iterate outwards from the mid point
            l, r = mid, mid
            while l > 0 and array[l] >= value - window:
                l -= 1
            while r < len(array) and array[r] <= value + window:
                r += 1
            return np.arange(l + 1, r)  # Return the range of indices
    return np.array([])  # Return an empty array if nothing is found

def find_3d_neighbors(coord_x, coord_y, time_stamps, target_idx, spatial_window, time_window):
    """
        Identifies neighboring points within a specified spatial and temporal window for a given target point in a 3D space (2D spatial + time).

        Parameters:
        coord_x (array-like): The x coordinates of points in the space.
        coord_y (array-like): The y coordinates of points in the space.
        time_stamps (array-like): The time stamps for each point, representing the temporal dimension.
        target_idx (int): The index of the target point in the arrays.
        spatial_window (float): The radius of the spatial window to consider around the target point.
        time_window (float): The radius of the temporal window to consider around the target time stamp.

        Returns:
        array-like: Indices of points that are neighbors to the target point within the specified spatial and temporal window.
    """

    temporal_window = find_range(time_stamps, time_stamps[target_idx], time_window)
    spatial_neighbors = temporal_window[np.where((coord_x[target_idx] - spatial_window <= coord_x[temporal_window]) & 
                                                 (coord_x[temporal_window] <= coord_x[target_idx] + spatial_window))[0]]
    spatial_neighbors = spatial_neighbors[np.where((coord_y[target_idx] - spatial_window <= coord_y[spatial_neighbors]) & 
                                                   (coord_y[spatial_neighbors] <= coord_y[target_idx] + spatial_window))[0]]

    return spatial_neighbors


def local_plane_fitting(x, y, ts, event_idx, neighborhood_size=3, time_threshold=500):
    """
        Implementing local plane fitting with iterative refinement for event-based data, assuming a spatiotemporal neighborhood.
        This function estimates the plane parameters fitting locally around a specified event and refines the fit iteratively.

        Parameters:
            x, y: Arrays of x and y coordinates of the events.
            ts: Array of timestamps for each event.
            event_idx: Index of the event to consider for local plane fitting.
            neighborhood_size: Dimension of the square neighborhood (default is 5x5).
            time_threshold: Duration to consider for temporal window around the event (default 1000 microseconds).

        Returns:
            final_plane_params: Parameters (a, b, c) of the refined fitted plane.
            neighborhood_idxs: Indices of events in the final neighborhood after refinement.
    """

    neighborhood_idxs = find_3d_neighbors(x, y, ts, event_idx, spatial_window=neighborhood_size, time_window= time_threshold)
    if len(neighborhood_idxs) >= 4:
        A = np.c_[x[neighborhood_idxs], y[neighborhood_idxs], np.ones(neighborhood_idxs.shape)]
        B = ts[neighborhood_idxs]

        plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    else:
        return None, neighborhood_idxs  

    eps = 10e6  
    th1, th2 = 1e-5, 0.05  

    while eps > th1:
        distances = np.abs(np.dot(A, plane_params) - B)
        inliers = distances <= th2
        if np.sum(inliers) < 4:
            break

        A = A[inliers]
        B = B[inliers]
        new_plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        eps = np.linalg.norm(new_plane_params - plane_params)
        plane_params = new_plane_params  
        neighborhood_idxs = neighborhood_idxs[inliers]
    return plane_params, neighborhood_idxs




def calculate_local_flow(x: np.ndarray, y: np.ndarray, t: np.ndarray, neighborhood_size: int = 3):
    """
    Calculate local flow using local plane fitting for each event.

    Parameters:
    x, y, t : Arrays of coordinates and timestamps of events.
    neighborhood_size: Size of the spatial neighborhood.

    Returns:
    local_flow: Array of local flow vectors for each event.
    """
    local_flow = np.zeros((len(x), 2), dtype=np.float32)
    ts_seconds = t * 1e-6

    for idx, (xi, yi, ti) in tqdm(enumerate(zip(x, y, t)), total=len(x)):
        plane_params, neighbors = local_plane_fitting(x, y, ts_seconds, idx)
        if plane_params is None:
            continue

        a, b = plane_params[:2]
        u_chap = np.linalg.norm([a, b])
        z_chap = np.sqrt(a**2 + b**2)

        # Calculating t_chap, differences, inliers in a vectorized manner
        t_chap = a * (xi - x[neighbors]) + b * (yi - y[neighbors])
        differences = np.abs(ts_seconds[neighbors] - ts_seconds[idx] - t_chap)
        inliers = differences < z_chap / 2

        inlier_count = np.sum(inliers)
        if inlier_count >= (0.5 * neighborhood_size**2):
            angle = np.arctan2(b, a)
            local_flow[idx] = np.array([u_chap, angle])
    
    return local_flow


def multi_spatial_scale_maxpooling(x: np.ndarray, y: np.ndarray, t: np.ndarray, local_flow: np.ndarray, time_threshold: int = 500):
    """
    Perform multi-spatial scale max-pooling on the local flow vectors.

    Parameters:
    x, y, t : Arrays of coordinates and timestamps of events.
    local_flow: Array of local flow vectors for each event.
    time_threshold: Time window to consider for each event.

    Returns:
    corrected_flow: Array of flow vectors for each event after max-pooling correction.
    """
    corrected_flow = np.zeros_like(local_flow)
    for idx, (xi, yi, ti) in tqdm(enumerate(zip(x, y, t)), total=len(x)):
        relevant_events = (t >= ti - time_threshold) & (t <= ti + time_threshold)
        event_window = np.where(relevant_events)[0]

        U_mean_values = []
        angle_mean_values = []
        sigma_values = range(10, 100, 10)  # Define spatial scales
        for sigma in sigma_values:
            spatial_window = event_window[(np.abs(x[event_window] - xi) <= sigma) & (np.abs(y[event_window] - yi) <= sigma)]
            if spatial_window.size > 0:
                U_mean = np.mean(local_flow[spatial_window, 0])
                angle_mean = np.mean(local_flow[spatial_window, 1])
            else:
                U_mean = 0
                angle_mean = 0

            U_mean_values.append(U_mean)
            angle_mean_values.append(angle_mean)

        best_sigma_index = np.argmax(U_mean_values)
        best_U = U_mean_values[best_sigma_index]
        best_angle = angle_mean_values[best_sigma_index]
        corrected_flow[idx] = np.array([best_U, best_angle])
    
    return corrected_flow



if __name__ == "__main__":
    try:
        # Retrieving the current path
        # name_data_file = 'datda/datamat.mat'
        name_data_file = select_file()
        data = sio.loadmat(name_data_file)

        # Parameters for visualization
        min_range, max_range = (0, 200000)
        # Access to the data in the .mat file
        ts = data['ts'].reshape(-1)[min_range:max_range]
        x  = data['x'] .reshape(-1)[min_range:max_range]
        y  = data['y'] .reshape(-1)[min_range:max_range]
        flow_local = calculate_local_flow(x, y, ts)
        corrected_flow = multi_spatial_scale_maxpooling(x, y, ts,flow_local)
        # Save data into data folder
        np.save('flow_local_out.npy'  , flow_local   )
        np.save('corrected_flow_out.npy', corrected_flow)
        print('done.')

    except Exception as e:
        print(f"An error occurred: {e}")