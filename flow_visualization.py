import numpy as np
import cv2 as cv
import scipy.io as sio
from utils import select_file

def calculate_image(x, y, flow_data, indices, dimensions):

    """
        Calculate the visualization image from the flow data.

        Parameters:
        - flow_data (np.ndarray): The flow data array where each row corresponds to a point and the second column contains flow values.
        - indices (np.ndarray): Array of indices to process in the current step.
        - dimensions (tuple): The dimensions (width, height) of the visualization image.

        Returns:
        - np.ndarray: The resulting image in BGR format ready for display.
    """
    image = np.zeros((dimensions[0], dimensions[1], 3), dtype=np.uint8)
    flow = flow_data[indices, 1]  # Assuming second column contains flow values
    # Normalize flow for better visualization
    normalized_flow = cv.normalize(flow, None, 0, 255, cv.NORM_MINMAX)
    x_coords = x[indices].astype(int)
    y_coords = y[indices].astype(int)

    for i, (x_coord, y_coord) in enumerate(zip(x_coords, y_coords)):
        if 0 <= x_coord < dimensions[0] and 0 <= y_coord < dimensions[1]:
            # Apply a color map for visualization
            color = cv.applyColorMap(np.uint8([normalized_flow[i]]), cv.COLORMAP_JET)[0, 0]
            image[x_coord, y_coord ] = color 

    return image

def visualize_flow(x, y, ts, EDL, ARMS, time_delay, step_size):

    """
        Visualize the flow data using OpenCV.

        Parameters:
        - x (np.ndarray): The x coordinates of the flow data points.
        - y (np.ndarray): The y coordinates of the flow data points.
        - ts (np.ndarray): The timestamps for the flow data points.
        - EDL (np.ndarray): The EDL flow data.
        - ARMS (np.ndarray): The ARMS flow data.
        - time_delay (int): Delay in milliseconds between frames.
        - step_size (int): Number of data points to process in each step.

        The function displays two windows showing the flow visualization for EDL and ARMS. Press 'q' to quit the display loop.
    """
    x = x.astype(int)
    y = y.astype(int)
    height, width = max(y) + 100, max(x) + 100

    for ind_min in range(0, len(x), step_size):
        ind_max = min(ind_min + step_size, len(x))
        indices = np.arange(ind_min, ind_max)
        # Update the images for EDL and ARMS
        EDL_image = calculate_image(x, y, EDL, indices, ( width, height ))
        ARMS_image = calculate_image(x, y, ARMS, indices, ( width, height ))

        cv.imshow('Rotated EDL Flow', EDL_image)
        cv.imshow('Rotated ARMS Flow', ARMS_image)
        if cv.waitKey(time_delay) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    try:
        name_data_file = select_file()
        data = sio.loadmat(name_data_file)

        # Parameters for visualization
        min_range, max_range = (0, 20000)
        # Access to the data in the .mat file
        ts = data['ts'].reshape(-1)[min_range:max_range]
        x  = data['x'] .reshape(-1)[min_range:max_range]
        y  = data['y'] .reshape(-1)[min_range:max_range]
        flow_local = np.load('flow_local_out.npy')
        corrected_flow = np.load('corrected_flow_out.npy')

        visualize_flow(x, y, ts, flow_local, corrected_flow, time_delay=1, step_size=1000)
    except Exception as e:
        print(f"An error occurred: {e}")