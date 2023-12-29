import numpy as np
import matplotlib.pyplot as plt
from utils import select_file, load_data

def plot_events(events, polarity=None, min_range=0, max_range=100000, show_polarity=False):
    """Plot the events in a 3D scatter plot."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select the events for plotting
    selected_events = events[min_range:max_range].transpose() 

    # Scatter plot with or without polarity
    if show_polarity and polarity is not None:
        pol_ = polarity[min_range:max_range]
        pol_color = np.where(pol_ == 1, "blue", "red")
        ax.scatter(*selected_events, color=pol_color, s=8)
    else:
        ax.scatter(*selected_events, s=8)

    # Set axis labels
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_zlabel('ts (s)')

    plt.show()

if __name__ == "__main__":
    try:
        # Load event data
        file_path = select_file()
        events, p = load_data(file_path)

        # Parameters for visualization
        min_range, max_range = (0, 100000)
        show_polarity = True

        # Plotting the events
        plot_events(events, polarity=p, max_range=max_range, show_polarity=show_polarity)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")