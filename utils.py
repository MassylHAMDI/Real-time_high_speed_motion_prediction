from tkinter import Tk
from tkinter.filedialog import askopenfilename
import scipy.io as sio
import numpy as np

def select_file():
    """ 
    This function creates a file dialog box using Tkinter for selecting a file. 

    Returns the selected file path.

    Returns:
    -------
        filename(str) : The name of the file path

    """
    # Create a Tkinter root window
    root = Tk()
    root.withdraw()

    while True:
        # Select a file using the filedialog module
        file_path = askopenfilename()   
        break
    return file_path


def load_data(file_path):
    """Load event data from a .mat file."""
    data = sio.loadmat(file_path)
    x  = data["x"].reshape(-1,)       
    y  = data["y"].reshape(-1,)       
    ts = data["ts"].reshape(-1,) 
    p  = data["p"].reshape(-1,)       
    events = np.vstack((x, y, ts* 10**(-6))).transpose() 
    return events, p