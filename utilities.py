import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_img_in_RGB(img: np.ndarray, size: tuple = (12, 10), img_name: str = None, name_size: int = None) -> None: 
    """
    Show image using matplotlib.

    Parameters
    ----------
    img : np.ndarray
        Image to show.
    size : tuple, optional
        Size of the image, by default (10, 8).
    img_name : str, optional
        Name of the image, by default None.
    name_size : int, optional
        Size of the name, by default None.

    Returns
    -------
    None
    """
    if type(img) != np.ndarray: raise TypeError("img must be a numpy array")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")

    plt.figure(figsize=size)
    plt.imshow(img)
    plt.axis('off') 

    if type(img_name) == str:
        if type(name_size) == int: plt.title(img_name, fontsize=name_size)
        else: plt.title(img_name)

    plt.show()


def show_imgs_in_RGB(imgs: list, cols: int = 1, size: tuple = (12, 10), img_names: list = None, name_size: int = None) -> None:
    """
    Show multiple images using matplotlib in a single plot.

    Parameters
    ----------
    imgs : list of np.ndarray
        List of images to show.
    cols : int, optional
        Number of columns in the plot, by default 1.
    size : tuple, optional
        Size of the figure, by default (12, 10).
    img_names : list of str, optional
        List of image names, by default None.
    name_size : int, optional
        Font size of the names, by default None.

    Returns
    -------
    None
    """
    if not all(isinstance(img, np.ndarray) for img in imgs): raise TypeError("All images must be numpy arrays")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")
    if not isinstance(cols, int) or cols <= 0: raise ValueError("cols must be a positive integer")

    rows = (len(imgs) + cols - 1) // cols

    plt.figure(figsize=size)

    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        if img_names and len(img_names) > i:
            plt.title(img_names[i], fontsize=name_size if name_size else None)
            
    plt.tight_layout()
    plt.show()


def show_img_in_grayscale(img: np.ndarray, size: tuple = (12, 10), img_name: str = None, name_size: int = None) -> None: 
    """
    Show image using matplotlib in grayscale.

    Parameters
    ----------
    img : np.ndarray
        Image to show.
    size : tuple, optional
        Size of the image, by default (12, 10).
    img_name : str, optional
        Name of the image, by default None.
    name_size : int, optional
        Size of the name, by default None.

    Returns
    -------
    None
    """
    if type(img) != np.ndarray: raise TypeError("img must be a numpy array")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")

    plt.figure(figsize=size)
    plt.imshow(img, cmap='gray')
    plt.axis('off') 

    if type(img_name) == str:
        if type(name_size) == int: plt.title(img_name, fontsize=name_size)
        else: plt.title(img_name)
        
    plt.show()


def show_keypoints_in_img(img: np.ndarray, keypoints: list, size: tuple = (12, 10), img_name: str = None, name_size: int = None, color: tuple = (0, 0, 0), flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) -> None:
    """
    Show image with keypoints using matplotlib.

    Parameters
    ----------
    img : np.ndarray
        Image to show.
    keypoints : list
        List of keypoints.
    size : tuple, optional
        Size of the image, by default (12, 10).
    img_name : str, optional
        Name of the image, by default None.
    name_size : int, optional
        Size of the name, by default None.
    color : tuple, optional
        Color of the keypoints, by default (0, 0, 0).
    flags : int, optional
        Flags for drawing keypoints, by default cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS.

    Returns
    -------
    None
    """
    if type(img) != np.ndarray: raise TypeError("img must be a numpy array")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")
    
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color, flags)
    
    plt.figure(figsize=size)
    plt.imshow(img_with_keypoints)
    plt.axis('off')

    if type(img_name) == str:
        if type(name_size) == int: plt.title(img_name, fontsize=name_size)
        else: plt.title(img_name)

    plt.show()
