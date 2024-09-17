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


def draw_matches(img1: np.ndarray, img2: np.ndarray, kp1: list, kp2: list, matches: list, size: tuple = (12, 10), img_name: str = None, name_size: int = None, match_color: tuple = None, points_color: tuple = None, flags: int = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) -> None:
    """
    Draw matches between two images using matplotlib.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    kp1 : list
        List of keypoints of the first image.
    kp2 : list
        List of keypoints of the second image.
    matches : list
        List of matches.
    size : tuple, optional
        Size of the image, by default (12, 10).
    img_name : str, optional
        Name of the image, by default None.
    name_size : int, optional
        Size of the name, by default None.
    match_color : tuple, optional
        Color of the matches, by default None.
    points_color : tuple, optional
        Color of the keypoints, by default None.
    flags : int, optional
        Flags for drawing matches, by default cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS.

    Returns
    -------
    None
    """
    if type(img1) != np.ndarray or type(img2) != np.ndarray: raise TypeError("img1 and img2 must be numpy arrays")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")
  
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=match_color, singlePointColor=points_color, flags=flags)

    plt.figure(figsize=size)
    plt.imshow(img_matches)
    plt.axis('off')

    if type(img_name) == str:
        if type(name_size) == int: plt.title(img_name, fontsize=name_size)
        else: plt.title(img_name)

    plt.show()


def draw_correspondences(img1: np.ndarray, img2: np.ndarray, pts_1: np.ndarray, pts_2: np.ndarray, size: tuple = (12, 10), img_name: str = None, name_size: int = None, pts_1_color: tuple = (255, 0, 0), pts_2_color: tuple = (0, 255, 0), lines_color: tuple = (255, 0, 255)) -> None:
    """
    Draw correspondences between two images using matplotlib.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    pts_1 : np.ndarray
        Points of the first image.
    pts_2 : np.ndarray
        Points of the second image.
    size : tuple, optional
        Size of the image, by default (12, 10).
    img_name : str, optional
        Name of the image, by default None.
    name_size : int, optional
        Size of the name, by default None.
    pts_1_color : tuple, optional
        Color of the points of the first image, by default (255, 0, 0).
    pts_2_color : tuple, optional
        Color of the points of the second image, by default (0, 255, 0).
    lines_color : tuple, optional
        Color of the lines, by default (255, 0, 255).

    Returns
    -------
    None
    """
    if type(img1) != np.ndarray or type(img2) != np.ndarray: raise ValueError("img1 and img2 must be numpy arrays")
    if type(size) != tuple or len(size) != 2: raise ValueError("size must be a tuple with 2 elements")
    if size[0] <= 0 or size[1] <= 0: raise ValueError("size elements must be greater than 0")
    if type(pts_1) != np.ndarray or type(pts_2) != np.ndarray: raise ValueError("pts_1 and pts_2 must be numpy arrays")
    if pts_1.shape[1] != 2 or pts_2.shape[1] != 2: raise ValueError("pts_1 and pts_2 must have 2 columns")

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1, :] = img1
    combined_img[:h2, w1:w1 + w2, :] = img2

    pts2_shifted = pts_2.copy()
    pts2_shifted[:, 0] += w1  

    for i in range(len(pts_1)):
        pt1 = tuple(pts_1[i])
        pt2 = tuple(pts2_shifted[i])

        cv2.circle(combined_img, pt1, 5, pts_1_color, -1)  
        cv2.circle(combined_img, pt2, 5, pts_2_color, -1) 

        cv2.line(combined_img, pt1, pt2, lines_color, 2) 

    plt.figure(figsize=size)
    plt.imshow(combined_img)
    plt.axis('off')

    if type(img_name) == str:
        if type(name_size) == int: plt.title(img_name, fontsize=name_size)
        else: plt.title(img_name)

    plt.show()