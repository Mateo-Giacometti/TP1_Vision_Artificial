import numpy as np
import matplotlib.pyplot as plt

def show_imgs(imgs: list, cols: int = 1, size: tuple = (12, 10), img_names: list = None, name_size: int = None) -> None:
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