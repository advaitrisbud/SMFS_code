def subtract_background(montage_folder, file_name):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio
    from scipy.signal import argrelextrema
    import pandas as pd
    from scipy.optimize import curve_fit
    import cv2
    import csv
    from scipy.signal import savgol_filter
    from matplotlib import transforms
    from scipy.fft import fft, fftfreq
    import pywt
    from skimage import data, restoration, util
    import glob, os

    # def plot_result(image, background):
    #     fig, ax = plt.subplots(nrows=1, ncols=3)

    #     ax[0].imshow(image, cmap='gray')
    #     ax[0].set_title('Original image')
    #     ax[0].axis('off')

    #     ax[1].imshow(background, cmap='gray')
    #     ax[1].set_title('Background')
    #     ax[1].axis('off')

    #     ax[2].imshow(image - background, cmap='gray')
    #     ax[2].set_title('Result')
    #     ax[2].axis('off')

    #     fig.tight_layout()
    
    
    
    i=0
    print(i)
    for montage in montage_folder:

        image = cv2.imread(montage, cv2.IMREAD_UNCHANGED)
        print(image.shape)
        # plt.imshow(image)
        # plt.show()
        background = restoration.rolling_ball(image, radius=50)

        imageio.imwrite(montage_folder+file_name+'_montage_'+str(i)+'.png', image-background)
        print(i)
        i+=1

    # plot_result(image, background)
    # plt.show()