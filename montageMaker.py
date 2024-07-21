def make_montage(file_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio
    from scipy.signal import argrelextrema
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    import pandas as pd
    import os
    from mpl_toolkits.mplot3d import Axes3D
    import cv2
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    # /Users/advait/Documents/SMFS_Lab/Mn:ZnCdS_Data
    file_name_1 = os.path.basename(file_path)
    file_name = os.path.splitext(file_name_1)[0]
    def baseline_als(y, lam, p, niter=10):
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.diags(w, 0, shape=(L, L))
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return z

    # baseline = baseline_als(y_smooth, lam=1e5, p=0.01)
        # plt.plot(x_smooth, baseline, label='Baseline', color='red')

    """"""""

    degree = 5
        



    def read_tiff_file(file_path):
        # Read the TIFF file
        tiff_reader = imageio.get_reader(file_path)

        # Initialize an empty list to store frames as NumPy arrays
        frames_list = []

        # Iterate through frames and convert each to a NumPy array
        for frame in tiff_reader:
            frame_array = np.array(frame)
            frames_list.append(frame_array)

        # Close the TIFF reader
        tiff_reader.close()

        return frames_list

    frames = read_tiff_file(file_path)
    frames = np.array(frames)
    frames = np.squeeze(frames)
    size = frames.shape
    num_frames = size[0]
    rows = size[1]
    columns = size[2]
    # print(size, frames.dtype)
    '''''Create max intensity image'''
    max_projection = np.max(frames, axis=0)
    # print(max_projection.shape)
    '''''Finding the zeroth order maxima'''
    averaged_over_rows = np.mean(max_projection, axis=0)
    # print(averaged_over_rows.shape)
    # print(averaged_over_rows_and_frames.shape)
    zeroth_order = np.argmax(averaged_over_rows)
    # plt.plot(averaged_over_rows)
    # plt.show()
    '''''Locating the particles along the zeroth order'''
    # zeroth_order_column = frames[:, :, zeroth_order-3:zeroth_order+3]
    # zeroth_order_column = np.mean(zeroth_order_column, axis=0)
    zeroth_column = max_projection[:, zeroth_order-3:zeroth_order+3]
    zeroth_column_mean = np.mean(zeroth_column, axis=1)
    # print(zeroth_column.shape)
    # print(zeroth_order_column.shape)
    zeroth_order_column_norm = cv2.normalize(zeroth_column_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    zeroth_order_column_norm = np.mean(zeroth_order_column_norm, axis=1)
    coeffs = np.polyfit(np.arange(len(zeroth_order_column_norm)), zeroth_order_column_norm, degree)
    poly_baseline = np.polyval(coeffs, np.arange(len(zeroth_order_column_norm)))
    corrected_spectrum = zeroth_order_column_norm-poly_baseline
    # plt.plot(corrected_spectrum)
    # plt.show()
    _, binary_zeroth_order = cv2.threshold(corrected_spectrum, 40, 255, cv2.THRESH_BINARY)
    # # plt.imshow(binary_zeroth_order)
    # # plt.show()
    binary_zeroth_order = np.mean(binary_zeroth_order, axis=1)
    # plt.plot(binary_zeroth_order)
    # plt.show()
    particle_location = np.where(binary_zeroth_order!=0)[0]

    a = [[particle_location[0]]]
    ind = 0
    for i in range(1, len(particle_location)):
        if particle_location[i]-particle_location[i-1] <=10:
            a[ind].append(particle_location[i])
        else:
            ind+=1
            a.append([particle_location[i]])
            
    print(a)

    def make_montage(par_loc, zeroth_order_line):
        for i in range(len(par_loc)):
            particle = frames[:, par_loc[i], zeroth_order_line:]
            background_1 = frames[:, np.max(par_loc[i])+5:np.max(par_loc[i])+10, zeroth_order_line:]
            print(background_1.shape)
            background_1_mean = np.mean(background_1, axis=1)
            background_2 = frames[:, np.min(par_loc[i])-10:np.min(par_loc[i])-5:, zeroth_order_line:]
            background_2_mean = np.mean(background_2, axis=1)

            average_background = (background_1_mean+background_2_mean)/2
            # print(particle.dtype)
            particle_mean = np.mean(particle, axis=1).astype(np.uint16)
            particle_subtracted = particle_mean-average_background
            # plt.imshow(particle_subtracted)
            # plt.show()
            # print(particle_mean.dtype)
            # plt.plot(np.arange(1002), particle_mean[1, :])
            # plt.show()
            # print(file_name)
            imageio.imwrite('/Users/advait/Documents/SMFS_Lab/MnDopedData/ProcessedData/'+file_name+"/"+file_name+'_montage_'+str(i)+'.tif', particle_subtracted)
            # print("hello world")

    make_montage(a, zeroth_order_line=zeroth_order)
    '''''Visualise as a surface'''
    # overall_map = np.mean(frames, axis=0)
    # x = np.arange(overall_map.shape[0])
    # y = np.arange(overall_map.shape[1])
    # x, y = np.meshgrid(x, y)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(np.transpose(x), np.transpose(y), overall_map, cmap='viridis')
    # fig.colorbar(surf)
    # plt.show()

    '''''Binary thresholding'''
    # averaged_movie = cv2.normalize(np.mean(frames, axis=0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # _, binary_image = cv2.threshold(averaged_movie, 10, 255, cv2.THRESH_BINARY)

    # spliced_binary_image = binary_image[:, zeroth_order+5:]
    # non_zero_blobs = np.array(np.where(spliced_binary_image==255))
    # plt.scatter(non_zero_blobs[1, :], non_zero_blobs[0, :])
    # # print(spliced_binary_image.shape)
    # plt.imshow(spliced_binary_image)
    # plt.show()
    # plt.imshow(binary_image)
    # plt.show()
    # if binary_image.dtype != np.uint8:
    #     binary_image = cv2.normalize(binary_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    '''''Blob detection'''
    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    # params.maxInertiaRatio = 1
    # params.minArea = 90 # Adjust based on the size of particles
    # params.maxArea = 400  # Adjust based on the size of particles

    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(binary_image)



    # # Draw detected keypoints
    # output_image = cv2.drawKeypoints(averaged_movie, keypoints, 0, 
    #                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # # Show the results
    # plt.figure(figsize=(10, 10))
    # plt.imshow(output_image, cmap='gray')
    # plt.title('Detected Particles')
    # plt.show()