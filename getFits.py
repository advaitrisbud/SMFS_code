def get_fits(csv_file, montage_folder):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio
    from scipy.signal import argrelextrema
    import pandas as pd
    from scipy.optimize import curve_fit
    import cv2, os
    import csv
    from scipy.signal import savgol_filter
    from matplotlib import transforms
    import glob
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    # from scipy.stats import signaltonoise
    # def read_montage(montage_path):
    #     originalImage = cv2.imread(montage_path)
    #     grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    #     image = np.array(grayImage)
    #     height, width = image.shape

    #     wavelength = 1.5*np.arange(width)-40.5

    #     image[:, :300] = np.zeros(image[:, :300].shape)
    #     return image, wavelength


    # def moving_average(data, window_size):
    #     cumsum = np.cumsum(np.insert(data, 0, 0))
    #     return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # def smoothen_image(image_array, x, window_size):
    #     smoothed_array = np.copy(image_array[:, 4:])
    #     for i in range(image_array.shape[0]):
    #         smoothed_array[i, :] = moving_average(image_array[i, :], window_size)
    #     x_smooth = x[(window_size-1):]
    #     # print(smoothed_array.shape, x_smooth.shape)
    #     return smoothed_array, x_smooth


    # def gaussian(x, mu, sigma, a, y0):
    #     return y0 + a * np.exp(- (x - mu)**2 / (2 * sigma**2))

    # def write_parameters_to_csv(parameters, file_path):
    #     """Write fitting parameters to a CSV file."""
    #     # Open the file in write mode
    #     with open(file_path, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # Write the headers (if any)
    #         writer.writerow(['Parameter', 'Value'])
    #         # Write each parameter and its value
    #         for key, value in parameters.items():
    #             writer.writerow([key, value])

    # def fit_gaussian(img, x_smooth, csv_file):
        
    #     for i in range(img.shape[0]):
    #         maxima = np.argmax(img[i, :])
    #         first_peak = x_smooth[maxima]
    #         print(img[i, :].shape, x_smooth.shape)
    #         initial_guess_gauss = [1, first_peak, np.max(img[i, :]), np.min(img[i, :])]
    #         params_gauss, params_covar_gauss = curve_fit(gaussian, x_smooth, img[i, :], p0 = initial_guess_gauss, maxfev=10000)
    #         param_dict = {i: params_gauss[i] for i in range(len(params_gauss))}
    #         write_parameters_to_csv(param_dict, csv_file)




    # def final(image_path, csv_path):
    #     img, x_arr = read_montage(image_path)
    #     img_smooth, x_smooth = smoothen_image(img, x_arr, 5)
    #     fit_gaussian(img_smooth, x_smooth, csv_path)
        
    # final("./Particle_0.png", "./fit_params.csv")

    def write_parameters_to_csv(mean, sigma, amp, y0, ratio, file_path):
            """Write fitting parameters to a CSV file."""
            # Open the file in write mode
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write each parameter and its value
                # writer.writerow(' ')
                writer.writerow([mean, sigma, amp, y0, ratio])

    def get_frame(image, frame):
            return image[frame, :]

    def moving_average(data, window_size):
            cumsum = np.cumsum(np.insert(data, 0, 0))
            return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

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

    def gaussian(x, mu, sigma, a, y0):
            return y0 + a * np.exp(- (x - mu)**2 / (2 * sigma**2))

    def fitting(montage_path, csv_path, frame_number):
        originalImage = cv2.imread(montage_path, cv2.IMREAD_UNCHANGED)
        # print(originalImage.dtype)

        # grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        # print(grayImage.dtype)
        # normalisedGrayImage = cv2.normalize(grayImage, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        image = np.array(originalImage)
        height, width = image.shape
        
        

        # print(height, width)
        
        
        # plt.figure(figsize=(8, 6))
        """"""
        
        y_data = get_frame(image, frame_number)
        # print(y_data.shape)
        # y_data = y_data[200:]
        wavelength = 1.5*np.arange(width)-40.5
        # wavelength = wavelength[200:]
        # h = 6.626*10**-34
        # c = 3*10**8
        # e = 1.6*10**-19
        # wavenumber = 1/(wavelength*10**-7)
        # y_wavenumber = y_data*(wavelength*10e-9)**2

        # Smooth the data first
        

        # Define window size
        window_size = 5

        # Compute moving average
        y_smooth = moving_average(y_data, window_size)
        y_smooth = y_smooth[300:]
        # Adjust x to match the length of y_smooth
        x_smooth = wavelength[(window_size-1):]
        x_smooth = x_smooth[300:]
        # y_smooth = savgol_filter(y_data, 5, 3)
        # wavelength = wavelength[300:]
        # y_smooth = y_smooth[300:]
        # plt.plot(wavelength, y_data, label='Noisy Data')
        # plt.plot(x_smooth, y_smooth, label = 'Smooth Function')
        # plt.grid()
        # plt.legend()

        """"""""

        

        

        baseline = baseline_als(y_smooth, lam=1e5, p=0.01)
        # plt.plot(x_smooth, baseline, label='Baseline', color='red')

        """"""""

        degree = 5
        coeffs = np.polyfit(x_smooth, baseline, degree)
        poly_baseline = np.polyval(coeffs, wavelength)
        corrected_spectrum = y_smooth-baseline

        """"""""

        # plt.figure(figsize=(8,6))
        # plt.plot(x_smooth, y_smooth, label='Original Spectrum')
        # plt.plot(x_smooth, baseline, label='Baseline', color='red')
        # plt.plot(x_smooth, corrected_spectrum, label='Baseline Corrected Spectrum', color='green')
        # plt.xlabel("Wavenumber (cm^-1)")
        # plt.ylabel("Normalised Intensity")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        """"""""

        # plt.subplot(1, 2, 1)
        roi = corrected_spectrum
        roi_wavelength = x_smooth
        # plt.plot(roi_wavelength, roi)

        # plt.subplot(1, 2, 2)

        # # define transformed line
        # plt.hist(y_smooth[150:300], bins=3, orientation = 'horizontal', color = 'g')


        """"""""
        # Fitting single gaussian

        

        

        maxima = np.argmax(roi)
        first_peak = roi_wavelength[maxima]

        initial_guess_gauss = [first_peak, 10, np.max(roi), np.min(roi)]
        param_bounds = ([400, 0, 0, -np.inf], 
                        [900, np.inf, np.inf, np.inf])
        params_gauss, params_covar_gauss = curve_fit(gaussian, roi_wavelength, roi, p0 = initial_guess_gauss,bounds=param_bounds, maxfev=10000)
        mu = params_gauss[0]
        var = params_gauss[1]
        amplitude = params_gauss[2]
        floor = params_gauss[3]
        # print(initial_guess_gauss)
        # print("Fitted Parameters for Gaussian:")
        # print(params_gauss)

        # # y_fit = gaussian(np.linspace(-20000, 20000, 1000), *params_gauss)
        y_fit = gaussian(roi_wavelength, *params_gauss)
        

        # print(mu, var, amplitude, floor, np.mean(roi), np.max(y_fit), np.mean(roi)/np.max(y_fit))

        # plt.figure(figsize=(8, 6))
        # # # # # plt.subplot(1, 2, 1)
        # plt.plot(roi_wavelength, roi, label='Baseline Corrected Spectrum', color='green')
        # plt.plot(roi_wavelength, y_fit, color='red', label='Fitted Gaussian')
        # # # # plt.subplot(1, 2, 2)
        # # # # plt.hist(roi, bins=20, orientation='horizontal')
        # # # # plt.plot(np.linspace(-20000, 20000, 1000), y_fit)
        # # # # plt.axhline(roi.mean(), color='k', linestyle='dashed', linewidth=1)
        # plt.legend()
        # plt.xlabel('Wavelength (nm)')
        # plt.ylabel('Normalised Intensity')
        # plt.title(f'Gaussian Fit, frame number {frame_number}')
        # plt.grid(True)
        # plt.imsave("./"+str(frame_number)+".png")
        # plt.show()
        residuals = roi**2 - y_fit
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((roi**2 - np.mean(roi**2))**2)
        r_squared_gauss_1 = 1 - (ss_res / ss_tot)
        write_parameters_to_csv(mu, var, amplitude, floor, r_squared_gauss_1, csv_path)
        # print(r_squared_gauss_2)
    # file_name = "4-GLS-CLR-100ms"
    # folder = "/Users/advait/Documents/SMFS_Lab/MnZnCdS_Data_To_Transfer/Montages/"+file_name+"/"
    suffix = "*.tif"
    # bad = [22, 42, 44, 48, 49, 50]
    # print(height, width)
    # fitting(montage_path=montage, csv_path=csv_file, frame_number=49)


    file_paths = glob.glob(os.path.join(montage_folder, suffix))

    # for montage in file_paths:
    #     img = imageio.imread(montage)
    #     height, width = img.shape
    #     for frame in range(height):
    #         fitting(montage_path=montage, csv_path=csv_file, frame_number=frame)
        # plt.plot(row)
        # plt.show()

    j=0
    for montage in file_paths:
        j = j + 1
        bad_frames = []
        # csv_file = "/Users/advait/Documents/SMFS_Lab/MnZnCdS_Data_To_Transfer/Montages/"+file_name+"/"+file_name+"_Params.csv"
        
        img = imageio.imread(montage)
        # print(img.shape)
        # print(img)
        # plt.imshow(img)
        # plt.show()
        csv_file_path = csv_file + "_Particle_" + str(j) + "_Params.csv"
        write_parameters_to_csv(" ", " ", " ", " ", " ", csv_file_path)
        image = np.array(img)
        height, width = image.shape
        for i in range(height-1):
            try:
                fitting(montage_path=montage, csv_path=csv_file_path, frame_number=i+1)
            
            except Exception as issue:
                bad_frames.append(i+1)
                write_parameters_to_csv(issue, "bad", "bad", "bad", "bad", csv_file_path)
                continue
        


    # print(bad_frames)
    # plt.plot()
    # with open(csv_file, 'a', newline='') as csvfile:
    #             writer = csv.writer(csvfile)
    #             # Write each parameter and its value
    #             writer.writerow(' ')