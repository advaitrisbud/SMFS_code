import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# csv_file = "/Users/advait/Documents/SMFS_Lab/MnZnCdS_Data_To_Transfer/Montages/1-GLS-CLR-100ms/1-GLS-CLR-100ms_Particle_1_Params.csv"
mean_data = np.array([])
sigma_data = np.array([])
R2_data = np.array([])
for i in range(3):
    i = i + 1
    csv_file = "/Users/advait/Documents/SMFS_Lab/MnZnCdS_Data_To_Transfer/Montages/3-GLS-CLR-100ms/3-GLS-CLR-100ms_Particle_"+str(i)+"_Params.csv"
    data = pd.read_csv(csv_file)

    columns = ["Mean", "Sigma", "Max Intensity", "Floor", "R2"]
    data.columns = columns
    # print(data.head())
    # data_good_fit = data[data["R2"]>0]
    # print(data_good_fit.shape)
    mean = pd.to_numeric(data["Mean"])
    linewidth = pd.to_numeric(data["Sigma"])
    rsqr = pd.to_numeric(data["R2"])
    mean = np.array(mean)
    linewidth = np.array(linewidth)
    rsqr = np.array(rsqr)
    # print(mean, linewidth, rsqr)
    mean_data = np.append(mean_data, mean)
    sigma_data = np.append(sigma_data, linewidth)
    R2_data = np.append(R2_data, rsqr)

    # plt.scatter(2*linewidth, mean)
    # plt.xlabel("Linewidth (nm)")
    # plt.ylabel("Mean (nm)")
    # plt.title("Mean against Linewidth for Particle 1")
    # plt.grid()
    # plt.show()
# print(mean_data)
plt.scatter(mean_data, sigma_data)
# plt.hist(sigma_data, bins=20, density=True,color='b') 

# mu, std = norm.fit(sigma_data)
# xmin, xmax = plt.xlim()

# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)

# plt.plot(x, p, 'k', linewidth=2)
# plt.xlabel("Linewidth (nm)")
# plt.xlabel("Mean (nm)")
plt.title("Mean against Linewidth for Particle 1")
plt.grid()
plt.show()