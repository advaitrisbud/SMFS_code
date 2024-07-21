from montageMaker import make_montage
from backgroundSubtraction import subtract_background
from getFits import get_fits
import os
import glob

print("Hello world")
def movie_to_fit(movie, save_data):
    movie_base = os.path.basename(movie)
    movie_name = os.path.splitext(movie_base)[0]

    folder = save_data+movie_name
    os.makedirs(folder, exist_ok=True)

    make_montage(movie)
    print("Montages made")


    # subtract_background(folder, movie_name)
    # print("Montages adjusted")
    csv_file_path = folder + "/" + movie_name
    get_fits(csv_file=csv_file_path, montage_folder=folder)

def process_in_folder(movie_folder, extension):
    search_pattern = os.path.join(movie_folder, f"*.{extension}")
    files = glob.glob(search_pattern)
    print(len(files))
    for file_path in files:
        print(file_path)
        movie_to_fit(file_path, folder_to_save_data)

folder_to_save_data = "/Users/advait/Documents/SMFS_Lab/MnDopedData/ProcessedData/"
folder_containing_movies = "/Users/advait/Documents/SMFS_Lab/MnDopedData/"

process_in_folder(folder_containing_movies, 'tif')