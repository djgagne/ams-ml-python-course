from urllib.request import urlretrieve
import os
from os.path import exists, join
import tarfile


if not exists("data"):
    os.mkdir("data")
csv_tar_file = "https://storage.googleapis.com/track_data_ncar_ams_3km_csv_small/track_data_ncar_ams_3km_csv_small.tar.gz"
nc_tar_file = "https://storage.googleapis.com/track_data_ncar_ams_3km_nc_small/track_data_ncar_ams_3km_nc_small.tar.gz"
print("Get csv files")
urlretrieve(csv_tar_file, join("data", csv_tar_file.split("/")[-1]))
print("Get nc files")
urlretrieve(nc_tar_file, join("data", nc_tar_file.split("/")[-1]))
print("Extract csv tar file")
csv_tar = tarfile.open(join("data", csv_tar_file.split("/")[-1]))
csv_tar.extractall("data/")
csv_tar.close()
print("Extract nc tar file")
nc_tar = tarfile.open(join("data", nc_tar_file.split("/")[-1]))
nc_tar.extractall("data/")
nc_tar.close()
