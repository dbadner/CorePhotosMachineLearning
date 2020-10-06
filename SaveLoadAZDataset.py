import h5py
import numpy as np
import LoadNistDatasets as impData

#load handwritten training data data from CSV
objImp = impData.LoadNistDatasets()
(azData, azLabels) = objImp.load_az_dataset_csv(r'input\a_z_handwritten_data.csv')

#create hdf5 file to write to
f = h5py.File(r'input/az_dataset.hdf5', "w")

#create and write datasets
f.create_dataset("azData", data=azData)
f.create_dataset("azLabels", data=azLabels)
f.close()

#retrieve datasets
f2 = h5py.File(r'input/az_dataset.hdf5', 'r')
azDataRead = f2['azData'][:]
azLabelsRead = f2['azLabels'][:]
f2.close()