import os
import numpy as np
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import nrrd
import cv2

def get_labels_and_clinical(base_dir):
    train_labels_patientid = np.array(pd.read_csv(os.path.join(base_dir, 'label_train.csv'))['ID'])
    train_labels = np.array(pd.read_csv(os.path.join(base_dir, 'label_train.csv'))['Label'])
    test_labels_patientid = np.array(pd.read_csv(os.path.join(base_dir, 'label_test.csv'))['ID'])
    test_labels = np.array(pd.read_csv(os.path.join(base_dir, 'label_test.csv'))['Label'])
    train_clinical = pd.read_csv(os.path.join(base_dir, 'label_train.csv')).to_numpy()[:,3:]
    test_clinical = pd.read_csv(os.path.join(base_dir, 'label_test.csv')).to_numpy()[:,3:]

    return train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical

def get_data_splits(train_data_path, test_data_path, train_labels_patientid, train_labels, test_labels_patientid, test_labels, train_clinical, test_clinical, config):

    all_train_patients = np.array(os.listdir(train_data_path))
    all_test_patients = np.array(os.listdir(test_data_path))

    kf = StratifiedKFold(n_splits=config.k_splits, shuffle=config.shuffle_data, random_state=config.random_seed)
    idx_split = [x for x in kf.split(all_train_patients, train_labels)]

    train_patients = all_train_patients[idx_split[config.fold_no][0]]
    val_patients = all_train_patients[idx_split[config.fold_no][1]]
    test_patients = all_test_patients

    # Generate image paths for train, validation, and test sets
    train_img_paths = generate_img_paths(train_data_path, train_patients)
    val_img_paths = generate_img_paths(train_data_path, val_patients)
    test_img_paths = generate_img_paths(test_data_path, test_patients)

    # Extract labels and clinical data for each dataset
    label_train = extract_labels_or_clinical(train_labels_patientid, train_labels, train_img_paths)
    label_val = extract_labels_or_clinical(train_labels_patientid, train_labels, val_img_paths)
    label_test = extract_labels_or_clinical(test_labels_patientid, test_labels, test_img_paths)

    clinical_train = extract_labels_or_clinical(train_labels_patientid, train_clinical, train_img_paths)
    clinical_val = extract_labels_or_clinical(train_labels_patientid, train_clinical, val_img_paths)
    clinical_test = extract_labels_or_clinical(test_labels_patientid, test_clinical, test_img_paths)

    return train_img_paths, val_img_paths, test_img_paths, label_train, label_val, label_test, clinical_train, clinical_val, clinical_test

def generate_img_paths(data_path, patients):
    patients_paths = [os.path.join(data_path, x) for x in patients]
    img_paths = [glob.glob(os.path.join(x, "*.nrrd")) for x in patients_paths]
    img_paths = [item for sublist in img_paths for item in sublist]
    return img_paths

def extract_labels_or_clinical(patientids, data, img_paths):
    return data[np.squeeze(np.vstack([np.where(patientids == int(x.split("\\")[-2]))[0] for x in img_paths]))]
