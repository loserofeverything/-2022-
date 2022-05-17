import cv2
import random
import numpy as np
import os

class hw_4(object):
    
    def __init__(self, dirs, partition):
        self.dir = dirs
        self.partition = partition

    def create_database(self, database_path="Grp13Dataset/", partition = 0.8):
        """
        输入训练集根目录
        返回训练样本矩阵，其形状为(图片数量,1)
        其每一行为一张图片堆叠成的一列(M*N,1)的列向量
        """

        test_label_array = []
        train_label_array = []
        col_array_train = []
        col_array_test = []
        lsdir = os.listdir(database_path)


        for subdir in lsdir:
            lsimg = os.listdir(os.path.join(database_path, subdir))
            offset = int(len(lsimg) * partition)
            random.shuffle(lsimg)
            train_part = lsimg[:offset]
            test_part = lsimg[offset:]
            for img in train_part:
                train_image_path = database_path + str(subdir) + '/' + str(img)
                train_image = cv2.imread(train_image_path,cv2.IMREAD_GRAYSCALE) 
                train_image = train_image/255
                train_label_array.append(int(str(subdir)[11:])-1)
                col_trainimg_vector = np.reshape(np.array(train_image), (-1, 1), order="F")
                col_array_train.append(col_trainimg_vector)
            for img in test_part:
                test_image_path = database_path + str(subdir) + '/' + str(img)
                test_image = cv2.imread(test_image_path,cv2.IMREAD_GRAYSCALE) 
                test_image = test_image/255
                test_label_array.append(int(str(subdir)[11:])-1)
                col_testimg_vector = np.reshape(np.array(test_image), (-1, 1), order="F")
                col_array_test.append(col_testimg_vector)

        col_trainvector_set = np.mat(np.array(col_array_train))
        col_testvector_set = np.mat(np.array(col_array_test))

        return col_trainvector_set, col_testvector_set, train_label_array, test_label_array

    def cal_eigen_paramp(self, covariance_mat, diff_mat,param_p = 8):
        eigenvalues, eigen_vector = np.linalg.eig(covariance_mat)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues)]#特征值由大到小排序
        eigen_vector = eigen_vector[:, np.argsort(-eigenvalues)]

        
        # chosen_eigenvalues = eigenvalues[:index_p]
        chosen_eigenVec = eigen_vector[:,:param_p]
        chosen_eigenVec = np.dot(np.transpose(diff_mat), chosen_eigenVec)
        return chosen_eigenVec
    
    def cal_eigen_parama(self, covariance_mat, diff_mat, param_a = 0.99):
        eigenvalues, eigen_vector = np.linalg.eig(covariance_mat)
        print(eigenvalues.shape)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues)]#特征值由大到小排序
        eigen_vector = eigen_vector[:, np.argsort(-eigenvalues)]
        total_sum = np.sum(eigenvalues)
        index_p = 1
        
        while np.sum(eigenvalues[:index_p])/total_sum < param_a:
            index_p += 1
        
        # chosen_eigenvalues = eigenvalues[:index_p]
        chosen_eigenVec = eigen_vector[:,:index_p]
        chosen_eigenVec = np.dot(np.transpose(diff_mat), chosen_eigenVec)
        return chosen_eigenVec

    def recognition(self, train_label, test_label, omega_train, omega_test):

        acc = 0
        euclidean_distance = []
        recognition = []
        for testIndex in range(omega_test.shape[1]):
            testi = omega_test[:,testIndex]
            mindistance = float("inf")
            for trainIndex in range(omega_train.shape[1]):
                traini = omega_train[:,trainIndex]
                if np.linalg.norm(testi -traini) ** 2 < mindistance:
                    mindistance = np.linalg.norm(testi -traini) ** 2
                    target_index = trainIndex
            
            if test_label[testIndex] == train_label[target_index]:
                acc += 1
            recognition.append(train_label[target_index])
            
        return acc/omega_test.shape[1]

    def cal_mean_face(self, dataset):
        return(np.mean(dataset, axis = 0))


    def cal_diff(self, dataset, mean_face_vec):
        diff_dataset = np.zeros((dataset.shape[0], dataset.shape[1]))
        for i in range(dataset.shape[0]):
            diff_dataset[i, :] = dataset[i, :] - mean_face_vec
        return diff_dataset