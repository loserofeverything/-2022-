import matplotlib.pyplot as plt  
import cv2 
from fatherClass import hw_4
import os
import numpy as np
import random


class PCA(hw_4):

    def __init__(self, train_face_ds, test_face_ds, train_label, test_label):
        self.train_face_ds = train_face_ds
        self.test_face_ds = test_face_ds
        self.train_label = train_label
        self.test_label = test_label

    def cal_eigen_paramp(self, covariance_mat, diff_mat,param_p = 8):
        eigenvalues, eigen_vector = np.linalg.eig(covariance_mat)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues)]#特征值由大到小排序
        eigen_vector = eigen_vector[:, np.argsort(-eigenvalues)]

        
        # chosen_eigenvalues = eigenvalues[:index_p]
        chosen_eigenVec = eigen_vector[:,:param_p]
        chosen_eigenVec = np.dot(np.transpose(diff_mat), chosen_eigenVec)
        return chosen_eigenVec

    def cal_mean_face(self, dataset):
        return(np.mean(dataset, 0))


    def cal_diff(self, dataset, mean_face_vec):
        diff_dataset = np.zeros((dataset.shape[0], dataset.shape[1]))
        for i in range(dataset.shape[0]):
            diff_dataset[i, :] = dataset[i, :] - mean_face_vec
        return diff_dataset

    def cal_covariance_mat(self, diff_mat):
        # length = diff_mat.shape[0]
        # return np.dot(diff_mat, np.transpose(diff_mat))/length
        return np.dot(diff_mat, np.transpose(diff_mat))


    def cal_omega(self, eigen_vector, diff_mat):
        omega_mat = np.dot(np.transpose(eigen_vector), np.transpose(diff_mat))
        return omega_mat

  


    def main(self):
        
        #计算平均脸
        mean_train = self.cal_mean_face(self.train_face_ds)


        #计算差值脸
        diff_train = self.cal_diff(self.train_face_ds, mean_train)
        print(diff_train.shape)
        diff_test = self.cal_diff(self.test_face_ds, mean_train)

        #构建协方差矩阵
        Covariance_mat = self.cal_covariance_mat(diff_train)

        # print(Covariance_mat.shape)
        param_a_list = np.linspace(0.1, 1, 30)
        param_p_list = np.arange(1, 11)
        acc = []

        for param_p in param_p_list:

            eigen_vec = self.cal_eigen_paramp(Covariance_mat, diff_train,param_p)

            #[Ω1,Ω2,...,Ωn]
            omega_train = self.cal_omega(eigen_vec, diff_train)
            omega_test = self.cal_omega(eigen_vec, diff_test)

            print(omega_train.shape , omega_test.shape, len(self.train_label), len(self.test_label))
            
            acc.append(self.recognition(self.train_label, self.test_label, omega_train, omega_test))
            print("acc:",acc)

        return acc


        # plt.xlabel("number of principal components")
        # plt.ylabel("Accuracy")
        # plt.plot(param_p_list, np.array(acc))
        # plt.legend(labels = ["PCA"], loc ='best')
        # plt.savefig("figs/PCA_finalver.png")
        # plt.show()



