import matplotlib.pyplot as plt  
from fatherClass import hw_4
import numpy as np


class LDA(hw_4):

    def __init__(self, train_face_ds, test_face_ds, train_label, test_label):
        self.train_face_ds = train_face_ds
        self.test_face_ds = test_face_ds
        self.train_label = train_label
        self.test_label = test_label

    #计算某一类的类内平均向量
    def cal_meanVec_within_cluster(self, X,y):
        mean_vectors = []
        for class_label in np.unique(y):
            mean_vectors.append(self.cal_mean_face(X[y == class_label]))
        
        return mean_vectors
        
    # 计算类内散度矩阵
    def cal_Smat_within(self, X, y):
        n_dim = X.shape[1]
        S_Within = np.zeros([n_dim, n_dim])
        mean_vectors = self.cal_meanVec_within_cluster(X, y)
        
        for class_label in np.unique(y):
            within_scatter = np.zeros([n_dim, n_dim])
            
            for sample in X[y == class_label]:
                sample, vec = sample.reshape(n_dim, 1), mean_vectors[class_label].reshape(n_dim, 1)
                within_scatter += np.dot(sample - vec, (sample - vec).T)
            S_Within += within_scatter
        
        return S_Within

    # 计算类间散度矩阵
    def cal_Smat_between(self, X, y):
        n_dim = X.shape[1]
        S_Between = np.zeros([n_dim, n_dim])
        mean_global = self.cal_mean_face(X)
        mean_vectors = self.cal_meanVec_within_cluster(X, y)
        
        for class_label in np.unique(y):
            N = X[y == class_label].shape[0]
            mean_global, vec = mean_global.reshape(n_dim, 1), mean_vectors[class_label].reshape(n_dim, 1)
            S_Between += N * np.dot(vec - mean_global, (vec - mean_global).T)
        
        return S_Between

    def cal_eigen_paramp(self, input_mat, param_p = 8):
        eigenvalues,eigen_vector=np.linalg.eig(input_mat)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues)]#特征值由大到小排序
        eigen_vector = eigen_vector[:, np.argsort(-eigenvalues)]
        chosen_eigenVec = eigen_vector[:,:param_p]
        print(chosen_eigenVec.shape,param_p)
        for i in range(param_p):
            chosen_eigenVec[:, i] = chosen_eigenVec[:, i] / np.linalg.norm(chosen_eigenVec[:, i])    # 归一化
        return chosen_eigenVec

    
    #实施LDA步骤的主程序
    def main(self, kind):
        
        #计算训练数据的类间散度矩阵和类内散度矩阵
        smat_between_train = self.cal_Smat_between(self.train_face_ds, self.train_label)
        smat_within_train = self.cal_Smat_within(self.train_face_ds, self.train_label)
        param_p = kind - 1
        # ((Sw)**-1)(SB)w = λw
        eigVecs = self.cal_eigen_paramp(np.dot(np.linalg.inv(smat_within_train), smat_between_train), param_p)
        train_projection = np.dot(np.transpose(eigVecs), np.transpose(self.train_face_ds))
        test_projection = np.dot(np.transpose(eigVecs), np.transpose(self.test_face_ds))
        acc = self.recognition(self.train_label, self.test_label, train_projection, test_projection)
        return acc
