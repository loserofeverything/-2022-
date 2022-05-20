import cv2
import random
import numpy as np
import os
from fatherClass import hw_4
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances



class KPCA(hw_4):

    def __init__(self, train_face_ds, test_face_ds, train_label, test_label):
        self.train_face_ds = train_face_ds
        self.test_face_ds = test_face_ds
        self.train_label = train_label
        self.test_label = test_label


    def kernel_mat_gen_1(self, X, gamma):

        #|xi - xj|**2
        dists=pdist(X) ** 2
        mat=squareform(dists)
        #k(xi, xj) = exp(-gamma * || xi - xj ||**2)
        K=np.exp(-mat/(2*(gamma**2)))
        return K

    def kernel_mat_gen_2(self, Y,X,gamma):
        #|xi - xj|**2
        mat=euclidean_distances(Y,X) ** 2
        #k(xi, xj) = exp(-gamma * || xi - xj ||**2)
        K=np.exp(-mat/(2*(gamma**2)))
        return K

    def cal_eigen_paramp(self, K_, param_p = 8):
        eigenvalues,eigen_vector=np.linalg.eig(K_)
        eigenvalues = eigenvalues[np.argsort(-eigenvalues)]#特征值由大到小排序
        eigen_vector = eigen_vector[:, np.argsort(-eigenvalues)]
        chosen_eigenVec = eigen_vector[:,:param_p]/np.sqrt(eigenvalues[:param_p])
        for i in range(param_p):
            chosen_eigenVec[:,i] = chosen_eigenVec[:,i]/np.linalg.norm(chosen_eigenVec[:, i])
        return chosen_eigenVec

    def main(self):
        #设原始数据有m行，n列，则每一行看成X，跟自身及其他m-1行运算，得到m*m的矩阵
        
        sigma=pdist(self.train_face_ds)**2
        sigma=np.sqrt(squareform(sigma))
        sigma[sigma<=0]=float("inf")
        sigma=np.min(sigma,axis=0)
        sigma=5*np.mean(sigma)
        print(sigma)


        #生成核心近似矩阵
        K=self.kernel_mat_gen_1(self.train_face_ds, sigma)
        N=K.shape[0]
        one_n=np.ones([N,N])/N

        #生成中心化核心近似矩阵
        #K' = K - 1nK - K1n + 1nK1n
        K_=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)


        param_p_list = np.arange(1, 11)
        acc = []
        for param_p in param_p_list:
            #得到 (1/sqrt(λ))u
            eigvecs=self.cal_eigen_paramp(K_, param_p)


            K_test=self.kernel_mat_gen_2(self.test_face_ds, self.train_face_ds, sigma)

            #根据公式(v.T)(Φ(x')) = (1/sqrt(λ))(u.T)[k(x1,x')...k(xN,x')].T 
            #计算图片向量在特征空间中的投影
            train_projection=np.transpose(np.dot(K,eigvecs))

            test_projection=np.transpose(np.dot(K_test,eigvecs))
            print('train,test',train_projection.shape,test_projection.shape)
            acc.append(self.recognition(self.train_label, self.test_label, train_projection, test_projection))
            print(acc)

        return acc