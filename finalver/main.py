import imp
import cv2
import random
import numpy as np
import os
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances
from KPCA import KPCA
from PCA import PCA
import matplotlib.pyplot as plt
from fatherClass import hw_4
from LDA import LDA
import datetime

hw4 = hw_4("Grp13Dataset/", 0.8)

train_face_ds, test_face_ds, train_label, test_label = hw4.create_database(hw4.dir, hw4.partition)

print(np.max(train_face_ds))
pca = PCA(train_face_ds, test_face_ds, train_label, test_label)
kpca = KPCA(train_face_ds, test_face_ds, train_label, test_label)

pca_acc = pca.main()
kpca_acc = kpca.main()

param_p_list = np.arange(1, 11)

plt.xlabel("number of principal components")
plt.ylabel("Accuracy")
plt.plot(param_p_list, np.array(pca_acc))
plt.plot(param_p_list, np.array(kpca_acc))
plt.legend(labels = ["PCA", "KPCA"], loc ='best')
plt.savefig("figs/PCA_KPCA" + "_"  +  datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
plt.show()

accuracy = []
partition_list = np.linspace(0.1,0.9,9)
for parti in partition_list:
    hw4.partition = parti
    train_face_ds, test_face_ds, train_label, test_label = hw4.create_database(hw4.dir, hw4.partition)
    pca_4_lda = PCA(train_face_ds, test_face_ds, train_label, test_label)

    c = len(np.unique(test_label))
    N = train_face_ds.shape[0]

    pca_4_lda_p =  N - c

    #计算平均脸
    mean_train = pca_4_lda.cal_mean_face(train_face_ds)
    #计算差值脸
    diff_train = pca_4_lda.cal_diff(train_face_ds, mean_train)
    diff_test = pca_4_lda.cal_diff(test_face_ds, mean_train)
    #构建协方差矩阵
    Covariance_mat = pca_4_lda.cal_covariance_mat(diff_train)
    #(5600, 287) -> (287, 5600)
    eigen_vec = pca_4_lda.cal_eigen_paramp(Covariance_mat, diff_train, pca_4_lda_p)

    train_projection = pca_4_lda.cal_omega(eigen_vec, diff_train)
    test_projection = pca_4_lda.cal_omega(eigen_vec, diff_test)

    lda = LDA(np.transpose(train_projection), np.transpose(test_projection), train_label, test_label)

    # lda = LDA(train_face_ds, test_face_ds, train_label, test_label)
    accuracy.append (lda.main(c))
    print(accuracy)

plt.xlabel("train/test")
plt.ylabel("Accuracy")
plt.plot(partition_list, np.array(accuracy))
plt.legend(labels = ["PCA-LDA"], loc ='best')
plt.savefig("figs/PCA-LDA" + "_"  +  datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg")
plt.show()
