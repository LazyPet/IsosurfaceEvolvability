import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import cv2
import ot
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance

"""
Continous Measure, Similarity Measure 和聚类指标计算, 聚类损失, 透明度损失等
"""
def sigmoid(x, alpha=0.5, mu=0):
    return 1 / (1 + np.exp(-alpha * (x - mu)))

class Core_Calculater():
    def __init__(self):
        self.loss = None             # 损失值

    def CSM(self, img1, img2, a=0.5, C=1):
        if not (isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray)):
            raise ValueError("CSM must imput image numpy array")
        
        # imgcv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # imgcv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



        mssim = ssim(img1 , img2)

        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)

        # 获取所有边缘点的坐标
        edge1_points = np.argwhere(edges1 > 0)  # 获取边缘点的 (y, x) 坐标
        edge2_points = np.argwhere(edges2 > 0)  # 获取边缘点的 (y, x) 坐标

        # 分别提取 X 和 Y 坐标
        edges1_x_coords = edge1_points[:, 1]  # X 坐标
        edges1_y_coords = edge1_points[:, 0]  # Y 坐标
        edges2_x_coords = edge2_points[:, 1]  # X 坐标
        edges2_y_coords = edge2_points[:, 0]  # Y 坐标

        nsmim = self.CaculateWassersteinDistanceDiscrete(edges1_x_coords, edges1_y_coords, edges2_x_coords, edges2_y_coords)

        return mssim, nsmim
    
    def CaculateWassersteinDistanceDiscrete(self, edges1_x_coords, edges1_y_coords, edges2_x_coords, edges2_y_coords, C=1):
        """
        分别计算边缘1与边缘2的x、y各自分布的Wasserstein距离，并进行指数归一化。
        
        :param edges1_x_coords: 边缘1的x坐标
        :param edges1_y_coords: 边缘1的y坐标
        :param edges2_x_coords: 边缘2的x坐标
        :param edges2_y_coords: 边缘2的y坐标
        :param C: 归一化常数
        :return: 两个方向 Wasserstein 距离的归一化结果，分别为 'wd_x' 和 'wd_y'
        """
        # x坐标归一化，用于返回归一化后的结果（二维数组，与ssim对应）

        # 确保输入为数组或列表
        edges1_x_coords = list(edges1_x_coords)
        edges1_y_coords = list(edges1_y_coords)
        edges2_x_coords = list(edges2_x_coords)
        edges2_y_coords = list(edges2_y_coords)

        # 分别计算 x 和 y 坐标的 Wasserstein 距离
        x_num = len(edges1_x_coords)
        y_num = len(edges2_x_coords)

        if len(edges1_x_coords) == 0 or len(edges2_x_coords) == 0:
            wd_final = 0.001  # 如果有空输入，返回0
        else:
            # 计算 Wasserstein 距离
            wd_x = wasserstein_distance(edges1_x_coords, edges2_x_coords)
            wd_y = wasserstein_distance(edges1_y_coords, edges2_y_coords)

            # 计算最终的 Wasserstein 距离（考虑边界像素数量变化率）
            if abs(y_num - x_num)==0:
                wd_final = 0.001
            else:
                wd_final = (wd_x * wd_y) / abs(y_num - x_num)

            # 对最终的 Wasserstein 距离进行指数归一化
        # print(f"{wd_final}  {abs(y_num - x_num)}")
        normalized_wd = np.exp(-(wd_final*wd_final)/C)

        return normalized_wd

    def compute_gradient_variance(self, points):
        """
        计算点集在y方向上的梯度值的方差
        """
        y_values = points[:, 1]  # 获取y方向的值
        gradients = abs(np.diff(y_values))  # 计算y方向上的差分（梯度）
        return np.var(gradients)  # 返回方差
    

    def CalculateSparseClusterLossOne(self, dense_points, sparse_points, boundaries, n):
        """
        计算稀疏聚类损失
        
        :param density_difference: 稠密和稀疏点的密度差 

        """
        # 计算稠密集和稀疏集的y方向上的梯度值的方差
        dense_gradient_variance = self.compute_gradient_variance(dense_points)
        sparse_gradient_variance = self.compute_gradient_variance(sparse_points)
        cluster_gradient_variance = (dense_gradient_variance * dense_points.shape[0] + sparse_gradient_variance * sparse_points.shape[0])

        # 计算每对相邻点的中点的欧几里得距离的方差
        midpoints = (boundaries[:-1] + boundaries[1:]) / 2
        midpoint_distances = np.linalg.norm(np.diff(midpoints, axis=0), axis=1)
        distance_variance = np.var(midpoint_distances)


        # 提取 boundaries 中相邻点的 x 坐标
        x_coordinates = boundaries[:, 0]  # 假设 boundaries 的形状是 (n, 2)，即每个点有 x 和 y 坐标

        # 计算相邻 x 坐标之间的差异
        x_distances = np.diff(x_coordinates)

        # 计算这些 x 坐标差异的方差
        distance_variance_x = np.var(x_distances)

        # 计算稀疏聚类损失 
        loss = 1/n * cluster_gradient_variance * distance_variance * distance_variance_x
        return loss

    def CalculateSparseClusterLoss(self, smims, boundaries, n, tv = 0.95):
        """
        计算稀疏聚类损失
        
        :param density_difference: 稠密和稀疏点的密度差
        :param smims: 演化指数，二维数组
        :param boundaries: 相似性指数稀疏聚类得到的边界集 二维数组
        :param tv: 演化指数阈值
        :return: 稀疏聚类损失
        """
        boundaries = np.delete(boundaries, 0, axis=0)  # 删除第一行，即边界集的第一行，因为第一行是无效的
        boundaries = np.delete(boundaries, -1, axis=0)  # 删除最后一行，即边界集的最后一行，因为最后一行是无效的

        if len(boundaries) == 0:
            return 0

        # 根据阈值确定演化指数
        mask = smims[:, 1] < tv
        smims = smims[mask]

        # 遍历边界集，计算稀疏聚类损失
        loss = 0
        num_boundaries = len(boundaries)
        for i in range(len(boundaries)):
            bx = boundaries[i][0]

            # 计算距离 bx 最近的 smim
            min_distance = 1
            min_index = -1
            for j in range(len(smims)):
                distance = abs(smims[j][0] - bx)
                if distance < min_distance:
                    min_index = j
                    min_distance = distance

            # 计算稀疏聚类损失
            a = abs(boundaries[i][1] - smims[min_index][1])
            b = abs(boundaries[i][0] - smims[min_index][0]) + 0.01
            loss += a / b

        # 综合稀疏聚类损失和密度差异
        loss = loss / num_boundaries
        print(f"loss: {loss}")
        return loss


        
    


    
    

        


