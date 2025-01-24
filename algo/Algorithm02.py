from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from scipy.spatial.distance import cdist
from scipy.ndimage import uniform_filter, gaussian_filter


current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df

from dh.image_handle import ImageHandle
from core.surface_renderer import ImageDataSurfaceRenderer, UnstructuredGridSurfaceRenderer
from core.core_calculater import Core_Calculater

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score as mis
import matplotlib.pyplot as plt
from algo.Algorithm01 import UnstructuredGridASO, StructuredGridASO

class AlgorithmStepTwo:
    """
    对算法步骤中的每一个区间，计算视觉信息，并投影
    """
    def init(self):
        self.surfaceRenderer = None

        self.frequencyDt = None
        self.regionsBinIndices = None
        self.regionsBinCenter = None
        self.regionsFrequencies = None

        self.totalBinCenter = None
        self.totalFrequencies = None

        # 通过算法步骤一(aso),得到的区域进一步加工
        self.regionsBinIndicesArray = None
        self.regionsFrequenciesArray = None
        self.regionsDensitiesArray = None

        # 一些渲染需要的信息
        self.dataBounds = None
        self.renderData = None
        self.cameraParams = None

        # 需要计算得出的视觉信息
        self.regionsSSIMS = None
        self.totalitySSIMS = None
        self.imageArray = None
        self.smoothTotalitySSIMS = None

        # 需要计算得出的演化信息
        self.totalitySMIMS = None

        # 返回值
        self.boundaryPoints = None
        self.boundaryIndices = None
        self.representivePoints = None
        self.representiveIndices = None
        self.keyPoints = []

        # 超参数
        self.percentile = None
        self.keyValueX = None

        # 传递给步骤3的数据
        self.viewPointPoint = None

        # 数学计算
        self.calculator = None

        # 用于图的绘制
        self.max_val_x = None
        self.min_val_x = None
        self.max_val_y = None
        self.min_val_y = None
        self.densePoints = None
        self.sparsePoints = None
        self.cluster_parameter = None
        self.min_loss = None
        self.Xspace = None              # isovalue 和对应的 mssim值构成的二维数组
        self.densityArray = None        # 密度数组

        # 用于传输函数的生成
        self.scalarRange = None
        self.keyPoints = None

    def SetClusterParamenter(self, p): self.percentile = p   

    def SetKeyValueParamenter(self, p): self.keyValueX = p

    def GetXspace(self): return self.Xspace

    def GetFrequencyRegionsPoints(self): return self.regionsBinCenter

    def GetVisionRegionsPoints(self): return  self.boundary_bin_center

    def GetKeyValuePoints(self): return self.keyPoints

    def GetFrequencyDt(self): return self.frequencyDt

    def GetBinDistance(self): return self.binDistance

    def GetViewPoint(self): return self.viewPoint

    def GetRenderData(self):return self.renderData

    def GetBinCenters(self):return self.totalBinCenter

    def InitASTFromASO(self, aso):
        a,b = aso.GetTotalityInfor()
        c,d,e = aso.GetRegionsInfor()
        self.regionsBinIndices = c
        self.regionsBinCenter = d
        self.regionsFrequencies = e
        self.totalBinCenter = a
        self.totalFrequencies = b
        self.frequencyDt = aso.GetFrequencyDt()

        self.cameraParams = aso.GetCameraParams()
        self.dataBounds = aso.GetScalarBounds()
        self.renderData = aso.GetRenderData()

        self.regionsSSIMS = []
        self.totalitySSIMS = []
        self.totalitySMIMS = []
        self.boundaryPoints = None

        self.regionsBinIndicesArray = []
        self.regionsFrequenciesArray = []
        self.regionsDensitiesArray = []

        self.binDistance = aso.GetBinDistance()

        self.InitRegionsInfor()

        self.calculator = Core_Calculater()
        self.imageArray = []
        self.keyPoints = []

        self.scalarRange = aso.GetDataLoader().GetScalarRange()
        self.dataSpacing = aso.GetDataLoader().GetDataSpacing()
    
    def InitRegionsInfor(self):
        regionsFrequenciesArray = []
        regionsIndicesArray = []
        indices = self.regionsBinIndices

        for i in range(indices.shape[0] - 1):
            regionStart = indices[i]
            regionEnd = indices[i + 1]
            # 对于每一个划分的区间
            regionFrequencies = []
            regionBinIndices = []
            while regionStart < regionEnd:
                regionFrequencies.append(self.totalFrequencies[regionStart])
                regionBinIndices.append(self.totalBinCenter[regionStart])
                regionStart += 1

            regionsFrequenciesArray.append(np.array(regionFrequencies))
            regionsIndicesArray.append(np.array(regionBinIndices))

        # Convert to numpy arrays
        self.regionsBinIndicesArray = np.array(regionsIndicesArray, dtype=object)
        self.regionsFrequenciesArray = np.array(regionsFrequenciesArray, dtype=object)

        # Ensure densities are properly normalized
        self.regionsDensitiesArray = np.array(self.regionsFrequenciesArray) / np.sum(self.totalFrequencies)

    
    def CaculateTotalityVisionInfor(self, show = True):
        # 设置摄像机参数
        b = self.dataBounds               # 乘以dataSpacings
        self.dataBounds = [b[0], b[1]*self.dataSpacing[0],
                           b[2], b[3]*self.dataSpacing[1],
                           b[4], b[5]*self.dataSpacing[2]]
        print(f"dataBounds: {self.dataBounds}")
        self.viewPoint = self.surfaceRenderer.SetCameraPosition(self.dataBounds)  # viewport
        self.surfaceRenderer.SetCameraParams(self.cameraParams)

        # 渲染等值面图像
        for i in range(self.totalBinCenter.shape[0]):
            counterValue = self.totalBinCenter[i]
            self.surfaceRenderer.SetContourValue(counterValue)
            self.surfaceRenderer.SetRenderData(self.renderData)
            self.surfaceRenderer.Render(axes=False, border=False, offscreen= not show)
        
            cimage = ImageHandle.StoreRenderImageToNumpy(self.surfaceRenderer.GetRenderWindow())
            self.imageArray.append(cv2.cvtColor(cimage, cv2.COLOR_BGR2GRAY))


    def CaculateSimilatyAndContinousInfor(self):
        # 对于相邻的两张图像计算连续性和相似性
        self.totalitySSIMS.append(1)
        self.totalitySMIMS.append(1)
        for i in range(len(self.imageArray)-1):
            mssim, nseim = self.calculator.CSM(self.imageArray[i], self.imageArray[i+1])
            self.totalitySSIMS.append(mssim)
            self.totalitySMIMS.append(nseim)
        self.totalitySSIMS = np.array(self.totalitySSIMS)
        self.totalitySMIMS = np.array(self.totalitySMIMS)


    def TotalityVisionDensityProject(self, show = True):
        x = np.delete(self.totalitySSIMS, 0)
        y = self.totalFrequencies/np.sum(self.totalFrequencies)
        
        y = np.delete(y, 0)
        # plt.title("Totality Vision-Frequency Projection")
        plt.scatter(x, y, s=2, c='b', marker='o')

        keyPoints = []
        for i in range(y.shape[0]):
            if x[i] < self.keyValueX: 
                originx = self.totalBinCenter[i]
                if (originx >= 0.15 and originx < 0.16) or (originx > 5.61 and originx <= 5.62):
                    plt.scatter(x[i], y[i], s=2,color='red', marker='o')
                    plt.text(x[i], y[i], f'{originx:.2f}', fontsize=12, color='red', ha='right', va='bottom')
                else :
                    # plt.text(x[i], y[i], f'{originx:.2f}', fontsize=12, color='green', ha='right', va='bottom')
                    pass
                keyPoints.append(originx)
        self.keyPoints = np.array(keyPoints)

        if show: plt.show()
        else: return
        
    def ShowTotalityInfo(self):
        x = np.delete(self.totalBinCenter,0)
        y = np.delete(self.totalitySSIMS,0)
        plt.scatter(x, y, s=2, c='b', marker='o')
        plt.show()
    
    def ShowTotalityEvolutionInfo(self):
        x = np.delete(self.totalBinCenter,0)
        y = np.delete(self.totalitySMIMS,0)
        plt.plot(x, y)
        plt.show()
    
    def SoomthTotalitySSIMS(self):
        # 均值平滑
        y = self.totalitySSIMS
        x = self.totalBinCenter
        section, smoothY = 6, []
        n = int(section / 2)
        
        for i in range(y.shape[0]):
            if i - n >= 0 and i + n + 1 <= y.shape[0]: 
                subY = y[i - n:i + n + 1]
                submean = np.mean(subY)
            else: 
                submean = y[i]
            smoothY.append(submean)
        
        self.smoothTotalitySSIMS = np.array(smoothY)
        # plt.title("Smoothed Totality SSIMS Information")
        # plt.scatter(x, self.smoothTotalitySSIMS, s=2, c='r', marker='o')
        # plt.show()

    def AutoBoundaryPointDetection(self, n, neigh_size=4, show=True):
        """
        根据系数聚类确定边界
        param:   show: 是否显示结果
        return:  边界点坐标及其对应的下标
        """
        x = self.totalBinCenter
        y = self.totalitySSIMS

        # 归一化x和y并投影到二维空间
        y[0] = y[1]  # 因为第一帧的结构相似性指数为1，所以让它等于第二帧的
        min_val_x = np.min(x)
        max_val_x = np.max(x)
        x = (x - min_val_x) / (max_val_x - min_val_x)
        min_val_y = np.min(y)
        max_val_y = np.max(y)
        y = (y - min_val_y) / (max_val_y - min_val_y)
        dx = (x[1] - x[0]) * 2  # 计算x轴的步长，用于通过聚类结果找到边界

        self.Xspace = np.column_stack((x, y))  # Stack x and y into a 2D array

        # 保存尺寸用于画图
        self.min_val_x, self.max_val_x = min_val_x, max_val_x
        self.min_val_y, self.max_val_y = min_val_y, max_val_y

        # 消除首尾点对密度计算的影响
        first = np.tile(self.Xspace[0], (neigh_size // 2, 1))  # 重复 Xspace[0]，生成二维数组
        last = np.tile(self.Xspace[-1], (neigh_size // 2, 1))  # 重复 Xspace[-1]，生成二维数组
        Yspace = np.concatenate((first, self.Xspace, last), axis=0)  # 合并 first, Xspace, last, Yspace用于密度计算

        # 使用K临近计算密度阈值, 并根据密度阈值进行稀疏聚类
        neighbors = NearestNeighbors(n_neighbors=neigh_size)
        neighbors_fit = neighbors.fit(Yspace)
        distances, indices = neighbors_fit.kneighbors(Yspace)
        density_array = 1 / distances.mean(axis=1)
        self.densityArray = density_array[neigh_size // 2:-neigh_size // 2]
        boundaryIndices = []

        # 调整密度阈值，并根据密度阈值进行稀疏聚类
        p = 10
        dp = 1
        self.min_loss = 9999
        while p <= 90:
            p += dp  # 步长 调整稀疏聚类结果
            density_threshold = np.percentile(density_array, p)  # i 为百分比参数
            self.density_threshold = density_threshold
            dense_mask = density_array >= density_threshold
            sparse_mask = ~dense_mask
            dense_points = Yspace[dense_mask]
            sparse_points = Yspace[sparse_mask]

            # 根据稀疏聚类结果进行边界点提取
            boundary_points = []
            boundary_indices = []  # 用于存储边界点的原始索引

            for i in range(dense_points.shape[0] - 1):
                x0 = dense_points[i, 0]
                x1 = dense_points[i + 1, 0]
                
                if x1 - x0 > dx*2:  
                    boundary_points.append(dense_points[i])
                    boundary_indices.append(np.where(np.all(self.Xspace == dense_points[i], axis=1))[0][0])  # 记录边界点的索引
                    boundary_points.append(dense_points[i + 1])
                    boundary_indices.append(np.where(np.all(self.Xspace == dense_points[i + 1], axis=1))[0][0])  # 记录边界点的索引

            boundary_points.append(dense_points[0])
            boundary_indices.append(np.where(np.all(self.Xspace == dense_points[0], axis=1))[0][0])  # 记录第一个边界点的索引
            boundary_points.append(dense_points[-1])
            boundary_indices.append(np.where(np.all(self.Xspace == dense_points[-1], axis=1))[0][0])  # 记录最后一个边界点的索引

            # 包含首尾边界, 并聚合数组
            bool_front, bool_end = False, False
            front, end = self.Xspace[0], self.Xspace[-1]
            for i in range(len(boundary_points)):
                if np.array_equal(boundary_points[i], front): bool_front = True
                if np.array_equal(boundary_points[i], end): bool_end = True

            if not bool_front:
                boundary_points.append(front)
                boundary_indices.append(np.where(np.all(self.Xspace == front, axis=1))[0][0])  # 记录首个边界点的索引
            if not bool_end:
                boundary_points.append(end)
                boundary_indices.append(np.where(np.all(self.Xspace == end, axis=1))[0][0])  # 记录最后一个边界点的索引

            boundary_points = np.array(boundary_points)

            # 排序边界点和边界点的索引
            sorted_indices = np.argsort(boundary_indices)  # 获取按索引排序的顺序
            boundary_points = boundary_points[sorted_indices]  # 按照排序顺序排序边界点
            boundary_indices = np.array(boundary_indices)[sorted_indices]  # 按照排序顺序排序边界点索引
           

            # 计算损失
            smims = self.totalitySMIMS
            boundaries = np.array(boundary_points)
            
            
            loss = self.calculator.CalculateSparseClusterLossOne(dense_points, sparse_points, boundaries, n)

            print(f"Parameter: {p} boundaries.shape[0] - 1: {boundaries.shape[0] - 1} n: {n} Loss: {loss}")
            if loss < self.min_loss and (boundaries.shape[0] - 1) == n:
                self.min_loss = loss
                self.boundaryPoints = boundary_points
                self.densePoints = dense_points
                self.sparsePoints = sparse_points
                self.cluster_parameter = p - dp
                boundaryIndices = boundary_indices  # 保存对应的索引
               
        # 若边界重合
        i=0
        self.boundaryIndices = []
        while(i < len(boundaryIndices)-1):
            if boundaryIndices[i] == boundaryIndices[i+1]:
                self.boundaryIndices.append(boundaryIndices[i])
                self.boundaryIndices.append(boundaryIndices[i]+2)
                i += 2
            else:
                self.boundaryIndices.append(boundaryIndices[i])
                i += 1
        self.boundaryIndices.append(boundaryIndices[-1])

        self.boundary_bin_center = self.totalBinCenter[self.boundaryIndices]  # 记录边界点的原始坐标
           
        print(f"New boundary points: {self.boundary_bin_center}")
        if not show:
            return 

        if self.boundaryPoints is None:
            return ValueError("The Cluster NUM is not correct. Please re-adjust the parameter.")

        return 
    
    def ShowAutoBoundaryPointDetection(self):
        """
        显示自动分割的相似性信息
        """
        # 示例数据
        scale_x = (self.max_val_x - self.min_val_x)
        scale_y = (self.max_val_y - self.min_val_y)

        x = np.delete(self.totalBinCenter, 0)
        y = np.delete(self.totalitySMIMS, 0)

        # 创建第一个坐标轴
        fig, ax1 = plt.subplots()

        # 创建第二个坐标轴（共享x轴，但独立的y轴）

        # 绘制垂直线（在第二个坐标轴上）
        boundary_x_list = []
        for i, xsplit in enumerate(self.boundaryPoints[:, 0]):
            xsplit = xsplit * scale_x + self.min_val_x
            # 只在第一次绘制垂直线时设置标签
            if i == 0:
                ax1.axvline(x=xsplit, color='r', linestyle='--', linewidth=1, label='boundary isovalue')
            else:
                ax1.axvline(x=xsplit, color='r', linestyle='--', linewidth=1)
            boundary_x_list.append(xsplit)  # originx

        # 绘制散点图（在第二个坐标轴上）
        ax1.scatter(self.densePoints[:, 0] * scale_x + self.min_val_x, self.densePoints[:, 1] * scale_y + self.min_val_y, c='blue', s=2, label='dense isovalue')
        ax1.scatter(self.sparsePoints[:, 0] * scale_x + self.min_val_x, self.sparsePoints[:, 1] * scale_y + self.min_val_y, c='green', s=2, label='sparse isovalue')
        ax1.set_ylabel('MSSIM(scatter)', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlabel('isovalue', fontsize=12)

        ax2 = ax1.twinx()
         # 绘制折线图（在第一个坐标轴上）
        ax2.plot(x, y, alpha=0.28, c = 'gray', linewidth=1.5)
        
        ax2.set_ylabel('NSEIM(line)', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')

        

        # 获取图例的所有handles和labels
        handles, labels = ax1.get_legend_handles_labels()

        # 找到 'boundary isovalue' 的 handle，并将其移到最底部
        boundary_handle = handles.pop(0)
        boundary_label = labels.pop(0)

        # 将 'boundary isovalue' 的 handle 添加到图例底部
        handles.append(boundary_handle)
        labels.append(boundary_label)

        # 设置标题
        title_str = "number of evolutionary processes: " + str(self.boundaryPoints.shape[0] - 1) 
        ax1.set_title(title_str, fontsize=14)

        # 显示图例，并让 'boundary isovalue' 显示在底部
        ax2.legend(handles=handles, labels=labels, loc='lower right')
        plt.savefig('img/AutoBoundaryPointDetection.png', dpi=600)
        # 显示图形
        plt.show()
        return self.boundaryIndices

    def AutoBoundaryPointAjustion(self, threshold=0.98, threshold2 = 0.9, n=5):
        '''
        根据SIMMS信息调整边界点
        '''
        smims = np.array(self.totalitySMIMS)  # 获取结构相似性指数数组
        boundary_indices = self.boundaryIndices[1:-1]  # 获取当前的边界点索引

        new_boundary_indices = []  # 用于存储新的边界点索引

        for i in range(len(boundary_indices)):
            x = boundary_indices[i]

            if smims[x] < threshold2:  # 如果当前边界点的SIMMS大于阈值，则跳过
                new_boundary_indices.append(x)
                continue
            # 向左向右找最近的突变点
            left = x - 1
            right = x + 1
            count = 0

            best_point = x  # 初始为原始边界点
            max_ratio = 0  # 用来记录最大欧几里得距离与x方向距离的比值

            # Zspace: 合并SMIMS信息与X坐标
            Zspace = np.column_stack((self.Xspace[:, 0], smims))  # 转换坐标轴
               
            while left > 0 and right < smims.shape[0] and count < n:
                # 计算欧几里得距离

                if smims[left] > threshold and smims[right] > threshold:  # 如果左边界点的SIMMS大于阈值，则跳过
                    count += 1
                    left -= 1
                    right += 1
                    continue

                distance_left = np.linalg.norm(Zspace[left] - Zspace[x])  # 原始边界点与左边界点的欧几里得距离
                distance_right = np.linalg.norm(Zspace[right] - Zspace[x])  # 原始边界点与右边界点的欧几里得距离

                # 计算x方向的距离
                x_distance_left = abs(Zspace[left][0] - Zspace[x][0])  # 左边界点与原始边界点x方向的距离
                x_distance_right = abs(Zspace[right][0] - Zspace[x][0])  # 右边界点与原始边界点x方向的距离

                # 防止除以0的情况
                if x_distance_left != 0:
                    ratio_left = distance_left / x_distance_left
                else:
                    ratio_left = 0

                if x_distance_right != 0:
                    ratio_right = distance_right / x_distance_right
                else:
                    ratio_right = 0

                # 计算距离最近边界点的距离
                min_distance_left = np.min(np.abs(self.Xspace[left, 0] - self.Xspace[self.boundaryIndices, 0]))
                min_distance_right = np.min(np.abs(self.Xspace[right, 0] - self.Xspace[self.boundaryIndices, 0]))


                # 乘上最近边界点的距离, 稀疏集取1/min_distance_left,若是稠密集，则尽可能远，若是稀疏集合，则近
             
                weighted_ratio_left = ratio_left * min_distance_left* (self.Xspace[left][1]*10)
                weighted_ratio_right = ratio_right * min_distance_right * (self.Xspace[right][1]*10)
                

                # 比较加权后的比值
                if weighted_ratio_left > max_ratio:  # 如果左边界点的加权比值更大，更新
                    max_ratio = weighted_ratio_left
                    best_point = left

                if weighted_ratio_right > max_ratio:  # 如果右边界点的加权比值更大，更新
                    max_ratio = weighted_ratio_right
                    best_point = right

                count += 1
                left -= 1
                right += 1

            # 将找到的最优边界点加入新边界点索引
            new_boundary_indices.append(best_point)

        # 加入首个和最后一个边界点
        new_boundary_indices.insert(0, self.boundaryIndices[0])  
        new_boundary_indices.append(self.boundaryIndices[-1])  
        

        # 获取新的边界点坐标
        self.boundaryIndices = np.array(new_boundary_indices)
        self.boundaryPoints = self.Xspace[new_boundary_indices]
        print(f"self {self.boundaryPoints}")
        return 
    
    # def AutoRepresentativePointDetectionImprove(self):
    #     """
    #     根据优化后的边界、ssim、smim信息确定代表等值面
    #     """
    #     boundaries = self.boundaryIndices  # 获取优化后的边界点坐标
    #     representive_indices = []

    #     # 将图像数据转换为 NumPy 数组
    #     mssim_list = []
    #     for i in range(len(boundaries) - 1):
    #         begin, end = boundaries[i], boundaries[i + 1]

    #         # 获取当前区域的子数组
    #         if i == len(boundaries) - 2:  # 最后一个区域
    #             end += 1  # 确保包含最后一个点
    #             sub_image = self.imageArray[begin:end]
    #         else:
    #             sub_image = self.imageArray[begin:end]

    #         # 创建一个矩阵存储 ssims 的结果
    #         area_mssim_matrix = np.zeros((end - begin, end - begin))
    #         print(f"Region {i}: begin={begin}, end={end}, size={end - begin}")
    #         # 计算区域内所有点之间的 SSIM 值
    #         for j in range(end - begin):
    #             for k in range(j, end - begin):  # 避免重复计算
    #                 ssim_value = ssim(sub_image[j], sub_image[k])
    #                 area_mssim_matrix[j, k] = ssim_value
    #                 area_mssim_matrix[k, j] = ssim_value  # 对称矩阵

    #         # 计算每个点的总相似度（按列求和）
    #         area_mssim_list = np.sum(area_mssim_matrix, axis=1)/(end - begin)
    #         mssim_list.extend(area_mssim_list)
    #         # 找到最大相似度的索引
    #         max_index = np.argmax(area_mssim_list)
    #         representive_indices.append(max_index + begin)

    #     self.representivePoints = self.Xspace[representive_indices]
    #     self.representiveIndices = representive_indices
    #     self.regionSimilarity = np.array(mssim_list)
    #     print(f"Representative Points: {self.totalBinCenter[representive_indices]}")
    


    def process_region(self, i, boundaries, image_array):
        """
        处理每个区域的计算：计算区域内点之间的 SSIM 值
        返回每个区域的 MSSIM 列表和最大相似度的索引
        """
        begin, end = boundaries[i], boundaries[i + 1]

        # 获取当前区域的子数组
        if i == len(boundaries) - 2:  # 最后一个区域
            end += 1  # 确保包含最后一个点
            sub_image = image_array[begin:end]
        else:
            sub_image = image_array[begin:end]

        # 创建一个矩阵存储 ssims 的结果
        area_mssim_matrix = np.zeros((end - begin, end - begin))
        print(f"Region {i}: begin={begin}, end={end}, size={end - begin}")

        # 计算区域内所有点之间的 SSIM 值
        for j in range(end - begin):
            for k in range(j, end - begin):  # 避免重复计算
                ssim_value = ssim(sub_image[j], sub_image[k])
                area_mssim_matrix[j, k] = ssim_value
                area_mssim_matrix[k, j] = ssim_value  # 对称矩阵

        # 计算每个点的总相似度（按列求和）
        area_mssim_list = np.sum(area_mssim_matrix, axis=1) / (end - begin)

        # 找到最大相似度的索引
        max_index = np.argmax(area_mssim_list)
        
        # 返回当前区域的 MSSIM 列表、最大相似度的索引和区域的索引
        return area_mssim_list, max_index + begin, i

    def AutoRepresentativePointDetectionImprove(self):
        """
        根据优化后的边界、ssim、smim信息确定代表等值面
        """
        boundaries = self.boundaryIndices  # 获取优化后的边界点坐标
        representive_indices = []
        mssim_list = [None] * (len(boundaries) - 1)  # 保证每个区域的结果有正确的位置

        # 创建线程池来处理每个区域
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(boundaries) - 1):
                futures.append(executor.submit(self.process_region, i, boundaries, self.imageArray))

            # 收集每个线程的结果
            for future in as_completed(futures):
                area_mssim_list, max_index, region_idx = future.result()

                # 确保按区域索引顺序更新 mssim_list 和 representive_indices
                mssim_list[region_idx] = area_mssim_list
                representive_indices.append(max_index)

        # 更新代表性点
        self.representivePoints = self.Xspace[representive_indices]
        self.representiveIndices = representive_indices
        self.regionSimilarity = np.concatenate(mssim_list)  # 按照区域的顺序合并
        print(f"Representative Points: {self.totalBinCenter[representive_indices]}")

    def ShowRengionSimilarity(self, m):
        """
        显示区域相似性信息
        """
        scale_x = (self.max_val_x - self.min_val_x)
        scale_y = (self.max_val_y - self.min_val_y)


        plt.scatter(self.totalBinCenter, self.regionSimilarity, c='blue', s=2, alpha=0.8)
        
        for i, xsplit in enumerate(self.boundaryPoints[:, 0]):
            xsplit = xsplit * scale_x + self.min_val_x
            # 只在第一次绘制垂直线时设置标签
            if i == 0:
                plt.axvline(x=xsplit, color='r', linestyle='--', linewidth=1, label='boundary isovalue')
            else:
                plt.axvline(x=xsplit, color='r', linestyle='--', linewidth=1)

        plt.scatter(self.representiveIndices, self.regionSimilarity[self.representiveIndices], c='red', s=25, marker='^', label='represent isovalue')

        # 获取图例的所有handles和labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # 找到 'boundary isovalue' 的 handle，并将其移到最底部
        boundary_handle = handles.pop(0)
        boundary_label = labels.pop(0)

        # 将 'boundary isovalue' 的 handle 添加到图例底部
        handles.append(boundary_handle)
        labels.append(boundary_label)

        title_str = "number of evolutionary processes: " + str(m)
        plt.title(title_str, fontsize=14)
        plt.xlabel('isovalue',fontsize=12)
        plt.ylabel('MSSIM(scatter)',fontsize=12)
        plt.legend(handles=handles, labels=labels, loc='lower right')
        plt.savefig('img/RegionSimilarity.png', dpi=600)

        # 显示图形
        plt.show()
        return self.representiveIndices


    def ShowAutoRepresentativePointDetectionImprove(self):
        """
        显示最终的边界等值和代表等值的结果
        """
        """
        显示自动分割的相似性信息
        """
        # 示例数据
        scale_x = (self.max_val_x - self.min_val_x)
        scale_y = (self.max_val_y - self.min_val_y)

        x = np.delete(self.totalBinCenter, 0)
        y = np.delete(self.totalitySMIMS, 0)

        # 创建第一个坐标轴
        fig, ax1 = plt.subplots()

        # 绘制折线图（在第一个坐标轴上）
        ax1.plot(x, y, alpha=0.28, c = 'gray', linewidth=1.5)
        ax1.set_xlabel('isovalue', fontsize=12)
        ax1.set_ylabel('NSEIM(line)', color='black', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='black')

        # 创建第二个坐标轴（共享x轴，但独立的y轴）
        ax2 = ax1.twinx()

        # 绘制垂直线（在第二个坐标轴上）
        boundary_x_list = []
        for i, xsplit in enumerate(self.boundaryPoints[:, 0]):
            xsplit = xsplit * scale_x + self.min_val_x
            # 只在第一次绘制垂直线时设置标签
            if i == 0:
                ax2.axvline(x=xsplit, color='r', linestyle='--', linewidth=1, label='boundary isovalue')
            else:
                ax2.axvline(x=xsplit, color='r', linestyle='--', linewidth=1)
            boundary_x_list.append(xsplit)  # originx

        # 绘制散点图（在第二个坐标轴上）
        ax2.scatter(self.densePoints[:, 0] * scale_x + self.min_val_x, self.densePoints[:, 1] * scale_y + self.min_val_y, c='blue', s=2, label='dense isovalue', alpha=0.8)
        ax2.scatter(self.sparsePoints[:, 0] * scale_x + self.min_val_x, self.sparsePoints[:, 1] * scale_y + self.min_val_y, c='green', s=2, label='sparse isovalue',alpha=0.8)
        ax2.set_ylabel('MSSIM(scatter)', color='black', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='black')

        # 绘制代表等值
        representive_x_list = []
       
        ax2.scatter(self.representivePoints[:, 0] * scale_x + self.min_val_x, self.representivePoints[:, 1] * scale_y + self.min_val_y, c='red', s=25, marker='^', label='represent isovalue')
        
        # self.representivePoints = self.Xspace[self.midpointRepresentation]
        # ax2.scatter(self.midpointRepresentation, self.representivePoints[:, 1] * scale_y + self.min_val_y, c='red', s=25, marker='^', label='represent isovalue')
        # 获取图例的所有handles和labels
        handles, labels = ax2.get_legend_handles_labels()

        # 找到 'boundary isovalue' 的 handle，并将其移到最底部
        boundary_handle = handles.pop(0)
        boundary_label = labels.pop(0)

        # 将 'boundary isovalue' 的 handle 添加到图例底部
        handles.append(boundary_handle)
        labels.append(boundary_label)

        # 设置标题
        title_str = "number of evolutionary processes: " + str(self.boundaryPoints.shape[0] - 1) 
        ax1.set_title(title_str, fontsize=14)

        # 显示图例，并让 'boundary isovalue' 显示在底部
        ax2.legend(handles=handles, labels=labels, loc='lower right')
        plt.savefig('img/AutoRepesentativePointDetection.png', dpi=600)
        # 显示图形
        plt.show()
        


    def AutoRepresentativePointDetection(self):
        """
        根据优化后的边界、ssim、smim信息确定代表等值面
        """
        boundaries = self.boundaryIndices                               # 获取优化后的边界点坐标
        representive_indices = []
        for i in range(len(boundaries) - 1):
            begin, end = boundaries[i], boundaries[i + 1]

            area_mssim_list = []
            for j in range(begin, end):
                area_mssim_sum = 0
                for k in range(begin, end):
                    ssimValue = ssim(self.imageArray[j], self.imageArray[k])
                    area_mssim_sum += ssimValue
                area_mssim_list.append(area_mssim_sum)
            max_area_mssim = max(area_mssim_list)
            representive_indices.append(area_mssim_list.index(max_area_mssim) + begin)

        self.representivePoints = self.Xspace[representive_indices]
    
    def ShowAutoRepresentativePointDetection(self):
        """
        显示最终的边界等值和代表等值的结果
        """
        # 示例数据
        scale_x = (self.max_val_x - self.min_val_x)
        scale_y = (self.max_val_y - self.min_val_y)

        x = np.delete(self.totalBinCenter, 0)
        y = np.delete(self.totalitySMIMS, 0)

        # 创建第一个坐标轴
        fig, ax1 = plt.subplots()

        # 绘制折线图（在第一个坐标轴上）
        ax1.plot(x, y, alpha=0.2, label='Line Plot')
        ax1.set_xlabel('X 轴')
        ax1.set_ylabel('左侧 Y 轴', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # 创建第二个坐标轴（共享x轴，但独立的y轴）
        ax2 = ax1.twinx()

        # 绘制散点图（在第二个坐标轴上）
        ax2.scatter(self.densePoints[:, 0] * scale_x + self.min_val_x, self.densePoints[:, 1] * scale_y + self.min_val_y, c='blue', s=2, label='Dense Points')
        ax2.scatter(self.sparsePoints[:, 0] * scale_x + self.min_val_x, self.sparsePoints[:, 1] * scale_y + self.min_val_y, c='green', s=2, label='Sparse Points')
        ax2.set_ylabel('右侧 Y 轴', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # 绘制垂直线（仍在第一个坐标轴上）边界等值
        boundary_x_list = []
        for xsplit in self.boundaryPoints[:, 0]:
            xsplit = xsplit * scale_x + self.min_val_x
            ax1.axvline(x=xsplit, color='r', linestyle='--', linewidth=1, label=f'x = {xsplit:.2f}')
            boundary_x_list.append(xsplit)  # originx

        # 绘制垂直线，代表等值
        representive_x_list = []
        for xsplit in self.representivePoints[:, 0]:
            xsplit = xsplit * scale_x + self.min_val_x
            ax1.axvline(x=xsplit, color='g', linestyle='--', linewidth=1, label=f'x = {xsplit:.2f}')
            representive_x_list.append(xsplit)  # originx
        
        # 设置标题
        title_str = "Parameter: " + str(self.cluster_parameter) + " Loss: " + str(self.min_loss)
        ax1.set_title(title_str)

        # 显示图例
        # ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')

        # 显示图形
        plt.show()
    
    def GenerateImproveTFJson(self):
        """
        根据mssim和nseim分割的演化区域生成传输函数
        """
        evolutionBoundaries = self.boundaryIndices
      
        actualEvolutionBoundaries = self.totalBinCenter[evolutionBoundaries]

        actualRepresentation = []
        for i in range(len(actualEvolutionBoundaries)-1):
            midpoint = (actualEvolutionBoundaries[i] +actualEvolutionBoundaries[i + 1]) / 2
            actualRepresentation.append(midpoint)


        print("EvolutionBoundaries: ",  actualEvolutionBoundaries)
        print("ActualRepresentation: ", actualRepresentation)

        actualRepresentation = np.array(actualRepresentation)
        self.midpointRepresentation = actualRepresentation
        # 构造传输函数控制点
        f = np.array(actualEvolutionBoundaries)
        d = np.array(actualRepresentation)

        # self.representivePoints = self.Xspace[actualRepresentation]

        f_list = []
        d_list = []
        for i in range(f.shape[0]):
            f_list.append([f[i], 0.0, 0.5, 0.0])
        
        for i in range(d.shape[0]):
            d_list.append([d[i], 0.5, 0.5, 0.0])
        
        pointList = f_list + d_list
        pointArray = np.array(pointList)
        pointArray= pointArray[pointArray[:, 0].argsort()].ravel()
    
        jsonData = {}
        jsonData["ColorSpace"] = "Diverging"
        jsonData["Name"] = "Test"
        jsonData["Points"] = pointArray.tolist()
        jsonData["RGBPoints"] = [
			0.0,
			0.23137254902000001,
			0.298039215686,
			0.75294117647100001,
			127.5,
			0.86499999999999999,
			0.86499999999999999,
			0.86499999999999999,
			255.0,
			0.70588235294099999,
			0.015686274509800001,
			0.149019607843
		]
        jsonDatas = []
        jsonDatas.append(jsonData)
        with open('itf/tf2.json', 'w') as json_file:
            json.dump(jsonDatas, json_file, indent=4)  # indent 参数用于格式化输出

  

class UnstructuredGridAST(AlgorithmStepTwo):

    def __init__(self):
        super().__init__()
        self.surfaceRenderer = UnstructuredGridSurfaceRenderer()
        self.surfaceRenderer.unstructuredGrid = True

class StructuredGridAST(AlgorithmStepTwo):
    def __init__(self):
        super().__init__()
        self.surfaceRenderer = ImageDataSurfaceRenderer()
        self.surfaceRenderer.imageData = True

def test1():
    dataFile = df.velocity()
    # dataFile = df.velocityQ()
    # dataFile = df.tangaroa()
    aso = UnstructuredGridASO(dataFile)
    aso.LoadScalarArray("U")
    aso.SetFrequncyParamenter(0.01)
    aso.SetContinuousParamenter(5)
    aso.TotalityFrequncyInfo()
    aso.AutoReginDevision()
    aso.ShowTotalityDevisionInfor()

    ast = UnstructuredGridAST()
    ast.InitASTFromASO(aso)
    ast.SetClusterParamenter(20)
    ast.SetKeyValueParamenter(0.8)

    ast.CaculateTotalityVisionInfor()
    ast.TotalityVisionDensityProject()

    ast.AutoKeyPointDetection()

    # ast.CaculateRegionsVisionInfor()
    # ast.RegionsVisionDensityProject()

def test2():
    # dataFile = df.engine()
    dataFile = df.tooth()
    # dataFile = df.tacc()
    # dataFile = df.duct()
    # dataFile = df.tornado()
    aso = StructuredGridASO(dataFile)
    aso.LoadScalarArray()
    aso.SetFrequncyParamenter(2)
    aso.SetContinuousParamenter(5)
    aso.TotalityFrequncyInfo()
    aso.AutoReginDevision()
    aso.ShowTotalityDevisionInfor()

    ast = StructuredGridAST()
    ast.InitASTFromASO(aso)
    ast.SetClusterParamenter(10)
    ast.SetKeyValueParamenter(0.9)

    ast.CaculateTotalityVisionInfor()
    ast.AutoKeyPointDetection()
    ast.TotalityVisionDensityProject()

    # ast.CaculateRegionsVisionInfor()
    # ast.RegionsVisionDensityProject()

if __name__ == '__main__':
    # test1()
    test2()
    

        
