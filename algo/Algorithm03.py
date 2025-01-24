import json
import os
import sys
import time
import cv2
import numpy as np
import vtk


current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df

from dh.data_loader import DataLoader
from dh.image_handle import ImageHandle
from core.surface_renderer import ImageDataSurfaceRenderer, UnstructuredGridSurfaceRenderer

from skimage.metrics import structural_similarity as ssim
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from algo.Algorithm01 import UnstructuredGridASO, StructuredGridASO
from algo.Algorithm02 import StructuredGridAST, UnstructuredGridAST

class AlgorithmStepThree():
    """
    该类实现区域组合算法，将频率信息区域、视觉信息区域、关键数值点合成为布点
    """
    def __init__(self):
        # 输入
        self.frequencyRegionsPoints = None
        self.visionRegionsPoints = None
        self.keyValuePoints = None
        self.frequencyDt = None

        # 输出
        self.fixPoints = None
        self.dynamicPoints = None

        # 参数
        self.keyPointParamenter = None

        # 透明度必须参数
        self.viewPoint = None
        self.renderData = None
        self.binCenters = None

    def SetKeyPointParamenter(self, p): self.keyPointParamenter = p

    def InitASEFromAST(self, ast): 
        self.frequencyRegionsPoints = ast.GetFrequencyRegionsPoints()
        self.visionRegionsPoints = ast.GetVisionRegionsPoints()
        self.keyValuePoints = ast.GetKeyValuePoints()
        self.frequencyDt = ast.GetFrequencyDt()
        self.binDistance = ast.GetBinDistance()
        self.viewPoint = ast.GetViewPoint()
        self.renderData = ast.GetRenderData()
        self.binCenters = ast.GetBinCenters()

    def CaculateTFPoints(self, vRegin = False, fixKey = False): 
        f = self.frequencyRegionsPoints
        v = self.visionRegionsPoints
        k = self.keyValuePoints
        d = self.keyPointParamenter
        dt = self.binDistance/2
        
        print(f"最终f: {f}")
        print(f"最终v: {v}")
  
        if vRegin:
            fv = np.union1d(f, v)
        else:
            fv = f

        # 关键点
        fixKeyPoints = []
        k = np.array(k)
        for i in range(k.shape[0]):
            krange = [k[i]-dt, k[i]+dt]
            for j in range(k.shape[0]):
                # 若子区间范围内已经有了点，则不加该关键点的固定控制点
                # if k[j] >= krange[0] and k[j] <= krange[1]:
                #     # break
                #     pass
                # else:
                #     fixKeyPoints.append(k[i]-dt)
                #     fixKeyPoints.append(k[i]+dt)
                fixKeyPoints.append(k[i]-dt)
                fixKeyPoints.append(k[i]+dt)   
                    
        fixKeyPoints = np.array(fixKeyPoints)
        fixKeyPoints = np.unique(fixKeyPoints)

        # dynamicPoint
        dynamicPoints = []
        for i in range(fv.shape[0]-1):
            dynamicPoint = fv[i] + (fv[i+1]-fv[i])/2
            dynamicPoints.append(dynamicPoint)
        dynamicPoints = np.array(dynamicPoints)
        
        if fixKey == False:
            self.fixPoints = fv
            self.dynamicPoints = dynamicPoints
            # print("fixPoints:", self.fixPoints)
            # print("dynamicPoints:", self.dynamicPoints)
            return
        else:
            # self.fixPoints = np.union1d(fv, fixKeyPoints)
            self.fixPoints = np.union1d(fv, fixKeyPoints)
            self.dynamicPoints = np.union1d(dynamicPoints,k)

        # print("fixPoints:", self.fixPoints)
        # print("dynamicPoints:", self.dynamicPoints)
        
    def ConcretePointList(self):
        f = self.fixPoints
        d = self.dynamicPoints

        f_list = []
        d_list = []
        for i in range(f.shape[0]):
            f_list.append([f[i], 0.0, 0.5, 0.0])
        
        for i in range(d.shape[0]):
            d_list.append([d[i], 0.5, 0.5, 0.0])
        
        pointList = f_list + d_list
        pointArray = np.array(pointList)
        pointArray= pointArray[pointArray[:, 0].argsort()].ravel()
        return pointArray.tolist()
    

    def OpacityForDynamicPoints(self):
        # 计算代表动态point的等值面距离视点的距离
        print(f"视点: {self.viewPoint}")

        area_arr = []
        gradual_point_nums = []
        distance = []
        for i in range(self.dynamicPoints.shape[0]):
            counterValue = self.dynamicPoints[i]
            # 创建一个vtkMarchingCubes来生成等值面
            surface = vtk.vtkMarchingCubes()
            surface.SetInputData(self.renderData)
            surface.ComputeNormalsOn()      # 计算法线
            surface.SetValue(0, counterValue)
            surface.Update()

            num_points =surface.GetOutput().GetPoints().GetNumberOfPoints()
            distance_calculator = vtk.vtkImplicitPolyDataDistance()
            distance_calculator.SetInput(surface.GetOutput())

            mass_properties = vtk.vtkMassProperties()
            mass_properties.SetInputData(surface.GetOutput())
            area = mass_properties.GetSurfaceArea()

            viewPoint = self.viewPoint
            min_distance = distance_calculator.EvaluateFunction(viewPoint)
            binEdges = [self.binCenters[0],self.binCenters[-1]]

            # 提取等值面的所有点
            # viewPoint = np.array(self.viewPoint)
            # points = surface.GetOutput().GetPoints()
            # num_points = points.GetNumberOfPoints()

            # # 遍历每个点计算距离
            # min_distance = float("inf")  # 初始化最小距离为正无穷
            # for i in range(num_points):
            #     point = np.array(points.GetPoint(i))  # 获取点坐标 [x, y, z]
            #     distance = np.linalg.norm(viewPoint - point)  # 计算欧几里得距离
            #     if distance < min_distance:
            #         min_distance = distance  # 更新最小距离
            area_arr.append(area)
            distance.append(abs(min_distance))

            print(f"视点到等值面 {counterValue} 最近点的最短距离: {min_distance} 数量：{num_points} 面积：{area}")
           
        # 渐变等值数量
        for i in range(self.fixPoints.shape[0]):
            if i <= 0: continue
            pointNum = self.fixPoints[i]-self.fixPoints[i-1]
            gradual_point_nums.append(pointNum)
            print(f"poinNum：{pointNum}")
        
        # 面积归一化
        area_arr = np.array(area_arr)
        normalized_area_arr = (area_arr - np.min(area_arr)) / (np.max(area_arr) - np.min(area_arr))

        # 渐变点归一化
        gradual_arr = np.array(gradual_point_nums)
        normalized_gradual_arr = (gradual_arr - np.min(gradual_arr)) / (np.max(gradual_arr) - np.min(gradual_arr))

        distance_arr = np.array(distance)
        alpha = 1/(np.max(distance)-np.min(distance))
        # 指数衰减计算
        exponential_decay = np.exp(-alpha * distance_arr)

        # 对指数衰减结果进行归一化
        normalized_distance = (exponential_decay - np.min(exponential_decay)) / (np.max(exponential_decay) - np.min(exponential_decay))
        
        print(f"areas: {normalized_area_arr}")
        print(f"gradual: {normalized_gradual_arr}")
        print(f"distance: {normalized_distance}")

        opacity = 0.5 * normalized_distance + 0.2 * normalized_gradual_arr + 0.3 * normalized_area_arr

        unopacity = (1-opacity)*0.8
        print(f"opacity: {unopacity}")
        

    def WriteJSONFile(self):
        """
        写入JSON文件，在paraview中可视化
        """ 
        PointList = self.ConcretePointList()

        jsonData = {}
        jsonData["ColorSpace"] = "Diverging"
        jsonData["Name"] = "Test"
        jsonData["Points"] = PointList 
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
        with open('itf/tf3.json', 'w') as json_file:
            json.dump(jsonDatas, json_file, indent=4)  # indent 参数用于格式化输出

