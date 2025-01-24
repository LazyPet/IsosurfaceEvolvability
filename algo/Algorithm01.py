import json
import os
import sys
import cv2
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df

from dh.data_loader import DataLoader
from dh.image_handle import ImageHandle
from core.surface_renderer import SurfaceRenderer
from core.volume_renderer import VolumeRenderer

from skimage.metrics import structural_similarity as ssim
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class AlgorithmStepOne:
    """
    算法步骤1，进行频率统计，并作频率区间划分
    """
    def __init__(self, dataFile):
        self.dataFile = dataFile

        self.dataLoader = DataLoader()
        self.scalarName = None
        self.scalarArray = None
        self.scalarDims = None
        self.scalarRange = None

        self.surfaceRenderer = SurfaceRenderer()
        self.volumeRenderer = VolumeRenderer()

        self.totalFrequency = None
        self.totalProbDensity = None
        self.totalBinCenter = None

        self.regionsIndices = []
        self.regionsFrequencies = []
        self.regionsBinCenter = []
        self.regionsProbDensity = []

        self.dt = None  # 超参数
        self.continuous = None  # 超参数
    
    def SetFrequncyParamenter(self, p): self.dt = p

    def SetContinuousParamenter(self, p): self.continuous = p

    def LoadScalarArray(self, scalarName = None): pass  # 虚函数, 子类实现

    def GetCameraParams(self): return self.cameraParams

    def GetRegionsInfor(self): return self.regionsIndices, self.regionsBinCenter, self.regionsFrequencies

    def GetTotalityInfor(self): return self.totalBinCenter, self.totalFrequency

    def GetDataLoader(self): return self.dataLoader

    def GetBinDistance(self): return self.binDistance

    def GetFrequencyDt(self): return self.dt
    
    def TotalityFrequncyInfo(self):
        dt = self.dt
        # 计算总体区域的频率分布
        binCenters, frequency = self.RegionFrequencyInfo(dt, self.scalarRange[0], self.scalarRange[1])
        self.binDistance = (self.scalarRange[1]-self.scalarRange[0])/dt
        self.totalProbDensity = np.array(frequency/ (np.array(frequency).sum()), dtype=float)
        self.totalFrequency = np.array(frequency, dtype=int)
        self.totalBinCenter = np.array(binCenters,dtype=float)

    def RegionFrequencyInfo(self, dt, start, end):
        # 计算区域的频数分布
        # binWidth = dt
       
        # binNumber = int((end - start) / binWidth) + 1
        binNumber = dt
        binEdges = np.linspace(start, end, binNumber + 1)
        binCenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        # frequency, _ = np.histogram(self.scalarArray, bins=binEdges, density = False)
        frequency, _ = np.histogram(self.scalarArray, bins=binEdges, density = False)
        return binCenters, frequency
    
    def SoomthTotalityFreq(self):
        # 均值平滑
        y = self.totalFrequency
        section, smoothY = 6, []
        n = int(section / 2)
        
        for i in range(y.shape[0]):
            if i - n >= 0 and i + n + 1 <= y.shape[0]: 
                subY = y[i - n:i + n + 1]
                submean = np.mean(subY)
            else: 
                submean = y[i]
            smoothY.append(submean)
        
        self.self.totalFrequency = np.array(smoothY)

    def AutoReginDetection(self):
        x, y = self.totalBinCenter, self.totalFrequency
        # k为上一个区域内的斜率，count表示现有的斜率持续平滑点的个数，record表示记录持续平滑开始的位置
        k, count, record = 0, 0, -1            
        continuous = self.continuous                # 连续平滑的阈值, 超过这个阈值将会进行区域判断
        csymbol, psymbol = 0, 0                     # 分别记录当前持续平滑点的斜率符号和上一个区间的斜率符号
        regions = [0]                               # 记录区域划分点, 初始点为0，不算在任何区域，只是方便算法处理
        regions2 = [0]
        for i in range(1,self.totalFrequency.shape[0]):
            # if i < 1: continue                      # 线性拟合至少需要两个点
            last = regions[len(regions)-1]          # 取上一个区域的划分点

            # t = np.argmin(y[last:i])                        # 取last和izio之间的最小值
            # p = (y[i]-y[last + t])/(x[i]-x[last + t])       # 计算斜率
            p = (y[i]-y[last])/(x[i]-x[last])
            if abs(p) - abs(k) < 0 :                # 若斜率变得平滑，即斜率的绝对值变小
                if count == 0: record = i-1         # 记录最开始的平滑点为待定区域划分点
                count += 1                          # 连续平滑计数器加1 
            else:
                count = 0                           # 斜率变得不平滑，重置计数器
                record = -1                         # 重置记录点
            
            # 平滑点到了阈值即判断是否为区域划分点
            if count > continuous:                  # 若持续了continuous个平滑点
                lastFreq = self.totalFrequency[last]                # 取上一个区域的频率
                recordFreq = self.totalFrequency[record]            # 取待定区域的频率
                nextFreq = self.totalFrequency[record+continuous]   # 取后续的频率
                

                if recordFreq - lastFreq > 0: psymbol = 1           # 判断符号, 上升趋势
                else: psymbol = -1                                  # 下降趋势
                if nextFreq - recordFreq > 0: csymbol = 1           # 判断符号, 上升趋势
                else: csymbol = -1                                  # 下降趋势

                if psymbol * csymbol > 0:                           # 若符号相同，则为同一区域
                    i = record + 1                                  # 重置i为平滑点的持续区间后下一个点
                    count = 0                                       # 重置计数器
                    record = -1                                     # 重置记录点
                else:                                               # 符号不同，则为不同区域
                    if psymbol > 0 and csymbol <0:                  # 若从上升到下降
                        regions2.append(record)                     # 记录区域划分点
                    regions.append(record)
                    i = record + 1                                  # 重置i为第一个平滑点的下一个点
                    record = -1                                     # 重置记录点
                    count = 0                                       # 重置计数器
            k = p
        regions.append(self.totalFrequency.shape[0]-1)                                         # 最后一个点也作为区域划分点
        regions2.append(self.totalFrequency.shape[0]-1)
        print("RegionsIndices: ", regions)
        self.regionsIndices = np.array(regions2)
        print("RegionsIndicesMaximum: ", regions2)

    def AutoReginDevision(self):
        """
        将autoRegionDetection的到的结果进一步转化存储，以便使用
        """
        self.AutoReginDetection() 
        rengions = self.regionsIndices

        for i in range(len(rengions)):
            index = self.totalBinCenter[rengions[i]]
            freq = self.totalFrequency[rengions[i]]
            density = self.totalProbDensity[rengions[i]]

            self.regionsBinCenter.append(index)
            self.regionsFrequencies.append(freq)
            self.regionsProbDensity.append(density)

        self.regionsBinCenter = np.array(self.regionsBinCenter, dtype=float)
        self.regionsFrequencies = np.array(self.regionsFrequencies, dtype=float)
        self.regionsProbDensity = np.array(self.regionsProbDensity, dtype=float)
        
        # print("RegionsBinCenter: ", self.regionsBinCenter)
        # print("RegionsFrequencies: ", self.regionsFrequencies)
        # print("RegionsProbDensity: ", self.regionsProbDensity)
        
    def ShowTotalityInfor(self, log=True):
        x, y = self.totalBinCenter, self.totalProbDensity
        if log: 
            epslon = 1e-4
            y = np.log2(y + epslon)
        plt.scatter(x, y, s=2, alpha=0.8, c='blue')
        # plt.grid(True)
        plt.show()
        
    def ShowTotalityDevisionInfor(self, log=True, show=True):
        if show==False: return
        
        xsplits = self.regionsBinCenter
        for i,xsplit in enumerate(xsplits):
            if i == 0: 
                plt.axvline(x=xsplit, color='r', linestyle='--', linewidth=1, label='boundary isovalue')
            else:
                plt.axvline(x=xsplit, color='r', linestyle='--', linewidth=1)
            # plt.axvline(x=xsplit, color='r', linestyle='--', linewidth=1)
        
        x, y = self.totalBinCenter, self.totalProbDensity
        if log: 
            epslon = 1e-4
            y = np.log2(y + epslon)

        plt.scatter(x, y, s=2, alpha=0.8, c='blue')
        
        title = f"continuous parameter: {self.continuous}"
        plt.xlabel("isovalue", fontsize=12)  # 根据实际数据调整 x 轴标签
        plt.ylabel("log$_2$(probability density + ε)",fontsize=12)  # 根据实际数据调整 y 轴标签
        plt.title(title,fontsize=14)
        plt.legend(loc='lower right')
        plt.savefig(f"img/totality_devision_infor_{self.continuous}.png", dpi=600)
        plt.show()
        
    def CameraParamentalSelect(self, res_scale=2):
        self.volumeRenderer.SetRenderData(self.dataFile, res_scale)
        self.volumeRenderer.Render()
        self.cameraParams = self.volumeRenderer.GetCameraParams()
    

    def GenerateFundamentalTFJson(self):
        """
        利用基于频率的分割，生成基础传输函数
        """
        binWidth = self.binDistance
        evolutionBoundaries = self.regionsIndices
        scalarRange = self.dataLoader.GetScalarRange()
        minValue = scalarRange[0]

        actualEvolutionBoundaries = self.totalBinCenter[evolutionBoundaries]

        actualRepresentation = []
        for i in range(len(actualEvolutionBoundaries)-1):
            midpoint = (actualEvolutionBoundaries[i] +actualEvolutionBoundaries[i + 1]) / 2
            actualRepresentation.append(midpoint)

        print("EvolutionBoundaries: ",  actualEvolutionBoundaries)
        print("ActualRepresentation: ", actualRepresentation)


        # 构造传输函数控制点
        f = np.array(actualEvolutionBoundaries)
        d = np.array(actualRepresentation)

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
        with open('itf/tf1.json', 'w') as json_file:
            json.dump(jsonDatas, json_file, indent=4)  # indent 参数用于格式化输出


       



class UnstructuredGridASO(AlgorithmStepOne):
    def __init__(self, dataFile):
        super().__init__(dataFile)
        self.dataLoader.LoadVTKFile(dataFile.path)
    

    def LoadScalarArray(self, scalarName = None):
        self.scalarName = scalarName
        self.unstructuredGrid = self.dataLoader.GetUnstructuredGrid()
        self.scalarArray = self.dataLoader.GetScalarNumpyFromScalarName(self.scalarName)
        self.scalarRange = self.dataLoader.GetScalarRangeFromScalarName(self.scalarName)
        self.scalarBounds = self.dataLoader.GetScalarBounds()
        
        print("DatasetType: Unstructedgrid")
        print("Scalar Bounds：", self.scalarBounds)
    
    def GetScalarBounds(self): return self.scalarBounds
    
    def GetRenderData(self): return self.unstructuredGrid


class StructuredGridASO(AlgorithmStepOne):
    def __init__(self, dataFile):
        super().__init__(dataFile)
        self.dataLoader.LoadRawFile(dataFile.path, dataFile.dim, dataFile.dtype)
    
    def LoadScalarArray(self, scalarName=None):
        self.imageData = self.dataLoader.GetImageData()
        self.scalarArray = self.dataLoader.GetScalarsNumpy()
        self.scalarRange = self.dataLoader.GetScalarRange()
        self.scalarDims = self.dataLoader.GetScalarDims()
    
    def GetScalarBounds(self): return np.array([0,self.scalarDims[0],0,self.scalarDims[1],0,self.scalarDims[2]])

    def GetRenderData(self): return self.imageData


if __name__ == '__main__':
    dataFile = df.velocity()
    # dataFile = df.tangaroa()
    aso = UnstructuredGridASO(dataFile)
    aso.LoadScalarArray("U")
    aso.SetFrequncyParamenter(0.025)
    aso.SetContinuousParamenter(5)
    aso.TotalityFrequncyInfo()
    aso.AutoReginDevision()
    aso.ShowTotalityDevisionInfor(log=False)

    # dataFile = df.tornado()
    # aso = StructuredGridASO(dataFile)
    # aso.LoadScalarArray()
    # aso.TotalityFrequncyInfo(0.05)
    # aso.AutoReginDevision()
    # aso.ShowTotalityDevisionInfor(log=True)
