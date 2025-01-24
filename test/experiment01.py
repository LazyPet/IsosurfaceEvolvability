import json
import os
import sys
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df
from algo.Algorithm01 import UnstructuredGridASO, StructuredGridASO
from algo.Algorithm02 import StructuredGridAST, UnstructuredGridAST
from algo.Algorithm03 import AlgorithmStepThree

# 参数对比实验：图像尺寸


def test2():
    startTime = time.time()

    # dataFile = df.engine()
    # dataFile = df.bonsai()
    # dataFile = df.turbulent_combustion()
    # dataFile = df.manix()
    dataFile = df.tooth()
    # dataFile = df.carp()
    # dataFile = df.hcci()
    # dataFile = df.hydrogen()

    # dataFile = df.tacc()
    # dataFile = df.duct()
    # dataFile = df.velocity()
    # dataFile = df.tornado()
    n = 255
    m = 7
    boundary_bin_center_list = []
    representive_bin_center_list = []
    res_scale_list = [1, 2, 3, 4, 5]
    for i in range(0,len(res_scale_list)):

        aso = StructuredGridASO(dataFile)
        aso.LoadScalarArray()
        readTime = time.time()
        print(f"文件读取时间: {readTime-startTime} 秒")
        aso.SetFrequncyParamenter(255)        # 设置频率参数
        aso.SetContinuousParamenter(5)        # 设置连续值参数

        aso.TotalityFrequncyInfo()            # 计算频率信息
        aso.ShowTotalityDevisionInfor(log=True)
        # aso.ShowTotalityDevisionInfor()
        freqCTime = time.time()
        print(f"频率计算时间: {freqCTime-readTime} 秒")
        aso.AutoReginDevision()             # 自动分割
        aso.ShowTotalityDevisionInfor(show=True)     # 显示分割信息
        aso.GenerateFundamentalTFJson()

        aso.CameraParamentalSelect(res_scale = res_scale_list[i])
        freqDTime = time.time()
        print(f"频率分割时间: {freqDTime-freqCTime} 秒")

        ast = StructuredGridAST()
        ast.InitASTFromASO(aso)             # 初始化
        ast.SetClusterParamenter(30)        # 设置聚类参数
        ast.SetKeyValueParamenter(0.95)      # 设置关键值参数

        # ast.SelectViewParamenter()          # 设置视角参数
        ast.CaculateTotalityVisionInfor(show=True)   # 计算视觉信息
        

        ast.ShowTotalityInfo()
        # ast.ShowTotalityEvolutionInfo()
        ast.AutoBoundaryPointDetection(m, neigh_size=9)
        
        ast.ShowAutoBoundaryPointDetection()
        ast.AutoBoundaryPointAjustion(threshold=0.95, threshold2 = 0.9, n = round(n/(m*m)))
        boundart_bin_center = ast.ShowAutoBoundaryPointDetection()
        
        ast.AutoRepresentativePointDetectionImprove()
        representive_bin_center = ast.ShowRengionSimilarity(m)

        boundary_bin_center_list.append(boundart_bin_center)
        representive_bin_center_list.append(representive_bin_center)
        print(f"边界点：{boundart_bin_center}")
        print(f"代表点：{representive_bin_center}")

    print(f"边界点列表：{boundary_bin_center_list}")
    print(f"代表点列表：{representive_bin_center_list}")

    endTime = time.time()
    
if __name__ == '__main__':
    
    test2()