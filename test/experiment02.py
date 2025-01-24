import json
import os
import sys
import time
import cv2
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df
from algo.Algorithm01 import UnstructuredGridASO, StructuredGridASO
from algo.Algorithm02 import StructuredGridAST, UnstructuredGridAST
from algo.Algorithm03 import AlgorithmStepThree

# 性能实验
def test2():

    # dataFile = df.engine()
    # dataFile = df.bonsai()
    # dataFile = df.turbulent_combustion()
    dataFile = df.manix()
    # dataFile = df.tooth()
    # dataFile = df.carp()
    # dataFile = df.hcci()
    # dataFile = df.hydrogen()

    # dataFile = df.tacc()
    # dataFile = df.duct()
    # dataFile = df.velocity()
    # dataFile = df.tornado()
    n = 255
    m = 6
    c = 5
    r = 5

    aso = StructuredGridASO(dataFile)
    aso.LoadScalarArray()


    aso.SetFrequncyParamenter(n)                 # 设置频率参数
    aso.SetContinuousParamenter(c)                  # 设置连续值参数

    time1 = time.time()
    aso.TotalityFrequncyInfo()                  # 计算频率信息
    aso.AutoReginDevision()                     # 自动分割
    aso.GenerateFundamentalTFJson()
    time2 = time.time()

    print(f"基于频率分割的传输函数生成时间: {time2-time1} 秒，{(time2-time1)*1000} 毫秒")

    
    
    aso.ShowTotalityDevisionInfor(show=True)     # 显示分割信息
    aso.CameraParamentalSelect(res_scale = r)


    ast = StructuredGridAST()
    ast.InitASTFromASO(aso)                         # 初始化
    ast.SetClusterParamenter(30)                    # 设置聚类参数
    ast.SetKeyValueParamenter(0.95)                 # 设置关键值参数

    # ast.SelectViewParamenter()                     # 设置视角参数
    ast.CaculateTotalityVisionInfor(show=True)        # 计算视觉信息
    time2 = time.time()
    ast.CaculateSimilatyAndContinousInfor()
    time3 = time.time()
    print(f"连续相似性度量计算时间：{time3-time2} 秒，{(time3-time2)*1000} 毫秒")
    ast.ShowTotalityInfo()


    time4 = time.time()
    ast.AutoBoundaryPointDetection(m, neigh_size=9)
    ast.ShowAutoBoundaryPointDetection()
    ast.AutoBoundaryPointAjustion(threshold=0.999, threshold2 = 0.9, n = round(n/(m*m)))
    time5 = time.time()
    print(f"自动边界检测时间：{time5-time4} 秒，{(time5-time4)*1000} 毫秒")


    ast.ShowAutoBoundaryPointDetection()
    ast.GenerateImproveTFJson()

    time6 = time.time()
    # ast.AutoRepresentativePointDetectionImprove()
    time7 = time.time()
    print(f"代表等值提取时间：{time7-time6} 秒，{(time7-time6)*1000} 毫秒")


    # ast.ShowRengionSimilarity(m)
    ast.ShowAutoRepresentativePointDetectionImprove()
    


if __name__ == '__main__':
    # test1()
    
    test2()
