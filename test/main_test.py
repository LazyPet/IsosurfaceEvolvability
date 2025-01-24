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

def test1():
    startTime = time.time()

    dataFile = df.velocity2()
   
    aso = UnstructuredGridASO(dataFile)
    aso.LoadScalarArray("U")

    readTime = time.time()
    print(f"文件读取时间: {readTime-startTime} 秒")

    aso.SetFrequncyParamenter(255)
    aso.SetContinuousParamenter(5)
    
    aso.TotalityFrequncyInfo()
    aso.ShowTotalityInfor()

    freqCTime = time.time()
    print(f"频率计算时间: {freqCTime-readTime} 秒")

    aso.AutoReginDevision()
    aso.ShowTotalityDevisionInfor(show=True)
    aso.CameraParamentalSelect(res_scale = 2)

    freqDTime = time.time()
    print(f"频率分割时间: {freqDTime-freqCTime} 秒")

    ast = UnstructuredGridAST()
    ast.InitASTFromASO(aso)
    ast.SetClusterParamenter(18)
    ast.SetKeyValueParamenter(0.9)

    ast.CaculateTotalityVisionInfor()
    ast.CaculateSimilatyAndContinousInfor()
    ast.ShowTotalityInfo()
    visCTime = time.time()
    print(f"视觉计算时间：{visCTime-freqDTime} 秒")

    ast.AutoBoundaryPointDetection(5)
    ast.ShowAutoBoundaryPointDetection()
    ast.AutoBoundaryPointAjustion(threshold=0.96, n = 6)
    ast.ShowAutoBoundaryPointDetection()


    ast.TotalityVisionDensityProject(show=True)

    visDTime = time.time()
    print(f"视觉分割时间：{visDTime-visCTime} 秒")
    
    ase = AlgorithmStepThree()
    ase.InitASEFromAST(ast)
    ase.SetKeyPointParamenter(1)        # 设置关键点参数

    ase.CaculateTFPoints(fixKey=False)   # 计算传输函数布点
    mergeTime = time.time()
    print(f"区域合成时间：{mergeTime-visDTime}")
    ase.WriteJSONFile()                 # 写入JSON文件

    endTime = time.time()
    execution_time = endTime - startTime
    print(f"程序运行时间: {execution_time} 秒")


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
    c = 5
    r = 3

    aso = StructuredGridASO(dataFile)
    aso.LoadScalarArray()


    aso.SetFrequncyParamenter(n)              # 设置频率参数
    aso.SetContinuousParamenter(c)              # 设置连续值参数

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
    # ast.ShowAutoBoundaryPointDetection()
    ast.AutoBoundaryPointAjustion(threshold=0.95, threshold2 = 0.9, n = round(n/(m*m)))
    time5 = time.time()
    print(f"自动边界检测时间：{time5-time4} 秒，{(time5-time4)*1000} 毫秒")


    ast.ShowAutoBoundaryPointDetection()
    ast.GenerateImproveTFJson()

    time6 = time.time()
    ast.AutoRepresentativePointDetectionImprove()
    time7 = time.time()
    print(f"代表等值提取时间：{time7-time6} 秒，{(time7-time6)*1000} 毫秒")


    ast.ShowRengionSimilarity(m)
    ast.ShowAutoRepresentativePointDetectionImprove()
    

    visDTime = time.time()
    ast.TotalityVisionDensityProject(show=True)  # 密度投影

    projectTime = time.time()
    print(f"密度投影时间：{projectTime-visDTime} 秒")
    
    

    ase = AlgorithmStepThree()
    ase.InitASEFromAST(ast)
    ase.SetKeyPointParamenter(1)        # 设置关键点参数

    ase.CaculateTFPoints(vRegin=True, fixKey=False)   # 计算传输函数布点
    mergeTime = time.time()
    print(f"区域合成时间：{mergeTime-visDTime}")

    ase.OpacityForDynamicPoints()

    ase.WriteJSONFile()                 # 写入JSON文件

    endTime = time.time()
    execution_time = endTime - startTime
    print(f"程序运行时间: {execution_time} 秒")


if __name__ == '__main__':
    # test1()
    
    test2()
