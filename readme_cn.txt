#######################################################################################################################################

# 项目架构

--algo-- : 算法模块，主流程实现
algorithm01：基于频率信息的传输函数实现和视角选取
algorithm02：基于相似性和演化性的边界等值检测和代表等值提取
algorithm03：基于边界等值和代表等值的体渲染传输函数生成

--core-- : 核心模块，主要实现表面渲染、体渲染和损失计算的相关功能
core_calculater: 损失计算模块，实现各种损失函数的计算,聚类指标的计算
surface_render: 表面渲染模块，用于渲染等值面
volume_render: 体渲染模块，用于体绘制

-- dh -- : 数据处理模块，主要实现数据读取、预处理、增强、划分等功能
data_files: 数据集封装类，用于程序索引数据
data_info: 数据集信息封装类，用于提供调试信息
data_loader: 数据集加载封装类，用于加载数据集
image_handle: 用于处理渲染得到的等值面图像

--test-- : 测试模块，主要实现算法模块的单元测试、功能测试和性能测试
main_test: 环境配置好后应该能正常运行
experiment01: 参数对比实验
experiment01_result: 根据不同分辨率多次运行experiment01的可视化结果
experiment02: 性能实验
experiment03: 泛用性实验

--dataset--: 数据集
bonsai
carp
engine
hcci_oh
manix
tooth

--img--: 实验结果图像
AutoBoundaryPointDetection.png：演化边界分割结果
AutoRepesentativePointDetection.png： 代表等值提取结果
RegionSimilarity.png： 区域平均相似性
resolution_effect_on_boundary_value.png: 分辨率对演化边界的分割影响
totality_devision_infor_5.png: 频率演化性分割结果

--itf--: 实验结果传输函数
tf1:基础传输函数控制点
tf2:基于演化性的传输函数控制点
tf3:最终传输函数的控制点

environment.yaml : 环境配置文件, 使用anaconda导入

#######################################################################################################################################


#######################################################################################################################################

# 使用指南

1. 使用git将存储库克隆到本地
2. 使用anaconda导入environment配置环境
3. 使用vscode打开项目并配置好python相关插件
4. 运行test/main_test.py检查环境配置是否正确
5. 运行text下文件

#######################################################################################################################################


#######################################################################################################################################

# 重要函数

algorithm01：AutoReginDetection(): 基于频率的演化性划分
algorithm02：AutoBoundaryPointDetection(): 基于相似性的等值面演化边界检测
algorithm02：AutoBoundaryPointAjustion(): 基于连续性的等值面演化边界调整
algorithm03：CaculateTFPoints()：生成最终的传输函数控制点

core_calculater：CSM(): 相似性度量和连续性度量的计算
core_calculater: CaculateWassersteinDistanceDiscrete(): 基于wassertein的连续性度量计算
core_calculater: CalculateSparseClusterLossOne(): 聚类损失计算


#######################################################################################################################################

# 其他事项！！！

程序在执行到用户选择视角阶段时，请输入"a,z,s,x,d,c"来调整视角
carp数据集和hcci_oh由于太大无法上传至github, 请到 https://klacansky.com/open-scivis-datasets/ 获取

#######################################################################################################################################
感谢阅读