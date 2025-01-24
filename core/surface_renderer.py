import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path
import math
import numpy as np
import vtk
from core.volume_renderer import CameraParamenter

"""
不用于GUI
"""
class SurfaceRenderer:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()

        self.imageData = False
        self.unstructuredGrid = False

        self.renderData = None

        self.contourValue = None
        self.contourIndex = 0

        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.renderWindowInteractor.SetInteractorStyle(style)
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)

        self.surface = None
        self.actor = None

        self.cameraParams = CameraParamenter()

    def Render(self, axes=False, border=False, offscreen=False): pass   # 虚函数

    def SetRenderData(self, renderData): self.renderData = renderData

    def ReleaseActor(self):
        self.renderer.RemoveActor(self.actor)
        self.surface = None
        self.actor = None     

    def SetImageData(self, imageData): self.imageData = imageData

    def SetUnstructuredGrid(self, unstructuredGrid): self.unstructuredGrid = unstructuredGrid

    def SetContourValue(self, value): self.contourValue = value
    
    def GetContourSurface(self): return self.surface

    def GetRenderer(self): return self.renderer

    def GetRenderWindow(self): return self.renderWindow

    def GetRenderInteractor(self): return self.renderWindowInteractor
    
    def SetCameraParams(self, cameraParams): 
        self.cameraParams = cameraParams
        camera = self.renderer.GetActiveCamera()
        self.x_res = None
        self.y_res = None
        camera.SetViewAngle(cameraParams.view_angle)
        camera.SetFocalPoint(cameraParams.focal_point)
        camera.SetViewUp(cameraParams.view_up)
        camera.SetPosition(cameraParams.view_point)

        self.renderWindow.SetSize(cameraParams.x_res, cameraParams.y_res)
        self.renderWindow.SetPosition(0, 0)


    def SetCameraPosition(self, bounds):
        b = bounds
        dims =  [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
        center = [b[0] + (b[1] - b[0]) / 2, b[2] + (b[3] - b[2]) / 2, b[4] + (b[5] - b[4]) / 2]
        

        # 设置焦点
        self.renderer.GetActiveCamera().SetFocalPoint(center[0], center[1], center[2])

        # 获取最长的两条边
        tdims = dims.copy()
        longAxisIndex = np.argmax(tdims)
        longAxis = dims[longAxisIndex]
        tdims[longAxisIndex] = 0
        shortAxisIndex = np.argmax(tdims)
        shortAxis = dims[shortAxisIndex]

        # 计算摄像机位置
        max_dim = max(tdims)
        self.renderer.GetActiveCamera().SetViewAngle(60)
        camera_distance = max_dim / (2 * np.tan(np.radians(30))) 

        # 设置相机位置和视角方向
        viewPoint = None
        if longAxisIndex == 0 and shortAxisIndex == 1:
            camera_distance = (dims[2] / 2) * np.tan(np.radians(60)) 
            viewPoint = [0 - camera_distance, center[1], center[2]]
            
            self.u_res = int(dims[1] * 3)
            self.v_res = int(dims[2] * 3)

            # self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
            # viewPoint = [center[0], center[1], center[2] + dims[2] / 2 + camera_distance]
            # self.renderer.GetActiveCamera().SetPosition(viewPoint[0],viewPoint[1],viewPoint[2])
        elif longAxisIndex == 0 and shortAxisIndex == 2:
            self.renderer.GetActiveCamera().SetViewUp(0, 0, 1)
            viewPoint = [center[0], center[1] - camera_distance, center[2]]
            self.renderer.GetActiveCamera().SetPosition(viewPoint[0],viewPoint[1],viewPoint[2])
        elif longAxisIndex == 1 and shortAxisIndex == 2:
            self.renderer.GetActiveCamera().SetViewUp(1, 0, 0)
            viewPoint = [center[0] - camera_distance, center[1], center[2]]
            self.renderer.GetActiveCamera().SetPosition(viewPoint[0],viewPoint[1],viewPoint[2])
        elif longAxisIndex == 2 and shortAxisIndex == 0:
            self.renderer.GetActiveCamera().SetViewUp(1, 0, 0)
            viewPoint = [center[0], center[1] + dims[1] / 2 + camera_distance, center[2]]
            self.renderer.GetActiveCamera().SetPosition(viewPoint[0],viewPoint[1],viewPoint[2])
        elif longAxisIndex == 2 and shortAxisIndex == 1: # 长轴为z,短轴为y
            self.renderer.GetActiveCamera().SetViewUp(0, 1, 0)
            viewPoint =[center[0] + dims[0] / 2 + camera_distance, center[1], center[2]]
            self.renderer.GetActiveCamera().SetPosition(viewPoint[0],viewPoint[1],viewPoint[2])
        
        # 设置渲染分辨率
        # if self.unstructuredGrid:                   # 如果是非结构数据
        #     if longAxis > 6:
        #         self.xresolution = int(longAxis * 100)
        #         self.yresolution = int(shortAxis * 100)
        #     elif longAxis > 4:
        #         self.xresolution = int(longAxis * 100)
        #         self.yresolution = int(shortAxis * 100)
        #     else :
        #         self.xresolution = int(longAxis * 200)
        #         self.yresolution = int(shortAxis * 200)
        # else:                                        # 如果是结构数据        
        #     if longAxis > 600:
        #         self.xresolution = int(longAxis * 1)
        #         self.yresolution = int(shortAxis * 1)
        #     elif longAxis > 250:
        #         self.xresolution = int(longAxis * 1)
        #         self.yresolution = int(shortAxis * 1)
        #     else :
        #         self.xresolution = int(longAxis * 4)
        #         self.yresolution = int(shortAxis * 4)

        # 设置渲染分辨率
        if self.unstructuredGrid:                   # 如果是非结构数据
            if longAxis > 6:
                self.xresolution = int(longAxis * 100)
                self.yresolution = int(shortAxis * 100)
            elif longAxis > 4:
                self.xresolution = int(longAxis * 100)
                self.yresolution = int(shortAxis * 100)
            else :
                self.xresolution = int(longAxis * 1600)
                self.yresolution = int(shortAxis * 1600)
        else:                                        # 如果是结构数据        
            if longAxis > 600:
                self.xresolution = int(longAxis * 1/2.0)
                self.yresolution = int(shortAxis * 1/2.0)
            elif longAxis > 250:
                self.xresolution = int(longAxis * 1)
                self.yresolution = int(shortAxis * 1)
            else :
                self.xresolution = int(longAxis * 2)
                self.yresolution = int(shortAxis * 2)
        print("resolution: ", self.xresolution, self.yresolution)
        self.renderer.ResetCameraClippingRange()
        self.renderWindow.SetSize(self.xresolution, self.yresolution)
        self.renderWindow.SetPosition(0, 0)
        return viewPoint


class ImageDataSurfaceRenderer(SurfaceRenderer):
    def __init__(self):
        super().__init__()

    def SetRenderData(self, imageData):
        self.renderData = imageData

    def Render(self, axes=False, border=False, offscreen=False):
        # 创建一个vtkMarchingCubes来生成等值面
        self.surface = vtk.vtkMarchingCubes()
        self.surface.SetInputData(self.renderData)
        self.surface.ComputeNormalsOn()      # 计算法线
        self.surface.SetValue(self.contourIndex, self.contourValue)
        self.surface.Update()

        # 创建一个vtkPolyDataMapper来映射等值面数据
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.surface.GetOutputPort())
        mapper.ScalarVisibilityOff()  # 关闭标量可见性，使用单一颜色

        # 创建一个vtkActor来表示等值面
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)
            # print("Actor removed")
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(1, 0, 0)  # 设置等值面颜色

        # 将等值面actor添加到renderer
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色
        
        # 显示坐标轴
        if axes:
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(100, 100, 100)
            axes.AxisLabelsOn()
            self.renderer.AddActor(axes)
        
        # 显示边框
        if border:
            outline = vtk.vtkOutlineFilter()
            outline.SetInputData(self.renderData)
            outlineMapper = vtk.vtkPolyDataMapper()
            outlineMapper.SetInputConnection(outline.GetOutputPort())
            outlineActor = vtk.vtkActor()
            outlineActor.SetMapper(outlineMapper)
            outlineActor.GetProperty().SetColor(1, 1, 1)
            self.renderer.AddActor(outlineActor)    
        
        if offscreen:
            self.renderWindow.OffScreenRenderingOn()

        # 启动渲染
        
        self.renderWindow.Render()

class UnstructuredGridSurfaceRenderer(SurfaceRenderer):
    def __init__(self):
        super().__init__()
    
    def SetRenderData(self, unstructuredGrid):
        self.unstructuredGrid = unstructuredGrid

    def Render(self, axes=False, border=False, offscreen=False):
        # 创建一个vtkContourFilter来生成等值面
        self.surface = vtk.vtkContourFilter()
        self.surface.SetInputData(self.unstructuredGrid)
        self.surface.ComputeNormalsOn()      # 计算法线
        self.surface.SetValue(self.contourIndex, self.contourValue)
        self.surface.Update()

        # 创建一个vtkPolyDataMapper来映射等值面数据
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.surface.GetOutputPort())
        mapper.ScalarVisibilityOff()  # 关闭标量可见性，使用单一颜色

        # 创建一个vtkActor来表示等值面
        if self.actor is not None:
            self.renderer.RemoveActor(self.actor)
            print("Actor removed")
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetColor(1, 0, 0)  # 设置等值面颜色

        # 将等值面actor添加到renderer
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色
        
        # 显示坐标轴
        if axes:
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(100, 100, 100)
            axes.AxisLabelsOn()
            self.renderer.AddActor(axes)
        
        # 显示边框
        if border:
            outline = vtk.vtkOutlineFilter()
            outline.SetInputData(self.unstructuredGrid)
            outlineMapper = vtk.vtkPolyDataMapper()
            outlineMapper.SetInputConnection(outline.GetOutputPort())
            outlineActor = vtk.vtkActor()
            outlineActor.SetMapper(outlineMapper)
            outlineActor.GetProperty().SetColor(1, 1, 1)
            self.renderer.AddActor(outlineActor)    
        
        if offscreen:
            self.renderWindow.OffScreenRenderingOn()

        # 启动渲染
        self.renderWindow.SetSize(self.xresolution, self.yresolution)
        self.renderWindow.SetPosition(0, 0)
        self.renderWindow.Render()
    


