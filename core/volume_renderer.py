import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path

import dh.data_files as df
import vtk
import numpy as np


class CameraParamenter():
    """
    摄像机参数
    """
    def __init__(self):
        self.view_angle = None
        self.focal_point = None
        self.view_point = None
        self.view_up = None
        self.x_res = None
        self.y_res = None
    
    def SetCameraParamenter(self, view_angle, focal_point, view_up, view_point, x_res, y_res):
        self.view_angle = view_angle
        self.focal_point = focal_point
        self.view_up = view_up
        self.view_point = view_point
        self.x_res = x_res
        self.y_res = y_res

# 自定义交互样式
class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, interactor, data_bounds, data_spacing, resolution_scale=1.0):
        super().__init__()
        self.renderer = renderer
        self.interactor = interactor
        self.InitDataInfo(data_bounds, data_spacing, resolution_scale)
        self.camera_parameter = CameraParamenter()
        self.AddObserver("KeyPressEvent", self.onKeyPress)
    
    def GetCameraParameter(self): return self.camera_parameter
    
    def InitDataInfo(self, bounds, spacing, resolution_scale=1.0):
        b = bounds
        # 计算维度和几何中心
        self.spacing = [spacing[0], spacing[1], spacing[2]]
        self.dims =  [b[1]-b[0], b[3]-b[2], b[5]-b[4]]      
        self.dims = [self.dims[0]*self.spacing[0], self.dims[1]*self.spacing[1], self.dims[2]*self.spacing[2]]  # 复制一份数据维度
        self.center = [b[0] + (b[1] - b[0]) / 2, b[2] + (b[3] - b[2]) / 2, b[4] + (b[5] - b[4]) / 2]
        self.center = [self.center[0]*spacing[0], self.center[1]*spacing[1], self.center[2]*spacing[2]]  # 转为列表
        self.resolution_scale = resolution_scale
        self.u_res = 600
        self.v_res = 400

        print(f"Data bounds: {b}")
        print(f"Data dimensions: {self.dims}")
        print(f"Data center: {self.center}")

    def onKeyPress(self, obj, event):
        key = self.interactor.GetKeySym()
        print(f"Pressed key: {key}")  # 打印按键，便于调试
        if key == 'a':  # 设置为 +x 方向
            obj =  self.set_camera_view('+x')
        elif key == 'z':  # 设置为 x 方向
            return self.set_camera_view('-x')
        elif key == 's':  # 设置为 -y 方向
            return self.set_camera_view('+y')
        elif key == 'x':  # 设置为 y 方向
            return self.set_camera_view('-y')
        elif key == 'd':  # 设置为 z 方向
            return self.set_camera_view('+z')
        elif key == 'c':  # 设置为 -z 方向
            return self.set_camera_view('-z')
        elif key == 'i':  # 向上旋转 90°
            return self.set_camera_rotation('up')
        elif key == 'o':  # 向下旋转 90°
            return self.set_camera_rotation('down')
        elif key == 'k':
            return self.set_camera_rotation('left')
        elif key == 'l':
            return self.set_camera_rotation('right')
        elif key == 'm':
            return self.set_camera_rotation('front')
        elif key == 'n':
            return self.set_camera_rotation('back')

    def set_camera_rotation(self, direction):
        """根据方向旋转摄像机 90°"""
        camera = self.renderer.GetActiveCamera()
        # 获取当前的 viewUp 向量
        current_view_up = camera.GetViewUp()

        if direction == 'up':  # 向上旋转 90°
            # 交换当前的 viewUp 向量，旋转 90°
            self.view_up = [self.view_up[1], self.view_up[0], self.view_up[2]]
        elif direction == 'down':  # 向下旋转 90°
            # 交换当前的 viewUp 向量，旋转 -90°
            self.view_up = [-self.view_up[1], -self.view_up[0], self.view_up[2]]
        elif direction == 'left':  # 向左旋转 90°
            # 交换当前的 viewUp 向量，绕Z轴旋转 90°
            self.view_up = [self.view_up[2], self.view_up[1], -self.view_up[0]]
        elif direction == 'right':  # 向右旋转 90°
            # 交换当前的 viewUp 向量，绕Z轴旋转 -90°
            self.view_up = [-self.view_up[2], self.view_up[1], self.view_up[0]]
        elif direction == 'front':
            self.view_up = [self.view_up[2], -self.view_up[1], self.view_up[0]]
        elif direction == 'back':
             self.view_up = [self.view_up[2], -self.view_up[1], -self.view_up[0]]


        # 设置新的 viewUp 向量
        self.camera_parameter.SetCameraParamenter(60, self.center, self.view_up, self.view_point, self.u_res, self.v_res)
        camera.SetViewUp(self.view_up[0], self.view_up[1], self.view_up[2])

        # 更新相机视角
        self.renderer.ResetCameraClippingRange()
        self.interactor.GetRenderWindow().Render()
    

    def set_camera_view(self, direction):
        """根据给定方向设置相机视角"""
        camera = self.renderer.GetActiveCamera()
        self.dims = [self.dims[0], self.dims[1], self.dims[2]]  # 复制一份数据维度
        # 动态设置 viewUp，避免它与相机方向平行
        if direction == '-x' or direction == '+x':  # y,z面
            self.view_up = [0, 0, 1]  # 保持z轴为上方向
        elif direction == '-y' or direction == 'y': # x,z面
            self.view_up = [0, 0, 1]  # 保持z轴为上方向
        elif direction == 'z' or direction == '-z': # x,y面
            self.view_up = [0, 1, 0]  # 保持y轴为上方向
        
        # 根据方向设置相机的位置和焦点
        viewPoint = None
        center, dims = self.center, self.dims
        if direction == '+x':    # 从+x方向看, 看的是y,z平面
            camera_distance = (dims[2] / 2) * np.tan(np.radians(60)) 
            viewPoint = [dims[0] + camera_distance, center[1], center[2]]
            
            self.u_res = int(self.dims[1] * self.resolution_scale)
            self.v_res = int(self.dims[2] * self.resolution_scale)

        elif direction == '-x':    # 从-x方向看, 看的是y,z平面
            camera_distance = (dims[2] / 2) * np.tan(np.radians(60)) 
            viewPoint = [0 - camera_distance, center[1], center[2]]
            
            self.u_res = int(self.dims[1] * self.resolution_scale)
            self.v_res = int(self.dims[2] * self.resolution_scale)
        
        elif direction == '+y':    # 从+y方向看, 看的是x,z平面
            camera_distance = (dims[2] / 2) * np.tan(np.radians(60)) 
            viewPoint = [center[0], dims[1] + camera_distance, center[2]]
            
            self.u_res = int(self.dims[0] * self.resolution_scale)
            self.v_res = int(self.dims[2] * self.resolution_scale)

        elif direction == '-y':    # 从-y方向看, 看的是x,z平面
            camera_distance = (dims[2] / 2) * np.tan(np.radians(60)) 
            viewPoint = [center[0], 0 - camera_distance, center[2]]
            
            self.u_res = int(self.dims[0] * self.resolution_scale)
            self.v_res = int(self.dims[2] * self.resolution_scale)
        
        elif direction == '+z':    # 从z方向看, 看的是x,y平面
            camera_distance = (dims[1] / 2) * np.tan(np.radians(60)) 
            viewPoint = [center[0], center[1], dims[2] + camera_distance]
            
            self.u_res = int(self.dims[0] * self.resolution_scale)
            self.v_res = int(self.dims[1] * self.resolution_scale)

        elif direction == '-z':    # 从-z方向看, 看的是x,y平面
            camera_distance = (dims[1] / 2) * np.tan(np.radians(60)) 
            viewPoint = [center[0], center[1], 0 - camera_distance]
            
            self.u_res = int(self.dims[0] * self.resolution_scale)
            self.v_res = int(self.dims[1] * self.resolution_scale)
    
        elif direction == 'up':  # 向上旋转 90°
            self.view_up = [self.view_up[1], self.view_up[0], self.view_up[2]]

        elif direction == 'down':  # 向下旋转 90°
            self.view_up = [-self.view_up[1], -self.view_up[0], self.view_up[2]]

        print(f"view_up: {self.view_up}")
        self.view_point = viewPoint
        self.camera_parameter.SetCameraParamenter(60, self.center, self.view_up, viewPoint, self.u_res, self.v_res)
        camera.SetViewAngle(60)  # 保持相机的视角不变
        camera.SetViewUp(self.view_up[0],self.view_up[1],self.view_up[2])  # 保持z轴为上方向
        camera.SetFocalPoint(self.center[0], self.center[1], self.center[2])  # 重置焦点
        camera.SetPosition(viewPoint[0], viewPoint[1], viewPoint[2])
        self.interactor.GetRenderWindow().SetSize(self.u_res, self.v_res)
        self.renderer.ResetCameraClippingRange()
        print(f"res_x {self.u_res}, res_y {self.v_res}")
        print(f"view_point: {viewPoint}")
        
        self.interactor.GetRenderWindow().Render()
        return self.camera_parameter


class VolumeRenderer:
    def __init__(self):
        self.path = None                # 体数据文件路径
        self.dims = None                # 体数据数据维度
        self.bounds = None
        self.center = None              # 体数据数据几何中心    
        self.spacing = None         # 体数据数据间距
        self.u_res = None               # 窗口x方向分辨率
        self.v_res = None               # 窗口y方向分辨率 
        self.resolution_scale = None    # 体数据数据分辨率缩放比例

        self.tf_boundaries = None       # 颜色映射函数
        self.tf_features = None         # 透明度映射函数

        self.reader = None              # 体数据读取器
        self.style = None               # 自定义交互样式
    
    def GetCameraParams(self):
        return self.style.GetCameraParameter()

    def SetRenderData(self, data, res_scale = 2.0):
        self.path = data.path
        self.dims = data.dim
        self.center = [i/2 for i in data.dim]
        if data.spacing is None:
            self.spacing = [1.0,1.0,1.0]
        else:
            self.spacing = data.spacing
        self.bounds = [0, data.dim[0]-1, 0, data.dim[1]-1, 0, data.dim[2]-1]
        if data.dtype == 'uint8':
            self.dtype = vtk.VTK_UNSIGNED_CHAR
        elif data.dtype == 'uint16':
            self.dtype = vtk.VTK_UNSIGNED_SHORT
        elif data.dtype == 'float32':
            self.dtype = vtk.VTK_FLOAT

        self.resolution_scale = res_scale


    def Render(self):
        print(f"Volume rendering...data_spacing: {self.spacing}")
        print(f"Volume rendering...data_bounds: {self.bounds}")
        print(f"Volume rendering...data_center: {self.center}")
        print(f"Volume rendering...data_dim: {self.dims}")
        self.reader = vtk.vtkImageReader()
        self.reader.SetFileName(self.path)
        self.reader.SetFileDimensionality(3)
        self.reader.SetDataScalarType(self.dtype)
        self.reader.SetDataExtent(int(self.bounds[0]), int(self.bounds[1]), int(self.bounds[2]), int(self.bounds[3]), int(self.bounds[4]), int(self.bounds[5])) 
        self.reader.SetDataSpacing(self.spacing[0], self.spacing[1], self.spacing[2])
        self.reader.Update()

        # 翻转
        
        # 创建体积渲染器
        mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
        mapper.SetInputConnection(self.reader.GetOutputPort())
        # mapper.SetBlendModeToComposite()


         # Transferfunction
        dataArray = self.reader.GetOutput().GetPointData().GetScalars()
        scalarRange = dataArray.GetRange()
        colorTransferFunction = vtk.vtkColorTransferFunction()
        colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
        colorTransferFunction.AddRGBPoint(20, 0.4, 0.4, 1.0)
        colorTransferFunction.AddRGBPoint(30, 0.9, 0.4, 1.0)
        colorTransferFunction.AddRGBPoint(scalarRange[1], 0.4, 0.4, 1.0)

        opacityTransferFunction = vtk.vtkPiecewiseFunction()
        opacityTransferFunction.AddPoint(0, 0.3) 
        opacityTransferFunction.AddPoint(0,0.4)
        opacityTransferFunction.AddPoint(0,0.5)
        opacityTransferFunction.AddPoint(scalarRange[1], 0.1) 

        gradientOpacityTransferFunction = vtk.vtkPiecewiseFunction()
        gradientOpacityTransferFunction.AddPoint(0, 0.0)
        gradientOpacityTransferFunction.AddPoint(10, 0.5)

        # Actor the volume
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorTransferFunction)
        volumeProperty.SetScalarOpacity(opacityTransferFunction)
        volumeProperty.SetGradientOpacity(gradientOpacityTransferFunction)
        volumeProperty.ShadeOn()
        volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
        
        volume = vtk.vtkVolume()
        volume.SetProperty(volumeProperty)
        volume.SetMapper(mapper)

        # 渲染设置
        renderer = vtk.vtkRenderer()
        renderer.AddVolume(volume)
        renderer.SetBackground(0.1, 0.2, 0.4)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(100, 100, 100)
        axes.AxisLabelsOn()
        renderer.AddActor(axes)

        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(400, 300)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # 设置自定义交互风格
        self.style = CustomInteractorStyle(renderer, renderWindowInteractor,self.bounds,self.spacing, self.resolution_scale)
        renderWindowInteractor.SetInteractorStyle(self.style)

        renderWindowInteractor.Start()
  

   
    

def main():
    # 读取原始数据
    renderer = VolumeRenderer()
    # data = df.tooth()
    data = df.manix()
    renderer.SetRenderData(data, 3)
    renderer.Render()
    print("Done!")
    camera_parameter = renderer.style.GetCameraParameter()
    print(camera_parameter.view_angle)


    


if __name__ == '__main__':
    main()
