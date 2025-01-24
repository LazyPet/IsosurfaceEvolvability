import os
import vtk
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from vtkmodules.vtkCommonColor import vtkNamedColors

class ImageHandle:

    @staticmethod
    def WriteImage(fileName, renWin, rgba=True):
        if fileName:
            # Select the writer to use.
            path, ext = os.path.splitext(fileName)
            ext = ext.lower()
            if not ext:
                ext = '.png'
                fileName = fileName + ext
            if ext == '.bmp':
                writer = vtk.vtkBMPWriter()
            elif ext == '.jpg':
                writer = vtk.vtkJPEGWriter()
            elif ext == '.pnm':
                writer = vtk.vtkPNMWriter()
            elif ext == '.ps':
                if rgba:
                    rgba = False
                writer = vtk.vtkPostScriptWriter()
            elif ext == '.tiff':
                writer = vtk.vtkTIFFWriter()
            else:
                writer = vtk.vtkPNGWriter()

            windowto_image_filter = vtk.vtkWindowToImageFilter()
            windowto_image_filter.SetInput(renWin)
            windowto_image_filter.SetScale(1)  # image quality
            if rgba:
                windowto_image_filter.SetInputBufferTypeToRGBA()
            else:
                windowto_image_filter.SetInputBufferTypeToRGB()
                # Read from the front buffer.
                windowto_image_filter.ReadFrontBufferOff()
                windowto_image_filter.Update()

            writer.SetFileName(fileName)
            writer.SetInputConnection(windowto_image_filter.GetOutputPort())
            writer.Write()
        else:
            raise RuntimeError('Need a filename.')
    
    
    @staticmethod
    def StoreRenderImageToNumpy(render_window):
        # 创建一个vtkWindowToImageFilter来捕获渲染窗口内容
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.Update()

        # 创建一个vtkImageExport来导出图像数据
        image_exporter = vtk.vtkImageExport()
        image_exporter.SetInputConnection(window_to_image_filter.GetOutputPort())

        # 获取图像的维度
        width, height, _ = window_to_image_filter.GetOutput().GetDimensions()
        num_components = window_to_image_filter.GetOutput().GetNumberOfScalarComponents()

        # 创建numpy数组来存储图像数据
        image_array = np.zeros((height, width, num_components), dtype=np.uint8)

        # 设置ImageExport的输出指针为numpy数组的内存指针
        image_exporter.SetExportVoidPointer(image_array)
        image_exporter.Export()

        # VTK图像数据的原点在左下角，而numpy数组的原点在左上角，因此需要翻转
        image_array = np.flipud(image_array)

        return image_array


    @staticmethod
    def calculate_curvature(contour):
        
        num_points = len(contour)
        curvatures = np.zeros(num_points)
        
        for i in range(num_points):
            # 获取前一个点、当前点和下一个点的坐标
            prev_point = contour[i - 1][0]
            curr_point = contour[i][0]
            next_point = contour[(i + 1) % num_points][0]
            
            # 计算两个向量
            vec1 = prev_point - curr_point
            vec2 = next_point - curr_point
            
            # 计算两个向量的夹角
            angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            
            # 确保角度在 -pi 到 pi 之间
            if angle > np.pi:
                angle -= 2 * np.pi
            elif angle < -np.pi:
                angle += 2 * np.pi
            
            # 计算曲率：曲率 = 角度 / 向量长度
            length = np.linalg.norm(vec1) + np.linalg.norm(vec2)
            curvatures[i] = 2 * np.abs(np.sin(angle)) / length if length != 0 else 0
        
        return curvatures


    @staticmethod
    def CaculatImageEntropy(image: np.ndarray, frequncy)->float:
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用Canny边缘检测提取边缘
        edges = cv2.Canny(gray_image, 100, 200)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sumContours = 0
        lenContours = 0
        # 计算每个轮廓的曲率
        contour_curvatures = []
        for contour in contours:
            curvatures = ImageHandle.calculate_curvature(contour)
            contour_curvatures.append((contour, curvatures))
            sumContours += curvatures.sum()
            lenContours += len(curvatures)
        
        # plt.imshow(image, cmap='gray')
        # for contour, curvatures in contour_curvatures:
        #     for i, point in enumerate(contour):
        #         plt.scatter(point[0][0], point[0][1], c='white', s=curvatures[i] * 10)
        # plt.show()
        # cv2.waitKey(0)
        if lenContours == 0:
            return 0 
        return (sumContours/lenContours)
    

    @staticmethod
    def RenderImageWithPil(image_array):
        """
        使用PIL渲染numpy数组成图像
        :param image_array: numpy数组，形状为(height, width, channels)
        """
        image = Image.fromarray(image_array)
        image.show()


    @staticmethod
    def ImageNumpyMSE(image1, image2):
        """
        计算两张图像之间的均方误差（MSE）
        """
        # 检查图像尺寸是否相同
        if image1.shape != image2.shape:
            raise ValueError("The dimensions of the two images are not equal.")
        
        # 计算均方误差
        mse = np.mean((image1 - image2) ** 2)
        return mse


    @staticmethod
    def ImageMSE(imagePath1, imagePath2):
        imageA = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
        imageB = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)
        assert imageA.shape == imageB.shape, "Images must have the same dimensions"
        # 计算均方误差
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err
