import os
import sys
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from dh.data_infor import DataInfor

current_dir = os.path.dirname(os.path.abspath(__file__))
sibling_dir = os.path.join(current_dir, '..')  # 获取兄弟目录的路径
sys.path.append(sibling_dir)  # 添加兄弟目录到 sys.path


class DataLoader:
    def __init__(self):
        self.reader = None
        self.imageData = None
        self.unstructuredGrid = None
        self.pointData = None
        self.scalars = None
        self.scalarsNp = None
        self.dimensions = None

        self.dataInfor = DataInfor()

    def LoadVTIFile(self, filePath, dims, dtype):
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(filePath)
        self.reader.Update()
        self.imageData = self.reader.GetOutput()

        if self.imageData is None:
            raise Exception("Failed to read the VTI file.")
        print("ImageData dimensions: ", self.imageData.GetDimensions())
        
        self.pointData = self.imageData.GetPointData()
        if self.pointData is None:
            raise Exception("No point data found in the VTI file.")
        print("Number of points: ", self.imageData.GetNumberOfPoints())

        self.scalars = self.pointData.GetScalars()
        if self.scalars is None:
            raise Exception("No scalar data found in the VTI file.")
        print("Scalar type: ", self.scalars.GetDataTypeAsString())

        self.scalarsNp = vtk_to_numpy(self.scalars)
        self.dimensions = self.imageData.GetDimensions()
        self.scalarRange = self.scalars.GetRange()

        # 保存数据信息TODO
        self.ftype = ".vti"
        self.dtype = dtype
        self.dims = dims

        self.dataRange = self.scalars.GetRange()
        self.dataClass = "image data"

        self.pointNum = self.imageData.GetNumberOfPoints()
        self.cellNum = self.imageData.GetNumberOfCells()

        dataInfor = {
            "ftype": self.ftype,
            "dtype": self.dtype,
            "dims": self.dims,
            "dataRange": self.dataRange,
            "dataClass": self.dataClass,
            "pointNum": self.pointNum,
            "cellNum": self.cellNum
        }
        self.dataInfor.Create(dataInfor)

        return self.dataInfor

    
    def LoadRawFile(self, filePath, dims, dtype):
        self.reader = vtk.vtkImageReader()
        self.reader.SetFileName(filePath)
        self.reader.SetFileDimensionality(3)
        self.reader.SetDataExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
        if dims[0] == 128 and dims[1] == 128 and dims[2] == 115:
            self.reader.SetDataSpacing(0.49, 0.49, 0.70)
            print("manix")
        elif dims[0] == 256 and dims[1] == 256 and dims[2] == 512:
            self.reader.SetDataSpacing(0.78125, 0.390625, 1)
            print("carp")
        else:
            self.reader.SetDataSpacing(1, 1, 1)

        if dtype == 'float32':
            self.reader.SetDataScalarTypeToFloat()
        elif dtype == 'uint16':
            self.reader.SetDataScalarTypeToUnsignedShort()
        elif dtype == 'uint8':
            self.reader.SetDataScalarTypeToUnsignedChar()
        else:
            raise Exception("Unsupported data type.")
        
        self.reader.Update()

        self.imageData = self.reader.GetOutput()
        if self.imageData is None:
            raise Exception("Failed to read the VTI file.")
        print("ImageData dimensions: ", self.imageData.GetDimensions())
        
        self.pointData = self.imageData.GetPointData()
        if self.pointData is None:
            raise Exception("No point data found in the VTI file.")
        print("Number of points: ", self.imageData.GetNumberOfPoints())

        self.scalars = self.pointData.GetScalars()
        if self.scalars is None:
            raise Exception("No scalar data found in the VTI file.")
        print("Scalar type: ", self.scalars.GetDataTypeAsString())

        self.scalarsNp = vtk_to_numpy(self.scalars)
        self.dimensions = self.imageData.GetDimensions()
        self.scalarRange = self.scalars.GetRange()
        print("Scalar Range: ", self.scalarRange)

        # 保存数据信息
        self.ftype = ".raw"
        self.dtype = dtype
        self.dims = dims

        self.dataRange = self.scalars.GetRange()
        self.dataClass = "image data"

        self.pointNum = self.imageData.GetNumberOfPoints()
        self.cellNum = self.imageData.GetNumberOfCells()

        dataInfor = {
            "ftype": self.ftype,
            "dtype": self.dtype,
            "dims": self.dims,
            "dataRange": self.dataRange,
            "dataClass": self.dataClass,
            "pointNum": self.pointNum,
            "cellNum": self.cellNum
        }
        self.dataInfor.Create(dataInfor)

        return self.dataInfor


    def LoadVTKFile(self, filePath):
        self.reader = vtk.vtkUnstructuredGridReader()
        self.reader.SetFileName(filePath)
        self.reader.Update()

        # 获取标量值名称
        self.unstructuredGrid = self.reader.GetOutput()
        scalarNameNum = self.unstructuredGrid.GetPointData().GetNumberOfArrays()
        self.scalarNames = []
        self.scalarRanges = {}
        for i in range(scalarNameNum):
            name = self.unstructuredGrid.GetPointData().GetArrayName(i)
            scalarRange = self.unstructuredGrid.GetPointData().GetScalars(name).GetRange()
            self.scalarNames.append(name)
            self.scalarRanges[name] = scalarRange
        
        print("Scalar Ranges: ", self.scalarRanges)

        # 其他信息
        self.Bounds = self.unstructuredGrid.GetBounds()
        self.pointNum = self.unstructuredGrid.GetNumberOfPoints()
        self.cellNum = self.unstructuredGrid.GetNumberOfCells()
        
        print("PointNum: ", self.pointNum)
        print("CellNum: ",self.cellNum)
    
    def GetImageData(self):
        return self.imageData

    def GetDataSpacing(self):
        return self.imageData.GetSpacing()

    def GetUnstructuredGrid(self):
        return self.unstructuredGrid
    
    def GetScalarsNumpy(self):
        return self.scalarsNp
    
    def GetScalarNumpyFromScalarName(self, scalarName):
        # 通过标量值名称获取标量值
        if scalarName not in self.scalarNames:
            raise Exception("Scalar name not found in the VTK file.")
        self.pointNum = self.unstructuredGrid.GetNumberOfPoints()
        self.pointData = self.unstructuredGrid.GetPointData().GetArray(scalarName);
        if self.pointData is None:
            raise Exception("No scalar data found in the VTK file.")
        
        self.scalarsNp = vtk_to_numpy(self.pointData)
        return self.scalarsNp
    
    def GetScalarRangeFromScalarName(self, scalarName): return self.scalarRanges[scalarName]

    def GetScalarBounds(self): return self.Bounds

    def GetScalarDims(self): return self.dimensions
    
    def GetScalarRange(self): return self.scalarRange
    
    def GetScalarRanges(self): return self.scalarRanges

    def GetScalarNames(self): return self.scalarNames
    
    def GetDataInfor(self): return self.dataInfor