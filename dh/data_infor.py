from io import StringIO
import sys
class DataInfor:
    def __init__(self):
        self.ftype = None
        self.dtype = None
        self.dims = None

        self.dataRange = None
        self.dataClass = None

        self.pointNum = None
        self.cellNum = None

    def Create(self, dataInfor):
        self.ftype = dataInfor['ftype']
        self.dtype = dataInfor['dtype']
        self.dims = dataInfor['dims']

        self.dataRange = dataInfor['dataRange']
        self.dataClass = dataInfor['dataClass']

        self.pointNum = dataInfor['pointNum']
        self.cellNum = dataInfor['cellNum']
        
    def PrintInfo(self):
        output = StringIO()
        sys.stdout = output

        print(f"文件类型: {self.ftype}")
        print(f"数据类型: {self.dtype}")
        print(f"数据维度: {self.dims}")
        print(f"数据范围: {self.dataRange[0]:.2f} ~ {self.dataRange[1]:.2f}")
        print(f"数据类别: {self.dataClass}")
        print(f"顶点数量: {self.pointNum}")
        print(f"单元数量: {self.cellNum}")

        sys.stdout = sys.__stdout__
        return output.getvalue()