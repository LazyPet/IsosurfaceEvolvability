DATA_FILE_ROOT = "dataset/"
class DataFiles:
    def __init__(self):
        self.path = DATA_FILE_ROOT + "tooth_103x94x161_uint8.raw"
        self.dim = (103, 94, 161)
        self.dtype = "uint8"
        self.ftype = None 

class tooth(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "tooth_103x94x161_uint8.raw"
        self.dim = (103, 94, 161)
        self.spacing = (1.0, 1.0, 1.0)
        self.dtype = "uint8"
        self.ftype = "raw"

class foot(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "foot_256x256x256_uint8.raw"
        self.dim = (256,256,256)
        self.dtype = "uint8"
        self.ftype = "raw"

class carp(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "carp_256x256x512_uint16.raw"
        self.dim = (256,256,512)
        self.spacing = (0.78125,0.390625,1)
        self.dtype = "uint16"    
        self.ftype = "raw"

class bonsai(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "bonsai_256x256x256_uint8.raw"
        self.dim = (256,256,256)
        self.spacing = (1.0, 1.0, 1.0)
        self.dtype = "uint8"
        self.ftype = "raw"
    

class engine(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "engine_256x256x128_uint8.raw"
        self.dim = (256, 256, 128)  
        self.dtype = "uint8"
        self.spacing = (1.0, 1.0, 1.0)
        self.ftype = "raw"

class duct(DataFiles):    
    def __init__(self):
        self.path = DATA_FILE_ROOT + "duct_193x194x1000_float32.raw"
        self.dim = (193, 194, 1000)
        self.dtype = "float32"
        self.ftype = "raw"

class tacc(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "tacc_turbulence_256x256x256_float32.raw"
        self.dim = (256, 256, 256)
        self.dtype = "float32"
        self.ftype = "raw"

class tornado(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "tornado_128x128x128_float32.raw"
        self.dim = (128, 128, 128)
        self.dtype = "float32"
        self.ftype = "raw"
    
class velocity(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "velocity_800x300x100_float32.raw"
        self.dim = (800, 300, 100)
        self.dtype = "float32"
        self.ftype = "raw"

class velocity2(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "velocity.vtk"
        self.dim = (800, 300, 100)
        self.dtype = "float32"
        self.ftype = "vtk"

class manix(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "manix_128x128x115_uint8.raw"
        self.dim = (128, 128, 115)
        self.spacing = (0.49,0.49,0.7)
        self.dtype = "uint8"
        self.ftype = "raw"

class hcci(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "hcci_oh_560x560x560_float32.raw"
        self.dim = (560, 560, 560)
        self.spacing = (1.0, 1.0, 1.0)
        self.dtype = "float32"    
        self.ftype = "raw"  

class hydrogen(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "hydrogen_atom_128x128x128_uint8.raw"
        self.dim = (128, 128, 128)
        self.spacing = (1.0, 1.0, 1.0)
        self.dtype = "uint8"
        self.ftype = "raw"

class tooth2(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "tooth2_256x256x161_uint8.raw"
        self.dim = (256, 256, 161)
        self.dtype = "uint8"
        self.ftype = "raw"
        self.spacing = (1.0, 1.0, 1.0)

class threeSphere(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "three_spheres_c_128x128x128_float.raw"
        self.dim = (128, 128, 128)
        self.dtype = "float32"
        self.ftype = "raw"

class turbulent_combustion(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "turbulent_combustion_128x256x32_uint8.raw"
        self.dim = (128, 256, 32)
        self.dtype = "uint8"
        self.ftype = "raw"

class boston_teapot(DataFiles):
    def __init__(self):
        self.path = DATA_FILE_ROOT + "boston_teapot_256x256x178_uint8.raw"
        self.dim = (256, 256, 178)
        self.dtype = "uint8"
        self.ftype = "raw"
        self.spacing = (1.0, 1.0, 1.0)
