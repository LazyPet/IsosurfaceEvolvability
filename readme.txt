#######################################################################################################################################

# Topic

This code repository is for paper "Evolutionary Isosurface Analysis for Enhanced Transfer Function
Design in Direct Volume Rendering" in The Visual Computer

#######################################################################################################################################


#######################################################################################################################################

# Project Architecture

--algo-- : Algorithm Module, implements the main process
- algorithm01: Transmission function implementation based on frequency information and viewpoint selection
- algorithm02: Boundary isosurface detection and representative isovalue extraction based on similarity and evolutionary properties
- algorithm03: Volume rendering transmission function generation based on boundary isosurfaces and representative isovalues

--core-- : Core Module, primarily implements surface rendering, volume rendering, and loss function calculations
- core_calculater: Loss calculation module, implements various loss functions and clustering metrics
- surface_render: Surface rendering module, for rendering isosurfaces
- volume_render: Volume rendering module, for volume rendering

--dh-- : Data Processing Module, primarily implements data reading, preprocessing, augmentation, and segmentation
- data_files: Dataset encapsulation class, used for indexing data
- data_info: Dataset information encapsulation class, provides debugging information
- data_loader: Dataset loading encapsulation class, for loading datasets
- image_handle: For processing rendered isosurface images

--test-- : Testing Module, primarily implements unit tests, functional tests, and performance tests for the algorithm module
- main_test: Should run normally once the environment is properly configured
- experiment01: Parameter comparison experiment
- experiment01_result: Visualization results of running experiment01 at different resolutions
- experiment02: Performance experiment
- experiment03: Generalization experiment

--dataset-- : Datasets
- bonsai
- carp
- engine
- hcci_oh
- manix
- tooth

--img-- : Experimental Result Images
- AutoBoundaryPointDetection.png: Evolutionary boundary segmentation result
- AutoRepesentativePointDetection.png: Representative isovalue extraction result
- RegionSimilarity.png: Regional average similarity
- resolution_effect_on_boundary_value.png: Effect of resolution on evolutionary boundary segmentation
- totality_devision_infor_5.png: Frequency evolutionary segmentation result

--itf-- : Experimental Result Transfer Functions
- tf1: Basic transfer function control points
- tf2: Transfer function control points based on evolutionary properties
- tf3: Final transfer function control points

environment.yaml: Environment configuration file, to be imported using Anaconda

#######################################################################################################################################


#######################################################################################################################################

# Usage Instructions

1. Clone the repository to your local machine using Git
2. Use Anaconda to import the environment configuration
3. Open the project in VSCode and configure the Python-related plugins
4. Run test/main_test.py to check if the environment is set up correctly
5. Run the files in the test folder

#######################################################################################################################################


#######################################################################################################################################

# Important Functions

- algorithm01: AutoReginDetection(): Evolutionary segmentation based on frequency
- algorithm02: AutoBoundaryPointDetection(): Isosurface evolutionary boundary detection based on similarity
- algorithm02: AutoBoundaryPointAjustion(): Isosurface evolutionary boundary adjustment based on continuity
- algorithm03: CaculateTFPoints(): Generate final transfer function control points

- core_calculater: CSM(): Calculation of similarity and continuity metrics
- core_calculater: CaculateWassersteinDistanceDiscrete(): Continuity metric calculation based on Wasserstein distance
- core_calculater: CalculateSparseClusterLossOne(): Clustering loss calculation

#######################################################################################################################################
# Other Notes !!!
When the program reaches the viewpoint selection stage, please input "a, z, s, x, d, c" to adjust the viewpoint.
The carp and hcci_oh datasets are too large to upload to GitHub. Please download them from https://klacansky.com/open-scivis-datasets/


Thank you for reading.
