import numpy as np
import matplotlib.pyplot as matplotlib
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

dataSetPath = NuScenes(version='v1.0-mini', dataroot='D:\\v1.0-mini', verbose=False)

theScene = dataSetPath.scene[2]
theSample = dataSetPath.get('sample', theScene['first_sample_token'])

pointCloud = LidarPointCloud.from_file(dataSetPath.get_sample_data_path(theSample['data']['LIDAR_TOP']))

points = pointCloud.points[:3, :]

withinXRange = (points[0, :] >= -50.0) & (points[0, :] <= 50.0)
withinYRange = (points[1, :] >= -50.0) & (points[1, :] <= 50.0)

pointsInsideXYRange = points[:, withinXRange & withinYRange]

xCord = np.floor((pointsInsideXYRange[0, :] + 50.0) / 0.1).astype(int)
yCord = np.floor((pointsInsideXYRange[1, :] + 50.0) / 0.1).astype(int)

occupancyGrid = np.zeros((1000, 1000))
occupancyGrid[yCord, xCord] = 1.0

figure , axes = matplotlib.subplots(3, 3, figsize = (14, 11))

imageFrontLeft = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_FRONT_LEFT']))
axes[0, 0].imshow(imageFrontLeft)
axes[0, 0].set_title('CAM_FRONT_LEFT')
axes[0, 0].axis('off')

imageFront = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_FRONT']))
axes[0, 1].imshow(imageFront)
axes[0, 1].set_title('CAM_FRONT')
axes[0, 1].axis('off')

imageFrontRight = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_FRONT_RIGHT']))
axes[0, 2].imshow(imageFrontRight)
axes[0, 2].set_title('CAM_FRONT_RIGHT')
axes[0, 2].axis('off')

imageBackLeft = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_BACK_LEFT']))
axes[1, 0].imshow(imageBackLeft)
axes[1, 0].set_title('CAM_BACK_LEFT')
axes[1, 0].axis('off')

imageBack = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_BACK']))
axes[1, 1].imshow(imageBack)
axes[1, 1].set_title('CAM_BACK')
axes[1, 1].axis('off')

imageBackRight = matplotlib.imread(dataSetPath.get_sample_data_path(theSample['data']['CAM_BACK_RIGHT']))
axes[1, 2].imshow(imageBackRight)
axes[1, 2].set_title('CAM_BACK_RIGHT')
axes[1, 2].axis('off')

axes[2, 0].imshow(occupancyGrid, cmap = 'gray', origin = 'lower')
axes[2, 0].set_title('Lidar Scan')

axes[2, 0].axis('off')
axes[2, 1].axis('off')
axes[2, 2].axis('off')

matplotlib.show()
