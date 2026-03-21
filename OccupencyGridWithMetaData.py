import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

dataSetPath = NuScenes(version='v1.0-mini', dataroot = 'D:\\v1.0-mini', verbose = False)

def allSamples(dataSetPath, sample_token):
    theSample = dataSetPath.get('sample', sample_token)
    
    pointCloud = LidarPointCloud.from_file(dataSetPath.get_sample_data_path(theSample['data']['LIDAR_TOP']))

    points = pointCloud.points[:3, :]

    leftRightRange = 50.0
    frontBackRange = 50.0

    withinXRange = (points[0, :] >= -leftRightRange) & (points[0, :] <= leftRightRange)
    withinYRange = (points[1, :] >= -frontBackRange) & (points[1, :] <= frontBackRange)

    pointsInsideXYRange = points[:, withinXRange & withinYRange]

    xMin, xMax = -leftRightRange, leftRightRange
    yMin, yMax = -frontBackRange, frontBackRange

    imageWidth, imageHeight = 1000, 1000

    xCord = np.floor((pointsInsideXYRange[0, :] - xMin) / 0.1).astype(int)
    yCord = np.floor((pointsInsideXYRange[1, :] - yMin) / 0.1).astype(int)

    occupancyGrid = np.zeros((imageHeight, imageWidth))
    occupancyGrid[yCord, xCord] = 1.0

    camera = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    for i in range(6):
        camType = camera[i]
        camToken = theSample['data'][camType]
        
        camData = dataSetPath.get('sample_data', camToken)
        calibratedSensor = dataSetPath.get('calibrated_sensor', camData['calibrated_sensor_token'])
        egoPose = dataSetPath.get('ego_pose', camData['ego_pose_token'])
        
        print(f"{camType}")
        print(f"Filename: {camData['filename']}")
        print(f"Camera Intrinsic: {np.array(calibratedSensor['camera_intrinsic'])}")
        print(f"Sensor Translation: {calibratedSensor['translation']}")
        print(f"Sensor Rotation: {calibratedSensor['rotation']}")
        print(f"Ego Pose Translation: {egoPose['translation']}")
        print(f"Ego Pose Rotation: {egoPose['rotation']}")
        print("\n")
        
    return occupancyGrid

for sample in dataSetPath.sample:
    grid = allSamples(dataSetPath, sample['token'])
