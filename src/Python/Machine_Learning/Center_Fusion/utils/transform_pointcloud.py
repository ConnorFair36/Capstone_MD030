import numpy as np
from .image import getAffineTransform, getGaussianRadius
from .ddd import get3dBox, project3DPoints


# taken from src/lib/dataset/generic_dataset.py and src/lib/dataset/datasets/nuscenes.py
class PointCloudProcessor:
    def __init__(self, config):
        self.config = config

    def processPointCloud(
        self, pc_2d, pc_3d, img, transMatInput, transMatOutput, img_info
    ):
        """
        Process the point cloud data

        Args:
            pc_2d: the 2D point cloud data [x, y, d]
            pc_3d: the 3D point cloud data
            img: the original image
            transMatInput: the transformation matrix from the original image to the input image
            transMatOutput: the transformation matrix from the input image to the output image
            img_info: the image info

        Returns:
            pc_2d: masked 2D point cloud data [x, y, d]
            pc_3d: masked 3D point cloud data
            depthMap: the depthmap of the point cloud [d, vel_x, vel_z]
        """
        # initialize the depth map
        outputWidth, outputHeight = self.config.MODEL.OUTPUT_SIZE[::-1]
        transformedPoints, mask = self.transformPointCloud(
            pc_2d, transMatOutput, outputWidth, outputHeight
        )
        isOneHot = self.config.DATASET.ONE_HOT_PC
        maxDistance = int(self.config.DATASET.MAX_PC_DIST)
        depthMap = self.getDepthMap(maxDistance, isOneHot)

        if mask is not None:
            pc_N = sum(mask)
            pc_2d = pc_2d[:, mask]
            pc_3d = pc_3d[:, mask]
        else:
            pc_N = pc_2d.shape[1]

        # generate point cloud channels
        if self.config.DATASET.PC_ROI_METHOD == "pillars":
            boxesInput2D, pillar_wh = self.getPcPillarsSize(
                img_info, pc_3d, transMatInput, transMatOutput
            )
            if self.config.DEBUG:
                self.debugPillar(
                    img,
                    pc_2d,
                    transMatInput,
                    transMatOutput,
                    boxesInput2D,
                    pillar_wh,
                )
        elif self.config.DATASET.PC_ROI_METHOD == "points":
            depthMap = self.drawPcPoints(
                depthMap,
                transformedPoints[:2],  # x, y
                transformedPoints[2],  # depth
                maxDistance,
                isOneHot,
                pc_3d,
            )
            return transformedPoints, pc_3d, depthMap

        for i in range(pc_N):
            point = transformedPoints[:, i]
            depth = point[2]
            center = point[:2]
            method = self.config.DATASET.PC_ROI_METHOD
            if method == "pillars":
                box = [
                    max(center[1] - pillar_wh[1, i], 0),  # y1
                    center[1],  # y2
                    max(center[0] - pillar_wh[0, i] / 2, 0),  # x1
                    min(center[0] + pillar_wh[0, i] / 2, outputWidth),  # x2
                ]

            elif method == "heatmap":
                radius = (1.0 / depth) * 250 + 5
                radius = getGaussianRadius((radius, radius))
                radius = max(0, int(radius))
                x, y = int(center[0]), int(center[1])
                left, right = min(x, radius), min(outputWidth - x, radius + 1)
                top, bottom = min(y, radius), min(outputHeight - y, radius + 1)
                box = [y - top, y + bottom, x - left, x + right]

            else:
                raise ValueError(f"Invalid PC_ROI_METHOD: {method}")

            box = np.round(box).astype(np.int32)
            depthMap = self.drawPcHeat(
                depthMap, box, depth, maxDistance, isOneHot, pc_3d[:, i]
            )

        return transformedPoints, pc_3d, depthMap

    def transformPointCloud(
        self, pc_2d, transformMat, img_width, img_height, filter_out=True
    ):
        """
        Transform 2D point cloud using transformation matrix

        Args:
            pc_2d: 2D point cloud # [x, y] (2, N) or [x, y, d] (3, N)
            transformMat: transformation matrix (2, 3)
            img_width(int): output image width
            img_height(int): output image height
            filter_out: filter out points outside image

        Returns:
            out: transformed points # [x, y] (2, N) or [x, y, d] (3, N)
            mask: filtered points
        """
        if pc_2d.shape[1] == 0:
            return pc_2d, []

        pc_t = np.expand_dims(pc_2d[:2, :].T, 0)  # [3,N] -> [1,N,2]
        transformedPoints = cv2.transform(pc_t, transformMat)
        transformedPoints = np.squeeze(transformedPoints, 0).T  # [1,N,2] -> [2,N]

        # remove points outside image
        if filter_out:
            mask = (
                (transformedPoints[0, :] < img_width)
                & (transformedPoints[1, :] < img_height)
                & (0 < transformedPoints[0, :])
                & (0 < transformedPoints[1, :])
            )
            out = np.concatenate((transformedPoints[:, mask], pc_2d[2:, mask]), axis=0)
        else:
            mask = None
            out = np.concatenate((transformedPoints, pc_2d[2:, :]), axis=0)

        return out, mask

    def getDepthMap(self, maxDistance: int, isOneHot: bool) -> np.ndarray:
        """
        This function will return the empty depth map of the point cloud

        Args:
            maxDistance(int): the maximum distance of the point cloud
        """
        depChannelSize = maxDistance * 3 if isOneHot else 3
        depthMap = np.zeros(
            (depChannelSize, *self.config.MODEL.OUTPUT_SIZE), np.float32
        )
        return depthMap

    def getPcPillarsSize(self, img_info, pc_3d, transMatInput, transMatOutput):
        """
        Get the size of the point cloud pillars for every point

        Args:
            img_info: image infomation
            pc_3d: 3D point cloud in camera coordinate [x, y, z] (>=3, N)
            transMatInput: transformation matrix from origin image to input size
            transMatOutput: transformation matrix from input to output size

        Returns:
            pillar_wh: width and height of the point cloud pillars [w, h] (2, N)
        """
        pillar_dims = self.config.DATASET.PILLAR_DIMS
        boxesInput2D = None  # for debug

        # for i, center in enumerate(pc_3d[:3, :].T):
        centers = pc_3d[:3, :].T
        B, K = 1, len(centers)
        pillar_dims = np.array(pillar_dims).reshape(1, 1, 3)
        pillar_dims = pillar_dims.repeat(K, 1)  # (B, K, 3)

        # Create a 3D pillar at pc location for the full-size image
        centers = np.array(centers).reshape(B, K, 3)  # (B, K, 3)
        boxOrigin3D = get3dBox(pillar_dims, centers, np.zeros((B, K)))  # (B, K, 8, 3)
        calib = np.array(img_info["calib"]).reshape(1, 1, 3, 4)
        calib = calib.repeat(K, 1)  # (B, K, 3, 4)
        boxOrigin2D = project3DPoints(boxOrigin3D, calib)  # (B, K, 8, 2)
        pointsOrigin2D = boxOrigin2D.reshape((-1, 2)).T  # (B, K, 8, 2) -> (2, B*K*8)

        # save the box for debug plots
        if self.config.DEBUG:
            pointsInput2D, _ = self.transformPointCloud(
                pointsOrigin2D,
                transMatInput,
                self.config.MODEL.INPUT_SIZE[1],
                self.config.MODEL.INPUT_SIZE[0],
                filter_out=False,
            )  # (2, B*K*8)
            boxesInput2D = pointsInput2D.T.reshape(
                (-1, 8, 2)
            )  # (2, B*K*8) -> (B * K, 8, 2)

        # transform points
        pointsOutput2D, _ = self.transformPointCloud(
            pointsOrigin2D,
            transMatOutput,
            self.config.MODEL.OUTPUT_SIZE[1],
            self.config.MODEL.OUTPUT_SIZE[0],
            filter_out=False,
        )  # (2, B*K*8)

        boxOutput2D = pointsOutput2D.T.reshape(
            (B, -1, 8, 2)
        )  # (2, B*K*8) -> (B, K, 8, 2)

        # get the bounding box in [xyxy] format
        bbox = np.stack(
            [
                np.min(boxOutput2D[:, :, :, 0], 2),
                np.min(boxOutput2D[:, :, :, 1], 2),
                np.max(boxOutput2D[:, :, :, 0], 2),
                np.max(boxOutput2D[:, :, :, 1], 2),
            ],
            axis=-1,
        )  # (B, K, 4)

        # store height and width of the 2D box
        # pillar_wh = np.zeros((2, pc_3d.shape[1]))
        pillar_wh = np.concatenate(
            [bbox[:, :, 2] - bbox[:, :, 0], bbox[:, :, 3] - bbox[:, :, 1]]
        )

        return boxesInput2D, pillar_wh
        ...

    def drawPcHeat(self, depthMap, box, depth, maxDist, isOneHot, pc_3d, *_):
        """
        This function will draw the heat value of the point cloud on depth map
        Add depth, x velocity, and z velocity to depth map

        Args:
            depthMap(np.ndarray): the depth map of the point
            box(np.ndarray): the bounding box of the object (y1, y2, x1, x2)
            depth(float): the depth value of the point
            maxDist(int): the maximum distance of the point cloud
            isOneHot(bool): whether the point cloud is one hot
            pc_3d(np.ndarray): the 3D point cloud data in camera coordinate

        Returns:
            depthMap(np.ndarray): the depth map
        """
        if isOneHot:
            depthLayer = int(depth)
            xVelLayer = depthLayer + maxDist
            zVelLayer = depthLayer + maxDist * 2

            depthMap[depthLayer, box[0] : box[1], box[2] : box[3]] = depth
            depthMap[xVelLayer, box[0] : box[1], box[2] : box[3]] = pc_3d[8]
            depthMap[zVelLayer, box[0] : box[1], box[2] : box[3]] = pc_3d[9]
        else:
            depthMap[0, box[0] : box[1], box[2] : box[3]] = depth
            depthMap[-2, box[0] : box[1], box[2] : box[3]] = pc_3d[8]
            depthMap[-1, box[0] : box[1], box[2] : box[3]] = pc_3d[9]

        return depthMap