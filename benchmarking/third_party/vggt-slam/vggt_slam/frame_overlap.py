import argparse
import torch
import numpy as np
import cv2

# from core.raft import RAFT
# from core.utils import flow_viz
# from core.utils.utils import InputPadder

device = 'cuda'

def load_raft():
    print("Loading RAFT ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = "/home/<user>/RAFT/models/raft-sintel.pth")
    parser.add_argument('--path', default = "not_used")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args, _ = parser.parse_known_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()
    print("RAFT Loaded")
    return model

def get_raft_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def get_uniform_points(h,w, delta):
    uniform_points = []
    for r in range(h):
        if r%delta == 0:
            for c in range(w):
                if c%delta == 0:
                    uniform_points.append([c,r])
    return uniform_points

def get_sparse_flow(img, flo, origin_points):
    flo = np.transpose(flo, (1,2,0))
    h,w,_ = img.shape
    flow_coords = []
    p0 = []
    p1 = []
    for point in origin_points:
        if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
            point[0] = 0
            point[1] = 0

        p0.append([point[0], point[1]])
        p1.append([point[0]+(flo[point[1],point[0],0]), point[1]+(flo[point[1],point[0],1])])
        flow_coords.append(([point[1], point[0]], [point[1]+(flo[point[1],point[0],1]), point[0]+(flo[point[1],point[0],0])]))

    p0 = np.expand_dims(np.asarray(p0, dtype=np.float64), axis=0)
    p1 = np.expand_dims(np.asarray(p1, dtype=np.float64), axis=0)

    return p0, p1

def visualize_flow(img, flow_up, origin_points):
    p0, p1 = get_sparse_flow(img, flow_up, origin_points)
    img = np.copy(img)
    for i in range(p0.shape[1]):
        p0_x = p0[0,i,0]
        p0_y = p0[0,i,1]
        p1_x = p1[0,i,0]
        p1_y = p1[0,i,1]
        color = (0,0,0)
        cv2.arrowedLine(img, (int(p0_x), int(p0_y)), (int(p1_x), int(p1_y)), color=color, thickness=2, tipLength=1)
    cv2.imshow('image', img[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

class FrameTrackerRaft:
    def __init__(self):
        self.raft_model = load_raft()
        self.count = 0
        self.last_kf = None
    
    def get_optical_flow(self, image):
        """
        get optical flow of points from img1 to img2
        """
        image1 = get_raft_image(self.last_kf)
        image2 = get_raft_image(image)

        with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.raft_model(image1, image2, iters=20, test_mode=True)
        return flow_up[0,:,:].cpu().numpy()

    def compute_disparity(self, image, min_disparity, visualize=False):
        if self.last_kf is None:
            self.last_kf = image
            # First frame, no previous frame to compare to, return True so the first frame is a keyframe.
            return True
        flow = self.get_optical_flow(image)
        flow_magnitude = np.linalg.norm(flow, axis=0)

        mean_disparity = np.mean(flow_magnitude)

        if visualize:
            visualize_flow(image, flow, get_uniform_points(image.shape[0], image.shape[1], 15))

        if mean_disparity > min_disparity:
            self.last_kf = image
            # cv2.imshow("img",image)
            # cv2.waitKey()
            self.count += 1
            return True
        return False

class FrameTracker:
    def __init__(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None

    def initialize_keyframe(self, image):
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

    def compute_disparity(self, image, min_disparity, visualize=False):
        if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < 10:
            self.initialize_keyframe(image)
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_kf = self.kf_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_kf) < 10:
            self.initialize_keyframe(image)
            return True

        # Measure displacement from keyframe to current frame
        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disparity = np.mean(displacement)

        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_kf, good_next):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)

        if mean_disparity > min_disparity:
            self.initialize_keyframe(image)
            return True
        else:
            return False