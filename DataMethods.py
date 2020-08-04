import cv2 as cv
import numpy as np
from ImageMethods import *
from BallTrackingMethods import *
import matplotlib.pyplot as plt
import glob


# --------------------------
# METHODS FOR MODIFYING DATA
# --------------------------

# COLLECT DATA



class FNames:
    positive_folder = "ball_data/"
    negative_folder = "nonball_data/"

    ball_data_combined_base = "{}ball{}_data_{}.npy" # folder, pos/neg, type
    ball_data_for_place_base = "{}ball{}_data_{}_{}_{}.npy" # folder, pos/neg, place, num, type

    ball_pos_for_place_base = "ball_pos_data/ball_pos_data_{}_{}_full.csv"
    video_for_place_base = "videos/Tennis_{}_{}.mov"


def loadVidCap(vid_place, vid_num):
    vid_name = FNames.video_for_place_base.format(vid_place,vid_num)
    cap = cv.VideoCapture(vid_name)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return cap, frame_count

def loadBallCenters(vid_place, vid_num):
    csv_name = FNames.ball_pos_for_place_base.format(vid_place, vid_num)
    ball_centers = np.loadtxt(csv_name).astype(np.uint32)
    return ball_centers


def collectBallPosData(vid_place, vid_num):
    scale = 2.0
    
    cv.destroyAllWindows()
    vid_name = FNames.video_for_place_base.format(vid_place,vid_num)
    cap = cv.VideoCapture(vid_name)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # frame#, x, y,
    ball_loc_arr = np.array([[0,0,0]])

    SKIP_FRAMES = 0
    nextF = False
    cur_frame = 0

    ret, frame = cap.read()
    frame = resize(frame,1/scale)
    shape = frame.shape
    print(shape)
    
    def mouse_drawing(event, x, y, flags, params):
        nonlocal nextF
        nonlocal ball_loc_arr
        nonlocal shape
        if event == cv.EVENT_LBUTTONDOWN:
    #         print("Left click")
    #         print(cur_frame,x,y)
            y+=1


            circles.append((x, y))
            print(x,y)
            print(int(16/scale), int((shape[0] - 16)/scale))
            if (y in range(8, shape[0] - 8)) and (x in range(8, shape[1] - 8)):
                ball_loc_arr = np.concatenate((ball_loc_arr, np.array([[cur_frame,x*scale,y*scale]]))) 

        if event == cv.EVENT_RBUTTONDOWN:
    #         print("Right click")
            nextF = True
            print(ball_loc_arr)

    
    cv.namedWindow("Frame", flags=cv.WINDOW_NORMAL)
    cv.setMouseCallback("Frame", mouse_drawing)

    circles = []
    for i in range(SKIP_FRAMES):
        cap.read()
        cur_frame+=1

    while True:

        if nextF:
    #         print(nextF)
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize(frame,0.5)
            circles = []
            cur_frame+=1
            
            nextF = False

        for center_position in circles:
            cv.circle(frame, center_position, 2, (0, 0, 255), -1)

        cv.imshow("Frame", frame)
    #     cv.resizeWindow("Frame", 960, 540)

        key = cv.waitKey(1)
        if key == 27:
            break
        elif key & 0xFF == ord('q'):
            circles = []
    cap.release()
    cv.destroyAllWindows()
    
    return (FNames.ball_pos_for_place_base.format(vid_place, vid_num), ball_loc_arr[1:].astype(int))

    

def collectPositiveDataForVid(vid_place, vid_num, distort, num_samples=None, iterations=1, verbose=False):
    csv_name = FNames.ball_pos_for_place_base.format(vid_place, vid_num)
    vid_name = FNames.video_for_place_base.format(vid_place, vid_num)

    ball_centers = np.loadtxt(csv_name).astype(np.uint32)
    num_rows = ball_centers.shape[0]

    cap = cv.VideoCapture(vid_name)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    img_size = 32
    if num_samples == None:
        data_qnty = ball_centers.shape[0] * int(iterations)
    else:
        data_qnty = num_samples
    ball_data = np.zeros((data_qnty,img_size,img_size,3)).astype(np.uint8)

    for it in range(iterations):
        cap = cv.VideoCapture(vid_name)

        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            rows_idx = np.where(ball_centers[:, 0] == frame_idx)

            for row in rows_idx[0]:
                row_ = int(row + it * num_rows)
                print(row_)
                x = ball_centers[row, 1]
                y = ball_centers[row, 2]
                if distort:
                    ball_data[row_] = distortedCapture(frame, x, y, img_size)
                else:
                    if (y in range(16, 1080 - 16) and (x in range(16, 1920 - 16))):
                        ball_data[row] = frame[y - 16:y + 16, x - 16:x + 16]
                    
                if verbose:
                    plt.figure()
                    if distort:
                        plt.imshow(cv.cvtColor(ball_data[row_], cv.COLOR_BGR2RGB))
                    else:
                        plt.imshow(cv.cvtColor(ball_data[row], cv.COLOR_BGR2RGB))
                
                if row == ball_data.shape[0]-1 or row_ == ball_data.shape[0]-1:
                    return (FNames.ball_data_for_place_base.format(FNames.positive_folder,"+", vid_place, vid_num,"full"), ball_data)




def collectNegativeDataForVid(vid_place, vid_num, distort, data_qnty=300, verbose=False):
    csv_name = FNames.ball_pos_for_place_base.format(vid_place, vid_num)
    vid_name = FNames.video_for_place_base.format(vid_place, vid_num)

    ball_centers = np.loadtxt(csv_name).astype(np.uint32)
    cap = cv.VideoCapture(vid_name)

    img_size = 32
    ball_data = np.zeros((data_qnty, img_size, img_size, 3)).astype(np.uint8)

    data_count = 0

    for i in range(10):
        print("frame #", + i)
        ret, frame = cap.read()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        labels_good, labels, stats_good, centroids_good_size, frame_open = thresholdHSVBallClusters(frame)

        #     print(stats)

        cur_ball_centers = ball_centers[(ball_centers[:, 0] == i)]
        for ball_center in cur_ball_centers:
            cv.circle(frame_rgb, (int(ball_center[1]), int(ball_center[2])), 5, (0, 0, 255), 3)

        for j in range(centroids_good_size.shape[0]):
            x = int(centroids_good_size[j, 0])
            y = int(centroids_good_size[j, 1])
            
            if cur_ball_centers.size > 0:
                distances = cur_ball_centers[:, 1:] - centroids_good_size[j].astype(int)
                distances = distances * distances
                distances = distances[:, 0] + distances[:, 1]
                if (np.min(distances) < 400):
                    cv.circle(frame_rgb, (x, y), 10, (255, 0, 0), 3)
            else:

                if (data_count < data_qnty and (y in range(16, 1080 - 16) and (x in range(16, 1920 - 16)))):
                    if distort:
                        ball_data[data_count] = distortedCapture(frame_rgb, x, y, img_size)
                    else:
                        ball_data[data_count] = frame_rgb[y - 16:y + 16, x - 16:x + 16]
                    data_count += 1

        #             cv.circle(frame_rgb, (x, y), 10, (0,255,255), 3)

        if verbose:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(22, 8)
            # plt.imshow(frame_open,cmap = "gray")
            # plt.figure()
            # plt.imshow(frame_zoomedin)
            ax1.imshow(frame_rgb)
            ax2.imshow(frame_open, cmap="gray")
            plt.show()

        if (stats_good.shape[0] == 1):
            break
    return (FNames.ball_data_for_place_base.format(FNames.negative_folder,"-", vid_place, vid_num,"full"), ball_data)


def splitTrainingTestingData(case, split = 0.9):
    if "+" in case:
        folder = FNames.positive_folder
    else:
        folder = FNames.negative_folder
    
    data_files = glob.glob('{}/*full.npy'.format(folder))
    print("data_files: ", data_files)
    try:
        data_files.remove(FNames.ball_data_combined_base.format(folder, case, "full"))
    except:
        pass
    
    for file in data_files:
        ball_data = np.load(file)
        print("full ball data shape: ", ball_data.shape)
        split_int = int(ball_data.shape[0] * split)
        train = ball_data[:split_int]
        test = ball_data[split_int:]

        np.save(file[0:-8]+"train.npy", train)
        np.save(file[0:-8]+"test.npy", test)



def combineData(case, type_):
    # case should be: "+" or "-"
    # type_ should be one of:  'train', 'test', 'full'

    if "+" in case:
        folder = FNames.positive_folder
    else:
        folder = FNames.negative_folder

    data_files = glob.glob('{}/*{}.npy'.format(folder, type_))

    if (FNames.ball_data_combined_base.format(folder,case,type_) in data_files):
        data_files.remove(FNames.ball_data_combined_base.format(folder,case,type_))

    data = np.load(data_files[0])

    for i in range(1, len(data_files)):
        data2 = np.load(data_files[i])
        data = np.concatenate((data, data2))
    print(data.shape)
    return (FNames.ball_data_combined_base.format(folder,case,type_), data)




def valShiftDataHSV(data, idx_HSV, type_shift="max", amount=None,  verbose=False):
    # idx_HSV should be 0, 1, or 2 for the type of HSV value that needs transforming
    # data = np.load("ball_data/ball+_data_full.npy")
    data = data.copy()
    for i in range(int(data.shape[0])):
        hsv = cv.cvtColor(data[i, :, :, :], cv.COLOR_BGR2HSV)
        small_shifted = hsv.copy().astype(np.uint32)

        if amount==None:
            if type_shift == "max":
                max_v = np.max(hsv[:, :, idx_HSV])
                small_shifted[:, :, idx_HSV] = small_shifted[:, :, idx_HSV] + int(255 - max_v)
            if type_shift == "avg_raise":
                thresh = 150
                mean_v = np.mean(hsv[:, :, idx_HSV])
                if mean_v > thresh:
                    mean_v = thresh
                small_shifted[:, :, idx_HSV] = small_shifted[:, :, idx_HSV] + int(thresh - mean_v)
        else:
            small_shifted[:, :, idx_HSV] = small_shifted[:, :, idx_HSV] + amount
        
        small_shifted[(small_shifted[:, :, idx_HSV] > 255), idx_HSV] = 255
        small_shifted[(small_shifted[:, :, idx_HSV] < 0), 2] = 0

        rgb = cv.cvtColor(small_shifted.astype(np.uint8), cv.COLOR_HSV2RGB)
        bgr = cv.cvtColor(small_shifted.astype(np.uint8), cv.COLOR_HSV2BGR)
        data[i, :, :, :] = bgr

    if verbose:
        for i in range(int(data.shape[0]/10)):
            rgb = cv.cvtColor(data[i*10], cv.COLOR_HSV2RGB)
            plt.figure()
            plt.imshow(rgb)
    return data



def preprocessing(data):
    data2 = DataMethods.valShiftDataHSV(data, 2, type_shift="avg_raise")
    data3 = DataMethods.valShiftDataHSV(data2, 1, amount=50)
    return data3


def createYData(x_data=None, case="+"):
    col1 = np.ones((x_data.shape[0])).reshape(x_data.shape[0],1)
    col2 = np.zeros((x_data.shape[0])).reshape(x_data.shape[0],1)

    if case !="+":
        return np.concatenate((col2,col1), axis=1)
    
    return np.concatenate((col1,col2), axis=1)
