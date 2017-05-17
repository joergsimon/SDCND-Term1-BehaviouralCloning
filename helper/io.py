import helper.constants as const
import cv2
import numpy as np

def load_img(csv_line, idx):
    source_path = csv_line[idx]
    filename = source_path.split('/')[-1]
    current_path = const.IMG_DATA_PATH + filename
    image = cv2.imread(current_path)
    return image

def add_image_and_flipped(csv_line, idx, measurment, images, angles):
    if const.DEBUG:
        print('add_image_and_flipped', csv_line, idx, measurment)
    image = load_img(csv_line, idx)
    images.append(image)
    angles.append(measurment)
    
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurment
    images.append(image_flipped)
    angles.append(measurement_flipped)
    if const.DEBUG:
        print('add_image_and_flipped', image.shape, measurment, image_flipped.shape, measurment, measurement_flipped)

# trans_image is taken almost 1x1 from
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    #tr_y = 40*np.random.uniform()-40/2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320,160))
    #print('new image shape', image_tr.shape)

    return image_tr,steer_ang

def make_trans_image(csv_line, idx, measurment, images, angles):
    image = load_img(csv_line, idx)
    image, measurment = trans_image(image, measurment, 100)
    images.append(image)
    angles.append(measurment)

def process_line(csv_line, images, angles, add_left_right, add_trans):
    source_path = csv_line[const.IDX_CENTER_IMG]
    if source_path == "center":
        return

    measurement = float(csv_line[const.IDX_STEER_ANGLE])

    add_image_and_flipped(csv_line, const.IDX_CENTER_IMG, 
                            measurement, images, angles)
    
    correct_to_right = measurement + const.STEER_CORRECTION_CONSTANT
    correct_to_left = measurement - const.STEER_CORRECTION_CONSTANT
    
    if add_left_right:
        add_image_and_flipped(csv_line, const.IDX_LEFT_IMG, 
                                correct_to_right, images, angles)
        
        add_image_and_flipped(csv_line, const.IDX_RIGHT_IMG, 
                                correct_to_left, images, angles)
    
    if add_trans:
        make_trans_image(csv_line, const.IDX_CENTER_IMG, 
                                measurement, images, angles)
        make_trans_image(csv_line, const.IDX_LEFT_IMG, 
                                correct_to_right, images, angles)
        make_trans_image(csv_line, const.IDX_RIGHT_IMG, 
                                correct_to_left, images, angles)