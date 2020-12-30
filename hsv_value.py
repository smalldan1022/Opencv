import cv2
import numpy as np
import matplotlib.pyplot as plt

def Eliminate_white(img_path):

    rgb_image = cv2.imread(img_path)

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV) 
    lower_white = np.array([180,7,160]) 
    upper_white = np.array([222,20,255]) 
    mask = cv2.inRange(hsv, lower_white, upper_white) 

    mask = np.expand_dims(mask, axis=2)

    mask = np.concatenate((mask,mask,mask), axis=2)

    return hsv, mask


def main():

    hsv_CK18, CK18_mask = Eliminate_white("/data/data/Pathology專案/2_style_transfer/Method_3_segmentation/U-2-Net_CK18/S12-062035 R1 Re4CK18 _07.32.14.ndpi_aligned.png/17/453.png")
    hsv_HE, HE_mask = Eliminate_white("/data/data/Pathology專案/2_style_transfer/Method_3_segmentation/U-2-Net_CK18/S12-062035 R1 Re4HE _15.56.30.ndpi.png/17/453.png")

    plt.figure(0)
    plt.subplot(221)
    plt.title("hsv_CK18")
    plt.imshow(hsv_CK18[:,:,::-1])
    plt.subplot(222)
    plt.title("CK18_mask")
    plt.imshow(CK18_mask[:,:,::-1])

    plt.subplot(223)
    plt.title("hsv_HE")
    plt.imshow(hsv_HE[:,:,::-1])
    plt.subplot(224)
    plt.title("HE_mask")
    plt.imshow(HE_mask[:,:,::-1])
    plt.show()

if __name__ == "__main__":

    main()