import pandas as pd
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# Fill the image holes
def Fill_hole(img):
    
    img_copy = img.copy()

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  

    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  

    cv2.drawContours(img_copy,contours,-1,(255,255,255),-1)

    return img_copy


# Set the correct size of the buttom image and upper image , set the channel value of the upper image
def Set_channel_values(infer, origin, R, G, B):
    
    infer = infer / 255
    infer[infer>0.5] = 1
    infer[infer<=0.5] = 0

    (h, w, _) = infer.shape

    infer = cv2.resize(infer, (w//4, h//4))

    (h, w, _) = infer.shape

    origin = origin[0:h,0:w]

    print(origin.shape, infer.shape)


    infer[:,:,0][infer[:,:,0]==1] = B
    infer[:,:,1][infer[:,:,1]==1] = G
    infer[:,:,2][infer[:,:,2]==1] = R

    infer = np.array(infer, dtype=np.uint8)
    
    return infer, origin


###################################### the main function

def main():

    # Change the file dir here
    files = glob.glob("YOUR_FILEPATH")

    for file in files:
        
        print(file)
        
        # Get the image path
        origin_path = glob.glob(file+"YOUR_FILEPATH")[0]
        infer_path = glob.glob(file+"YOUR_FILEPATH")[0]
        
        # Get the image type you want
        origin = cv2.imread(origin_path)

        infer = cv2.imread(infer_path)

        infer_filled = Fill_hole(infer)
        
        # Set the RGB value here
        R, G, B = 125, 0, 0


        # Not filled
        infer_NF, origin_NF = Set_channel_values(infer, origin, R, G, B)

        notfilled_img = cv2.addWeighted(origin_NF, 0.8, infer_NF, 0.8,gamma=0)
        cv2.imwrite(file+"YOUR_FILENAME", notfilled_img)


        # filled
        infer_F, origin_F = Set_channel_values(infer_filled, origin, R, G, B)

        filled_img = cv2.addWeighted(origin_F, 0.8, infer_F, 0.8,gamma=0)
        cv2.imwrite(file+"YOUR_FILENAME", filled_img)

if __name__ == "__main__":
    
    main()
 
