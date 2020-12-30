import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt 


# ORB registration

def alignImages(im1, im2, MAX_FEATURES, GOOD_MATCH_PERCENT):

    MAX_FEATURES = MAX_FEATURES #default 500
    GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT #default 0.15
 

    print(im2.shape)
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    print("orb Created")
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    print("First detected")
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    print("Second detected")

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.png", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

 
if __name__ == '__main__':


    # FIXME: 這邊只能用level 3, 用level 2 "orb.detectAndCompute"會出現問題

    save_path = "/data/data/Pathology專案/公用data/trial_level_3/"

    PathologyFiles_l3 = glob.glob("/data/data/Pathology專案/公用data/CK18_level_3/*")
    PathologyFiles_l6 = glob.glob("/data/data/Pathology專案/公用data/CK18_level_6/*")

    pathology_dict = {}
    
    # 按照順序存取到pathology_dict, 同一張pathology用同一個key 對應到一個list
    # list包含 [level_6_CK18 , level_6_HE , level_2_CK18 , level_2_HE ]

    for ii in PathologyFiles_l6:
        key = ii.split("/")[-1].split(" ")[0]
        if key not in pathology_dict.keys():
            pathology_dict[key] = [ii]
        else:
            pathology_dict[key].append(ii)  


    for ii in PathologyFiles_l3:
        key = ii.split("/")[-1].split(" ")[0]
        if key not in pathology_dict.keys():
            pathology_dict[key] = [ii]
        else:
            pathology_dict[key].append(ii)


    ######################################################################################################
    
      # BOOKMARK: align the level 3 image through the h matrix got from level 6

    for key, pa in pathology_dict.items():
    
        # Read reference image
        refFilename = pa[1]
        print("\nFirst...Get the h matrix\n")
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        # Read image to be aligned
        aliFilename = pa[0]
        print("Reading image to align : ", aliFilename);  
        imAlign = cv2.imread(aliFilename, cv2.IMREAD_COLOR)

        print("Aligning images ...")
        # Registered image will be resotred in imReg. 
        # The estimated homography will be stored in h. 
        
        #TODO:Change the max feature
        MAX_FEATURES = 1000000000 
        GOOD_MATCH_PERCENT = 0.005
        imReg, h = alignImages(imAlign, imReference, MAX_FEATURES=MAX_FEATURES, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)
        height, width, channels = imReference.shape
        im1Reg = cv2.warpPerspective(imAlign, h, (width, height))


        # BOOKMARK: use level 6 to get the h matrix
      

        # Read reference image
        refFilename = pa[3]
        print("\nSecond...Get the aligned image\n")
        print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        # Read image to be aligned
        aliFilename = pa[2]
        print("Reading image to align : ", aliFilename);  
        imAlign = cv2.imread(aliFilename, cv2.IMREAD_COLOR)

        x, y, channels = imReference.shape

        imReference = cv2.resize(im1Reg, (y,x) )


        print("Aligning images ...")
        # Registered image will be resotred in imReg. 
        # The estimated homography will be stored in h. 
        
        #TODO:Change the max feature
        MAX_FEATURES=10000
        GOOD_MATCH_PERCENT = 0.01
        imReg, h = alignImages(imAlign, imReference, MAX_FEATURES=MAX_FEATURES, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)

        # Write aligned image to disk.
        os.chdir(save_path)
        pa_name = pa[2].split("/")[-1].replace(".png", "") 
        outFilename = pa_name + "_aligned.png"

        print("Saving aligned image : ", outFilename) 
        cv2.imwrite(outFilename, imReg)

        print("Done")

