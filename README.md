This repository contains two parts:
- First Part: Hough Detector low-level implementation(no libraries), with some slight modifications to speed up the computation time.
- Second Part: Contains multiple algorithms mainly: 
    - Harris Detector for corner detection.
    - Application of the Harris Detector on two pictures of the same scene, and the pairing of the corners previously determined using ZMSSD (Zero-mean sum of squared distances) method.
    - Application of RANSAC to find the Homography Matrix of the two pictures, and the creation of a panoramic picture containing two pictures.
    
  
