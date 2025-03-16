active mean=cv2.adaptive
Threshold(image,255,cv2.ADAPTIVE THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
adaptive_gaussian=cv2 adaptive
Threshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
otsu_thresh=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)