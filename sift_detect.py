import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

detector = cv2.SIFT()
FLANN_INDEX_KDTREE=0
flannParam = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

trainImg = cv2.imread('./training_image/<image file for training>', 0)
trainKP,trainDecs = detector.detectAndCompute(trainImg, None)

cam = cv2.imread(args['image'])

QueryImgBGR = cam
QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
matches = flann.knnMatch(queryDesc,trainDecs,k=2)

goodMatch  = []
for m,n in matches:
    if(m.distance<0.75*n.distance):
        goodMatch.append(m)

if(len(goodMatch) > 2):
    print("match")
    tp = []
    qp = []
    for m in goodMatch:
        tp.append(trainKP[m.trainIdx].pt)
        qp.append(queryKP[m.queryIdx].pt)
    tp,qp= np.float32((tp,qp))
    H,status = cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
    h,w = trainImg.shape
    trainingBorder = np.float32([[[0,0], [0,h-1],[w-1,h-1],[0,w-1]]])
    queryBorder = cv2.perspectiveTransform(trainingBorder,H)
    queryBorder = np.uint8(queryBorder)
    print queryBorder
    print queryBorder.shape
    points = np.array(queryBorder[0][:])
    print points
    print points.shape
    cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (255,0,0))
    cv2.imwrite('result.jpg', QueryImgBGR)

else:
    print("no match")
