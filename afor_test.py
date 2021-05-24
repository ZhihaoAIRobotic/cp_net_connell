import image
import cv2
a=image.Image.from_file('pcd0100r.png')
a.rotate(0.5,[100,300])
cv2.imshow('a',a.img)
cv2.waitKey(0)

