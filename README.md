# Imageprocessing
http://localhost:8891/notebooks/New%20folder/programp1.ipynb<br>
1)<br>
import cv2<br>
img=cv2.imread('flw1.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
OUTPUT:<br>
![image](https://user-images.githubusercontent.com/97939356/173812703-83a9050e-d7ab-4586-b13b-d21bf6856040.png)<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mping.imread('leaf1.jpg')<br>
plt.imshow(img)<br>
![image](https://user-images.githubusercontent.com/97939356/173813368-136d4654-07a2-40c2-8f6f-d616996be0e9.png)
from PIL import Image
img=Image.open('bf1.jpg')
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97939356/173813810-4609a005-d140-43af-a3a6-defd18a78376.png)
from PIL import ImageColor
img1=ImageColor.getrgb("yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)
![image](https://user-images.githubusercontent.com/97939356/173814159-22c9431c-03ac-485b-b6f7-214a880d96d2.png)
from PIL import Image
img=Image.new('RGB',(200,400),(255,255,0))
img.show()
![image](https://user-images.githubusercontent.com/97939356/173814503-e00acde1-af74-4c26-89f6-bae047e2f63e.png)
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('bf3.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
![image](https://user-images.githubusercontent.com/97939356/173814697-38f7d778-75bd-4973-baff-14cdfbc69aea.png)

