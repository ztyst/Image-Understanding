
# coding: utf-8

# In[62]:


import cv2
import numpy as np
import sys
import pickle
import math


# In[44]:


def readCalib(path):
    file = open(path,"r")
    line = file.readlines()
    f = float(line[0].strip("\n").split(":")[1])
    baseline = float(line[3].strip("\n").split(":")[1])
    px= float(line[1].strip("\n").split(":")[1])
    py = float(line[2].strip("\n").split(":")[1])
    return f, baseline, px, py


# In[45]:


def getDepth(path,f,baseline):
    img = cv2.imread(path)
    Z = f*baseline/img
    return Z
    


# In[138]:


f,baseline,px,py = readCalib("data/data/test/calib/004945_allcalib.txt")
Z = getDepth("data/data/test/results/004945_left_disparity.png",f,baseline)
cv2.imwrite("result/4945depth.png",Z)

f2,baseline2,px2,py2 = readCalib("data/data/test/calib/004964_allcalib.txt")
Z2 = getDepth("data/data/test/results/004964_left_disparity.png",f2,baseline2)
cv2.imwrite("result/4964depth.png",Z2)

f3,baseline3,px3,py3 = readCalib("data/data/test/calib/005002_allcalib.txt")
Z3 = getDepth("data/data/test/results/005002_left_disparity.png",f3,baseline3)
cv2.imwrite("result/5002depth.png",Z3)


# In[169]:


def parseData(input_path,output_path):    
    obj = pickle.load(open(input_path, "rb"))
    filtered_obj = {}
    score_list = []
    box_list = []
    class_list = []
    num_detection = 0
    for i in range(len(obj["detection_scores"])):
        if obj["detection_scores"][i] > 0.5:
            score_list.append(obj["detection_scores"][i])
            box_list.append(obj["detection_boxes"][i])
            class_list.append(obj["detection_classes"][i])
            num_detection += 1
    filtered_obj["num_detection"] = num_detection
    filtered_obj["detection_scores"] = score_list
    filtered_obj["detection_boxes"] = box_list
    filtered_obj["detection_classes"] = class_list
    f = open(output_path,"wb")
    pickle.dump(filtered_obj,f)
    f.close()
    return filtered_obj


# In[113]:


def visualize(obj,input_img_path):
    img = cv2.imread(input_img_path)
    w,h = img.shape[:2]
    for i in range(int(obj["num_detection"])):
        xmin = int(obj["detection_boxes"][i][1]*h)
        xmax = int(obj["detection_boxes"][i][3]*h)
        ymin = int(obj["detection_boxes"][i][0]*w)
        ymax = int(obj["detection_boxes"][i][2]*w)
        if int(obj["detection_classes"][i]) == 1:
            color = (255,0,0)
            label = "person"
        elif int(obj["detection_classes"][i]) == 2:
            color = (0,255,0)
            label = "bike"
        elif int(obj["detection_classes"][i]) == 3:
            color = (0,0,255)
            label = "car"
        elif int(obj["detection_classes"][i]) == 10:
            color = (255,255,0)   
            label = "traffic_light"
        else:
            continue
        
        cv2.line(img,(xmin,ymin),(xmax,ymin),color,3)
        cv2.line(img,(xmin,ymin),(xmin,ymax),color,3)
        cv2.line(img,(xmax,ymin),(xmax,ymax),color,3)
        cv2.line(img,(xmin,ymax),(xmax,ymax),color,3)
        cv2.putText(img,label,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
    return img  


# In[161]:


filtered_obj = parseData("result.pkl","./result/parsedResult004964.pkl")
output_img = visualize(filtered_obj,"./data/data/test/left/004964.jpg")
cv2.imwrite("./result/output004964.jpg",output_img)

filtered_obj2 = parseData("4945.pkl","./result/parsedResult004945.pkl")
output_img2 = visualize(filtered_obj2,"./data/data/test/left/004945.jpg")
cv2.imwrite("./result/output004945.jpg",output_img2)

filtered_obj3 = parseData("5002.pkl","./result/parsedResult005002.pkl")
output_img3 = visualize(filtered_obj3,"./data/data/test/left/005002.jpg")
cv2.imwrite("./result/output005002.jpg",output_img3)



# In[165]:


def segmentation(path,obj,Z,px,py,f):
    img = cv2.imread(path)
    w,h = img.shape[:2]
    segmentation = np.zeros(img.shape)
    for i in range(int(obj["num_detection"])):
        top_left = (int(obj["detection_boxes"][i][1]*h),int(obj["detection_boxes"][i][0]*w))
        bottom_right = (int(obj["detection_boxes"][i][3]*h),int(obj["detection_boxes"][i][2]*w))
        
        center = obj["center_of_mass"][i]
        for m in range(top_left[0],bottom_right[0]+1):
            for n in range(top_left[1],bottom_right[1]+1):
                
                subZ = Z[n,m,0]
                X = (m-px)*subZ/f
                Y = (n-py)*subZ/f
                dist = math.sqrt((center[0]-X)**2+(center[1]-Y)**2+(center[2]-subZ)**2)
                if dist < 3.0:
                    segmentation[n,m] = (255,0,0)

    return segmentation


# In[136]:


def compute3D(path,obj,Z,px,py,f):
    img = cv2.imread(path)
    w,h = img.shape[:2]
    location_list = []
    for i in range(int(obj["num_detection"])):
        top_left = (int(obj["detection_boxes"][i][1]*h),int(obj["detection_boxes"][i][0]*w))
        bottom_right = (int(obj["detection_boxes"][i][3]*h),int(obj["detection_boxes"][i][2]*w))
        center = ((top_left[0]+bottom_right[0])/2,(top_left[1]+bottom_right[1])/2)
        subZ = Z[int(center[1]),int(center[0]),0]
        X = (center[0]-px)*subZ/f
        Y = (center[1]-py)*subZ/f
        location = (X,Y,subZ)
        print(location)
        location_list.append(location)
    obj["center_of_mass"] =location_list
    return obj


# In[160]:


def alarm(path,obj,Z,px,py,f):
    img = cv2.imread(path)
    w,h = img.shape[:2]
    dist_list = []
    for i in range(int(obj["num_detection"])):
        top_left = (int(obj["detection_boxes"][i][1]*h),int(obj["detection_boxes"][i][0]*w))
        bottom_right = (int(obj["detection_boxes"][i][3]*h),int(obj["detection_boxes"][i][2]*w))
        segmentation = np.zeros(img.shape)
        center = obj["center_of_mass"][i]
        dist = math.sqrt(center[0]**2+center[1]**2+center[2]**2)
        dist_list.append((dist,i))
        
    dist_list.sort(key=lambda x:x[0])
    for j in range(len(dist_list)):
        idx = dist_list[j][1]
        label_int = int(obj["detection_classes"][idx])
        if label_int == 1:
            color = (255,0,0)
            label = "person"
        elif label_int == 2:
            color = (0,255,0)
            label = "bike"
        elif label_int == 3:
            color = (0,0,255)
            label = "car"
        elif label_int == 10:
            color = (255,255,0)   
            label = "traffic_light"
        else:
            continue
            
        X = obj["center_of_mass"][idx][0]
        if X >= 0:
            txt = "right"
        elif X < 0:
            txt = "left"
        print("There is a "+ str(label) + " "+ str(X) + " meters to your "+ txt + "\n")
        print("It is " + str(dist_list[j][0]) + " meters away from you \n")


# In[166]:


f,baseline,px,py = readCalib("data/data/test/calib/004964_allcalib.txt")
Z = getDepth("data/data/test/results/004964_left_disparity.png",f,baseline)
filtered_obj = parseData("result.pkl","./result/parsedResult004964.pkl")

new_obj = compute3D("./data/data/test/left/004964.jpg",filtered_obj,Z,px,py,f)
seg = segmentation("./data/data/test/left/004964.jpg",new_obj,Z,px,py,f)
cv2.imwrite('./result/seg004964.jpg',seg)

alarm("./data/data/test/left/004964.jpg",new_obj,Z,px,py,f)


# In[167]:


f,baseline,px,py = readCalib("data/data/test/calib/004945_allcalib.txt")
Z = getDepth("data/data/test/results/004945_left_disparity.png",f,baseline)
filtered_obj = parseData("4945.pkl","./result/parsedResult004945.pkl")

new_obj = compute3D("./data/data/test/left/004945.jpg",filtered_obj,Z,px,py,f)
seg = segmentation("./data/data/test/left/004945.jpg",new_obj,Z,px,py,f)
cv2.imwrite('./result/seg004945.jpg',seg)

alarm("./data/data/test/left/004945.jpg",new_obj,Z,px,py,f)


# In[168]:


f,baseline,px,py = readCalib("data/data/test/calib/005002_allcalib.txt")
Z = getDepth("data/data/test/results/005002_left_disparity.png",f,baseline)
filtered_obj = parseData("5002.pkl","./result/parsedResult005002.pkl")

new_obj = compute3D("./data/data/test/left/005002.jpg",filtered_obj,Z,px,py,f)
seg = segmentation("./data/data/test/left/005002.jpg",new_obj,Z,px,py,f)
cv2.imwrite('./result/seg005002.jpg',seg)

alarm("./data/data/test/left/005002.jpg",new_obj,Z,px,py,f)

