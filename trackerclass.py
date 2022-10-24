# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:38:04 2021

@author: Azul
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max



#define a function who gives the nearlyest maximun 
def nearMax(maximuns, oldMax):
    a=[np.linalg.norm(maximuns[i] - oldMax) for i in range(len(maximuns))]
    i=np.argmin(a)
    return maximuns[i]  

def observationAreaEdges(edges, upper_left_b, bottom_right_b, templateWidth):
    obs_area_edges = [list(upper_left_b), list(bottom_right_b)]
    negative_edges_ind = np.where(np.concatenate(obs_area_edges) < 0)[0]
    for i in negative_edges_ind :
        if i in [0, 1]:
            obs_area_edges[0][i] = int(edges[0][i])
            obs_area_edges[1][i] = int(obs_area_edges[0][i] + 2*templateWidth[0])
            
        if i in [2, 3]:
            obs_area_edges[1][i-2] = int(edges[1][i-2])
            obs_area_edges[0][i-2] = int(obs_area_edges[1][i-2] - 2*templateWidth[1])       
    
    return tuple(obs_area_edges[0]), tuple(obs_area_edges[1])

class tracker_in_video:
    
    def __init__(self, path):
        #open the video, shows the first frame, choose a box as template, create a template
        self.video = cv.VideoCapture(path)
        if (int(self.video.get(7)))>0:
            print('The video has been opened')
            print("fps {}".format(self.video.get(5)), "frames {}".format(self.video.get(7)))
            
        else:
            print("video not found")
                   
        #frames per second(fps)
        self.fps = self.video.get(5)
        #width and hight
        self.w, self.h = self.video.get(3), self.video.get(4)
        
        #DEFAULT VALUES
        self.templateCenter = [self.w//2, self.h//2]
        self.initialCenter = [self.w//2, self.h//2]
        self.templatewidth = [15, 15]
        self.observationWidth = [40, 40]
        self.linewidth = int(np.max([self.w, self.h])//300)  
        self.tamplate = -1
        self.edges = [[0, 0 ], [self.w-1, self.h -1 ]]     
            
    #SELECT A BOX
    def setTemplate(self, n0):
        # set tempplate where tracker starts
        self.video.set(1,n0)
        ret, frame = self.video.read()
        if ret:
          frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
          bbox = cv.selectROI(frame)                             #expected points in the box
          cv.waitKey(1)                                          #wait time in milliseconds
          cv.destroyAllWindows()
          (c,f,w,h) = [int(a) for a in bbox]
          
          self.h, self.w, self.c, self.f = h, w, c, f                      
          self.templateCenter=[int((c+c+w)/2), int((f+f+h)/2)]            # x, y coordinate where template starts
          self.templateWidth =[ int(w/2), int(h/2)]                       # template width
        else:
          print('video not found')
          
        return 
    
    
    #TEMPLATE AND OBSERVATION REGION
    def initialConditions(self, boolean, n0):
        try:
            self.video.set(1, n0) #fix in frame n0                          
            ret, frame = self.video.read()       
            imag=frame.copy()
            
            #draw rectangle in template
            leftSTemp, rightSTemp = self.templateCenter[0]- self.templateWidth[0], self.templateCenter[0]+ self.templateWidth[0]
            downSTemp, upSTemp = self.templateCenter[1]- self.templateWidth[1], self.templateCenter[1]+ self.templateWidth[1]            
            cv.rectangle(imag, (leftSTemp, downSTemp), (rightSTemp, upSTemp), [0,0,0], 2) 
            self.template=cv.cvtColor(frame[downSTemp:upSTemp, leftSTemp:rightSTemp], cv.COLOR_BGR2GRAY)
            
            #draw rectangle in observation area
            leftSObs, rightSTemp = self.templateCenter[0] - self.observationWidth[0] , self.templateCenter[0] + self.observationWidth[0]
            downSTemp, upSTemp = self.templateCenter[1] - self.observationWidth[1], self.templateCenter[1] + self.observationWidth[1]  
            cv.rectangle(imag, (leftSObs,  downSTemp), (rightSTemp, upSTemp), [0,0,255], 1)
            self.observationRegion=cv.cvtColor(frame[ downSTemp:upSTemp, leftSObs:rightSTemp], cv.COLOR_BGR2GRAY)
            if boolean==True:
                cv.imshow("tracker", imag)            
            print("Template and observation area were defined")
            
            self.initialCenter = self.templateCenter
            
        except:
            print("Somthing is wrong. Check if template and observation area center an width fits in the frame.")
            #self.template, self.observationRegion=0, 0
                    
        return

    #DEFINED TEMPLATE AND OBSERVATION REGION WITH SELECTION BOX 
    def initialConditionsSelecBox(self, n0):       
        self.setTemplate(n0)
        boolean = False
        self.initialConditions(boolean, n0)
                               
        return

        
    #this function makes the tracker 
    def corr(self, timeSleep, duration):
        
        #METHOD TO MATCH TEMPLATE
        method = cv.TM_CCOEFF
        
        #cap=cv.VideoCapture(path)
        self.video.set(1, duration[0])
        
        #INITIAL CONDITIONS
        h_t, w_t = self.templateWidth[1], self.templateWidth[0]              #width and high of template     
        h_v, w_v = self.observationWidth[1], self.observationWidth[0]     #width and high of observation area
        d_h, d_w = h_v-h_t, w_v-w_t
        x,y = self.templateCenter[0], self.templateCenter[1]
        max_loc= [int(w_v/2), int(h_v/2)]
                
        n=0                   #starts a frame counter of not match 
        c=duration[0]                  #starts a frame counter                            
        
        x = self.templateCenter[0]
        y = self.templateCenter[1]
        
        x_vec = [x]
        y_vec = [y]
        
        #COPY OF OBSERVATION AREA
        A = self.observationRegion.copy()
         
        #UPPER LEFT AND BOTTOM RIGHT
        upper_left_b = (x-w_v, y-h_v)   
        bottom_right_b =  (upper_left_b[0]+2*w_v, upper_left_b[1]+2*h_v)      
        
        #CHECK IF TEMPLATE IS DEFINED 
        if  np.mean(self.template) > 0:
            print("Tracker in progress")
            boolean = True
        else:
            print('Template not defined')
            boolean = False

        while boolean:
            ret, frame = self.video.read()    #cap.read(1)  returns a bool (True/False)
            c=c+1

            if ret:
                if c<=duration[1]:
                    
                    img = frame.copy()                          #clon the frame
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  #gray image
                    
                    #MATCH TEMPLATE WITH CORRELATION
                    try:
                        result = cv.matchTemplate(A, self.template, method) #cross correlation method for matched images 
                        
                    except:
                        #if observation area goes out of limits
                        upper_left_b, bottom_right_b = observationAreaEdges(self.edges, upper_left_b, bottom_right_b, self.templateWidth)
                        obs= gray[upper_left_b[1]:bottom_right_b[1], upper_left_b[0]:bottom_right_b[0]]                                
                        A=obs.copy()                     
                        
                        result = cv.matchTemplate(A, self.template, method) #cross correlation method for matched images 
                    
                    
                    #TAKE CORRELATION MAXIMUNS
                    try:
                        maximuns=peak_local_max(result, min_distance=5, threshold_rel=0.8)
                        max_loc=nearMax(maximuns, max_loc)                    
                        
                    except:
                        n=n+1
                        print("frame %s was skiped"%c)
                        if n>5:
                            print("Too many frames didn't match")
                            print("Tracker stoped in frame %s."%c) #see the frame where tracker fails 
                            break   
                                            
                    #REDEFINE OBSERVATION AREA AND TEMPLATE
                    #template
                    upper_left = (max_loc[1]+x-w_v, max_loc[0]+y-h_v) 
                    bottom_right = (upper_left[0]+2*w_t, upper_left[1]+2*h_t)                      
                    #template = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]    #define a new template
                    
                    #observation area
                    upper_left_b=(upper_left[0]-d_w, upper_left[1]-d_h) 
                    bottom_right_b = (upper_left_b[0]+2*w_v, upper_left_b[1]+2*h_v)
                    
                    obs= gray[upper_left_b[1]:bottom_right_b[1], upper_left_b[0]:bottom_right_b[0]]                                
                    A=obs.copy()

                    
                    #X Y COORDINATES                
                    x, y =int(upper_left[0])+w_t, int(upper_left[1])+h_t
                    x_vec.append(x)   #x positiom 
                    y_vec.append(y)   #y position
                    
                    #GRAPH TEMPLATE AND OBSERVATION AREA
                    cv.rectangle(img, upper_left, bottom_right, [0,0,0], self.linewidth) 
                    cv.rectangle(img, upper_left_b, bottom_right_b, [0,0,255], self.linewidth)
                    
                    cv.imshow("tracker", img)

                    if cv.waitKey(timeSleep) == ord("q"): #DECIMAL VALUE of q is 113.
                        break                    
                else:
                    print("Tracker ended")
                    break
                
            else: 
                print("The video has ended")
                break
        
                                    
        return np.array(x_vec), np.array(y_vec)
                
                

        
       
        
        
        
        