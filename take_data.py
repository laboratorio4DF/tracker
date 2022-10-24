# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:01:08 2021

@author: Azul
"""

from trackerclass import tracker_in_video
import matplotlib.pyplot as plt
import numpy as np
#%%
path = "/Users/azulbrigante/Documents/caminata/videos/facu_cande/"
file = "dia2_video2"
video = path + file + ".avi"
track = tracker_in_video(video)
fps = track.fps                  #frame rate
#%%
#TRACKER DURATION
initialTime = 8    #seconds
trackerDuration = 5 #seconds    

n0, l = int(fps*initialTime), int(fps*trackerDuration)
duration = [n0, n0+l]  #initial and final frame

#INITIAL IMAGE
#width  xy of observation area. The default observation area is [40, 40].
track.observationWidth = [45, 45]

#DEFINE INITIAL TEMPLATE (black) AND OBSERVATION AREA (red) WITH A SELECTION BOX.
track.initialConditionsSelecBox(n0) 

#TRACK WITH CORRELATION METHOD
timeSleep = 1 #ms
x_corr, y_corr = track.corr(timeSleep, duration)     

#TRAYECTORY'S GRAPH
plt.figure(1), plt.clf(), plt.grid(True)
plt.plot(x_corr, y_corr, ".-",color = "limegreen")
#%%
#SAVE DATA
n=1
np.savetxt("trayectory_%s.txt"%n, np.transpose([x_corr, y_corr]), header = "frames %s, center=%s, duration=%s, video=%s"%(len(x_corr), track.initialCenter, duration, file))


