import matplotlib.pyplot as plt
import numpy as np

robot_pos = np.load('robot_pos_beginning.npy',allow_pickle=True)
robot_pos = robot_pos.tolist()
print(len(robot_pos),len(robot_pos[1]))	#441*200*3?


plt.figure(figsize=(4,4))
plt.xlim(-2,2)
plt.ylim(-2,2)

pos_x = [] 
pos_y = []
#for j in range(len(robot_pos)):
for j in range(50):
    for i in range(len(robot_pos[j])):
        pos_x.append(robot_pos[j][i][0])
        pos_y.append(robot_pos[j][i][1])
#print(len(pos_x))
plt.plot(pos_x, pos_y)

r=0.3
center1=[0,0]	# unsafe
x1 = np.arange(center1[0]-r,center1[0]+r,0.0001)
y1 = np.sqrt(r**2-(x1-center1[0])**2)+center1[1]
x2 = np.arange(center1[0]-r,center1[0]+r,0.0001)
y2 = -1*np.sqrt(r**2-(x2-center1[0])**2)+center1[1]
plt.plot(x1,y1,x2,y2,color='r')

center2=[1.1,1.1]	# goal
x3 = np.arange(center2[0]-r,center2[0]+r,0.0001)
y3 = np.sqrt(r**2-(x3-center2[0])**2)+center2[1]
x4 = np.arange(center2[0]-r,center2[0]+r,0.0001)
y4 = -1*np.sqrt(r**2-(x4-center2[0])**2)+center2[1]
plt.plot(x3,y3,x4,y4,color='y')

plt.show()

