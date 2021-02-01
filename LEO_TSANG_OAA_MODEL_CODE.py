
####This is all written by Leo Tsang
#Email: Leotsang@hotmail.ca

#import all the packages
import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#######################################USER DEFINED FUNCTION#######################################

#function to determine where the closest field_putout would be, which is why our SS decided to go for putout play
#plug-in coordinates, it determines the distance of 2 closest predefined points for our SS to tag for an out

def closest_tag(x): 
    
    min_dist = []
    possible_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    for i in possible_options:
        min_dist.append(np.linalg.norm(x-field_coord[i]))
    
    min_dist = sorted(min_dist)
    
    return sum(min_dist[0:2]) / 2
    
#this function will calculate the probability between the hitter vs. our SS

def fieldvsbatt(x1,x2):
    
    u = x1-x2
    
    return norm.cdf(u/ (0.25*math.sqrt(2)))


#this function returns the potential bounce velocity of the ball

def bounce_velo(ev):
    ev_ft = 1.46667 * ev
    V = ev_ft * 0.546
    return V


def bounce_distance(ev):
    
    ev_ft = 1.46667 * ev
    deg = 30 * math.pi / 180
    V = ev_ft * 0.546 #coefficient of restititution of a baseball
    Vy = ev_ft * math.sin(deg)
    Vx = ev_ft * math.cos(deg)
    
    return (2 * Vx * Vy) / 32.1741 #returns the measurement in ft
    
#this function will be used to calculate the new coordinate of our player based off the hang_time

def new_coord(a,b,d2): #include points a and b as a np.array((x,y)), a & b are two points

    d = np.linalg.norm(a-b)
    c_x = a[0] - (d2*(a[0]-b[1]))/d
    c_y = a[1] - (d2*(a[1]-b[1]))/d

    return np.array((c_x,c_y))


#######################################END OF USER DEFINED FUNCTION#######################################

#######################################OAA_MODEL BEGINS#######################################

#######################################LOADING DATA / CLEANING DATA/ DATA SET UP#######################################
#read our dataset into pandas dataframe
df = pd.read_excel('shortstopdefense.xlsx')

#data cleaning
#check for NA or null Values

df.isnull().sum().sum()

#there is let's drop them. Can't find anything if there's no data.
df = df.dropna()
df = df.reset_index()

#look only at the ones fielded by the SS, since we're only looking at SS leaderboard
df = df[df['fielded_pos']==6]
df = df.reset_index()


#combine the coordinates to a column itself, so we can use numpy's distance formula for faster calculation

df['ball_land_coord']=list(zip(df['landing_location_x'], df['landing_location_y']))
df['player_coord']=list(zip(df['player_x'], df['player_y']))


#Set the bases, and plates in coordinates, 1-Pithcer, 2-Home, 3-1st base, 4-2nd base, 5- 3rd base, 6-SS
#this will come handy when we start calculating the distance and opportunity time
#since there's not one exact spot of SS coordiantes, we'll use the average of where they usually stand (ie. mean)
#we're going to also include midpoints to consider when the SS can tag the person out, or run to the nearby base whichever
#is closer would be used

second_base_coord = 127.28125 #127 3,3/8 inches
pos_6 = [np.average(df['player_x']), np.average(df['player_y'])] #take the average position of where most SS usually stands
field_coord = {1: [0,60.5], 2: [0,0], 2.5: [second_base_coord/4, second_base_coord/4],
               3:[second_base_coord/2, second_base_coord/2], 
               3.5: [(0+second_base_coord/2)/2, (second_base_coord+second_base_coord/2)/2], 
               4: [0, second_base_coord], 4.5: [(0-second_base_coord/2)/2, (second_base_coord+second_base_coord/2)/2],
               5: [-second_base_coord/2, second_base_coord/2], 5.5 : [-second_base_coord/4, second_base_coord/4],
               6: pos_6 }


#we need to dissect fielding_play
#this will be used to calculate the receiving end of our shortstop throw if he has any
df['fieldingplay'] = df['fieldingplay'].astype(int)
df['fieldingplay'] = df['fieldingplay'].astype(str)

#create a list of the plays segregated by the integer, this way we can see where the SS is throwing the ball to
list_test = []
for i in df['fieldingplay']:
    list_test.append([int(d) for d in i])
    
df['field_play'] = list_test


#the receiving end of SS fielded ball
pos_passed = []
for i in range(len(df)):
    if len(df['field_play'][i]) == 1:
        pos_passed.append(int(df['fielded_pos'][i]))
    else:
        pos_passed.append(int(df['field_play'][i][1]))

df['ball_passed'] = pos_passed

#######################################TRAJECTORY OF OUR BALL/ NEW PLAYER COORDINATE#######################################

#calculate distance from player and first bounce
df['player_ball_first_landed'] =np.sqrt((df['player_x'] - df['landing_location_x'])**2 + (df['player_y'] - df['landing_location_y'])**2) 

#bounce distance uses exit velocity and EOR to estimate the distance it can travel
bounce_dist = []

for i in range(len(df)):
    bounce_dist.append(bounce_distance(df['launch_speed'][i])) #here we recall the function that we created earlier
    
df['bounce_distance'] = bounce_dist

#incorporate hang-time to better determine intercept point
distance_between_hang = []

#the player travelled how much distance
df['hang_time_travel'] = df['hang_time'] * 27

player_new_coord = []
player_catch = [] #determine if the player has caught the ball
player_new_distance = []

#find the new coordinates, 
for i in range(len(df)):
    a = np.array(df['player_coord'][i])
    b = np.array(df['ball_land_coord'][i])
    c = new_coord(a,b,df['hang_time_travel'][i]) #the potential coordinate of where the fielder ran to
    
    if df['hang_time_travel'][i] >= df['player_ball_first_landed'][i] : #player had enough time to cover the distance before ball landed
        player_new_coord.append(b) #therefore he caught the ball, and this would be used as his intercept_point
        player_catch.append(True)
        player_new_distance.append(0) #therefore his distance is 0 from the ball
    else:
        player_new_coord.append(c)
        player_catch.append(False)
        player_new_distance.append(np.linalg.norm(c-b))
        
df['player_new_coord'] = player_new_coord
df['player_catch'] = player_catch #boolean that tells us if the ball was caught, therefore intercept-time = hang_time
df['player_new_distance'] = player_new_distance

#######################################INTERCEPT MODEL#######################################

#intercept point, if the bounce distance is further than our current SS, we assume the midway point is our intercept coordinate
#finding our intercept coordinate

intercept_coord = []
#player_x_intercept = []
#player_y_intercept = []
df['player_x_intercept'] = bounce_dist
df['player_y_intercept'] = bounce_dist
for i in range(len(df)):
    
    a = np.array(df['player_new_coord'][i])
    b = np.array(df['ball_land_coord'][i])
    
    if df['player_catch'][i] == True: #if it catch, then intercept point is just where the ball landed
        intercept_coord.append(df['ball_land_coord'][i])
    
    elif df['bounce_distance'][i] >= df['player_new_distance'][i]: #if ball bounces further than where we're at,
        #we'll take the midpoint as the intercept
        df['player_x_intercept'][i] = (df['player_new_coord'][i][0]+df['landing_location_x'][i])/ 2
        df['player_y_intercept'][i] = (df['player_new_coord'][i][1]+df['landing_location_y'][i])/ 2
        intercept_coord.append((df['player_x_intercept'][i],df['player_y_intercept'][i]))
        
    else:
        #here we need assume the bounce shall continue, and add it the inital landing spot 
        intercept_coord.append(new_coord(a,b,df['bounce_distance'][i]))
        
#our new intercept coord
df['intercept_coord'] = intercept_coord

#Intercept Model
#we are trying to find the distance between our fielder to intercept, and determine the time
#our fielder to reach there, given the hang_time

#find opportunity distance
opportunity_dist = []
intercept_home = []
for i in range(len(df)):
    
    a = np.array(df['player_new_coord'][i])
    b = np.array(df['intercept_coord'][i])
    
    opportunity_dist.append(np.linalg.norm(a-b)) #where player meets the intercept point
                            
df['opportunity_distance'] = opportunity_dist
                            
#######################################DISTANCE TIME MODEL#######################################

df['opportunity_time'] = df['hang_time'] + df['opportunity_distance']/ 27

#here we are calculating the distance of the ball to be thrown, or putout
#and the time it requires.
#assume: batter run-speed is 27 fps, throwing speed of our SS on avg is 132 fps
#you need 3/4 of a second to throw the ball

#SS time remaining, time remaining before batter reaches base
play_distance = []
play_time = []

for i in range(len(df)):
    #if df['ball_passed'][i] > 6:
    ##default play is to first base, for an forceplay
    #play_distance.append(np.linalg.norm(np.array(df['ball_land_coord'][i])-field_coord[3]))
    a = np.array(df['intercept_coord'][i])
    
    if df['ball_passed'][i] < 6 and df['ball_passed'][i] != 3:
        play_distance.append(np.linalg.norm(np.array(df['ball_land_coord'][i])-field_coord[df['ball_passed'][i]])) 
        play_time.append(play_distance[i]/ 132 + 0.75)
    
    elif df['ball_passed'][i] == 6 and df['fielded_scoring'][i] == 'f_putout':
        #distance = closest_tag(df['ball_land_coord'][i])
        play_distance.append(closest_tag(a))
        play_time.append(play_distance[i]/27)
    
    else:
        ##default play is to first base, for a forceplay
        play_distance.append(np.linalg.norm(np.array(df['ball_land_coord'][i])-field_coord[3])) 
        play_time.append(play_distance[i]/132 +0.75)
        
df['play_distance'] = play_distance
df['play_time'] = play_time


##Runner time remaining
df['runner_time'] = 90/ 27 - df['opportunity_time']

#######################################OAA CALCULATION#######################################

#using our predefined function, we can compare the distributions

OAA_odds = []

for i in range(len(df)):
    
    if df['fielded_scoring'][i] == 'f_putout' or df['fielded_scoring'][i] == 'f_assist':
        OAA_odds.append(fieldvsbatt(df['runner_time'][i],df['play_time'][i] * 1 * 0.975))
    else:
        OAA_odds.append((1-(fieldvsbatt(df['runner_time'][i],df['play_time'][i] * 1 * 0.975)))*-1)

df['OAA'] = OAA_odds

#let's get the aggregate of our leaderboard
df['opportunity'] = 1 # we assume if SS fielded the ball he is responsible for the way

df_pivot = df.pivot_table(index='playerid', values=['opportunity', 'OAA'], aggfunc='sum')
#sort our value
result = df_pivot.sort_values(('OAA'), ascending=False).round()

#export our results
result.to_csv('output.csv')
df.to_excel('out_data.xlsx')

#########################END OF OUR OAA MODEL ############################################


#######################################VISUALIZATION#######################################

#find the bunt plays in our data
bunt = df[df['is_bunt']==True]

int_x = []
int_y = []

for i in range(len(df)):
    int_x.append(df['intercept_coord'][i][0])
    int_y.append(df['intercept_coord'][i][1])

#our base coordinates by x, y    
base_x = []
base_y = []

for i in [2,3,4,5,1]:
    base_x.append(field_coord[i][0])
    base_y.append(field_coord[i][1])

#let's see the the play whether it was an out or not
res = []
for i in range(len(df)):
    if df['fielded_scoring'][i] == 'f_putout' or df['fielded_scoring'][i] == 'f_assist':
        res.append(1) #1 means it warranted an out
    else:
        res.append(0) #it was not an out


###just a heatmap and scatter plot of where our SS catches were


fig, ax = plt.subplots()
fig_dims = (500, 500)
sns.kdeplot(df['player_x'],df['player_y'], shade=True, shade_lowest=False, alpha=0.9, cbar=False, ax=ax, cmap="YlOrBr")
#sns.kdeplot(int_x,int_y, shade=True, shade_lowest=False, alpha=0.5, cbar=False, ax=ax, cmap="Oranges")
#sns.kdeplot(df['landing_location_x'],df['landing_location_y'], shade=True, shade_lowest=False, alpha=0.5, cbar=False, ax=ax, cmap="mako")
ax.scatter(bunt['landing_location_x'],bunt['landing_location_y'], color="red", s=5)
ax.scatter(df[df['filter']==1]['int_x'],df[df['filter']==1]['int_y'],color="g",marker = 'o', s=5)
ax.scatter(df[df['filter']==0]['int_x'],df[df['filter']==0]['int_y'],color="b",marker='x', s=5)
#ax.scatter(df['landing_location_x'],df['landing_location_y'], color="blue", s=5)
ax.scatter(base_x,base_y, color="black", s=40, marker='D')
plt.legend()
plt.xlabel('x coordinates')
plt.ylabel('y coordinate')

plt.savefig('visual.png', bbox_inches='tight', dpi=500)
plt.show()


