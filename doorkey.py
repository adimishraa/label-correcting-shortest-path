#!/usr/bin/env python
# coding: utf-8

# In[54]:


#required imports
import numpy as np
import matplotlib.pyplot as plt
import utils
import gym
import pickle
from utils import *


# In[55]:


#function to initliaze the various states, children of the states, and the total cost to reach each set
def initialize_states(env,info):
    list_states=[]
    gj=[]
    children={}
    #the direction states set
    dir_states=[[1,0],[0,1],[-1,0],[0,-1]]
    ct=0
    for i in range(0,env.height):
        for j in range(0,env.width):
            cell=env.grid.get(i,j)
            #for every cell which is not a wall add the states of every direction 
            if(cell is None or (cell.color!='grey')):
                #Get the states and the immediate children of each state
                for k,direction in enumerate(dir_states):
                    #child is for storing children of the current state
                    child=[]
                    st=(np.array([i,j]),np.array(direction))
                    #add the states whose cell position is not a wall
                    list_states.append(st)
                    #Add the left and right states as children for the current state
                    if(k!=len(dir_states)-1):
                        
                        child.append((st[0],np.array(dir_states[k-1])))
                        child.append((st[0],np.array(dir_states[k+1])))
                        
                    else:
                        child.append((st[0],np.array(dir_states[k-1])))
                        child.append((st[0],np.array(dir_states[0])))
                    #check if the cell in front of the current state cell is not a wall.
                    #If not add it to the children of the current state
                    next_cell=np.sum(st,0)
                    if(next_cell[0]>=0 and next_cell[0]<info['width']):
                        if(next_cell[1]>=0 and next_cell[1]<info['height']):
                            neighbor_cell=env.grid.get(next_cell[0],next_cell[1])
                            if(neighbor_cell is None or neighbor_cell.color!='grey'):
                                child.append((next_cell,st[1]))
                    children.update({ct:child})
                    ct+=1
                    #set the initial cost of every state to inf except the starting state( set to 0)
                    if((st[0]==env.agent_pos).all() and (st[1]==env.dir_vec).all()):
                           gj.append(0)
                    else:
                           gj.append(np.inf)
    #return all the states, the initial cost, and the children dictionary
    return list_states,gj,children 
#list_states contains a list of all the possible states of the agent
#gj is the list of initial costs of each state
#children is the dictionary with keys as the indices of each state and values as a list of states which are children 
#to the state pointed by the corresponding key


# In[56]:


#function to convert the children dictionary to one where the indices of each state 
# is stored instead of the state itself
def get_children_indices(children,list_states):
    children_indices={}
    for i,states in enumerate(list_states):
        index_list=[]
        for child in children[i]:
            for j,st in enumerate(list_states):
                if((st[0]==child[0]).all() and (st[1]==child[1]).all()):
                    index_list.append(j)
        #store index of each children to the index of each parent
        children_indices.update({i:index_list})
    #the keys of this dictionary are the indices of the parents states
    #and the values are lists of the indices of the children
    return children_indices
        


# In[57]:


#function to find the indices of vital states from list_states
def find_token_states(info,token,list_states,env,dir_states):
    token_states=[] #list to store the indices of the vital states
    check_states=[] #temporary list
    temp_state=np.array([])  #temporary state to store the cell calue which might lead to door or key
    #to find the index of the initial state in list_states
    if token=='agent':
        for i,(a,b) in enumerate(list_states):
            if((a==info['init_agent_pos']).all() and (b==info['init_agent_dir']).all()):
                return i
    #to find the indices of all possible states of agent when it reaches the goal cell
    elif token=='goal':
         for i,(a,b) in enumerate(list_states):
            if((a==info['goal_pos']).all()):
                token_states.append(i)
   #to find the indices of all possible states of agent which leads to the key cell 
    elif token=='key':
        for i,(a,b) in enumerate(list_states):
            if((a==info['key_pos']).all()):
                for direction in dir_states:
                    temp_state=a-direction
                    cell=env.grid.get(temp_state[0],temp_state[1])
                    if(cell is None or cell.color not in ['grey','green']):
                        check_states.append((temp_state,direction))
                break
        
        for i,(a,b) in enumerate(list_states):
                for ch in check_states:
                    if((a==ch[0]).all() and (b==ch[1]).all()):
                        token_states.append(i)
    #to find the indices of all possible states of agent which leads to the door cell
    elif token=='door':
        for i,(a,b) in enumerate(list_states):
            if((a==info['door_pos']).all()):
                for direction in dir_states:
                    temp_state=a-direction
                    cell=env.grid.get(temp_state[0],temp_state[1])
                    if(cell is None or cell.color not in ['grey','green']):
                        check_states.append((temp_state,direction))
                break
        
        for i,(a,b) in enumerate(list_states):
                for ch in check_states:
                    if((a==ch[0]).all() and (b==ch[1]).all()):
                        token_states.append(i)
     #to find the indices of all possible states of agent when it reaches the door cell
    elif token=='door_pos':
         for i,(a,b) in enumerate(list_states):
            if((a==info[token]).all()):
                token_states.append(i)
    #to find the indices of all possible states of agent when it reaches the key cell
    elif token=='key_pos':
         for i,(a,b) in enumerate(list_states):
            if((a==info[token]).all()):
                token_states.append(i)
    
    
    return token_states                   
 


# In[58]:


#use lc algorithm to find the shortest path
def lc_algorithm(cij,children_indices,s,tau,gj):
    OPEN=[] #OPEN stack is the same one used in lc algorithm
    Parent={} #this dictionary used to backtrack the path
    index_i=0 #index of the state popped from OPEN
    gtau=[0]*len(tau) #cost of all the possible final states
    dsum=0
    OPEN.append(s)
    #lc algorithm loop
    while (len(OPEN)>0):
        index_i=OPEN.pop()
        for i,t in enumerate(tau):
                gtau[i]=gj[t]
        for k,j in enumerate(children_indices[index_i]):
            dsum=(gj[index_i]+cij[index_i][k])
            if(dsum<gj[j] and dsum<min(gtau)):
                gj[j]=dsum
                Parent.update({j:index_i})
                if(j in tau):
                       gtau[tau.index(j)]=gj[j]
                elif j not in tau:
                    OPEN.append(j)                
    check=tau[np.argmin(gtau)]
    #check if the final state exists in Parent
    #if it does Parent and the total cost
    #if the final state doesn't exist return empty dictionary
    if(check in Parent.keys()):
        return Parent,gtau
    else:
        return {},np.inf


# In[59]:


#create a list of the states traversed from the end to the start
def shortest_path(gtau,tau,s,path):
    fin_state=tau[np.argmin(gtau)] #final state of the path tau is a list of possible final states
    states_in_path=[] #list of all the states traversed in the shortest path
    x=fin_state
    states_in_path.append(x)
    #check if x= starting state
    #s= starting state
    while(x!=s):
        x_new=path[x]
        states_in_path.append(x_new)
        x=x_new
    return states_in_path


# In[60]:


#use it when the door is not obstructing path ie no door in the shortest path or door key both on the way
#function to get the actions of the agent to travel the shortest path possible path if the door is not obstructing that
def policy_shortest_path(list_states,states_in_path,door_pos,key_pos,dir_states):
    policy=[] #list of all actions taken by the agent
    cur_dir=0 #storing the direction of the agent of a particular state
    next_dir=0 #storing the direction of the agent in the very next state

    for i in range(0,len(states_in_path)-1):
        # check if the next state is a cell which has the key
        if(states_in_path[i+1] in key_pos):
        # Add pick key to the list of actions
            policy.append(3)
        #check if the next stae has the location of the door
        elif(states_in_path[i+1] in door_pos):
        # add unlock door to the list of actions
            policy.append(4)
            
        if((list_states[states_in_path[i]][0]==list_states[states_in_path[i+1]][0]).all()):
            for k,direction in enumerate(dir_states):
                #check the direction of the current state
                if((list_states[states_in_path[i]][1]==direction).all()):
                    cur_dir=k
                    #check the direction of the next state
                if((list_states[states_in_path[i+1]][1]==direction).all()):
                    next_dir=k
                #check if the agent is turning left or right
            if(cur_dir==0 and next_dir==k):
                policy.append(1)
            elif(cur_dir==k and next_dir==0):
                policy.append(2)
            elif(next_dir>cur_dir):
                policy.append(2)
            elif(next_dir<cur_dir):
                policy.append(1)   
                #append move forward if the agent is moving forward
        else:
            policy.append(0)
    
    #return the list of actions
    return policy


# In[61]:


# function to create the info dictionary from env (same as the one given in utils.py)
def create_info(env):
    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec
        }
    
    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])    
    return info


# In[62]:


#function to split the shortest path into three parts:
#1. Initial state s to the states (key_token) leading to the position of the key (key_pos)
#2. From the new state in key_token to the states (door_token) leading to the position of the door (door_pos)
#3. From door_pos to the final goal (this part is not solved in this function but taken from a previous function)
# The states traversed in part 3 is given by states_door_to_goal
def key_door_goal(list_states,gj,children,states_door_to_goal,s,tau,key_token,door_token,dir_states,key_pos,door_pos,is_key_found):
    children_indices=get_children_indices(children,list_states) #get the children indices
    cij={} #transition cost from i to j
    gj_actual=list(gj) #copy of the initial costs
    policy_final=[] #list of final actions to be taken
    #populating the tranisition cost with the value 1
    for i in range(0,len(children_indices.keys())):
        cij.update({i:np.ones(len(children_indices[i]))})
    #1: to get the policies to traverse the shortest path from start to key
    #is_key_found is used to check if the first state is the state leading to the key\
    #if is_key_found=1 then skip the lc algorithm from intial state to the key_token
    if(is_key_found==0):
        #path 1 is a dictionary of state traversal from s to key_token
        #gtau1 is the total cost
        path1,gtau1=lc_algorithm(cij,children_indices,s,key_token,gj_actual)
        #check path exists
        if(bool(path1)==False):
            print('no path1')
            return [] #return no path
        #find the list of states traversed
        states_in_path1=shortest_path(gtau1,key_token,s,path1)[::-1] #reverse the list
        #create a list of the actions taken
        policy_final=policy_shortest_path(list_states,states_in_path1,door_pos,key_pos,dir_states)
        #add pick key action to the list
        policy_final.append(PK)
        #2: pick the key
        gj[s]=np.inf #set the cost of the initial state to infinity
        s=states_in_path1[-1] #s_new=key_token
        gj[s]=0 #set the cost of the current state as 0
    #3: from the current state to door
   #path2is a dictionary of state traversal from s_new to door_token
    #gtau2 is the total cost
    path2,gtau2=lc_algorithm(cij,children_indices,s,door_token,gj)
    #check path exists
    if(bool(path2)==False): 
        print('no path2')
        return []  #return no path
    #find the list of states traversed
    states_in_path2=shortest_path(gtau2,door_token,s,path2)[::-1] #reverse the list
    #create a list of the actions taken
    
    policy2=policy_shortest_path(list_states,states_in_path2,door_pos,key_pos,dir_states)
    
    for p in policy2:
        #check if the actions is not picking key or unlocking door (ensure that these actions are just useed once)
        if(p!=PK and p!=UD):
            policy_final.append(p) #apppend the final_policy
    #append unlock door action
    policy_final.append(UD)
    #append move forward action
    policy_final.append(MF)
    #4: door to goal
    #we have the states for this traversal
    #compute the actions
    policy3=policy_shortest_path(list_states,states_door_to_goal,door_pos,key_pos,dir_states)
    for p in policy3:
        #check if the actions is not picking key or unlocking door (ensure that these actions are just useed once)
        if(p!=PK and p!=UD):
            policy_final.append(p) #apppend the final_policy
    
    return policy_final   


# In[63]:


# This function is exactly similar to initialize_states , the only difference is that it is assumed that
# the door cell is a wall cell
#this function is used to find the shortest path from agent to goal without using key or door
def initialize_states_no_door(env,info):
    list_states=[]
    gj=[]
    children={}
    dir_states=[[1,0],[0,1],[-1,0],[0,-1]]
    ct=0
    for i in range(0,env.height):
        for j in range(0,env.width):
            cell=env.grid.get(i,j)
            if(cell is None or (isinstance(cell,gym_minigrid.minigrid.Wall) is not True)):
                if((isinstance(cell,gym_minigrid.minigrid.Door) is not True)):
                    for k,direction in enumerate(dir_states):
                            child=[]
                            st=(np.array([i,j]),np.array(direction))
                            list_states.append(st)
                            if(k!=len(dir_states)-1):
                        
                                child.append((st[0],np.array(dir_states[k-1])))
                                child.append((st[0],np.array(dir_states[k+1])))
                        
                            else:
                                child.append((st[0],np.array(dir_states[k-1])))
                                child.append((st[0],np.array(dir_states[0])))
                    
                            next_cell=np.sum(st,0)
                            if(next_cell[0]>=0 and next_cell[0]<info['width']):
                                if(next_cell[1]>=0 and next_cell[1]<info['height']):
                                    neighbor_cell=env.grid.get(next_cell[0],next_cell[1])
                                    if(neighbor_cell is None or (isinstance(neighbor_cell,gym_minigrid.minigrid.Door)                                                                 is not True)):
                                        child.append((next_cell,st[1]))
                            children.update({ct:child})
                            ct+=1
                            if((st[0]==env.agent_pos).all() and (st[1]==env.dir_vec).all()):
                                   gj.append(0)
                            else:
                                   gj.append(np.inf)
    return list_states,gj,children       


# In[64]:


#function to calculate the shortest path wwith no dor cell in the path
def get_states_no_door(env,info,door_pos,dir_states,tokens,list_states,gj,children):
    #list_states,gj,children=initialize_states_no_door(env,info,door_pos)
    children_indices=get_children_indices(children,list_states)
    cij={}
    for i in range(0,len(children_indices.keys())):
        cij.update({i:np.ones(len(children_indices[i]))})
    # define start and end
    s=find_token_states(info,tokens[0],list_states,env,dir_states)
    tau=find_token_states(info,tokens[3],list_states,env,dir_states)
    #use lc algortihm
    path,gtau=lc_algorithm(cij,children_indices,s,tau,gj)
    if(bool(path) is False):
        return []
    #reverse state transition from end to start
    states_in_path=shortest_path(gtau,tau,s,path)[::-1]
    return states_in_path


# In[65]:


def doorkey_problem(env):
    info=create_info(env)
    list_states,gj,children=initialize_states(env,info)  #initialize the list_states, state costs and children etc.
    dir_states=[np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1])] #list consisting all the directions
    tokens={0:'agent',1:'key',2:'door',3:'goal'} #tokens for find_tokens
    door_pos=find_token_states(info,'door_pos',list_states,env,dir_states) #find the door position
    key_pos=find_token_states(info,'key_pos',list_states,env,dir_states) #find the key position
    
    #compute the shortest path between the initial agent state to the goal
    states_in_path=get_states_no_door(env,info,door_pos,dir_states,tokens,list_states,gj,children)
    
    # check if there is a path possible
    if(bool(states_in_path) is False):
        return [] #return empty path 
    
    #check if the shortest path has door in the way
    if(any(item in states_in_path for item in door_pos) is False):
        #compute the policy for the above shortest path and return
        policy=policy_shortest_path(list_states,states_in_path,door_pos,key_pos,dir_states)
        print('no door in shortest path')
        return policy
    else:
        #if the door is present check if the cell containing the key is on the way to the goal in the shortest path
        if(any(item in states_in_path for item in key_pos) is True):
            #compute the policy for the above shortest path and return
            policy=policy_shortest_path(list_states,states_in_path,door_pos,key_pos,dir_states)
            print('key and door in shortest path')
            return policy
        else:
            print('check key-door-goal or no-door path')
            #find the states leading to the key
            key_token=find_token_states(info,tokens[1],list_states,env,dir_states)
            #find the state right before the agent enters the door cell
            for d in door_pos:
                if(d in states_in_path):
                    door_state_index=states_in_path.index(d)
                    break
            door_token=[] #list containing the state leading to the door
            s=find_token_states(info,tokens[0],list_states,env,dir_states)# initial agent state
            policy_key_door_goal=[] #policy for the path from begin to key to door to goal
            is_key_found=0
            #if the key is in front of s
            if(s in key_token):
                policy_key_door_goal.append(3) # append pick key
                is_key_found=1 #set the key has been found
            tau=find_token_states(info,tokens[1],list_states,env,dir_states) #find the goal state
            list_states,gj,children=initialize_states(env,info)  #initialize states
            door_token.append(states_in_path[door_state_index-1]) 
            states_door_to_goal=states_in_path[door_state_index:]
            #compute the shortest path from key to door to goal and append it to the policy to be  returned
            policy_key_door_goal.extend(key_door_goal(list_states,gj,children,states_door_to_goal,s,tau,key_token,door_token,dir_states,key_pos,door_pos,is_key_found))
             
                #initialize states with no door (wall in place of door)
            list_states,gj,children=initialize_states_no_door(env,info)  
            #get the state transition for this shortest path which has no door on the way
            states_in_path=get_states_no_door(env,info,door_pos,dir_states,tokens,list_states,gj,children)
            if(bool(states_in_path) is True):
                #find the policy for this shortest path which has no door on the way
                policy_no_door=policy_shortest_path(list_states,states_in_path,door_pos,key_pos,dir_states)
            else:
                #if the path doesn't exist then return the key_door_doal policy
                return policy_key_door_goal
            #compare which policy is better
            if(len(policy_no_door)>len(policy_key_door_goal)):
                policy=policy_key_door_goal
            else:
                policy=policy_no_door 
                        
    return policy
     


# In[96]:


def policy_to_words(policy):
    actions={0:"MF",1:"TL",2:"TR",3:"PK",4:'UD'}
    action_list=""
    for p in policy:
        action_list=action_list+actions[p]+"->"
    return action_list[:-2]


# In[ ]:


def main():
    env_path = './envs/example-8x8.env'
    env, info = load_env(env_path) # load an environment
    seq = doorkey_problem(env) # find the optimal action sequence
    actions=policy_to_words(seq)
    print(actions) #print the action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0]) # draw a GIF & save


if __name__ == '__main__':
    main()

