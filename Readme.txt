Doorkey.py contains the program for the optimal path algorithm
The program uses utils.py along with numpy and matplotlib

The program contains the following methods:
1) initialize_states(env,info)
2) get_children_indices(children,list_states)
3) find_token_states(info,token,list_states,env,dir_states)
4) lc_algorithm(cij,children_indices,s,tau,gj)
5) shortest_path(gtau,tau,s,path)
6) policy_shortest_path(list_states,states_in_path,door_pos,key_pos,dir_states)
7) create_info(env)
8)  key_door_goal(list_states,gj,children,states_door_to_goal,\
                  s,tau,key_token,door_token,dir_states,key_pos,door_pos,is_key_found)
9)initialize_states_no_door(env,info)
10) get_states_no_door(env,info,door_pos,dir_states,tokens,list_states,gj,children)
11) doorkey_problem(env)
12) policy_to_words(policy)

Execute the main() function to get the gif and optimal actions