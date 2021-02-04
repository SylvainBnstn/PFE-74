from Model_DQN_VF import DQN

### Initialise the DQN 

dqn = DQN("Data_Model_2.csv",0.9,0.5,0.83,1/10,2/10)

### executing the training part of the DQN 
def execute_train():
    ## setting 200 episodes for the trainig 
    return_trace, p_trace = dqn.dqn_training(200)
    ### plot the result by using the plot_result function 
    ### it shows the reward and the price for the training 
    dqn.plot_result(return_trace, p_trace)
    return return_trace, p_trace


### executing the test part 
def execute_test():
    ### plotting the result and return the sequences of reward, price and booked on the data 
    seq_reward, seq_price, seq_booked, reward_from_data = dqn.plot_result_test(dqn.price_grid_test)
    return seq_reward, seq_price, seq_booked, reward_from_data

### Calling these 2 functions for the trainging and the testing of our model 
rt , pt = execute_train()
seq_reward, seq_price, seq_booked, reward_from_data = execute_test()
