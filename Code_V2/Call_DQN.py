from Model_DQN_VF import DQN

dqn = DQN("Data_Model_2.csv",0.9,0.5,0.83,1/10,2/10)


def execute_train():
    return_trace, p_trace = dqn.dqn_training(200)
    dqn.plot_result(return_trace, p_trace)
    return return_trace, p_trace
    
def execute_test():
    #seq_reward_all_apt19, seq_price_all_apt19, seq_booked_all_apt19 = dqn.plot_result_test(dqn.data_test_2019, dqn.data_test_booked_2019, "2019")
    #seq_reward_all_apt20, seq_price_all_apt20, seq_booked_all_apt20 = dqn.plot_result_test(dqn.data_test_2020, dqn.data_test_booked_2020, "2020")
    
    #dqn.dqn_test(dqn.price_grid_test)
    #return seq_reward_all_apt19, seq_price_all_apt19, seq_booked_all_apt19, seq_reward_all_apt20, seq_price_all_apt20, seq_booked_all_apt20
    seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt, reward_from_data = dqn.plot_result_test(dqn.price_grid_test)
    return seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt, reward_from_data

rt , pt = execute_train()
seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt, reward_from_data = execute_test()

"""
# this function replace the demand of the customer
def get_booked(price):
    import random
    return random.randint(0,30)

def execute_interaction():
    state = dqn.env_initial_test_state([150,0])#init price 150 init demand 0
    reward_trace = [0] #reward à 0 au début
    p_trace = [state[0,0]]
    booked =[state[0,1]]
    
    for t in range(10):
        p, state= dqn.dqn_interaction(state)
        
        booking = get_booked(p)
        state[1,0] = booking
        reward = dqn.profit_t_d(state[0,0], state[1,0])

        reward_trace.append(reward)
        p_trace.append(p)
        booked.append(booking)
        
    print("PRICE: ", p_trace)
    print("REWARD: ", reward_trace)
    print("BOOKED: ", booked)
    return p_trace, reward_trace
"""
#seq_reward_all_apt19, seq_price_all_apt19, seq_booked_all_apt19, seq_reward_all_apt20, seq_price_all_apt20, seq_booked_all_apt20 = execute_test()
#execute_interaction()