import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from collections import deque

num_episodes = 1000

# Discount Rate
gamma = 0.95

# Learning Rate
alpha = 0.001

memory_size = 1000000
batch_size = 20

# Exploration Rate (For epsilon greedy strategy)
exploration_max= 1.0
exploration_min = 0.01
exploration_decay = 0.999
exploration_rate = exploration_max

# The memory variable storing the experience tuples
replay_memory = deque(maxlen=memory_size)

q_values = []
rewards_current_episode = 0
rewards_all_episodes = []

# This function creates the policy network for DQN 
def neural_net():
    model = Sequential()
    model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(action_space, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr = alpha))
    return model

# This function plays the game based on the current learning (experience)
# It also trains the policy network to estimate the optimal Q-function
def experience_replay():
    
    # If the number of tuples in the replay_memory is less than the batch size, ignore
    if len(replay_memory) < batch_size:
        return
    
    # Randomly sample a batch from the replay memory  
    batch = random.sample(replay_memory, batch_size)
    
    # For each experience tuple in batch
    for state, action, reward, state_next, done in batch:
        
        q_update = reward
        
        # If the episode is not terminated
        if not done:
            # Find the optimal Q-value using Bellmann Optimality Equation
            q_update = (reward + gamma * np.amax(model.predict(state_next)[0]))
        
        # Predict the Q-value using the policy network
        q_values = model.predict(state)
        
        # Update the Q-value of the optimal action 
        q_values[0][action] = q_update
        
        # Train the policy network to approximate the optimal Q-function 
        model.fit(state, q_values, verbose=0)
        
# Function to create the replay memory    
def remember(state, action, reward, next_state, done):
    # Append the experience tuple in the replay_memory
    replay_memory.append((state, action, reward, next_state, done))

# Function to choose the optimal action considering the epsilon greedy strategy 
def act(state):
    # Exploration-exploitation trade-off
    # Generating a random number between 0 and 1
    # If the generated number is less than the exploration rate, we explore otherwise we exploit
    if np.random.rand() < exploration_rate:
        return random.randrange(action_space)
    q_values = model.predict(state)
    return np.argmax(q_values[0])
    

# Creating the game environment
env = gym.make("CartPole-v1")
    
# Specifying the action space and state space
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
    
# Creating the deep neural network
model = neural_net()
    
for episode in range(num_episodes):
    print("Episode: ",str(episode),"   Reward: ",str(rewards_current_episode),"    Epsilon = ", str(exploration_rate)) 
    
    # Initialize new episode:
    # Reset the environment
    state = env.reset()
    
    #Reset the reward in the current episode to 0
    rewards_current_episode = 0  
    
    #Expand the dimension to make state a 2D list   
    state = np.reshape(state, [1, observation_space])
    
    while True:
       
        # Uncomment the line below to view the game being played
        #env.render()
        
        #Selecting the action to be taken  
        action = act(state)
        
        # Now that our action is chosen, we then take that action by calling step() on our env object
        # and passing our action to it. The function step() returns a tuple containing the new state, 
        # the reward for the action we took, whether or not the action ended our episode, and diagnostic 
        # information regarding our environment, which may be helpful for us if we end up needing to do any debugging.
        state_next, reward, done, info = env.step(action)
            
        # If the episode has not terminated, then the reward is positive else it is negative
        reward = reward if not done else -reward
        
        # we then update the rewards from our current episode by adding the reward we received for our previous action.
        rewards_current_episode += reward    
        
        # Expand the dimension to make state_next a 2D list
        state_next = np.reshape(state_next, [1, observation_space])
        
        # we store the agent’s experiences at each time step in a data set called the replay memory
        remember(state, action, reward, state_next, done)
            
        # This replay memory data set is what we’ll randomly sample from to train the network. 
        # The act of gaining experience and sampling from the replay memory that stores these 
        # experience is called experience replay
        experience_replay()
        
        # Once an episode is finished, we need to update our exploration_rate      
        exploration_rate *= exploration_decay
        exploration_rate = max(exploration_min, exploration_rate)
            
        # Next, we set our current state to the new_state that was returned to us once we took our last action
        state = state_next
            
        # If the episode has terminated, break the loop
        if done:
            break
    # We then just append the rewards from the current episode to the list of rewards from all episodes
    rewards_all_episodes.append(rewards_current_episode)
    
    # We’re good to move on to the next episode.    
  
######################################################################################################################    
# Calculate and print the average reward per hundred episodes:
    
rewards_per_ten_episodes = np.split(np.array(rewards_all_episodes),num_episodes/10)
count = 10

print("Average reward per ten episodes:\n")
for r in rewards_per_ten_episodes:
    print(count, ": ", str(sum(r/10)))
    count += 10  

########################################################################################################
#Printing the score graph
plt.figure(figsize=(12,6));    
    
episodes = range(len(rewards_all_episodes))
plt.plot(episodes, rewards_all_episodes, color = 'b', label='Reward for each Episode')
plt.legend()

plt.title('Score Graph')
plt.show()

######################################################################################################################    
# Watch our agent play the cartpole game:   
  
#Resest the environment
state = env.reset()
    
#Expand the dimension to make state a 2D list
state = np.expand_dims(state, axis = 0)

# Reset the reward for the current episode
reward_test = 0

while True:    
        
    # We then call render() on our env object, which will render the current state of the environment to the display
    env.render()
        
    #Selecting the action to be taken  
    action = act(state)
        
    # Now that our action is chosen, we then take that action by calling step() on our env object    
    state_next, reward, done, info = env.step(action)
               
    # we then update the rewards from our current episode by adding the reward we received for our previous action.
    reward_test += reward    
        
    # Expand the dimension to make state_next a 2D list
    state_next = np.expand_dims(state_next, axis = 0)
            
    # Next, we set our current state to the new_state that was returned to us once we took our last action
    state = state_next
            
    # If the episode has terminated, break the loop
    if done:
        break
print(" Reward: ",str(reward_test)) 

####################################################################################################          