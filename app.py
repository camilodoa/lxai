import gym

env = gym.make('CarRacing-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(1):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        print(env.action_space)
        observation, reward, done, info = env.step(action)
        # print(observation, reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
