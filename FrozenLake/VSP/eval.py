# read a map, receive a solution, determine if the solution reaches the goal

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import random
import os
import numpy as np
# from PIL import Image

count = 0
correct = 0

level = 5

check_map_dir = "../maps/level%d_text/"%(level)
check_answer_dir = "output/output_img_8/"
# check_answer_dir = "output/output_pure_text_1/"
# check_answer_dir = "output/output_table_2/"


# import ipdb; ipdb.set_trace()

for test_id in range(100):
    try:
        # print(test_id)
        # map
        with open(check_map_dir + '%d.txt'%(test_id), 'r') as f:
            contents = f.read()
            rows = contents.split('\n')

        # action
        with open(check_answer_dir + 'level%d/%d.txt'%(level, test_id), 'r') as f:
            contents = f.read()
            answer_index = contents.find('Action plan: ')
            if answer_index == -1:
                answer_index = contents.find('Action Plan: ')
            answer = contents[answer_index+13:]
            answer = answer.split(',')
            for answer_index in range(len(answer)):
                answer[answer_index] = answer[answer_index].replace('"', '')
                answer[answer_index] = answer[answer_index].replace("'", '')
                answer[answer_index] = answer[answer_index].replace("\n", '')
                answer[answer_index] = answer[answer_index].replace(".", '')
                answer[answer_index] = answer[answer_index].lstrip()
                answer[answer_index] = answer[answer_index].rstrip()

        env = gym.make('FrozenLake-v1', desc=rows, map_name="4x4", is_slippery=False)
        env.reset(seed=42)
        # temp = env.render()
        # import ipdb; ipdb.set_trace()

        action_number_map = {'L':0, 'D':1, 'R':2, 'U':3}
        for action_index in range(len(answer)):
            observation, reward, terminated, truncated, info = env.step(action_number_map[answer[action_index]])
            # temp = env.render()

            if terminated or truncated:
                if reward > 0:
                    correct += 1
                observation, info = env.reset()
                break
        count += 1

        env.close()
    except:
        # If not
        count += 1
        # import ipdb; ipdb.set_trace()
        pass

print("Total tested: %d"%(count))
print("Total correct: %d"%(correct))
