import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import PIL.Image
import time
import os

import random
import numpy as np
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # env.seed(seed)
seed_everything(1)

levels = [3,4,5,6,7,8]
in_context_example_num = 0 # 0, 1, 2, 4, 8

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


GOOGLE_API_KEY='YOUR-API-KEY'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')
if in_context_example_num > 0:
    output_path = "output/output_img_%d/"%(in_context_example_num)
    input_backup_path = "input/input_backup_img_%d/"%(in_context_example_num)
else:
    output_path = "output/output_img/"
    input_backup_path = "input/input_backup_img/"

os.makedirs(output_path, exist_ok=True)
os.makedirs(input_backup_path, exist_ok=True)

loop_time = 0

EXAMPLE_DICT = {
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
}

# Prepare examples
for level in levels:
    for example_id in range(8):
        example_path = "../example/level%d/"%(level)
        img_input = PIL.Image.open(example_path + "%d.png"%(example_id))
        with open("../example/level%d__sol/%d.txt"%(level, example_id), "r") as f:
            text_input = f.read()
        this_example = (img_input, text_input)
        EXAMPLE_DICT[level].append(this_example)

# import ipdb; ipdb.set_trace()
for level in levels:
    os.makedirs(output_path + "level%d"%(level), exist_ok=True)
    os.makedirs(input_backup_path + "level%d"%(level), exist_ok=True)
    start_idx = 0
    end_idx = 100
    runned_term = 0
    map_path = "../maps/level%d/"%(level)
    while True:
        try:
            curr_id = start_idx + runned_term
            if curr_id >= end_idx:
                break
            img_input = PIL.Image.open(map_path + "%d.png"%(curr_id))
            prompt_input_1 = '''
As a professional maze solver, your task is to analyze a grid-based map and devise an action plan that enables a player to reach the goal from the starting point without falling into any holes, using the fewest possible moves. Since coding is not within your skill set, your approach relies on logical reasoning of the map.

## Game Setup
- The game presents a fully observable grid-based map.
- The player starts at a specified grid square, with the goal located elsewhere on the map.
- Each grid square is either safe or contains a hole.
- Your goal is to guide the player to the goal while avoiding holes.
The following figure shows how the player, the holes (non-safe grid), the lands (safe grids), and the goals look like.
'''
            prompt_input_2 = '''
## Moving Rules
- The action plan involves a series of moves: 'L' (left), 'R' (right), 'U' (up), or 'D' (down).
- Each move transfers the player to the adjacent square in that direction, provided it is a safe square. The player cannot move more than one square at a time.
- Moving off the edge of the map has no effect. The player will remain at the same square.
- DO NOT MOVE INTO A HOLE! Falling into a hole results in defeat.
- Locating at the grid containing the goal results in victory.
We provide an example to further illustrate the rules.
'''
            prompt_input_3 = '''
In this provided example:
- The player is at Row 1, Column 1;
- The goal is at Row 4, Column 4;
- There are two holes: one at Row 1, Column 2, and another at Row 4, Column 1.
- The player can move DOWN. This is because moving down brings them to Row 2, Column 1, and this cell is safe (without holes).
- Moving UP has no effects. This is because the player is already in the topmost row.
- Similarly, moving LEFT has no effects because the player is already in the left-most column.
- Moving RIGHT places the player at Row 1, Column 2. Since there is a hole at this grid, this move results in a loss.

## Procedure and Output
Now you will solve the given maze. To solve it, please generate text EXACTLY FOLLOW THE FOLLOWING STEPS:
1. First, interpret map. List where the player is at now, where is the goal, and where are the holes.
2. Then, generate an action plan to navigate to the goal step by step. At each step, you should check:
    (a) Where the current move leads the player to (the row and column);
    (b) What is in that grid. Is it a hole? Is it the goal? Is it an empty space?
    (c) Determine if that is a safe action. If not, correct it and re-generate the action plan.
3. Next, verify if the steps successfully navigate the player to the goal without falling into the hole. If not, restart from step 2 and re-generate this step.
4. If succeed, output an aggregated plan using "Action plan: <PLAN>", where <PLAN> is a string concatenated action in each step. For example, "Action plan: L,L,R,U,D" meaning an action plan of left, left, right, up, and down. Double check the final action plan is consistent with the previous analysis.
Do not output any extra content after the above aggregated output.
'''
            prompt_examples = []
            image_examples = []
            if in_context_example_num > 0:
                prompt_examples.append("## Example:\n")
                example_indices = random.sample(range(8), in_context_example_num)
                for example_index in example_indices:
                    this_example = EXAMPLE_DICT[level][example_index]
                    image_examples.append(this_example[0])
                    prompt_examples.append(this_example[1] + "\n")
            prompt_input_4 = "\nPlease generate action plan for the following maze:"
            prompt_img_1 = PIL.Image.open('prompt-visual-images/system-figure-1.png')
            prompt_img_2 = PIL.Image.open('prompt-visual-images/system-figure-2.png')
            model_input_seq = [prompt_input_1, prompt_img_1, prompt_input_2, prompt_img_2, prompt_input_3]
            if in_context_example_num > 0:
                assert len(prompt_examples) == len(image_examples) + 1
                assert len(prompt_examples) == in_context_example_num + 1
                model_input_seq.append(prompt_examples[0])
                for example_index in range(in_context_example_num):
                    model_input_seq.append(image_examples[example_index])
                    model_input_seq.append(prompt_examples[example_index+1])
            model_input_seq += [prompt_input_4, img_input]

            response = model.generate_content(model_input_seq)
            with open(input_backup_path + "level%d/%d.txt"%(level, curr_id), "w") as f:
                contents = ""
                for input_prompt_index in range(len(model_input_seq)):
                    if type(model_input_seq[input_prompt_index]) == type("string"):
                        contents += model_input_seq[input_prompt_index]
                f.write(contents)
            with open(output_path + "level%d/%d.txt"%(level, curr_id), "w") as f:
                f.write(response.text)
            # import ipdb; ipdb.set_trace()
            # pass
            time.sleep(2)
            runned_term += 1
        except:
            time.sleep(2)
            pass






# pass



# cookie_picture = [{
#     'mime_type': 'image/png',
#     'data': Path('cookie.png').read_bytes()
# }]
# prompt = "Do these look store-bought or homemade?"

# response = model.generate_content(
#     model="gemini-pro-vision",
#     content=[prompt, cookie_picture]
# )
# print(response.text)