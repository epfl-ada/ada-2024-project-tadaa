# This code requires a CUDA envirement to run. 
from urllib.parse import unquote
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer



def llm_paths(sources, targets, links):
    """This function generates paths from each source to target pairwise in the lists (sources, targets).
    The generation is made by a large language model.

    Args:
        sources (list): the list of sources
        targets (list): the list of destinations
        links (dict): the links that can be visited from a each article

    Returns:
        dict: a dictionary containing the paths from each source to target
    """
    # make links in a readable format not html format
    links = {unquote(key): [unquote(word) for word in value] for key, value in links.items()}
    #choose model
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    #load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #set initial message
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba. You are a helpful assistant."},
    ]

    # sources = ['Asteroid', 'Brain', 'Theatre', 'Pyramid', 'Batman', 'Bird', 'Batman', 'Bird', 'Beer', 'Batman']
    # targets = ['Viking', 'Telephone', 'Zebra', 'Bean', 'Wood', 'Great_white_shark', 'The_Holocaust', 'Adolf_Hitler', 'Sun', 'Banana']
    responses = {}
    for source, target in tqdm(zip(sources, targets)):
        # run 100 times for each source target pair
        for i in tqdm(range(100)):

            new_source = source
            path = []
            count = 0
            path.append(new_source)

            while new_source != target:
                torch.cuda.empty_cache()
                count += 1
                if count > 50:
                    break

                # set the complete prompt
                new_messages = messages.copy()
                links_to_choose_from = [word for word in links[new_source] if word not in path]
                links_to_choose_from = [unquote(word) for word in links_to_choose_from]
                
                #define prompt
                prompt = "I will give you a list of elements. Choose one element in that list that you think is the most related to '" + target + "'." \
                "You should never, under no circumstances, chose an element that is not in the list otherwise you will fail forever." \
                " The first word of your response should be the answer which is an element of the list. " \
                "Here is the list of elements to chose from: " + "\n\n " \
                "Elements to chose from : " + str(links_to_choose_from)
                new_messages.append({"role": "user", "content": prompt})

                # tokenize the prompt
                text = tokenizer.apply_chat_template(
                    new_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # generate response
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=100
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                # decode response
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                #format the response
                response_word = response.split()[0].replace("*", "")

                # check of the response is valid (in the list)
                if response_word not in links_to_choose_from:
                    #give the model a second chance to generate a valid response
                    torch.cuda.empty_cache()
                    new_messages.append({"role": "assistant", "content": response})
                    new_prompt = "You chose a word that is not in the list of words to chose from. Please choose a word from " \
                                 "the list of words provided before related to the target word: " + target + ". If no word is related to the target word, still take a word from the list." \
                                                                                                             "Only return one word as your answer"
                    new_messages.append({"role": "user", "content": new_prompt})
                    text = tokenizer.apply_chat_template(
                        new_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=100
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # format the response 
                    response_word = response.split()[0].replace("*", "").replace(" ", "_")

                if response_word in links_to_choose_from:
                    # set the new source
                    new_source = response_word
                else:
                    # consider path as failed
                    break

                path.append(new_source)

            if len(path) > 0 and path[-1] == target:
                if source + "_" + target not in responses:
                    responses[source + "_" + target] = []
                responses[source + "_" + target].append(path)
            else:
                print("Failed", path, target)


    with open("./data/llm_responses.json", "w") as f:
        json.dump(responses, f)
    return responses


