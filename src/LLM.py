# This code requires a CUDA envirement to run. 
from urllib.parse import unquote
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
# os.environ['HF_HOME'] = '../HF'

def in_list(word, list):
    for element in list:
        if word in element or element in word:
            return element
    return None

def next_word(prompt, second_prompt, model, tokenizer, messages, links_to_choose_from):
    new_messages = messages.copy()
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
    response_word = response.split()[0].replace("*", "").replace(" ", "_")
    response_word = in_list(response_word, links_to_choose_from)    
    # check of the response is valid (in the list)
    if response_word is None:
        #give the model a second chance to generate a valid response
        torch.cuda.empty_cache()
        new_messages.append({"role": "assistant", "content": response})
        second_prompt = second_prompt
        new_messages.append({"role": "user", "content": second_prompt})
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
        response_word = in_list(response_word, links_to_choose_from)
        if response_word is not None:
            # set the new source
            hub_source = response_word
            new_source = hub_source
        else:
            new_source = None
    else:
        new_source = response_word
    return new_source



def llm_paths(sources, targets, links, hub = False, repeat = 30):
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
    # model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    #load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #set initial message
    messages = [
        {"role": "system", "content": "You are Llama, created by Meta. You are a helpful assistant."},
    ]

    responses = {}

    for source, target in tqdm(zip(sources, targets)):
        torch.cuda.empty_cache()      

        # run multiple times for each source target pair
        for i in tqdm(range(repeat)):
            path = [source]
            new_source = source
            links_to_choose_from = [word for word in links[new_source] if word not in path]
            links_to_choose_from = [unquote(word) for word in links_to_choose_from]
            if hub:
                hub_prompt = "I will give you a list of expressions. Choose one expression in that list that you think has the highest PageRank on wikipedia." \
                "You should never, under no circumstances, chose an element that is not in the list otherwise you will fail forever." \
                " The first word of your response should be the answer which is an element of the list. " \
                "Here is the list of elements to chose from: " + "\n\n " \
                "Elements to chose from : " + str(links_to_choose_from)
                second_prompt = "You chose a word that is not in the list of words to choose from. Please choose a word from " \
                    "the list of words provided. The word has to represent a general concept related to the target word: " + target + "." \
                    " If no word has a high pagerank, still take a word from the list and only from the list." \
                    " Only return one word as your answer"
                new_source = next_word(hub_prompt, second_prompt, model, tokenizer, messages, links_to_choose_from)
                if new_source is not None:
                    path.append(new_source)
                else:
                    new_source = source
            count = 0
            while new_source != target:
                torch.cuda.empty_cache()
                count += 1
                if count > 25:
                    break

                # set the complete prompt
                links_to_choose_from = [word for word in links[new_source] if word not in path]
                links_to_choose_from = [unquote(word) for word in links_to_choose_from]
                
                #define prompt
                prompt = "I will give you a list of expressions. Choose one expression in that list that you think is the most related to '" + target + "'." \
                "You should never, under no circumstances, chose an element that is not in the list otherwise you will fail forever." \
                " The first word of your response should be the answer which is an element of the list. " \
                "Here is the list of elements to chose from: " + "\n\n " \
                "Elements to chose from : " + str(links_to_choose_from)
                second_prompt = "You chose a word that is not in the list of words to choose from. Please choose a word from " \
                        "the list of words provided. The word has to represent a concept related to the target word: " + target + "." \
                        " If no word is related to the target word, still take a word from the list and only from the list." \
                        "Only return one word as your answer"

                new_source = next_word(prompt, second_prompt, model, tokenizer, messages, links_to_choose_from)
                if new_source is None:
                    break
                path.append(new_source)


            if len(path) > 0 and path[-1] == target:
                if source + "_" + target not in responses:
                    responses[source + "_" + target] = []
                responses[source + "_" + target].append(path)
                print("Success", path, target)  
            else:
                print("Failed", path, target)

        with open("llm_responses_qwen_simple_prompt_hub.json", "w") as f:
            json.dump(responses, f)
    return responses