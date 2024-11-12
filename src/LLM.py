
from urllib.parse import unquote
from unsloth import FastLanguageModel
import json
import torch



def llm_paths(sources, targets, links):

    links = {unquote(key): [unquote(word) for word in value] for key, value in links.items()}
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model, tokenizer = FastLanguageModel.from_pretrained(model_name)
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba. You are a helpful assistant."},
    ]

    # sources = ['Asteroid', 'Brain', 'Theatre', 'Pyramid', 'Batman', 'Bird', 'Batman', 'Bird', 'Beer', 'Batman']
    # targets = ['Viking', 'Telephone', 'Zebra', 'Bean', 'Wood', 'Great_white_shark', 'The_Holocaust', 'Adolf_Hitler', 'Sun', 'Banana']
    print(sources[0])
    responses = {}
    for source, target in zip(sources, targets):
        for i in range(10):

            new_source = source
            print(source)
            path = []
            count = 0
            path.append(new_source)
            while new_source != target:
                torch.cuda.empty_cache()

                count += 1
                if count > 50:
                    break
                new_messages = messages.copy()
                print(new_source)
                links_to_choose_from = [word for word in links[new_source] if word not in path]
                links_to_choose_from = [unquote(word) for word in links_to_choose_from]
                # print("links to choose from", links_to_choose_from)
                prompt = "I will give you a list of elements. Choose one element in that list that you think is the most related to '" + target + "'." \
                "You should never, under no circumstances, chose an element that is not in the list otherwise you will fail forever." \
                " The first word of your response should be the answer which is an element of the list. " \
                "Here is the list of elements to chose from: " + "\n\n " \
                "Elements to chose from : " + str(links_to_choose_from)
                new_messages.append({"role": "user", "content": prompt})
                # print(new_messages)
                # print(new_messages)
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
                # print("response:",response)
                response_word = response.split()[0].replace("*", "")
                print("response word", response_word)
                # print("links to choose from", links_to_choose_from)
                if response_word not in links_to_choose_from:
                    # print(response_word)
                    new_messages.append({"role": "assistant", "content": response})
                    new_prompt = "You chose a word that is not in the list of words to chose from. Please choose a word from " \
                                 "the list of words provided before related to the target word: " + target + ". If no word is related to the target word, still take a word from the list." \
                                                                                                             "Only return one word as your answer"
                    # print(new_messages)
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
                    # print(response)

                    response_word = response.split()[0].replace("*", "").replace(" ", "_")
                    print("corrected response", response_word)
                if response_word in links_to_choose_from:
                    new_source = response_word
                else:
                    break
                print(new_source)

                path.append(new_source)
            if len(path) > 0 and path[-1] == target:
                print(path)
                if source + "_" + target not in responses:
                    responses[source + "_" + target] = []
                responses[source + "_" + target].append(path)
            else:
                print("Failed", path, target)


    with open("llm_responses.json", "w") as f:
        json.dump(responses, f)
    return responses