import json, re
from genTemplates import *
import random
random.seed(1234)
import tiktoken
import evaluate
count_dict = {'Request Meeting': 0, 'Request Data': 0, 'Request Action': 0, 'Request Action Data': 0, 'Request Meeting Data': 0, 'Deliver Data': 0, 'Deliver Action Data': 0, 'Deliver Meeting Data': 0, 'Amend Data': 0, 'Amend Meeting Data': 0}
enc = tiktoken.encoding_for_model("gpt-4")
min_required = 2
OUTPUT_DELIMITER = "\nThe outputs are:\n"
eot = "\n"
bot = "\n"
MAX_PROMPT_LENGTH = 3700
max_gen_tokens = 300
K = 6


def create_dict(event_dict, temp_type):
    event_type = event_dict.pop('event_type')
    event_trigger = event_dict.pop('event_trigger')
    my_json = {}
    if(temp_type=="pred"):
        temp_type = "predicted_triggers"
    else:
        temp_type = "gold_triggers"
    my_json[temp_type] = {}
    my_json[temp_type]["arguments"] = []
    if(my_json.get(temp_type) is None):
        my_json[temp_type] = []
    my_json[temp_type]["span"] = re.sub('[ \t]+', ' ', event_trigger.replace(',', ' ,').replace('?', ' ?').replace(".", " .").strip())
    my_json[temp_type]["type"] = event_type
    my_json[temp_type]["indices"] = []

    for arg_key, arg_value in event_dict.items():
        arg_key, arg_value = arg_key.strip(), arg_value.strip()
        if(arg_key==arg_value):
            arg_value = ""
            continue#there is no arg in the template, so skip
        my_json[temp_type]["arguments"].append({"span": re.sub('[ \t]+', ' ', arg_value.replace(',', ' ,').replace('?', ' ?').replace(".", " .").strip()), "type":arg_key})
    return my_json  





def extract_args(template, delimiter = "|"):
        template = template.split()
        extracted_args = []
        idx = 0
        while(idx<len(template)):
            arg = []
            if(template[idx]==delimiter):
                idx += 1
                while(idx<len(template) and template[idx]!=delimiter):
                    arg.append(template[idx])
                    idx += 1
            if(len(arg)>0):
                extracted_args.append(re.sub('[ \t]+', ' ', ' '.join(arg).replace('|', "")))
            idx+=1
        return extracted_args

def extract_args_from_template(generated_templates, only_events=False):
        ret_outputs = []
        import re
        all_templates = generated_templates.split(eot)
        for template in all_templates:
            if(template.strip()==""):
                continue
            template = template.replace(bot, '').strip()
            event_template = r'Event (.+?) is triggered by \| (.+?) \| where , '
            event_trigger = re.match(event_template, template)
            try:
                event, trigger = event_trigger.groups()
                if(event is None):
                    event = ""
                if(trigger is None):
                    trigger = "" 
                dummy_event = template_function_call[event]({}, '', 'trigger')
            except Exception as e:
                print("Exception: ", e)
                continue#if the template is faulty or if there is a mistake in event name/trigger no need to check further

            if(not only_events):
                replace_string = event_template.replace('(.+?)', '{}').replace('\\', '').format(*(event_trigger.groups()))
                template = template.replace(replace_string, '')
                candidate_templates = dummy_event.masked_templates
                candidate_templates = [(re.sub('[ \t]+', ' ', x[0]), x[1]) for x in candidate_templates]
                template = re.sub('[ \t]+', ' ', template).replace("|,", "| ,")
                for candidate_template in candidate_templates:
                    ct = re.sub("[ \t]+", " ", candidate_template[0].replace(",", " ,").replace("?", " ?").replace(".", " ."))
                    ct = ct.replace('{}', '\| (.+?) \|')
                    matches = re.match(r'{}'.format(ct), template)
                    if(matches is not None):
                        attrib = candidate_template[1]
                        masked_template = re.sub('[ \t]+', ' ', template_function_call[event]({}, attrib, 'trigger').fill_template().replace(",", " ,").replace("?", " ?").replace(".", " .").strip())
                        attrib_names = extract_args(masked_template)
                        attrib_valus = extract_args(template)
                        display_output = '### '.join([(x + ":" + y) for x, y in zip(attrib_names, attrib_valus)])
                        ret_dict = dict([(x, y) for x, y in zip(attrib_names, attrib_valus)])
                        ret_dict['event_trigger'] = trigger
                        ret_dict['event_type'] = event
                        ret_outputs.append(ret_dict)
                        break
            else:
                ret_outputs.append(event_trigger.groups())
        return ret_outputs

import secrets

def extract_final(ops, pops):
    event_dict = {}
    for op in ops:
        formatted_json_gold = create_dict(op, "gold")
        if(event_dict.get("gold_triggers") is None):
            event_dict["gold_triggers"] = []
        event_dict["gold_triggers"].append(formatted_json_gold.pop("gold_triggers"))  
    if(event_dict.get("gold_triggers") is None):
        event_dict["gold_triggers"] = {}
    for pop in pops:#copy_dict:
        formatted_json_pred = create_dict(pop, "pred")
        if(event_dict.get("predicted_triggers") is None):
            event_dict["predicted_triggers"] = []
        event_dict["predicted_triggers"].append(formatted_json_pred.pop("predicted_triggers"))
    if(event_dict.get("predicted_triggers") is None):
        event_dict["predicted_triggers"] = {}
    return event_dict

import copy as cp
def sort_train_data_events_args(train_json_file):
    new_sorted_list = []
    for t_idx, train_prompt in enumerate(json.load(open(train_json_file))):
        train_prompt_, train_email_body = train_prompt["prompt"], train_prompt["email_body"]
        train_parts = train_prompt_.split(OUTPUT_DELIMITER)
        train_input, train_output = train_parts[0].strip(), train_parts[1].strip()
        trig_events = extract_args_from_template(train_output)#the events that we get from current train sample
        all_covered_eve_arg, events_covered_, args_covered=[], [], []
        for events_covered in trig_events:
            copy = cp.deepcopy(events_covered)
            event_covered = copy.pop("event_type")
            trigger = copy.pop("event_trigger")
            args_ = copy.keys()
            all_covered_eve_arg.extend(list(events_covered.keys()))
            events_covered_.append(event_covered)
            args_covered.extend(args_)
        all_covered_by_this = set(all_covered_eve_arg)
        train_prompt['covered_things'] = all_covered_by_this
        train_prompt["events_covered"] = list(set(events_covered_))
        train_prompt["args_covered"] = list(set(args_covered))
        new_sorted_list.append(train_prompt)
    return new_sorted_list 


def truncate_example(example, len_train_prompts):
    encoded_email = enc.encode(example)
    email_length = len(encoded_email)#we have this much token in email

    word_context_length, word_current_email_length = len("Context:"), len("Current Email:")

    allowed_test_prompt_len = 4000 - max_gen_tokens - len_train_prompts #so we fill the space of 4000 with prompts first and then number of tokens we can generate
    #whatever left is should be used for truncating test email
    truncated_email = encoded_email[-allowed_test_prompt_len:]
    decoded_test_example = enc.decode(truncated_email)
    if(decoded_test_example.find("Current Email:")<0):
        truncated_email = enc.encode("Context:\nCurrent Email:\n") + truncated_email[word_current_email_length:]
        decoded_test_example = enc.decode(truncated_email)
    if(decoded_test_example.find("Context:")<0):
        truncated_email = enc.encode("Context:\n") + truncated_email[word_context_length+word_current_email_length:]
        decoded_test_example = enc.decode(truncated_email)
    return decoded_test_example


from pprint import pprint
def prepare_prompt(train_json_file, test_json_file, model = 'davinci'):
    eval_dict = []
    global count_dict, K
    final_prompts_file = {}
    test_json = json.load(open(test_json_file))
    train_json = sorted(sort_train_data_events_args(train_json_file), key = lambda x: len(x["covered_things"]), reverse = True)
    covered = list({})
    train_id = 0
    discarded_set, accepted_set = [], []
    while(len(covered)<len(count_dict)):
        remaining_events = list(set([x for x in count_dict if x not in covered]))
        best_index, max_covers, min_length = -1, -1, 10000
        for t_idx, t in enumerate(train_json):
            json_covers = list(set(t["events_covered"]))
            events_this_can_cover = len(set([x for x in json_covers if x not in remaining_events]))
            if(events_this_can_cover>max_covers and len(t["prompt"]) < min_length):
                max_covers = events_this_can_cover
                best_index = t_idx
                min_length = len(t["prompt"])
        if(all([x in covered for x in set(train_json[best_index]["events_covered"])])):
            discarded_set.append(train_json[best_index])
            train_json.pop(best_index)
            continue
        covered.extend(set(train_json[best_index]["events_covered"]))
        covered = list(set(covered))
        accepted_set.append(train_json.pop(best_index))
        print("Covered: ", train_json[best_index]["events_covered"])
        train_id+=1
    print(f"It took {train_id} prompts to cover all events, covers is {covered}")
    need_truncation = 0
    max_test_prompt_length = 0
    for test_idx, test_prompt in enumerate(test_json):
        final_prompt = "Your task is to extract events from the Current Email, along with their corresponding triggers and arguments. You will need to identify the templates for each event and fill in the missing information with the extracted triggers and arguments. The unfilled templates are:\n"
        test_final_prompt = ""
        test_prompt, test_email_body = test_prompt["prompt"], test_prompt["email_body"]
        test_parts = test_prompt.split(OUTPUT_DELIMITER)
        test_input, test_output = test_parts[0].strip(), test_parts[1].strip()
        test_final_prompt += test_input
        test_final_prompt += "\nThe filled templates are:\n"
        test_final_prompt = test_final_prompt.replace("\n ", "\n")

        f = open("data/prompt_data.txt", "r")
        final_prompt  = "\n".join([line.strip() for line in f])
        train_prompts = copy.deepcopy(final_prompt)

        final_prompt = re.sub('[ \t]+',' ', final_prompt).replace(" \n ", '\n')

        final_prompt = re.sub('[ \t]+', ' ', final_prompt) + '\n' + test_final_prompt
        final_prompt = final_prompt.replace("\n ", "\n")

        if(len(enc.encode(final_prompt))>3700):

            truncated_test_data = truncate_example(test_input, len(enc.encode(train_prompts)))
            final_prompt = re.sub('[ \t]+',' ', train_prompts).replace(" \n ", '\n')
            final_prompt = re.sub('[ \t]+', ' ', final_prompt) + '\n' + truncated_test_data + "\nThe filled templates are:\n"
        final_prompt = final_prompt.replace("\n ", "\n")
        max_test_prompt_length = max(max_test_prompt_length, len(enc.encode(final_prompt)))
        print(final_prompt)
        print('-'*100)
        print(f"Done {test_idx+1} out of {len(test_json)}")
        try:
            output = get_response(final_prompt, model)
            if(model=='davinci'):
                output = outut['choices'][0]['text']
            else:
                output = output['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            output = ""
        print(output)
        print('Extracting from template: ')
        pops = extract_args_from_template(output, only_events=False)#takes all generated/input templates at once
        print(pops)
        print('Extracting from gold template: ')
        ops = extract_args_from_template(test_output, only_events=False)#takes all generated/input templates at once
        print(ops)
        event_dict = extract_final(ops, pops)
        event_dict["generated_prompt"] = final_prompt
        event_dict["reference_templates"] = test_output
        event_dict["generated_templates"] = output
        ebody = test_email_body.replace('?', ' ?').replace(".", " .").replace("'", " '").strip().split()

        current_email = ebody[((len(ebody) - ebody[-1::-1].index("[CONTEXT]") - 1) if "[CONTEXT]" in ebody else -1) +1:]
        event_dict["email_body"] = current_email
        eval_dict.append(event_dict)

    with open(f"prompts_output_{model}.json", "w") as f:
        json.dump(eval_dict, f, indent = 4)
    evaluate.trigger_scores(eval_dict)


import openai
from time import sleep
import backoff
from openai.error import RateLimitError

my_api_key = "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"

openai.api_key = my_api_key



@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff_davinci(prompt):
    got_res = False
    res = {'choices': []}
    try_time = 0
    while got_res is False and try_time < 100:
        try_time += 1
        try:
            res = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n\n"],
                n=1
            )
            got_res = True
        except RateLimitError:
            sleep(3)
        except Exception:
            sleep(3)
    return res


@backoff.on_exception(backoff.expo, RateLimitError)
def completion_with_backoff(prompt):
    got_res = False
    res = {"choices": []}
    try_time  = 0
    while not got_res and try_time<5:
        try_time += 1
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant to perform the text completion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=294,
                stop=["\n\n"])
            got_res = True
        except RateLimitError as e1:
            print('E1 message: ', e1)
            sleep(21)
        except Exception as e2:
            print("E2 message: ", e2)
            sleep(21)
    return res


def get_response(story, model):
    my_test_prompt = f"{story}"
    if(model=='davinci'):
        return completions_with_backoff_davinci(my_test_prompt)
    else:
        return completion_with_backoff(my_test_prompt)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model",type=str,required=True,help="GPT model to experiment with")
if __name__ == '__main__':
    prepare_prompt('data/icl_data/train_prompt_data.json', 'data/icl_data/dev_prompt_data.json')

