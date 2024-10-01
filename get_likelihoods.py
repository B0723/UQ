import argparse
import os
import pickle
import random
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForTokenClassification,pipeline
from tqdm import tqdm
from openai import OpenAI
#import wandb
client = OpenAI(api_key='sk-8f11a034d9b949948b9a9233cdc7a9a3',base_url='https://api.deepseek.com/v1')
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_model', type=str, default='opt-350m')
parser.add_argument('--generation_model', type=str, default='opt-350m')
parser.add_argument('--run_id', type=str, default='run_1')
args = parser.parse_args()
my_prompt='''
Extract the keywords from the following sentence. The keywords should directly capture the core and main idea of the sentence, using words exactly as they appear in the text. If there are no specific keywords to extract, return a list containing the entire sentence as a single element. You should use JSON format to your response.And the key named keywords
'''
device = 'cuda'
import config

# Set a seed value
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value

random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)

#Fix torch random seed
torch.manual_seed(seed_value)

os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

model = AutoModelForCausalLM.from_pretrained("/data2/yxbu/semantic_uncertainty/code/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c",
                                             torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("/data2/yxbu/semantic_uncertainty/code/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c",
                                          use_fast=True)
#wandb.init(project='nlg_uncertainty', id=args.run_id, config=args, resume='allow')
run_name = args.run_id

opt_models = ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b', 'opt-13b', 'opt-30b']

with open(f'{config.output_dir}/sequences/{run_name}/{args.generation_model}_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{config.output_dir}/sequences/{run_name}/{args.generation_model}_generations_similarities.pkl', 'rb') as infile:
    similarities_dict = pickle.load(infile)

def format_apiresp(resp):
    p1 = "```json((.|\n)*)```"
    p2 = "```json"
    p3 = "```"
    resp = re.search(p1, resp).string
    resp = re.sub(p2, '', resp)
    resp = re.sub(p3, '', resp)
    return resp.strip()
        
def get_keyword_positions(keywords_list,sentence):
    tokens_with_offsets = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    keywords_positions = {}
    offsets = tokens_with_offsets['offset_mapping']
    for keyword in keywords_list:
        keyword_positions = []
        # 对关键词进行分词
        keyword_tokens = tokenizer.tokenize(keyword)
        keyword_token_length = len(keyword_tokens)

        # 遍历分词后的句子 tokens，查找与关键词匹配的序列
        for i in range(len(offsets) - keyword_token_length + 1):
            # 提取句子中对应长度的 token 和偏移量
            match_tokens = tokens_with_offsets['input_ids'][i:i + keyword_token_length]
            match_offsets = offsets[i:i + keyword_token_length]

            # 提取偏移量对应的子字符串并合并成一个整体
            match_substring = sentence[match_offsets[0][0]:match_offsets[-1][1]]

            # 判断是否和关键词相同
            if match_substring.replace(" ", "").lower() == keyword.replace(" ", "").lower():
                keyword_positions.extend(list(range(i, i + keyword_token_length)))
                break  # 找到一个匹配后就跳出循环
        
        # 存储结果
        keywords_positions[keyword] = keyword_positions
    return keywords_positions
def get_neg_loglikelihoods(model, sequences):

    with torch.no_grad():
        result = []
        for sample in tqdm(sequences):
            result_dict = {}
            prompt = sample['prompt']
            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            else:
                generations = sample['generations'].to(device)

            id_ = sample['id']

            if 'cleaned_generated_texts' in sample:
                generated_texts = sample['cleaned_generated_texts']
            else:
                generated_texts = sample['generated_texts']

            average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            average_unconditioned_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_log_likelihoods = torch.zeros((generations.shape[0],))
            neg_unconditioned_log_likelihoods = torch.zeros((generations.shape[0],))
            pointwise_mutual_information = torch.zeros((generations.shape[0],))
            new_average_neg_log_likelihoods = torch.zeros((generations.shape[0],))
            sequence_embeddings = []
            new_sequence_embeddings = []
            for generation_index in range(generations.shape[0]):
                prompt = prompt[prompt != tokenizer.pad_token_id]
                generation = generations[generation_index][generations[generation_index] != tokenizer.pad_token_id]
                generation_text = generated_texts[generation_index]
                response = client.chat.completions.create(
                    model = 'deepseek-chat',
                    messages=[
                        {'role':'system','content':f'{my_prompt}'},
                        {'role':'user','content':f'{generation_text}'},
                    ],
                    stream = False,
                    max_tokens = 2000
                )
                print(response.choices[0].message.content)
                with open('/data2/yxbu/semantic_uncertainty/code/output.txt','a') as file:
                    file.write(response.choices[0].message.content + '\n')
                try:
                    keywords_list = format_apiresp(response.choices[0].message.content)
                    keywords_pisition = get_keyword_positions(keywords_list,generation_text)
                    all_positions = [position for positions in keywords_pisition.values() for position in positions]
                    all_positions = [x + len(prompt) for x in all_positions]
                except:
                    all_positions = [i for i in range(len(generation_text))]
                    all_positions = [x + len(prompt) for x in all_positions]

                # This computation of the negative log likelihoods follows this tutorial: https://huggingface.co/docs/transformers/perplexity
                target_ids = generation.clone()
                new_target_ids = target_ids.clone()
                target_ids[:len(prompt)] = -100
                new_target_ids[:len(prompt)] = -100
                for i in range(len(prompt),len(generation)):
                    if i not in all_positions:
                        new_target_ids[i]=-100
                new_model_output = model(torch.reshape(generation,(1,-1)),labels=new_target_ids,output_hidden_states=True)
                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids, output_hidden_states=True)
                generation_only = generation.clone()[(len(prompt) - 1):]
                unconditioned_model_output = model(torch.reshape(generation_only, (1, -1)),
                                                   labels=generation_only,
                                                   output_hidden_states=True)
                hidden_states = model_output['hidden_states']
                new_hidden_states = new_model_output['hidden_states']
                average_neg_log_likelihood = model_output['loss']
                new_average_neg_log_likelihood = new_model_output['loss']
                average_unconditioned_neg_log_likelihood = unconditioned_model_output['loss']
                new_average_neg_log_likelihoods[generation_index] = new_average_neg_log_likelihood
                average_neg_log_likelihoods[generation_index] = average_neg_log_likelihood
                average_unconditioned_neg_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood
                neg_log_likelihoods[generation_index] = average_neg_log_likelihood * (len(generation) - len(prompt))
                neg_unconditioned_log_likelihoods[generation_index] = average_unconditioned_neg_log_likelihood * (
                    len(generation) - len(prompt))
                pointwise_mutual_information[generation_index] = -neg_log_likelihoods[
                    generation_index] + neg_unconditioned_log_likelihoods[generation_index]

                average_of_last_layer_token_embeddings = torch.mean(hidden_states[-1], dim=1)
                new_average_of_last_layer_token_embeddings = torch.mean(new_hidden_states[-1], dim=1)
                sequence_embeddings.append(average_of_last_layer_token_embeddings)
                new_sequence_embeddings.append(new_average_of_last_layer_token_embeddings)

            most_likely_generation = sample['most_likely_generation_ids'].to(device)
            target_ids = most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_most_likely_gen = model_output['loss']
            most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            second_most_likely_generation = sample['second_most_likely_generation_ids'].to(device)
            target_ids = second_most_likely_generation.clone()
            target_ids[:len(prompt)] = -100
            model_output = model(torch.reshape(second_most_likely_generation, (1, -1)),
                                 labels=target_ids,
                                 output_hidden_states=True)
            hidden_states = model_output['hidden_states']
            average_neg_log_likelihood_of_second_most_likely_gen = model_output['loss']
            second_most_likely_generation_embedding = torch.mean(hidden_states[-1], dim=1)

            neg_log_likelihood_of_most_likely_gen = average_neg_log_likelihood_of_most_likely_gen * (
                len(most_likely_generation) - len(prompt))

            sequence_embeddings = torch.stack(sequence_embeddings)
            result_dict['prompt'] = prompt
            result_dict['generations'] = generations
            result_dict['average_neg_log_likelihoods'] = average_neg_log_likelihoods
            result_dict['new_average_neg_log_likelihoods'] = new_average_neg_log_likelihoods
            result_dict['neg_log_likelihoods'] = neg_log_likelihoods
            result_dict['sequence_embeddings'] = most_likely_generation_embedding
            result_dict['new_sequence_embeddings'] = new_sequence_embeddings
            result_dict['most_likely_sequence_embedding'] = most_likely_generation
            result_dict['average_unconditioned_neg_log_likelihoods'] = average_unconditioned_neg_log_likelihoods
            result_dict['neg_unconditioned_log_likelihoods'] = neg_unconditioned_log_likelihoods
            result_dict['pointwise_mutual_information'] = pointwise_mutual_information
            result_dict['average_neg_log_likelihood_of_most_likely_gen'] = average_neg_log_likelihood_of_most_likely_gen
            result_dict[
                'average_neg_log_likelihood_of_second_most_likely_gen'] = average_neg_log_likelihood_of_second_most_likely_gen
            result_dict['neg_log_likelihood_of_most_likely_gen'] = neg_log_likelihood_of_most_likely_gen
            result_dict['semantic_set_ids'] = torch.tensor(similarities_dict[id_[0]]['semantic_set_ids'], device=device)
            result_dict['id'] = id_
            result.append(result_dict)

        return result


likelihoods = get_neg_loglikelihoods(model, sequences)

with open(f'{config.data_dir}/sequences/{run_name}/{args.generation_model}_generations_{args.evaluation_model}_likelihoods.pkl',
          'wb') as outfile:
    pickle.dump(likelihoods, outfile)
