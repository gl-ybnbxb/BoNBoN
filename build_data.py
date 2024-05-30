
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import gc
import tqdm


# build the data loader
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, attention_masks):
        self.embeddings = embeddings
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.attention_masks[idx]

def build_dataloader(queries, tokenizer, batch_size=4, sampler='sequential'):
    query_tensor = tokenizer(queries, return_tensors='pt', padding=True)
    query_embedding_tensor = query_tensor['input_ids']
    query_attention_tensor = query_tensor['attention_mask']
    data = EmbeddingDataset(query_embedding_tensor, query_attention_tensor)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


# best and worst sampler
def best_and_worst_of_n_sampler(query_list, batch_size, max_len, 
                                model, tokenizer, rw_model, rw_tokenizer, device,
                                gen_kwargs, n_seq = [3,6,8], responses_best_and_worst=None):
    '''This is the function to sample the best and wrost sample'''
    # build the data loader
    dataloader = build_dataloader(query_list, tokenizer, batch_size=batch_size)
    print('Built the data loader!')

    # Initialize the response dictionary
    if not responses_best_and_worst:
        responses_best_and_worst = {}
        for k in n_seq:
            responses_best_and_worst[k] = defaultdict(lambda: defaultdict(list))
            
    # iterate within the dataset
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),desc='Generate data')
    for step, batch in pbar:
        emb_batch, mask_batch = batch
        curr_batch_size = len(emb_batch)

        # original queries in this batch
        query_batch_list = query_list[(step*batch_size):(step*batch_size+batch_size)]
        
        # build the dataset the same time for all n's and get the responses
        max_n = max(n_seq)
        embs_batch = emb_batch.repeat((max_n, 1))
        masks_batch = mask_batch.repeat((max_n, 1))
        print(embs_batch.shape)
        
        queries_len = embs_batch.shape[1] # length of prompts

        # move tensors to gpu
        embs_batch.to(device)
        masks_batch.to(device)

        # generate the responses
        output = model.generate(embs_batch, attention_mask=masks_batch, max_new_tokens=max_len, **gen_kwargs).squeeze()[:,queries_len:]
        responses = tokenizer.batch_decode(output, skip_special_tokens=True)
        print('Responses done.')
        
        # get the reward and the best of n samples for all n in n_seq together
        # batch tokenize and get rewards
        inputs = rw_tokenizer(query_batch_list*max_n, responses, return_tensors='pt', padding=True)
        inputs.to(device)
        scores_all = rw_model(**inputs).logits.cpu().detach() 
        scores_all = scores_all.reshape((max_n,curr_batch_size))
        print('Rewards done.')
        
        # get response pairs for each n
        for k in n_seq:
            scores_sub = scores_all[:k,:]
            min_indicies = scores_sub.argmin(dim=0)
            max_indicies = scores_sub.argmax(dim=0)
            # the response pairs: the former is the best response and the latter is the worst one
            response_bw_pairs = [[responses[i+curr_batch_size*max_indicies[i]], responses[i+curr_batch_size*min_indicies[i]]] for i in range(curr_batch_size)]
            
            # build the data as a dictionary
            for i, p in enumerate(query_batch_list):
                n_responses = len(responses_best_and_worst[k][p]['responses'])
                responses_best_and_worst[k][p]['pairs'].append((n_responses, n_responses + 1))
                responses_best_and_worst[k][p]['responses'].extend(response_bw_pairs[i])
                # responses_best_and_worst[k][p]['sft_target'] = response_bw_pairs[i][0]
            
        torch.cuda.empty_cache()
        gc.collect()  
    return responses_best_and_worst