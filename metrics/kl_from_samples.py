import torch
from tqdm import tqdm


def kl_divergence_from_pairs(prompts: list, responses: list, tokenizer, model, ref_tokenizer, ref_model, device):
    '''
    This is a function to compute the Kl divergence between two models by directly over prompt response pairs from the SFT model.
    prompts: a list of prompts
    responses: a list of responses generated from the new model corresponding to each prompt
    model: the new model
    tokenizer: the tokenizer of the new model
    ref_model: the reference model
    ref_tokenizer: the tokenizer of the reference model
    '''
    # Initialize the divergence
    # total_kl_divergence_reference, total_kl_divergence = 0, 0
    kl_divergence_list = []

    # Iteration
    with torch.no_grad():
         pbar = tqdm.tqdm(enumerate(zip(prompts,responses)), total=len(prompts),desc='Compute KL divergence...')
         for _, batch in pbar:
            #  print(batch)
             prompt, response = batch
             
             # The length of the prompt
             prompt_tensor = tokenizer(prompt, return_tensors='pt')
             prompt_len = prompt_tensor['input_ids'].shape[1]
             
             # Tokenize the whole sentence
             # Get log probabilities for the generated samples
             try:
                 model_inputs = tokenizer(prompt+response, return_tensors='pt')
                 model_inputs.to(device)
                 model_outputs = model(**model_inputs)
                 model_log_probs_all = model_outputs.logits.log_softmax(-1)
                 model_log_probs = torch.gather(model_log_probs_all[:,:-1,:], 2, model_inputs['input_ids'][:,1:].unsqueeze(2)).squeeze()
                 
                 ref_inputs = ref_tokenizer(prompt+response, return_tensors='pt')
                 ref_inputs.to(device)
                 ref_outputs = ref_model(**ref_inputs)
                 ref_log_probs_all = ref_outputs.logits.log_softmax(-1)
                 ref_log_probs = torch.gather(ref_log_probs_all[:,:-1,:], 2, ref_inputs['input_ids'][:,1:].unsqueeze(2)).squeeze()
                 
                 # Calculate kl-divergence
                 log_ratio = model_log_probs[(prompt_len-1):] - ref_log_probs[(prompt_len-1):]
                 kl_divergence = log_ratio.sum().item()  
                 kl_divergence_list.append(kl_divergence)
             except:
                 print(prompt+response)
                
    # Compute averages
    average_kl_divergence = sum(kl_divergence_list) / len(kl_divergence_list)
    print(f'Total counts: {len(kl_divergence_list)}')

    return average_kl_divergence, kl_divergence_list