
from build_data import *


if __name__ == '__main__':
    from argparse import ArgumentParser
    import pandas as pd
    import json
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

    parser = ArgumentParser()
    parser.add_argument('--maxlen', type=int, default=256, help='The max number of new tokens allowed')
    parser.add_argument('--batch_size', type=int, default=4, help='The sampling batch size')
    parser.add_argument('--prompt_set', type=str, default=None, help='The path of the prompt set')
    parser.add_argument('--n_seq', type=str, default='3,6,8', help='The sequence of n')
    parser.add_argument('--ref_model_dir', type=str, default='EleutherAI/pythia-2.8b', help='The path of the original reference model')
    parser.add_argument('--ref_weight_dir', type=str, default=None, help='The weight path of the reference model')
    parser.add_argument('--reward_model_dir', type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2', help='The path of the reward model')
    parser.add_argument('--save_data_path', type=str, default=None, help='The path to save the best and worst data')

    args = parser.parse_args()

    # specify the device
    NUM_OF_GPUS = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(device)
    
    # load the dataset
    df = pd.read_csv(args.prompt_set)
    prompts = df['Prompt'].tolist()

    # the sequence of n for best of n
    args.n_seq = [int(num) for num in args.n_seq.split(',')]

    
    # load the reference model and its tokenizer
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_dir)
    if args.ref_weight_dir:
        checkpoint = torch.load(args.ref_weight_dir)
        ref_model.load_state_dict(checkpoint['state'])

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_dir, padding=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0,
              "do_sample": True, "pad_token_id": tokenizer.eos_token_id, 'temperature': 1.0}

    # load the reward model
    rw_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_dir)
    rw_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_dir, padding_side='right')

    # Move all the models to gpus
    if NUM_OF_GPUS > 1:
        print(f"Using {NUM_OF_GPUS} GPUs!")
        ref_model = nn.DataParallel(ref_model)
        rw_model = nn.DataParallel(rw_model)
    ref_model.to(device)
    rw_model.to(device)
    ref_model.eval()
    rw_model.eval()
    print('Model loaded.')

    # run the experiment 
    if NUM_OF_GPUS > 1: 
        best_worst_ds = best_and_worst_of_n_sampler(prompts, batch_size=args.batch_size, max_len=args.maxlen, 
                                                    model = ref_model.module, tokenizer = tokenizer, rw_model = rw_model.module, rw_tokenizer= rw_tokenizer,
                                                    device = device, gen_kwargs = gen_kwargs, n_seq=args.n_seq)
    else:
        best_worst_ds = best_and_worst_of_n_sampler(prompts, batch_size=args.batch_size, max_len=args.maxlen, 
                                                    model = ref_model, tokenizer = tokenizer, rw_model = rw_model, rw_tokenizer= rw_tokenizer,
                                                    device = device, gen_kwargs = gen_kwargs, n_seq=args.n_seq)

    # save the data     
    for pn in args.n_seq:
        dict_to_save = dict(best_worst_ds[pn])

        full_filename = f'{args.save_data_path}/best-of-{pn}/bon_maxlen_{args.maxlen}.jsonl'
        with open(full_filename, 'w') as f:
            for key, value in dict_to_save.items():
                json_record = json.dumps({key: value})
                f.write(json_record + '\n')

        print(f'Best of {pn} data saved to {full_filename}')

    quit()