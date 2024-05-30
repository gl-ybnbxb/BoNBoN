import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tqdm import tqdm
import json
import argparse


NUM_OF_GPUS = torch.cuda.device_count()
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)


# the reward model
reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_model_name), AutoTokenizer.from_pretrained(reward_model_name, padding_side='right')

# Move all the models to gpus
if NUM_OF_GPUS > 1:
    print(f"Using {NUM_OF_GPUS} GPUs!")
    reward_model = nn.DataParallel(reward_model)
reward_model.to(device)
print('Model loaded.')




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_modelA", type=str)
    parser.add_argument("--run_modelB", type=str)
    parser.add_argument("--exp_name", type=str)

    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    #path = os.path.join("outputs", f"{args.run_modelA}.json")
    with open(args.run_modelA, "r") as f:
        generations_modelA = f.readlines()

    #path = os.path.join("outputs", f"{args.run_modelB}.json")
    with open(args.run_modelB, "r") as h:
        generations_modelB = h.readlines()

    #selected_indices = random.sample(range(len(generations_red)), 100)
    #generations_A = [generations_modelA[i] for i in selected_indices]
    #generations_B = [generations_modelB[i] for i in selected_indices]

    assert(len(generations_modelA) == len(generations_modelB))

    evaluations = []
    win_A = 0
    win_B = 0
    tie = 0

    for gen_A, gen_B in tqdm(zip(generations_modelA, generations_modelB), total=len(generations_modelA)):
        a = json.loads(gen_A)
        b = json.loads(gen_B)

        assert (a.keys() == b.keys())
        prompt = list(a.keys())[0]

        
        response_modelA = a[prompt][0]
        response_modelB = b[prompt][0]

        """
        user_prompt = USER_PROMPT.format(query=prompt, response_A=response_modelA, response_B=response_modelB)
        prompt = f"{B_INST} {B_SYS}{SYS_MESSAGE}{E_SYS}{user_prompt} {E_INST}"
        
        outputs = get_output(model, tokenizer, prompt, device)[0]
        print("Outputs", outputs)

        model_decision = outputs[outputs.rfind("More helpful:") + len("More helpful:") : outputs.rfind("More helpful:") + len("More helpful:") + 2].strip()
        print("Model_decision", model_decision)
        """

        # print("Response A", response_modelA, type(response_modelA))
        # print("Response B", response_modelB, type(response_modelB))

        inputs_A = reward_tokenizer(prompt, response_modelA, return_tensors='pt')
        score_A = reward_model(**inputs_A).logits[0].cpu().detach()

        inputs_B = reward_tokenizer(prompt, response_modelB, return_tensors='pt')
        score_B = reward_model(**inputs_B).logits[0].cpu().detach()

        # print("Prompt", prompt)
        # print("Response A", response_modelA )
        # print("Score Response A", score_A.item())
        # print("Response B", response_modelB )
        # print("Score Response B", score_B.item())

        evaluations.append(
            {
                "prompt": prompt,
                "responseA": response_modelA,
                "responseB": response_modelB,
                "scoreA": score_A.item(),
                "scoreB": score_B.item(),
            },
        )

        win_A += (score_A.item() > score_B.item())
        win_B += (score_A.item() < score_B.item())
        tie += (score_A.item() == score_B.item())
        print(win_A, win_B, tie)


    result = {
        "run_A": args.run_modelA,
        "run_B": args.run_modelB,
        "win_A_percent": float(win_A/len(generations_modelA))*100,
        "win_B_percent": float(win_B/len(generations_modelA))*100,
        "tie_percent": float(tie/len(generations_modelA))*100,
        "win_A": win_A,
        "win_B": win_B,
        "tie": tie,
        "total_evaluations": len(generations_modelA),
        "evaluations": evaluations,
    }

    # eval_path = os.path.join("/net/projects/veitch/llm_alignment/data/HH/best-of-hh/samples_Antrophic_HH_test/reward_model_deberta_evaluations_updated", f"eval_{args.exp_name}.json")
    eval_path = f"eval_{args.exp_name}.json"
    json.dump(result, open(eval_path, "w"), indent=2)