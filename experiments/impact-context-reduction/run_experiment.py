import argparse
from sampler import RandomSampler, KateSampler
from dotenv import load_dotenv
import os
import torch
from transformers import BitsAndBytesConfig


def main(model_id, sampler_name, k, with_class, device):
    
    print(f"Model ID: {model_id}")
    print(f"Sampler: {sampler_name}")
    print(f"k: {k}")
    print(f"with_class: {with_class}")
    print(f"Device: {device}")

    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_TOKEN")



    if sampler_name == "random":
        sampler = RandomSampler(model_id=model_id, access_token=access_token, is_with_class=with_class)

    elif sampler_name == "kate":
        model_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
        sampler = KateSampler(model_id=model_id, access_token=access_token, is_with_class=with_class, model_kwargs=model_kwargs ,device=device)
        
    else:
        raise ValueError("Invalid sampler")


    print("Running experiment...")
    results = sampler.run(k=k)
    if with_class:
        results.to_csv(f"data/results_{sampler_name}_{k}_in_class.csv", index=False)
    else:
        results.to_csv(f"data/results_{sampler_name}_{k}_general.csv", index=False)
        print("Saved")
    print("Done!")
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("-m", "--model_id", type=str, help="The model ID (string).")
    parser.add_argument("-s", "--sampler",type=str, help="The sampler (string).")
    parser.add_argument("-k", "--k", type=int, help="An integer k.")
    parser.add_argument('--with_class', dest='with_class', action='store_true', help='A boolean flag')
    parser.add_argument("-d", "--device", type=str, help="Flag indicating if it's using qunatization.")
    parser.set_defaults(with_class=False)


    args = parser.parse_args()

    main(args.model_id, args.sampler, args.k, args.with_class, args.device)