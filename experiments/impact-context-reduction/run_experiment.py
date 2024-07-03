import argparse
from sampler import RandomSampler, KateSampler
from dotenv import load_dotenv
import os

def main(model_id, sampler_name, k, with_class):
    
    print(f"Model ID: {model_id}")
    print(f"Sampler: {sampler_name}")
    print(f"k: {k}")
    print(f"with_class: {with_class}")

    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_TOKEN")



    if sampler_name == "random":
        sampler = RandomSampler(model_id=model_id, access_token=access_token, is_with_class=with_class)

    elif sampler_name == "kate":
        sampler = KateSampler(model_id=model_id, access_token=access_token, is_with_class=with_class)
    
    else:
        raise ValueError("Invalid sampler")


    print("Running experiment...")
    results = sampler.run(k=k)
    if with_class:
        results.to_csv(f"data/results_{sampler_name}_{k}_in_class.csv", index=False)
    else:
        results.to_csv(f"data/results_{sampler_name}_{k}_general.csv", index=False)
    print("Done!")
    return
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("-m", "--model_id", type=str, help="The model ID (string).")
    parser.add_argument("-s", "--sampler",type=str, help="The sampler (string).")
    parser.add_argument("-k", "--k", type=int, help="An integer k.")
    parser.add_argument("-c", "--with_class", type=bool, help="Whether to use class distances.")

    args = parser.parse_args()

    main(args.model_id, args.sampler, args.k, args.with_class)