import pandas as pd
import json, time, os, argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams

# load env file
import dotenv
dotenv.load_dotenv()

# login to huggingface
from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

def do_llm_inference(model_name_or_path, output_folder = "../../results/model_responses/llms/", data_source_type = "real", split_name = "all", batch_size = 32, num_gpus = 1, use_system_prompt = True, max_model_len = 4000):
    dataset = load_dataset("qcri-ai/HCTQA")
    
    # Convert each split to a pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Choose the split based on the split_name argument
    if split_name == "all":
        # Concatenate the DataFrames
        final_df = pd.concat([train_df, val_df, test_df], axis=0)
        final_df = final_df.reset_index(drop=True)
    elif split_name == "train":
        final_df = train_df
    elif split_name == "validation":
        final_df = val_df
    elif split_name == "test":
        final_df = test_df
    else:
        raise ValueError("Invalid split name. Choose from 'train', 'validation', 'test', or 'all'.")

    # If the data source type is "real", filter the DataFrame
    if data_source_type == "real":
        final_df = final_df[final_df["dataset_type"] == "realWorldHCTs"]
    elif data_source_type == "synthetic":
        final_df = final_df[final_df["dataset_type"] == "syntheticHCTs"]
    elif data_source_type == "all":
        pass
    else:
        raise ValueError("Invalid data source type. Choose from 'real', 'synthetic', or 'all'.")

    # Make batches
    batch_counter = 0
    test_batches = {}
    for i in range(0, final_df.shape[0], batch_size):
        test_batches[batch_counter] = {
            "ids" : [],
            "prompts": [],
            "prompts_without_system": [],
            "gts": []
        }
        for j in range(i, min(i + batch_size, len(final_df))):
            row = final_df.iloc[j]
            test_batches[batch_counter]["ids"].append(row['question_id'])
            test_batches[batch_counter]["prompts"].append(eval(row['prompt']))
            test_batches[batch_counter]["prompts_without_system"].append(eval(row['prompt_without_system']))
            test_batches[batch_counter]["gts"].append(row['answer'])

        batch_counter += 1

    
    # Initialize the LLM
    # special case for gemma-3
    try:
        if "gemma-3" in model_name:
            llm_model = LLM(model=model_name, gpu_memory_utilization=0.9, tensor_parallel_size=num_gpus, dtype="bfloat16", max_model_len=max_model_len)
        else:
            llm_model = LLM(model=model_name, gpu_memory_utilization=0.9, tensor_parallel_size=num_gpus, max_model_len=max_model_len)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model using VLLM (ensure enough GPU RAM and it is a VLLM supported model): {e}")

    # Initialize the sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128
    )

    model_responses = []
    print("Total Batches: ", len(test_batches))
    for i in range(0, len(test_batches), 1):
        s_time = time.time()
        if (use_system_prompt == False) or ("gemma" in model_name and "gemma-3" not in model_name): # gemma and gemm2 models do not support system prompts
            outputs = llm_model.chat(test_batches[i]["prompts_without_system"],
                    sampling_params=sampling_params,
                    use_tqdm=True)
        else:
            outputs = llm_model.chat(test_batches[i]["prompts"],
                    sampling_params=sampling_params,
                    use_tqdm=True)

        print(f"Batch {i} done of {len(test_batches)} in {time.time() - s_time} seconds")
        
        for j in range(len(outputs)):
            model_responses.append({
                "id" : test_batches[i]["ids"][j],
                "question" : test_batches[i]["prompts"][j][-1]["content"].split("Question:")[-1].split("?`")[0].strip() + "?",   
                "gt" : test_batches[i]["gts"][j],
                "response" : outputs[j].outputs[0].text
            })

    # Save the model responses to a JSON file
    with open(os.path.join(output_folder, f"{model_name.split('/')[-1]}--{data_source_type}HCTs--results.json"), 'w') as f:
        json.dump(model_responses, f)

    return True

if __name__ == "__main__":
    models_for_exps = {
        # Qwen
        "Qwen/Qwen1.5-7B-Chat" : 32768,
        "Qwen/Qwen2-7B-Instruct" : 32768,
        "Qwen/Qwen2.5-7B-Instruct" : 32768,
        "Qwen/Qwen2.5-72B-Instruct" : 32000,
        "Qwen/Qwen3-14B" : 32768,
        "Qwen/Qwen3-32B" : 32000, 
        # Llama
        "meta-llama/Meta-Llama-3-8B-Instruct" : 8000,
        "meta-llama/Llama-3.1-8B-Instruct" : 16000,
        "meta-llama/Llama-3.1-70B-Instruct" : 32000,
        "meta-llama/Llama-3.3-70B-Instruct" :  32000, 
        # # Gemma
        "google/gemma-2-9b-it" : 8192,
        "google/gemma-3-12b-it" : 32768,
        "google/gemma-2-27b-it" : 8000, 
        "google/gemma-3-27b-it" : 32000,
        # Mistal
        "mistralai/Mistral-7B-Instruct-v0.1" : 32768,
        "mistralai/Mistral-7B-Instruct-v0.2" : 32768,
        "mistralai/Mistral-7B-Instruct-v0.3" : 32768,
        "mistralai/Mathstral-7B-v0.1" : 32768,
        # Phi
        "microsoft/Phi-3-mini-4k-instruct" : 4096,
        "microsoft/Phi-3.5-vision-instruct" : 65536,
        "microsoft/Phi-4-mini-instruct" : 65536,
        # Misc
        "QCRI/Fanar-1-9B-Instruct" : 4096,
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM inference on HCT-QA dataset.")
    # required arguments - model_name_or_path
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model or model name.")
    # optional arguments
    parser.add_argument("--output_folder", type=str, default="../../results/model_responses/llms/", help="Output folder for model responses.")
    parser.add_argument("--data_source_type", type=str, default="real", choices=["real", "synthetic", "all"], help="Data source type to use for inference.")
    parser.add_argument("--split_name", type=str, default="all", choices=["train", "validation", "test", "all"], help="Split name to use for inference.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for inference.")
    parser.add_argument("--use_system_prompt", type=bool, default=True, help="Whether to use the system prompt or not.")
    args = parser.parse_args()

    if args.model_name_or_path == "all":
        for model_name, max_model_len in models_for_exps.items():
            do_llm_inference(model_name, args.output_folder, args.data_source_type, args.split_name, args.batch_size, args.num_gpus, args.use_system_prompt, max_model_len)
    else:
        if args.model_name_or_path not in models_for_exps:
            print(f"Model {args.model_name_or_path} not found in predefined models. Please provide a valid model name.")
            raise ValueError(f"Model {args.model_name_or_path} not found in predefined models.")
        max_model_len = models_for_exps.get(args.model_name_or_path, 4000)
        do_llm_inference(args.model_name_or_path, args.output_folder, args.data_source_type, args.split_name, args.batch_size, args.num_gpus, args.use_system_prompt, max_model_len)

    



