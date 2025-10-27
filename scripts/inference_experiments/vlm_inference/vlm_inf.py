
import sys, os, torch, argparse, json, math, time, gzip
from vllm.utils import cuda_device_count_stateless
import PIL, vllm, base64
from tqdm import tqdm
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Load dotenv
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
login(token=os.getenv("HUGGINGFACE_TOKEN"))

print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())

# Set torch multiprocessing to spawn
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Disable warnings
import warnings
warnings.filterwarnings("ignore")


####################################################################################################################
############################################# HELPER FUNCTIONS #####################################################
####################################################################################################################

def file_to_data_url(file_path: str):
    """
    Convert a local image file to a data URL.
    """    
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    _, extension = os.path.splitext(file_path)
    mime_type = f"image/{extension[1:].lower()}"
    
    return f"data:{mime_type};base64,{encoded_string}"


def make_image_prompt_dict(qaps_file):
    all_dfs = []

    ### Load QAPS File - ../realWorld_data_processing/realWorld_datasets/qaps/realWorld_HCT_qaps.json
    with open(qaps_file, "r") as f:
        qaps_list = json.load(f)
    
    all_qaps_dict = {}
    for table_entry in qaps_list:
        relative_path_to_table_image = table_entry["table_info"]["table_image_local_path"]
        for question_entry in table_entry['questions']:
            qap_id_t = question_entry['question_id']
            if os.path.exists(relative_path_to_table_image):
                all_qaps_dict[qap_id_t] = {
                    "image_path": relative_path_to_table_image,
                    "question": question_entry["question"]
                }
            
            elif os.path.exists(os.path.join("/".join(relative_path_to_table_image.split("/")[:1]), relative_path_to_table_image.split("/")[:1].replace("_","--"))):
                all_qaps_dict[qap_id_t] = {
                    "image_path": os.path.join("/".join(relative_path_to_table_image.split("/")[:1]), relative_path_to_table_image.split("/")[:1].replace("_","--")),
                    "question": question_entry["question"]
                }
            else:
                raise Exception("Image file not found:", relative_path_to_table_image)


    print("*** TOTAL QAPS: ***", len(all_qaps_dict))
    return all_qaps_dict


def do_inference(model_t, output_folder, qaps_file, num_gpus=1):

    # Ensure output_folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if model_t == "meta-llama/Llama-3.2-11B-Vision-Instruct": # Requires specific CUDA version to run with VLLM
        llm = vllm.LLM(model=model_t, 
                    tensor_parallel_size = num_gpus,
                    max_model_len=4096,
                    gpu_memory_utilization=0.8,
                    trust_remote_code=True,
                    max_num_seqs=16,
                    enforce_eager=True
                    )
    elif model_t == "mistralai/Pixtral-12B-2409":
        llm = vllm.LLM(model=model_t, 
                    trust_remote_code=True, 
                    tokenizer_mode="mistral", 
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size = num_gpus
                    )

    else:
        llm = vllm.LLM(model=model_t, 
                    trust_remote_code=True, 
                    gpu_memory_utilization=0.9,
                    tensor_parallel_size = num_gpus
                    )
    
    ### INFERENCE 
    
    ############################################################################################
    ############################################ QWEN
    ############################################################################################
    ### Load all question and image paths 
    all_qap_info_dict = make_image_prompt_dict(qaps_file)
    output_file_name = os.path.join(output_folder, f"{model_t.split('/')[-1]}--vision--results.jsonl")

    if model_t == "Qwen/Qwen2-VL-7B-Instruct":

        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info

        # Qwen does better with higher temperature and less max tokens
        sampling_params = vllm.SamplingParams(temperature=0.5, max_tokens=128)

        # Load the tokenizer
        processor = AutoProcessor.from_pretrained(model_t, trust_remote_code=True)

        #  Loop over all images and questions
        count = 0
        all_outputs_buffer = []
        for qap_id, qap_info in tqdm(all_qap_info_dict.items()):
            image_path_t = qap_info["image_path"]
            question_t = qap_info["question"]

            # Load image
            base64_image = base64.b64encode(open(image_path_t, "rb").read()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/jpeg;base64,{base64_image}",
                        },
                        {"type": "text", "text": f"You are a table question answering assistant. You are given an image of a table and a question. Your task is to answer the question based on the information in the table. The table structure may be complex and not a standard relational table so try to understand the structure of the table when answering the question. If the question cannot be answered using information from the table, return 'No Answer'. Do not provide any explanations, equations, code, or text explaining intermediate steps in figuring out the answer. Return only the final answer itself. If there are multiple values, separate each value with the ' || ' token. Question: {question_t}"},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference
            output = llm.generate(vllm.inputs.TokensPrompt(prompt_token_ids = inputs["input_ids"].flatten().tolist()), sampling_params=sampling_params)
            all_outputs_buffer.append({
                "id": qap_id,
                "response": output[0].outputs[0].text
            })

            # if buffer is size 50, write to file
            if count % 50 == 0:
                with open(output_file_name, "a") as f:
                    for o in all_outputs_buffer:
                        f.write(json.dumps(o) + "\n")
                all_outputs_buffer = []
                print(f"COMPLETED: {count} out of {len(all_qap_info_dict)} prompts - {(count*100)/len(all_qap_info_dict)}")

            count += 1


    ############################################################################################
    ############################################ PHI
    ############################################################################################

    ### Special Processing for microsoft/Phi-3-vision-128k-instruct | microsoft/Phi-3.5-vision-instruct
    elif model_t == "microsoft/Phi-3-vision-128k-instruct" or model_t == "microsoft/Phi-3.5-vision-instruct":

        all_outputs_buffer = []
        sampling_params = vllm.SamplingParams(temperature=0.01, max_tokens=128)
        
        count = 0
        for qap_id, qap_info in tqdm(all_qap_info_dict.items()): 
            image_path_t = qap_info["image_path"]
            question_t = qap_info["question"]

            ## PROMPTS
            prompt = f"USER: <|image_1|>\ You are a table question answering assistant. You are given an image of a table and a question. Your task is to answer the question based on the information in the table. The table structure may be complex and not a standard relational table so try to understand the structure of the table when answering the question. If the question cannot be answered using information from the table, return 'No Answer'. Do not provide any explanations, equations, code, or text explaining intermediate steps in figuring out the answer. Return only the final answer itself. If there are multiple values, separate each value with the ' || ' token. Question: {question_t} \nASSISTANT:"
            
            image = PIL.Image.open(image_path_t)

            # Single prompt inference
            outputs = llm.generate({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                },
                sampling_params=sampling_params
            )

            for o in outputs:
                generated_text = o.outputs[0].text
                all_outputs_buffer.append({
                    "id": qap_id,
                    "response": generated_text
                })

            # if buffer is size 50, write to file
            if count % 50 == 0:
                with open(output_file_name, "a") as f:
                    for o in all_outputs_buffer:
                        f.write(json.dumps(o) + "\n")
                all_outputs_buffer = []
                print(f"COMPLETED: {count} out of {len(all_qap_info_dict)} prompts - {(count*100)/len(all_qap_info_dict)}")

            count += 1

    ############################################################################################
    ############################################ PIXTRAL
    ############################################################################################

    ### Special Processing for mistralai/Pixtral-12B-2409

    elif model_t == "mistralai/Pixtral-12B-2409":

        all_outputs_buffer = []

        sampling_params = vllm.SamplingParams(temperature=0.01, max_tokens=128)
        count = 0

        for qap_id, qap_info in tqdm(all_qap_info_dict.items()):
            image_path_t = qap_info["image_path"]
            question_t = qap_info["question"]

            image_source = file_to_data_url(image_path_t)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"You are a table question answering assistant. You are given an image of a table and a question. Your task is to answer the question based on the information in the table. The table structure may be complex and not a standard relational table so try to understand the structure of the table when answering the question. If the question cannot be answered using information from the table, return 'No Answer'. Do not provide any explanations, equations, code, or text explaining intermediate steps in figuring out the answer. Return only the final answer itself. If there are multiple values, separate each value with the ' || ' token. Question: {question_t}"}, {"type": "image_url", "image_url": {"url": image_source}}]
                },
            ]

            outputs = llm.chat(messages, sampling_params=sampling_params)
            all_outputs_buffer.append({
                "id": qap_id,
                "response": outputs[0].outputs[0].text
            })
            
            # if buffer is size 50, write to file
            if count % 50 == 0:
                with open(output_file_name, "a") as f:
                    for o in all_outputs_buffer:
                        f.write(json.dumps(o) + "\n")
                all_outputs_buffer = []
                print(f"COMPLETED: {count} out of {len(all_qap_info_dict)} prompts - {(count*100)/len(all_qap_info_dict)}")

            count += 1
    
    ############################################################################################
    ############################################ GENERAL MODELS
    ############################################################################################

    elif model_t in [
            "HuggingFaceM4/Idefics3-8B-Llama3",
            "OpenGVLab/InternVL2_5-8B-MPO",
            "OpenGVLab/InternVL2-4B",
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "allenai/Molmo-7B-D-0924",
            "google/paligemma2-10b-ft-docci-448"
        ]:
        sampling_params = vllm.SamplingParams(temperature=0.01, max_tokens=128)
        # sampling_params = vllm.SamplingParams(temperature=0.1, max_tokens=128)

        all_outputs_buffer = []
        count = 0

        for qap_id, qap_info in tqdm(all_qap_info_dict.items()):
            image_path_t = qap_info["image_path"]
            question_t = qap_info["question"]

            prompt = f"USER: <image>\n You are a table question answering assistant. You are given an image of a table and a question. Your task is to answer the question based on the information in the table. The table structure may be complex and not a standard relational table so try to understand the structure of the table when answering the question. If the question cannot be answered using information from the table, return 'No Answer'. Do not provide any explanations, equations, code, or text explaining intermediate steps in figuring out the answer. Return only the final answer itself. If there are multiple values, separate each value with the ' || ' token. Question: {question_t} \nASSISTANT:"
            
            image = PIL.Image.open(image_path_t)

            # Single prompt inference
            outputs = llm.generate({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                },
                sampling_params=sampling_params
            )

            for o in outputs:
                generated_text = o.outputs[0].text
                all_outputs_buffer.append({
                    "id": qap_id,
                    "response": generated_text
                })

            # if buffer is size 50, write to file
            if count % 50 == 0:
                with open(output_file_name, "a") as f:
                    for o in all_outputs_buffer:
                        f.write(json.dumps(o) + "\n")
                all_outputs_buffer = []
                print("COMPLETED:", count)

            count += 1

    # Write remaining outputs to file
    with open(output_file_name, "a") as f:
        for o in all_outputs_buffer:
            f.write(json.dumps(o) + "\n")
    all_outputs_buffer = []

    return True


if __name__ == "__main__":
    
    all_models = [
        "OpenGVLab/InternVL2_5-8B-MPO",
        "OpenGVLab/InternVL2-4B",
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        "allenai/Molmo-7B-D-0924",
        "mistralai/Pixtral-12B-2409",
        "microsoft/Phi-3-vision-128k-instruct",
        "microsoft/Phi-3.5-vision-instruct",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to run inference on")
    if not parser.parse_args().model in all_models:
        print("Model not supported. Please choose from the following models:")
        for model in all_models:
            print(model)
        print("or choose 'all' to run all models.")
        sys.exit(1)
    parser.add_argument("--output_folder", type=str, default="../../results/model_responses/vlm/", help="Output folder to save results")
    parser.add_argument("--qaps_file", type=str, default = "../../datasets/realWorld_datasets/qaps/realWorld_HCT_qaps.json", help="Path to QAPS file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for inference. Most models require atleast 1 A100's to run.")

    args = parser.parse_args()

    # Check if qaps file exists
    if not os.path.exists(args.qaps_file):
        print("QAPS file not found. Please provide a valid path to the QAPS JSON file that is normally in the `datasets/realWorld_datasets/qaps` folder.")
        sys.exit(1)
    
    if args.model == "all":
        for model in all_models:
            do_inference(model, args.output_folder, args.qaps_file, args.num_gpus)
    else:
        do_inference(args.model, args.output_folder, args.qaps_file, args.num_gpus)

# Example command to run this script with a specific model
# python vllm_inference.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --num_gpus 2 