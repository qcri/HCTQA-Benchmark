import json
import re, os
import argparse
import pandas as pd

import json
import re, os
import pandas as pd

##########################
######### HELPER FUNCTIONS
##########################

def clean_gt_val(ground_truth_t):
    ground_truth = re.sub(r"(?<=\d),(?=\d)", "", ground_truth_t)
    ground_truth = ground_truth.lower().strip()
    ground_truth = [x.lower().strip() for x in ground_truth.replace("{", "").replace("}", "").replace("||", "|").split("|")]
    for j in range(len(ground_truth)):
        temp_val = ground_truth[j]
        for _ in range(5):
            temp_val = re.sub(r'(\d+),(\d+)', r'\1\2', temp_val)

        # Remove trailing .0, .00, .000, etc.
        temp_val = re.sub(r'(\d+)\.0+\b', r'\1', temp_val)
        temp_val = re.sub(r'(\d*\.\d*?[1-9])0+\b', r'\1', temp_val)

        ground_truth[j] = temp_val

    return ground_truth

import re

def split_label_number_for_gpt(text: str):
    """
    Split only if:
      - The substring before ':' ends with an alphabetic word (A–Z/a–z only)
      - There is at least one whitespace after ':'
      - Followed by a valid int/float (optional '-')
    Otherwise return [text].
    """
    text = text.strip()
    m = re.match(r"^([A-Za-z][A-Za-z\s]*[A-Za-z])\s*:\s+(-?\d+(?:\.\d+)?)$", text)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return [text]

def post_process_response(response: str, mode = None) -> str:
    response = response.lower().strip()
    
    # Define junk tokens
    junk_tokens = r"(?:<eos_token>|#input|# input|# solution:|#solution|# explanation:|#explanation|note:|table:)"
    match = re.search(junk_tokens, response)

    # Remove "answer=", "answer:", "answer is"
    response = re.sub(r'answer\s*[:=]\s*', '', response, flags=re.IGNORECASE)
    
    # Truncate before junk tokens
    if match:
        response = response[:match.start()]

    # Handle newline splits
    response = response.split("\n\n")[0]
    response = response.split("\n```\n")[0]

    # Remove backticks and newlines
    response = response.replace("`", "").replace("\n", "")

    # Remove commas in numbers (1,000 -> 1000)
    for _ in range(5):
        response = re.sub(r'(\d+),(\d+)', r'\1\2', response)

    # Also remove stray commas (e.g., "6249," or ",6249")
    response = re.sub(r'(^,|,$)', '', response.strip())

    # Remove trailing .0, .00, etc.
    response = re.sub(r'(\d+)\.0+\b', r'\1', response)
    response = re.sub(r'(\d*\.\d*?[1-9])0+\b', r'\1', response)

    response = [x.strip() for x in response.lower().strip().split("||")]

    if mode == "gpt":
        final_response = []
        for resp in response:
            split_resp = split_label_number_for_gpt(resp)
            final_response.extend(split_resp)
        return final_response
    
    elif mode == "hitab":
        # remove "%" from each response
        final_response = []
        for resp in response:
            resp = resp.replace("%", "").strip()
            final_response.append(resp)
        return final_response
    else:
        return response
    
def get_prec_rec_f1_cc(results, mode = None):

    cleaned_responses = []
    cleaned_gts = []
    for x in results:
        og_output = x.get('output', x.get('response', ''))
        og_gt = x.get('gt', x.get('ground_truth', ''))
        if og_output is None or og_output.lower().strip().startswith("no answer") or og_output.lower().strip().startswith("error"):
            continue
        
        gt_clean = clean_gt_val(og_gt)
        resp_clean = post_process_response(og_output, mode=mode)
        if mode == "hitab":
            if (
                len(gt_clean) == 1 and len(resp_clean) == 1
                and re.fullmatch(r'-?\d+(\.\d+)?', gt_clean[0])
                and re.fullmatch(r'-?\d+(\.\d+)?', resp_clean[0])
            ):
                gt_val = gt_clean[0].strip()
                resp_val = resp_clean[0].strip()
                gt_neg = gt_val.startswith('-')
                resp_neg = resp_val.startswith('-')

                # (A) Fix sign mismatch
                if gt_neg != resp_neg:
                    if gt_neg:
                        gt_clean[0] = gt_val.lstrip('-')
                    else:
                        resp_clean[0] = resp_val.lstrip('-')

                # (B) Fix percentage mismatch (0.x vs integer)
                try:
                    gt_f, resp_f = float(gt_clean[0]), float(resp_clean[0])
                    if (0 < gt_f < 1 and resp_f >= 1) or (0 < resp_f < 1 and gt_f >= 1):
                        if 0 < gt_f < 1:
                            gt_clean[0] = str(round(gt_f * 100, 6)).rstrip('0').rstrip('.')
                        else:
                            resp_clean[0] = str(round(resp_f * 100, 6)).rstrip('0').rstrip('.')
                except ValueError:
                    pass

        cleaned_gts.append(gt_clean)
        cleaned_responses.append(resp_clean)

    # For each response and GT pair calculate precision and recall
    def calculate_precision_recall(gt, response):
        gt_set = set(gt)
        response_set = set(response)
        
        # Calculate precision and recall
        precision = sum(1 for x in response_set if x in gt_set) / max(len(response_set), 1)
        recall = sum(1 for x in gt_set if x in response_set) / max(len(gt_set), 1)
        
        return precision, recall

    def calculate_f1(precision, recall):
        if precision + recall == 0:
            return 0
        else:
            return (2 * precision * recall) / (precision + recall)

    # Calculate precision, recall, and F1 score for each response
    precision_list = []
    recall_list = []
    f1_list = []
    cc_list = []
    for i in range(len(cleaned_responses)):
        precision, recall = calculate_precision_recall(cleaned_gts[i], cleaned_responses[i])
        f1 = calculate_f1(precision, recall)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        if recall == 1:
            cc_list.append(1)
        else:
            cc_list.append(0)
    
    # Calculate average precision, recall, F1 and cc score
    avg_precision = sum(precision_list) / max(len(precision_list), 1)
    avg_recall = sum(recall_list) / max(len(recall_list), 1)
    avg_f1 = sum(f1_list) / max(len(f1_list), 1)
    avg_cc = sum(cc_list) / max(len(cc_list), 1)

    return avg_precision, avg_recall, avg_f1, avg_cc
    
def results_to_per_dataset_results(results):
    # Create a dictionary to hold the results for each dataset
    dataset_results = {}

    # Iterate through the results and group them by dataset
    for result in results:
        dataset = result['id'].split("--")[0]
        if dataset not in dataset_results:
            dataset_results[dataset] = [result]
        else:
            dataset_results[dataset].append(result)
    return dataset_results


############################
######### MAIN SCORING FUNCS
############################

def score_results_main(folder_name, output_file, print_results=False):
    output_string = ""

    for fname in os.listdir(folder_name):
        with open(os.path.join(folder_name, fname)) as f:
            results = json.load(f)

        model_name = fname.split("--")[0]
        data_mode = fname.split("--")[1]
        if "gpt" in model_name.lower():
            mode = "gpt"
        elif "hitab" in model_name.lower():
            mode = "hitab"
        else:
            mode = None
        mean_precision, mean_recall, mean_f1, mean_cc = get_prec_rec_f1_cc(results, mode=mode)

        output_string += f"Model: {model_name} on {data_mode} on dataset ALL\n"
        output_string += f"Mean Precision: {mean_precision:.4f}\n"
        output_string += f"Mean Recall: {mean_recall:.4f}\n"
        output_string += f"Mean F1 Score: {mean_f1:.4f}\n"
        output_string += f"Mean CC Score: {mean_cc:.4f}\n"
        output_string += "*" * 50

        dataset_results = results_to_per_dataset_results(results)

        for dataset, results_d in dataset_results.items():
            model_name = fname.split("--")[0]
            data_mode = fname.split("--")[1]
            mean_precision, mean_recall, mean_f1, mean_cc = get_prec_rec_f1_cc(results_d)

            output_string += f"Model: {model_name} on {data_mode} on dataset {dataset}\n"
            output_string += f"Mean Precision: {mean_precision:.4f}\n"
            output_string += f"Mean Recall: {mean_recall:.4f}\n"
            output_string += f"Mean F1 Score: {mean_f1:.4f}\n"
            output_string += f"Mean CC Score: {mean_cc:.4f}\n"
            output_string += "*" * 50

    with open(output_file, "w") as f:
        f.write(output_string)

    if print_results:
        print(output_string)

    print(f"Results saved to {output_file}")
    
    return True

############ MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score model responses")
    parser.add_argument("--folder_name", type=str, required=True, help="Folder containing the model response files")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save the results")
    parser.add_argument("--print_results", action="store_true", help="Print results to console")
    args = parser.parse_args()
    folder_name = args.folder_name
    output_file = args.output_file
    print_results = args.print_results

    try:
        score_results_main(folder_name, output_file, print_results)
    except ValueError as e:
        print(f"Error during scoring: {e}")
    print(f"Results saved to {output_file}")

# Example usage:
# python score_responses.py --folder_name /path/to/folder --output_file /path/to/output.txt --print_results

