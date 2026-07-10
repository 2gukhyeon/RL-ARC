from eval.eval_utils import compute_pass_n, get_brier, get_ece, get_auroc, exact_match_score
import numpy as np
from math_verify import verify, parse
import re
from vllm import LLM, SamplingParams
import gc
from transformers import AutoTokenizer
from openai import OpenAI

client = OpenAI()
def confidence_extractor(response, **kwargs):  # answer confidence extractor
    """Extracts the confidence from the completions.

    Priority:
    1. <confidence>...</confidence>
    2. <answer_confidence>...</answer_confidence> from tabc_align
    3. Minimum value among all <step_confidence>...</step_confidence> tags
       from tabc_step

    Returns:
        (validity, confidence)
    """

    conf_pattern = r"<confidence>(.*?)</confidence>"

    # Get all <confidence>...</confidence> occurrences
    conf_matches = re.findall(
        conf_pattern,
        response,
        re.DOTALL | re.MULTILINE
    )

    # Get the last confidence, if exists
    last_confidence = conf_matches[-1] if conf_matches else ""

    # tabc_align
    if last_confidence == "":
        align_conf_pattern = (
            r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*"
            r"<answer_confidence>(.*?)</answer_confidence>"
        )

        dual_matches = re.findall(
            align_conf_pattern,
            response,
            re.DOTALL | re.MULTILINE
        )

        if dual_matches:
            _, last_confidence = dual_matches[-1]
        else:
            last_confidence = ""

    # tabc_step
    if last_confidence == "":
        step_conf_pattern = r"<step_confidence>(.*?)</step_confidence>"

        step_conf_matches = re.findall(
            step_conf_pattern,
            response,
            re.DOTALL | re.MULTILINE
        )

        if step_conf_matches:
            step_confidences = []

            for step_confidence in step_conf_matches:
                try:
                    confidence = float(step_confidence)

                    if confidence > 1 and confidence <= 100:
                        confidence = confidence / 100
                    elif confidence >= 0 and confidence <= 1:
                        confidence = confidence
                    else:
                        return 0, 0.0

                    step_confidences.append(confidence)

                except:
                    # Extract the first number in the step confidence string
                    first_number = re.search(
                        r'-?\d+(?:\.\d+)?',
                        step_confidence
                    )

                    if first_number:
                        confidence = float(first_number.group())

                        if confidence >= 0 and confidence <= 1:
                            confidence = confidence
                        elif confidence > 1 and confidence <= 100:
                            confidence = confidence / 100
                        else:
                            return 0, 0.0

                        step_confidences.append(confidence)

                    else:
                        return 0, 0.0

            if step_confidences:
                return 1, min(step_confidences)
            else:
                return 0, 0.0

    # Existing single-confidence handling
    if last_confidence == "":
        return 0, 0.0

    else:
        try:
            confidence = float(last_confidence)

            if confidence > 1 and confidence <= 100:
                return 1, confidence / 100

            elif confidence >= 0 and confidence <= 1:
                return 1, confidence

            else:
                return 0, 0.0

        except:
            # Extract the first number in the string
            first_number = re.search(
                r'-?\d+(?:\.\d+)?',
                last_confidence
            )

            if first_number:
                first_number = float(first_number.group())

                if first_number >= 0 and first_number <= 1:
                    return 1, first_number

                elif first_number > 1 and first_number <= 100:
                    return 1, first_number / 100

                else:
                    return 0, 0.0

            else:
                return 0, 0.0
            
# def confidence_extractor(response, **kwargs): # answer confidence extractor
#     """Extracts the confidence from the completions
#     If a float is found within confidence tags, it is processed as follows:
#     If the float is between 0 and 1, it is returned as is.
#     If the float is between 1 and 100, it is divided by 100 and returned.
#     If float is not directly found, the first number in the string is extracted and processed as above.
#     If no float is found, 0 is returned.    
#     """
#     conf_pattern = r"<confidence>(.*?)</confidence>"
#     # Get all <confidence>...</confidence> occurrences
#     conf_matches = re.findall(conf_pattern, response, re.DOTALL | re.MULTILINE)
#     # Get the last confidence, if exists
#     last_confidence = conf_matches[-1] if conf_matches else ""

#     if last_confidence == "": # for RLAR
#         align_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
#         dual_matches = re.findall(align_conf_pattern, response, re.DOTALL | re.MULTILINE)
#         if dual_matches:
#             _, last_confidence = dual_matches[-1]
#         else:
#             last_confidence = ""

#     if last_confidence == "":
#         return 0, 0.0
#     else:
#         try:
#             confidence = float(last_confidence)
#             if confidence > 1 and confidence <= 100:
#                 return 1, confidence/100
#             elif confidence >= 0 and confidence <= 1:
#                 return 1, confidence
#             else:
#                 return 0, 0.0
#         except:
#             # extract the first number in the string
#             first_number = re.search(r'-?\d+(?:\.\d+)?', last_confidence)
#             if first_number:
#                 first_number = float(first_number.group())
#                 if first_number >= 0 and first_number <= 1:
#                     return 1, first_number
#                 elif first_number > 1 and first_number <= 100:
#                     return 1, first_number/100
#                 else:
#                     return 0, 0.0
#             else:
#                 return 0, 0.0

def reasoning_confidence_extractor(response, **kwargs): # reasoning confidence extractor
    """Extracts the confidence from the completions
    If a float is found within confidence tags, it is processed as follows:
    If the float is between 0 and 1, it is returned as is.
    If the float is between 1 and 100, it is divided by 100 and returned.
    If float is not directly found, the first number in the string is extracted and processed as above.
    If no float is found, 0 is returned.    
    """
    reasoning_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>"
  
    reasoning_conf_matches = re.findall(reasoning_conf_pattern, response, re.DOTALL | re.MULTILINE)
    # Get the last confidence, if exists
    last_reasoning_confidence = reasoning_conf_matches[-1] if reasoning_conf_matches else ""

    if last_reasoning_confidence == "": # for RLAR
        align_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
        dual_matches = re.findall(align_conf_pattern, response, re.DOTALL | re.MULTILINE)
        if dual_matches:
            last_reasoning_confidence, _ = dual_matches[-1]
        else:
            last_reasoning_confidence = ""

    if last_reasoning_confidence == "":
        return 0, 0.0
    else:
        try:
            confidence = float(last_reasoning_confidence)
            if confidence > 1 and confidence <= 100:
                return 1, confidence/100
            elif confidence >= 0 and confidence <= 1:
                return 1, confidence
            else:
                return 0, 0.0
        except:
            # extract the first number in the string
            first_number = re.search(r'-?\d+(?:\.\d+)?', last_reasoning_confidence)
            if first_number:
                first_number = float(first_number.group())
                if first_number >= 0 and first_number <= 1:
                    return 1, first_number
                elif first_number > 1 and first_number <= 100:
                    return 1, first_number/100
                else:
                    return 0, 0.0
            else:
                return 0, 0.0

def gen_correctness_reward(completions, answer, **kwargs):
    """Reward function that checks if the answer is correct or not
    The answer must be present within the answer tags.
    For math datasets, the correctness is checked using huggingface math-verify.
    For factual datasets, the correctness is checked using exact match.

    """
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        # Get all <answer>...</answer> occurrences
        # print(content)
        ans_matches = re.findall(ans_pattern, content,
                                 re.DOTALL | re.MULTILINE)
        # Get the last answer, if exists
        last_answer = ans_matches[-1] if ans_matches else ""
        attempt = parse(last_answer)
        label = verify(e, attempt)
        if label ==0 :
            label = exact_match_score(last_answer, e)
        matches.append(float(label))

    return matches

def extract_answer(text):
    """
    # \boxed{204} → 204
    """
    if text is None:
        return None
    text = str(text)

    text = re.sub(r'\\boxed\{(\d+)\}', r'\1', text)

    nums = re.findall(r'-?\d+', text)
    return str(nums[-1]) if nums else None

def confidence_verifier(local_dataset, config, format_fn="confidence_format", format_pattern="tabc", **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    reasoning_confidence_levels = []
    reasoning_conf_format_levels = []
    metrics = {}
    n = config.n
    correctness_fn = gen_correctness_reward

    if f"{config.name}-class_output" in local_dataset.column_names:
        #If classification outputs are present (these come from classifier/probe)
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    ### CHECK CORRECTNESS ###

    for i in range(len(local_dataset)):
        eval_list, c_len_list, conf_list, conf_format_list, reasoning_conf_list, reasoning_conf_format_list  = [], [], [], [], [], []
        for j in range(n):
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            try:
                answer = local_dataset[i]["answer"]
            except:
                answer = extract_answer(local_dataset[i]["solution"]) # aime24
            pred = [{"role": "assistant", "content": pred_response}]

            args = {"completions": [pred], "answer": [answer]}

            actual_correctness = correctness_fn(**args)[0]
            conf_format, conf_level = confidence_extractor(pred_response)
            reasoning_conf_format, reasoning_conf_level = reasoning_confidence_extractor(pred_response)
            conf_format_list.append(conf_format)
            reasoning_conf_format_list.append(reasoning_conf_format)

            c_len_list.append(len(pred[0]["content"]))
            conf_list.append(conf_level)
            reasoning_conf_list.append(reasoning_conf_level)
            if actual_correctness == 1:
                eval_list.append(1)
            else:
                eval_list.append(0)

        evals.append(eval_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)
        reasoning_confidence_levels.append(reasoning_conf_list)
        reasoning_conf_format_levels.append(reasoning_conf_format_list)
  
    ### END OF CHECK CORRECTNESS ###
     
    ### COMPUTE PASS@K ###
    if n not in config.pass_k_vals:
        config.pass_k_vals.append(n)
    if 1 not in config.pass_k_vals:
        config.pass_k_vals.append(1)
    for k in config.pass_k_vals:
        if k <= n:
            pass_k, responses = compute_pass_n(evals, k)
            metrics[f"pass@{k}"] = pass_k

    ### END OF COMPUTE PASS@K ###

    if class_outputs is not None:
        #If classification outputs are present (these come from classifier/probe), then we use the corresponding confidence levels
        if type(class_outputs[0]) == list:
            confidence_levels = [[c[1]] for c in class_outputs]
            reasoning_confidence_levels = [[c[1]] for c in reasoning_class_outputs]
            print("Overriding confidence levels with classification outputs")
        else:
            confidence_levels = [ [c] for c in class_outputs]
            reasoning_confidence_levels = [[c[1]] for c in reasoning_class_outputs]

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-reasoning_confidence_levels"] = reasoning_confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels
    label_dict[f"{config.name}-reasoning_conf_format_adherence"] = reasoning_conf_format_levels

    correctness_array = np.array(evals).flatten()
    confidence_array = np.array(confidence_levels).flatten()
    reasoning_confidence_array = np.array(reasoning_confidence_levels).flatten()
    # the metrics related to answer confidence
    metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)
    
    # the metrics related to reasoning confidence
    metrics["brier_score (r)"] = get_brier(correctness_array, reasoning_confidence_array) 
    metrics["ece (r)"] = get_ece(correctness_array, reasoning_confidence_array)
    metrics["auroc (r)"] = get_auroc(correctness_array, reasoning_confidence_array)
    

    metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["reasoning_confidence level"] = np.mean(np.array(reasoning_confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels))
    metrics["reasoning_confidence format adherence"] = np.mean(
        np.array(reasoning_conf_format_levels))

    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return label_dict, metrics, responses, reasoning_confidence_array

def llm_confidence_verifier(local_dataset, config, judge_model="meta-llama/Llama-3.1-8B-Instruct", format_fn="confidence_format", **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []
    reasoning_confidence_levels = []
    reasoning_conf_format_levels = []
    
    metrics = {}
    n = config.n

    if f"{config.name}-class_output" in local_dataset.column_names:
        class_outputs = local_dataset[f"{config.name}-class_output"]
    else:
        class_outputs = None

    # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS. 
    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for j in range(n):
            pred = local_dataset[i][f"{config.name}-output_{j}"]
            ans_pattern = r"<answer>(.*?)</answer>"
            # Get all <answer>...</answer> occurrences
            ans_matches = re.findall(
                ans_pattern, pred, re.DOTALL | re.MULTILINE)
            # Get the last answer, if exists
            last_answer = ans_matches[-1] if ans_matches else ""
            if last_answer == "":
                last_answer = "I don't know"
            q_spec_ans.append(last_answer)
        extracted_answers.append(q_spec_ans)

    ####### DO LLM AS JUDGE SETUP #######
    sys_prompt = """
    You are a judge that will be given a question,ground truth answers and a model generated answer. There might be multiple ground truth answers. 
    The model generated answer is correct if it matches any of the ground truth answers.
    You will need to determine if the model generated answer is correct or not. 
    Your response should be a single word. 'YES' if the answer is correct and 'NO' if it is not.
    """

    prompts = []
    chosen_key = "question" if "question" in local_dataset.column_names else "problem"
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    
    #Generate prompts for each example
    for i in range(len(local_dataset)):
        for j in range(n):
            prompt = f"""
            Question: {local_dataset[i][chosen_key]}
            Ground Truth Answers: {local_dataset[i]["answer"]}
            Model Generated Answer: {extracted_answers[i][j]}
            """
            processed_prompt = [{'role': 'system', 'content': sys_prompt}, {
                'role': 'user', 'content': prompt}]
            tokenized_prompt = tokenizer.apply_chat_template(
                processed_prompt, truncation=False, add_generation_prompt=True)
            decoded_prompt = tokenizer.decode(tokenized_prompt)
            prompts.append(decoded_prompt)

    # Setup LLM and send prompts
    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=20)
    llm = LLM(model=judge_model, gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    ####### END OF LLM AS JUDGE SETUP #######

    ####### AGGREGATE RESPONSES #######

    responses = []
    for output in outputs:
        text_r = output.outputs[0].text
        if "yes" in text_r.lower():
            responses.append(1)
        else:
            responses.append(0)

    agg_responses = []
    # agg responses by taking groups of n and making a list of them
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i:i+n])
    
    
    ####### END OF AGGREGATE RESPONSES #######

    # Compute accuracy
    accuracy = np.mean(responses)
    print(f"Accuracy of {config.name} = {accuracy}")

    for i in range(len(local_dataset)):
        eval_list, c_len_list, conf_list, conf_format_list, reasoning_conf_list, reasoning_conf_format_list = [], [], [], [], [], []
        for j in range(n):
            pred_response = local_dataset[i][f"{config.name}-output_{j}"]
            pred = [{"role": "assistant", "content": pred_response}]

            actual_correctness = agg_responses[i][j]
            conf_format, conf_level = confidence_extractor(pred_response)
            reasoning_conf_format, reasoning_conf_level = reasoning_confidence_extractor(pred_response)
            conf_format_list.append(conf_format)
            reasoning_conf_format_list.append(reasoning_conf_format)

            c_len_list.append(len(pred[0]["content"]))
            conf_list.append(conf_level)
            reasoning_conf_list.append(reasoning_conf_level)
            if actual_correctness == 1:
                eval_list.append(1)
            else:
                eval_list.append(0)

        evals.append(eval_list)
        c_lengths.append(c_len_list)
        confidence_levels.append(conf_list)
        conf_format_levels.append(conf_format_list)
        reasoning_confidence_levels.append(reasoning_conf_list)
        reasoning_conf_format_levels.append(reasoning_conf_format_list)


    if n not in config.pass_k_vals:
        config.pass_k_vals.append(n)
    if 1 not in config.pass_k_vals:
        config.pass_k_vals.append(1)
    for k in config.pass_k_vals:
        if k <= n:
            pass_k, _ = compute_pass_n(evals, k)
            metrics[f"pass@{k}"] = pass_k

    if class_outputs is not None:
        if type(class_outputs[0]) == list:
            confidence_levels = [[c[1]] for c in class_outputs]
            reasoning_confidence_levels = [[c[1]] for c in reasoning_class_outputs]
            print("Overriding confidence levels with class outputs")
        else:
            confidence_levels = [ [c] for c in class_outputs]
            reasoning_confidence_levels = [[c[1]] for c in reasoning_class_outputs]

    correctness_array = np.array(evals).flatten()
    confidence_array = np.array(confidence_levels).flatten()
    reasoning_confidence_array = np.array(reasoning_confidence_levels).flatten()
    metrics["brier_score"] = get_brier(correctness_array, confidence_array) 
    metrics["ece"] = get_ece(correctness_array, confidence_array)
    metrics["auroc"] = get_auroc(correctness_array, confidence_array)

    # the metrics related to reasoning confidence
    metrics["brier_score (r)"] = get_brier(correctness_array, reasoning_confidence_array) 
    metrics["ece (r)"] = get_ece(correctness_array, reasoning_confidence_array)
    metrics["auroc (r)"] = get_auroc(correctness_array, reasoning_confidence_array)

    # take mean of c_lengths
    c_length_mean = np.mean(np.array(c_lengths))

    label_dict[f"{config.name}-evals"] = evals
    label_dict[f"{config.name}-c_lengths"] = c_lengths
    label_dict[f"{config.name}-confidence_levels"] = confidence_levels
    label_dict[f"{config.name}-reasoning_confidence_levels"] = reasoning_confidence_levels
    label_dict[f"{config.name}-conf_format_adherence"] = conf_format_levels
    label_dict[f"{config.name}-reasoning_conf_format_adherence"] = reasoning_conf_format_levels

    metrics["accuracy"] = metrics["pass@1"]
    metrics["completion length"] = c_length_mean
    metrics["confidence level"] = np.mean(np.array(confidence_levels))
    metrics["reasoning_confidence level"] = np.mean(np.array(reasoning_confidence_levels))
    metrics["confidence format adherence"] = np.mean(
        np.array(conf_format_levels))
    metrics["reasoning_confidence format adherence"] = np.mean(
        np.array(reasoning_conf_format_levels))

    print(f"Metrics of {config.name} =")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    del llm
    gc.collect()
    return label_dict, metrics, responses, reasoning_confidence_array


def reasoning_verifier(local_dataset, config, judge_model="gpt-4o-mini",
                       format_fn="confidence_format",
                       **kwargs):
    label_dict = {f"{config.name}-evals": []}
    evals = []
    c_lengths = []
    confidence_levels = []
    conf_format_levels = []

    
    metrics = {}
    n = config.n

    # FIRST EXTRACT OUT ALL ANSWERS FROM THE MODEL OUTPUTS. 

    # for i in range(len(local_dataset)):
    #     q_spec_ans = []
    #     for j in range(n):
    #         pred = local_dataset[i][f"{config.name}-output_{j}"]
    #         ans_pattern = r"<answer>(.*?)</answer>"
    #         # Get all <answer>...</answer> occurrences
    #         ans_matches = re.findall(
    #             ans_pattern, pred, re.DOTALL | re.MULTILINE)
    #         # Get the last answer, if exists
    #         last_answer = ans_matches[-1] if ans_matches else ""
    #         if last_answer == "":
    #             last_answer = "I don't know"
    #         q_spec_ans.append(last_answer)
    #     extracted_answers.append(q_spec_ans)
    extracted_reasons = []
    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_rea = []
        q_spec_ans = []
        for j in range(n):
            pred = local_dataset[i][f"{config.name}-output_{j}"]
            
            ans_pattern = r"<answer>(.*?)</answer>"
            # Get all <answer>...</answer> occurrences
            ans_matches = re.findall(
                ans_pattern, pred, re.DOTALL | re.MULTILINE)
            # Get the last answer, if exists
            last_answer = ans_matches[-1] if ans_matches else ""
            if last_answer == "":
                last_answer = "I don't know"
            q_spec_ans.append(last_answer)
            
            r_pattern = r"<think>(.*?)</think>"
            rea_matches = re.findall(
                r_pattern, pred, re.DOTALL | re.MULTILINE)
            # Get the last answer, if exists
            last_reason = rea_matches[-1] if rea_matches else ""
            if last_reason == "":
                last_reason = "I don't know"
            q_spec_rea.append(last_reason)
        extracted_reasons.append(q_spec_rea)
        extracted_answers.append(q_spec_ans)

    ####### DO LLM AS JUDGE SETUP #######
    reasoning_sys_prompt = """
    You are a judge who will be given a question, ground truth answers, and a model-generated reasoning process.
    The model-generated reasoning (thinking) process is considered correct if all of its content is related to at least one of the ground truth answers and is truthful and logically valid for generating the model-generated answer.
    You need to determine whether the model-generated reasoning (thinking) process is correct or not.
    Your response should be a single word: 'YES' if the reasoning (thinking) process is correct, and 'NO' if it is not.
    """

    reasoning_responses = []
    chosen_key = "question" if "question" in local_dataset.column_names else "problem"

    #Generate prompts for each example
    for i in range(len(local_dataset)):
        for j in range(n):
            user_prompt = f"""
            Question: {local_dataset[i][chosen_key]}
            Ground Truth Answers: {local_dataset[i]["answer"]}
            Model Generated Reasoning Process: {extracted_reasons[i][j]}
            """
   
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    temperature=0,
                    max_tokens=10,
                    messages=[
                        {
                            "role": "system",
                            "content": reasoning_sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )

                text_r = (
                    response.choices[0]
                    .message.content
                    .strip()
                )

                if "yes" in text_r.lower():
                    reasoning_responses.append(1)
                else:
                    reasoning_responses.append(0)

            except Exception as e:
                print(f"Error at sample {i}, output {j}: {e}")
                reasoning_responses.append(0)
  
    ####### END OF AGGREGATE RESPONSES #######

    # Compute accuracy
    r_accuracy = np.mean(reasoning_responses)
    print(f"r_Accuracy of {config.name} = {r_accuracy}")
    metrics["reasoning_relevance_accuracy"] = r_accuracy

    return metrics, reasoning_responses
