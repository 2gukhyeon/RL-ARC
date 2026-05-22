import math
import re
from math_verify import verify,parse
import numpy as np 
import string

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def format_reward(format_pattern,completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "tac":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    elif format_pattern == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "tabc_align":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<reasoning_confidence>.*?</reasoning_confidence>\s*<answer_confidence>.*?</answer_confidence>\s*\Z"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    if format_pattern == "tabc_align":
        confidence_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        matches = [1.0 if match else 0.0 for match in matches]
    else:
        confidence_pattern = r"<confidence>(.*?)</confidence>"
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        matches = [1.0 if match else 0.0 for match in matches]
    
    #if it matches, check if the confidence is between 0 and 1
    # for i,match in enumerate(matches):
    #     if match:
    #         content = completion_contents[i]
    #         if 'c' in format_pattern:
    #             confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <confidence>...</confidence> occurrences
    #             last_confidence = confidence_matches[-1] if confidence_matches else ""  # Get the last confidence, if exists
    #             if last_confidence == "":
    #                 matches[i] = 0.0
    #             else:
    #                 try:
    #                     confidence = float(last_confidence)
    #                     if confidence < 0 or confidence >1:
    #                         matches[i] = 0.0
    #                     else:
    #                         matches[i] = 1

    #                 except:
    #                     matches[i] = 0.0
    # if it matches, check if the confidence is between 0 and 1
    for i, match in enumerate(matches):
        if match:
            content = completion_contents[i]

            if 'c' in format_pattern:
                confidence_matches = re.findall(confidence_pattern, content, re.DOTALL | re.MULTILINE)

                if not confidence_matches:
                    matches[i] = 0.0
                    continue

                last_confidence = confidence_matches[-1]

                try:
                    # case 1: two confidences (tuple)
                    if isinstance(last_confidence, tuple):
                        reasoning_conf, answer_conf = last_confidence
                        reasoning_conf = float(reasoning_conf)
                        answer_conf = float(answer_conf)

                        if not (0 <= reasoning_conf <= 1 and 0 <= answer_conf <= 1):
                            matches[i] = 0.0
                        else:
                            matches[i] = 1

                    # case 2: single confidence
                    else:
                        confidence = float(last_confidence)

                        if 0 <= confidence <= 1:
                            matches[i] = 1
                        else:
                            matches[i] = 0.0

                except:
                    matches[i] = 0.0
    return matches

def accuracy_reward(format_pattern,completions,answer,source=None,**kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []
    format_rewards = format_reward(format_pattern,completions) 
    
    for content,e,fr in zip(completion_contents,eval_contents,format_rewards):
        if fr == 0:
            matches.append(0) 
        else:
            ans_matches = re.findall(ans_pattern, content, re.DOTALL | re.MULTILINE)  # Get all <answer>...</answer> occurrences
            last_answer = ans_matches[-1] if ans_matches else ""  # Get the last answer, if exists
            #if source exists in key and is equal to hotpot, then use the exact match score
            if source is not None and source[0] == 'hotpot':
                label = exact_match_score(last_answer,e)
            else:
                attempt = parse(last_answer)
                label = verify(e,attempt)
            matches.append(float(label))
    return matches

def brier_reward(format_pattern,completions,answer,source=None, **kwargs):
    """Reward function that checks if the completion is correct."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    single_conf_pattern = r"<confidence>(.*?)</confidence>"
    dual_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern,completions,answer,source) 
    format_rewards = format_reward(format_pattern,completions) 
    for content,cr,fr in zip(completion_contents,correctness_rewards,format_rewards):
        if fr == 0:
            matches.append(0)
            continue
        try:
            if format_pattern == "tabc_align": # two confidences
                confidence_matches = re.findall(dual_conf_pattern, content, re.DOTALL | re.MULTILINE)
                if not confidence_matches:
                    matches.append(0)
                    continue

                last_conf = confidence_matches[-1]

                reasoning_conf, answer_conf = last_conf
                conf = float(answer_conf)
                
            else: # baselines
                confidence_matches = re.findall(single_conf_pattern, content, re.DOTALL | re.MULTILINE)
                if not confidence_matches:
                    matches.append(0)
                    continue

                last_conf = confidence_matches[-1]
                conf = float(last_conf)

            if conf < 0 or conf > 1: # 실행가능성 낮음. 이미 format reawrd에서 걸러졌기 때문이다.
                matches.append(0)
                continue
            
            
            # brier = (cr - conf) ** 2
            brier = (cr - ((conf + float(reasoning_conf))*0.5)) ** 2 # arithmetic average


            # reasoning_conf = float(reasoning_conf)
            # geo_conf = math.sqrt(conf * reasoning_conf)
            # brier = (cr - geo_conf) ** 2 # geometic average
            
            reward = 1 - brier
            
            # if format_pattern == "tabc_align": # aligning reasoning and answer
            #     if cr > 0.5: # correct case
            #         align_weight = 0.4
            #         align_reward = align_weight * (conf - float(reasoning_conf))**2
            #         reward = 1 - brier - align_reward
            #     else: # incorrect case
            #         align_weight = 0
            #         align_reward = align_weight * (float(reasoning_conf)**2)
            #         reward = 1 - brier - align_reward
            matches.append(reward)

        except:
            print("Could not parse confidence:", content)
            matches.append(0)
    
    return matches

def mean_confidence_reward(format_pattern,completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    # confidence_pattern = r"<confidence>(.*?)</confidence>"
    single_conf_pattern = r"<confidence>(.*?)</confidence>"
    dual_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence = None
        if format_pattern == "tabc_align": # ours
            dual_matches = re.findall(dual_conf_pattern, content, re.DOTALL | re.MULTILINE)
            if dual_matches:
                _, answer_conf = dual_matches[-1]  # answer confidence 사용
                try:
                    confidence = float(answer_conf)
                    confidence = max(0.0, min(confidence, 1.0))
                except:
                    confidence = 0.0
            else:
                matches.append(0.0)
                continue

        else: # baseline (single confidence)
            single_matches = re.findall(single_conf_pattern, content, re.DOTALL | re.MULTILINE)
            if single_matches:
                try:
                    confidence = float(single_matches[-1])
                    confidence = max(0.0, min(confidence, 1.0))
                except:
                    confidence = 0.0
            else:
                matches.append(0.0)
                continue

        matches.append(confidence)
    return matches

def confidence_one_or_zero(format_pattern,completions,answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    single_conf_pattern = r"<confidence>(.*?)</confidence>"
    dual_conf_pattern = r"<reasoning_confidence>(.*?)</reasoning_confidence>\s*<answer_confidence>(.*?)</answer_confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer] 
    matches = []

    for content,e in zip(completion_contents,eval_contents):
        confidence = None

        # ours
        if format_pattern == "tabc_align": # two confidences
            dual_matches = re.findall(dual_conf_pattern, content, re.DOTALL | re.MULTILINE)
            
            if dual_matches:
                _, answer_conf = dual_matches[-1] # answer에 대해서만 평가를 진행한다.
                try:
                    confidence = float(answer_conf)
                    confidence = max(0.0, min(confidence, 1.0))
                except:
                    confidence = 0.0
            else:
                matches.append(0.0)
                continue

        else: # baseline
            single_matches = re.findall(single_conf_pattern, content, re.DOTALL | re.MULTILINE)

            if single_matches:
                try:
                    confidence = float(single_matches[-1])
                    confidence = max(0.0, min(confidence, 1.0))
                except:
                    confidence = 0.0
            else:
                matches.append(0.0)
                continue
                
        if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
            matches.append(1.0)
        else:
            matches.append(0.0)
    return matches


if __name__ == '__main__':
    s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "
 
    pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z" 
    match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    print(match)
    print(match[0])
