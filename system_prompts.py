TABC_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    " The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

TAC_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and analyzes its confidence about the solution and then provides the user with the final answer as well as its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

TABC_LONG_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level. "
    "The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags."
    "The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags."
    "Here are some guidelines for the analysis: "
    "1. Your task is to point out things where the model could be wrong in its thinking, or things where there might be ambiguity in the solution steps, or in the reasoning process itself.\n" 
    "2. You should not suggest ways of fixing the response, your job is only to reason about uncertainties.\n"
    "3. For some questions, the response might be correct. In these cases, It is also okay to have only a small number of uncertainties and then explictly say that I am unable to spot more uncertainties.\n"
    "4. Uncertainties might be different from errors. For example, uncertainties may arise from ambiguities in the question, or from the application of a particular lemma/proof. \n"
    "5. If there are alternate potential approaches that may lead to different answers, you should mention them.\n"
    "6. List out plausible uncertainties, do not make generic statements, be as specific about uncertainties as possible.\n"
    "7. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
    "The final format that must be followed is : <think> reasoning process here </think><answer> final answer here </answer> <analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>"
)

TABC_LONG_ALIGN_PROMPT = (
    "A conversation between a User and an Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process internally, then provides the user with a final answer, "
    "and analyzes its confidence level regarding the reasoning and solution.\n"
    "The final answer must be enclosed between <answer> </answer> tags. "
    "The analysis about confidence and uncertainty must be enclosed within <analysis> </analysis> tags. "
    "Two confidence scores must be provided:\n"
    "- Reasoning confidence, enclosed within <reasoning_confidence> </reasoning_confidence> tags.\n"
    "- Answer confidence, enclosed within <answer_confidence> </answer_confidence> tags.\n"
    "Reasoning confidence represents how truthfully and accurately the model reasons about the given question. "
    "Answer confidence represents how accurately the model answers the given question based on its reasoning process. "
    "Each confidence score must be a number between 0 and 1 (inclusive).\n"
    "Here are some guidelines for the analysis:\n"
    "1. Your task is to point out aspects where ambiguity may exist in the solution steps or in the reasoning process itself.\n"
    "2. You should NOT suggest ways to fix the response. Your role is only to analyze uncertainties.\n"
    "3. In some cases the response might be correct. In such cases, it is acceptable to identify only a small number "
    "of uncertainties and explicitly state that you are unable to find additional uncertainties.\n"
    "4. Uncertainties may differ from errors. For example, uncertainties may arise from ambiguities in the question "
    "or from the application of a particular lemma or proof.\n"
    "5. List plausible uncertainties. Avoid generic statements and be as specific as possible when describing uncertainties.\n"
    "6. The uncertainty analysis must be enclosed within <analysis> </analysis> tags.\n"
    "The final format that must be followed is:\n"
    "<think> reasoning process here </think>"
    "<answer> final answer here </answer>"
    "<analysis> analysis about confidence and uncertainty here </analysis>"
    "<reasoning_confidence> confidence level here (number between 0 and 1) </reasoning_confidence>"
    "<answer_confidence> confidence level here (number between 0 and 1) </answer_confidence>"
)

TABC_ALIGN_PROMPT = (
    "A conversation between a User and an Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process internally, then provides the user with a final answer, "
    "and analyzes its confidence level regarding the reasoning and solution.\n"
    "The final answer must be enclosed between <answer> </answer> tags. "
    "The analysis about confidence and uncertainty must be enclosed within <analysis> </analysis> tags. "
    "Two confidence scores must be provided:\n"
    "- Reasoning confidence, enclosed within <reasoning_confidence> </reasoning_confidence> tags.\n"
    "- Answer confidence, enclosed within <answer_confidence> </answer_confidence> tags.\n"
    "Reasoning confidence represents how truthfully and accurately the model reasons about the given question. "
    "Answer confidence represents how accurately the model answers the given question based on its reasoning process. "
    "Each confidence score must be a number between 0 and 1 (inclusive).\n"
    "The final format that must be followed is:\n"
    "<think> reasoning process here </think>"
    "<answer> final answer here </answer>"
    "<analysis> analysis about confidence and uncertainty here </analysis>"
    "<reasoning_confidence> confidence level here (number between 0 and 1) </reasoning_confidence>"
    "<answer_confidence> confidence level here (number between 0 and 1) </answer_confidence>"
)

GEN_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

DEEPSEEK_VERIFIER_PROMPT = (
    "You are given a question and a solution to it. You have to verify if the solution is correct and enclose your verification reasoning within <analysis> </analysis> tags. Your analysis should be a minimum of 300 characters and should sequentially go through the thinking solution step by step. Here are the guidelines for your analysis - "
    "1. Your analysis should also be in 'I' form as if you wrote the solution and are now verifying it. \n"
    "2. Your goal is not to solve the problem but instead to verify if the steps in the presented solution are correct. \n"
    "3. If there are ambiguities in the solution steps or if step introduces uncertainty, you should mention it in the analysis. \n"
    "4. Go through the solution sequentially in a step-by-step manner. \n"
    "5. The analysis should be 300 characters minimum. \n"
    "6. Enclose this uncertainty analysis within <analysis> </analysis> tags.\n"
)



def get_sys_prompt(sys_prompt_name):
    if sys_prompt_name == "gen":
        return GEN_PROMPT
    elif sys_prompt_name == "tac":
        return TAC_PROMPT
    elif sys_prompt_name == "tabc":
        return TABC_PROMPT
    elif sys_prompt_name == "tabc_long":
        return TABC_LONG_PROMPT
    elif sys_prompt_name == "deepseek_verifier":
        return DEEPSEEK_VERIFIER_PROMPT
    elif sys_prompt_name == "tabc_long_align":
        return TABC_LONG_ALIGN_PROMPT
    elif sys_prompt_name == "tabc_align":
        return TABC_ALIGN_PROMPT
    else:
        raise ValueError(f"Invalid system prompt name: {sys_prompt_name}")
    
