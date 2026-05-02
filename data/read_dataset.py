from datasets import Dataset


# dataset = Dataset.from_file("./trivia/test/data-00000-of-00001.arrow") # trivia
# dataset = Dataset.from_file("./gpqa/test/data-00000-of-00001.arrow") # trivia
dataset = Dataset.from_file("./hotpot_qa_vanilla/test/data-00000-of-00001.arrow") # hotpotqa
# dataset = Dataset.from_file("./") # hotpotqa

# print(load_and_clean(dataset[0]["problem"]))      # 첫 샘플
print(dataset[:5])     # 5개 샘플