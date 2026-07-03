from datasets import load_dataset

# dataset = load_dataset("STARRY3056/ChnSentiCorp")
dataset = load_dataset("clue/clue", "tnews")
print(dataset)
print(dataset["train"][0])
print(dataset["train"].features)
print(dataset["train"].column_names)


