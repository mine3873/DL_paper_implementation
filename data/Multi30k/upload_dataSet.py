from datasets import load_dataset

raw_datasets = load_dataset("bentrevett/multi30k")

print(raw_datasets)

print(raw_datasets["train"][0])

def save_corpus_to_file(dataset, lang, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(item[lang] + '\n')
            
save_corpus_to_file(raw_datasets["train"], "en", "train.en")
save_corpus_to_file(raw_datasets["train"], "de", "train.de")

save_corpus_to_file(raw_datasets["validation"], "en", "val.en")
save_corpus_to_file(raw_datasets["validation"], "de", "val.de")

save_corpus_to_file(raw_datasets["test"], "en", "test.en")
save_corpus_to_file(raw_datasets["test"], "de", "test.de")