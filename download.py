
from datasets import load_dataset
load_dataset('wikitext', 'wikitext-2-v1', split='train')
load_dataset('wikitext', 'wikitext-2-v1', split='validation')


ds = load_dataset(
    "allenai/c4", "en",
    data_files={
        "train": [
            "en/c4-train.00000-of-01024.json.gz",
            "en/c4-train.00001-of-01024.json.gz",
            "en/c4-train.00002-of-01024.json.gz",
            "en/c4-train.00004-of-01024.json.gz",
            "en/c4-train.00005-of-01024.json.gz",
            # 添加更多文件...
        ],
        "validation": "en/c4-validation.00000-of-00008.json.gz",
    }, 
    verification_mode="no_checks",
)


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/OPT-1.3b-Chat")

model = AutoModelForCausalLM.from_pretrained("KoalaAI/OPT-1.3b-Chat")
