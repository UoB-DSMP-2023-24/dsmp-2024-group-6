# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only")

sequence = "C S V A R A G G R R E T Q Y F"

# 使用tokenizer.encode方法
encoded_sequence = tokenizer.encode(sequence, add_special_tokens=True)

# 使用tokenizer.encode_plus方法，这会提供更多的信息，比如attention mask
encoded_dict = tokenizer.encode_plus(
    sequence,
    add_special_tokens=True,  # 添加special tokens，比如[BOS]和[EOS]
    max_length=512,  # 设定最大序列长度
    padding='max_length',  # 进行填充到最大长度
    return_attention_mask=True,  # 返回attention mask
    return_tensors='pt',  # 返回PyTorch tensors
)

# 输出编码结果
print("Encoded Sequence:", encoded_sequence)
print("Encoded Dictionary:", encoded_dict)

