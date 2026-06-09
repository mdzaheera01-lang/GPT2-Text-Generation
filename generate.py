from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = input("Enter a prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.8
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n====================")
print("GENERATED OUTPUT")
print("====================\n")
print(result)