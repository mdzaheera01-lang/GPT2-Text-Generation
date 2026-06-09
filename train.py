from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2 model...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Model loaded successfully!")

prompt = "Artificial Intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(generated_text)