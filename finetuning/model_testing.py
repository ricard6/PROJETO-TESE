import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model
model_path = "./stance_ft_bart_v1/final-model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label mapping
stance_labels = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}

def predict_stance(topic, argument):
    input_text = f"Topic: {topic} Argument: {argument}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=384)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probs).item()

    print(f"🔹 **Topic:** {topic}")
    print(f"🗣 **Argument:** {argument}")
    print(f"🎯 **Predicted Stance:** {stance_labels[predicted_class]}")
    print(f"📊 **Confidence:** {probs[predicted_class].item():.4f}")
    print(f"🔢 **All Probabilities:** {dict(zip(stance_labels.values(), probs.tolist()))}")
    print("-" * 80)

test_examples = [
    # Climate Change
    ("Climate Change", "Many countries are setting ambitious climate goals, but the real challenge is implementing them effectively."),
    ("Climate Change", "Climate change is a hoax pushed by scientists for funding."),
    ("Climate Change", "There is no doubt that human activities contribute to global warming."),
    ("Climate Change", "While global temperatures have risen, it’s important to consider natural climate cycles before making drastic policy decisions."),

    # Abortion
    ("Abortion", "Women should have the right to make decisions about their own bodies."),
    ("Abortion", "Abortion is the same as murder and should be banned."),
    ("Abortion", "Some religious traditions believe that life begins at conception, which is why abortion is controversial."),
    ("Abortion", "The abortion debate is complex, with legal, ethical, and medical considerations playing a role."),

    # Gun Control
    ("Gun Control", "Stricter gun laws will reduce crime and make society safer."),
    ("Gun Control", "The Second Amendment guarantees the right to bear arms, and gun control is unconstitutional."),
    ("Gun Control", "Gun violence statistics show a correlation between gun availability and homicide rates."),
    ("Gun Control", "Different states have different gun laws, and crime rates vary regardless of restrictions."),

    # Artificial Intelligence
    ("Artificial Intelligence", "AI is making our lives easier by automating tedious tasks."),
    ("Artificial Intelligence", "AI will destroy jobs and lead to mass unemployment."),
    ("Artificial Intelligence", "Some experts say AI could surpass human intelligence within the next few decades."),
    ("Artificial Intelligence", "The impact of AI on jobs is uncertain, with some roles being replaced while others are created."),
]
    

# Run model on test examples
for topic, argument in test_examples:
    predict_stance(topic, argument)
