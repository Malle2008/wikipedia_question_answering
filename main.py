import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

wikipedia.set_lang("sv")
querry = input("context: ")
wiki_data = wikipedia.summary(querry, 4)


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = wiki_data
print(wiki_data)

while True:
    question = input("question: ")

    questions = [
        question,
    ]

    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = torch.argmax(answer_start_scores)
        # Get the most likely end of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )

        print(f"Question: {question}")
        print(f"Answer: {answer}")


#I want to test git