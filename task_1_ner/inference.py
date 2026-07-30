import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class MountainPredictor:

    def __init__(self, model_path: str = None):

        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "saved_model")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model folder not found at the path: {model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text) -> dict:

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        id2label = self.model.config.id2label
        labels = [id2label[p.item()] for p in predictions]

        extracted_mountains = []
        current_mountain = []
        tokens_with_tags = []

        for token, label in zip(tokens, labels):

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            tokens_with_tags.append((token, label))

            clean_token = token[2:] if token.startswith('##') else token

            if token.startswith("##"):
                if current_mountain:
                    current_mountain[-1] += clean_token

            elif label == "B-MOUNTAIN":
                if current_mountain:
                    extracted_mountains.append(" ".join(current_mountain))
                    current_mountain = []
                current_mountain.append(clean_token)

            elif label == "I-MOUNTAIN" and current_mountain:
                current_mountain.append(clean_token)

            else:
                if current_mountain:
                    extracted_mountains.append(" ".join(current_mountain))
                    current_mountain = []

        if current_mountain:
            extracted_mountains.append(" ".join(current_mountain))
            current_mountain = []
            
        return {
            "text": text,
            "extracted_mountains": extracted_mountains,
            "tokens_with_tags": tokens_with_tags
        }

def main():
    predictor = MountainPredictor()

    test_sentences = [
        "Yesterday we reached the summit of Mount Everest.",
        "K2 is considered one of the most dangerous peaks.",
        "I enjoy walking in the city park with my dog.",
        "Mont Blanc offers stunning views of the Alps."
    ]

    for sentence in test_sentences:
        result = predictor.predict(sentence)
        print(result['text'])
        print(result['extracted_mountains'])

if __name__ == "__main__":
    main()