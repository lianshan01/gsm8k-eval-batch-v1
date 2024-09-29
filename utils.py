import re
from transformers import StoppingCriteria


# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, inputs):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.inputs = inputs
        self.prompt_len = inputs['input_ids'].size(1)

    def __call__(self, input_ids, scores, **kwargs):
        current_texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        res = []
        for current_text in current_texts:
            res.append(any(stop_string in current_text for stop_string in self.stop_strings))
        return all(res)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()