"""
OpenAI caller.

Class:
     OpenAICaller
"""
from openai import OpenAI


class OpenAICaller:
    def __init__(self, api_key: str, role: str = "You are a helpful assistant.", model_name: str = "gpt-4o-mini",
                 verbose: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.role = role
        self.supported_models = self.model_name
        self.chat = model_name[0:3] == "gpt"
        self.verbose = verbose

    def _create_msg_from_input_prompt(self, input_prompt: str):
        return [
            {"role": "system", "content": self.role},
            {"role": "user", "content": input_prompt}
        ]

    def make_call(self, prompt: str, max_answer_length: int, temperature: int = 1):
        client = OpenAI(api_key=self.api_key)

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=self._create_msg_from_input_prompt(prompt),
            stream=False,
            temperature=temperature,
            max_tokens=max_answer_length,
        )
        client.api_key = None

        return completion.choices[0].message.content
