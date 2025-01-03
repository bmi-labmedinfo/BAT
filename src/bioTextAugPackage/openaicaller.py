"""
OpenAI caller.

Class:
     OpenAICaller
"""
from openai import OpenAI
import tiktoken


class OpenAICaller:
    def __init__(self, api_key: str, role: str = "You are a helpful assistant.", model_name: str = "gpt-4o-mini",
                 max_cost=0.1, verbose: bool = False):
        self.api_key = api_key
        self.model_name = model_name
        self.max_cost = max_cost
        self.role = role
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        self.supported_models = self.model_name
        self.billing_history = 0
        self.chat = model_name[0:3] == "gpt"
        self.verbose = verbose

    def estimate_cost(self, input_prompt: str, max_tokens: int):
        cost_table = {
            self.model_name: {
                "input": 0.15,  # context window of 128K tokens
                "output": 0.60,  # 16K context
            }
        }
        output_cost = max_tokens * cost_table[self.model_name]["output"] / 1000000
        input_cost = self._num_tokens_from_messages(messages=self._create_msg_from_input_prompt(input_prompt)) * \
                     cost_table[self.model_name]["input"] / 1000000
        return input_cost + output_cost

    def _num_tokens_from_messages(self, messages):
        encoding = tiktoken.encoding_for_model(self.model_name)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def _create_msg_from_input_prompt(self, input_prompt: str):
        return [
            {"role": "system", "content": self.role},
            {"role": "user", "content": input_prompt}
        ]

    def make_call(self, prompt: str, max_answer_length: int, temperature: int = 1):
        client = OpenAI(api_key=self.api_key)
        call_estimated_cost = self.estimate_cost(prompt, max_answer_length)
        # if self.verbose: print(call_estimated_cost)
        if call_estimated_cost > self.max_cost:
            raise Exception("Expected cost exceeds single-run usage limit. No API call.")
        else:
            self.billing_history += call_estimated_cost

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=self._create_msg_from_input_prompt(prompt),
            stream=False,
            temperature=temperature,
            max_tokens=max_answer_length,
        )
        client.api_key = None

        single_call_cost = round(self.estimate_cost(input_prompt=prompt, max_tokens=max_answer_length), 6)
        cumulative_cost = self.billing_history
        if self.verbose:
            print(f"single_call_cost: {single_call_cost} - cumulative_cost: {cumulative_cost}")

        return completion.choices[0].message.content
