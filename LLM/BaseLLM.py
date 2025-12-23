"""Created on 09-14-2025 16:36:06 @author: denizyalimyilmaz"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Abstract Parent Class
class base_LLM:
    def __init__(self):
        self.prompt = ""
        self.context = ""
        self.model = "gpt-5"  # gpt-4o, gpt-4o-mini, gpt-4-turbo
        self.allowed_tools = []
        load_dotenv()

    def work(self):
        pass

    def getResponse(self, prompt, justMessage = True):
        """# Now forced to input prompt
        if not prompt: # if prompt isn't given default to class default
            prompt = self.prompt"""

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": prompt},
            ],
        )
        if(justMessage):
            return response.choices[0].message.content
        
        return response
    
    """ 
    def getResponseWithTools(self):
        client = OpenAI(api_key=OPENAI_API_KEY)

        # prepare tool definitions
        tools = []
        if "get_headlines" in self.allowed_tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_headlines",
                        "description": "Return the daily financial news headlines",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            )

        # first call
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.context},
                {"role": "user", "content": self.prompt},
            ],
            tools=tools if tools else None,
        )

        msg = response.choices[0].message

        # if the model requested a tool
        if msg.tool_calls:
            for call in msg.tool_calls:
                if call.function.name == "get_headlines":
                    result = get_headlines()

                    # call GPT again with tool output
                    followup = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.context},
                            {"role": "user", "content": self.prompt},
                            msg,
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": str(result),
                            },
                        ],
                    )
                    return followup.choices[0].message.content

        # no tool call, return normal reply
        return msg.content
"""

"""
class headline_classifier_LLM(base_LLM):
    # This llm will take all of the headlines and classify them for their tickets.
    self.allowed_tools = [""]

    self.prompt =  
        Please classify headlines for every ticket that will be affected
   
    self.context = f 
        You are the Consulter agent.
        Your job is to analyze financial news headlines and produce structured suggestions for portfolio actions. 

        You read the news provided to you and respond with clear, concise recommendations such as 'buy', 'sell', or 'hold' along with reasoning.

        Always output in JSON with fields: {{"action": "buy|sell|hold", "asset": "ticker or name", "confidence": 0-100, "reason": "short explanation"}}.

        Here's headlines of today: {get_headlines()} 
    pass
"""


if __name__ == "__main__":
    # print(get_headlines())
    llm = base_LLM()
    print(llm.getResponse("hey gpt what's 5x5 ?"))
