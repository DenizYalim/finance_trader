from BaseLLM import base_LLM

# 
class traderLLm(base_LLM):
    def __init__(self):
        super().__init__()
        self.allowed_tools = ["get_headlines"]

        self.prompt = """
            please provide me with portfolio suggestions based on news of today.
        """
        self.context = f"""
            You are the Consulter agent.
            Your job is to analyze financial news headlines and produce structured suggestions for portfolio actions. 

            You read the news provided to you and respond with clear, concise recommendations such as 'buy', 'sell', or 'hold' along with reasoning.

            Always output in JSON with fields: {{"action": "buy|sell|hold", "asset": "ticker or name", "confidence": 0-100, "reason": "short explanation"}}.
        """

    def work(self, data = None):
        prompt = self.prompt
        if data:
            print("Data was used by trader!")
            prompt += f"Here's some data about the markets to help you make better suggestions {data}"

        return self.getResponse(prompt=prompt)

if __name__ == "__main__":
    trader = traderLLm()
    print(trader.work())