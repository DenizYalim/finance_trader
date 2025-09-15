
class consulter_LLM(base_LLM):
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

            Here's headlines of today: {get_headlines()}
        """
