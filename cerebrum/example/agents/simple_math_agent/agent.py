from cerebrum.llm.apis import llm_chat

class SimpleMathAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.messages = []

    def run(self, task_input):
        self.messages.append({
            "role": "user",
            "content": task_input
        })

        response = llm_chat(
            agent_name=self.agent_name,
            messages=self.messages,
            base_url="http://localhost:8000"
        )

        result = response["response"]["response_message"]
        print(f"Answer: {result}")
        return result
