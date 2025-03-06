from core.chatbot_base import ChatbotBase

from openai import OpenAI

class DeepseekV3OpenRouter(ChatbotBase):
    """
        Deepseek V3 over OpenRouter chatbot.
    """

    def __init__(self, api_key, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(api_key, remote_tls_port)

        # Create client
        self._client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key)

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
        """

        # Send prompt
        response = ''
        stream = self._client.chat.completions.create(extra_body={}, model='deepseek/deepseek-chat', messages=[ { 'role': 'user', 'content': prompt } ], stream=True, temperature=temperature)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        # Return response
        return response

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 1.0
        return 1.0
