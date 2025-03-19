from core.chatbot_utils import ChatbotBase
from core.chatbot_utils import LocalPortSaverTransport

from openai import OpenAI, AzureOpenAI
import httpx

class AzureGPT4o(ChatbotBase):
    """
        Deepseek V3 over OpenRouter chatbot.
    """

    def __init__(self, api_key, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(api_key, remote_tls_port)

        # Create client that also saves the local port
        self._transport = LocalPortSaverTransport()
        self._client = AzureOpenAI(
            azure_endpoint=f'https://stg-mc-ncus-openai-api.openai.azure.com',
            api_key=api_key,
            api_version="2024-02-01",
            http_client=httpx.Client(transport=self._transport))

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """

        # Send prompt
        response = ''
        stream = self._client.chat.completions.create(
            extra_body={},
            model='gpt-4o-adhoc',
            messages=[ { 'role': 'user', 'content': prompt } ],
            stream=True,
            temperature=temperature
        )
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        # Return response
        return (response, self._transport.get_local_port())

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 0.7
        return 0.7
