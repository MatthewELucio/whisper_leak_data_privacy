from core.chatbot_base import ChatbotBase

import google.generativeai as genai
import asyncio

class Gemini(ChatbotBase):
    """
        Google Gemini chatbot.
    """

    def __init__(self, api_key, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Call superclass
        super().__init__(api_key, remote_tls_port)

        # Create model
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel('gemini-2.0-flash')

    async def _get_response(self, prompt, temperature):
        """
            Gets a response asynchronously.
        """

        # Run asynchronously
        generation_config = genai.GenerationConfig(temperature=temperature)
        chat = self._model.start_chat(history=[])
        response = await chat.send_message_async(prompt, generation_config=generation_config)
        return response.text

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
        """

        # Make sure we have an asyncio loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Get the response
        return loop.run_until_complete(chat.send_message_async(prompt, temperature))

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # For now we just return the default of 1.0
        return 1.0
