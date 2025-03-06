from core.chatbot_base import ChatbotBase

import google.generativeai as genai
import asyncio
import random

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
        self._model = genai.GenerativeModel('gemini-1.5-flash')

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

        # Send prompt with a clean histoy
        chat = self._model.start_chat(history=[])
        loop.run_until_complete(chat.send_message_async(prompt))

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # Return a random number between 0 and 2 at a 0.1 granularity
        return random.randint(0, 20) / 10
