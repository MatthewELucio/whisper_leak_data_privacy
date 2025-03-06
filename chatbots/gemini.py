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
        self._model = genai.GenerativeModel('gemini-2.0-flash')

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
        generation_config = genai.GenerationConfig(temperature=temperature)
        chat = self._model.start_chat(history=[])
        loop.run_until_complete(chat.send_message_async(prompt, generation_config=generation_config))

    def get_temperature(self):
        """
            Gets the temperature of the model.
        """

        # Return a random number between 0 and 2 at a 0.1 granularity in a Bell-Curve
        while True:
            
            # Generate a random value from a normal distribution centered around 1.0
            value = random.gauss(1.0, 0.3)

            # Clip the value to the specified range
            if 0.0 <= value and value <= 2.0:
                return value

