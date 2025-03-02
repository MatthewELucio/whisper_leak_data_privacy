from core.chatbot_base import ChatbotBase

import google.generativeai as genai

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

    def send_prompt(self, prompt):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
        """

        # Send prompt with a clean histoy
        chat = self._model.start_chat(history=[])
        response = chat.send_message(prompt)

