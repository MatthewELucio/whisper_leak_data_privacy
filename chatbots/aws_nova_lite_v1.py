import boto3
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from core.chatbot_utils import ChatbotBase, LocalPortSaverTransport

class AmazonNovaLiteV1(ChatbotBase):
    """
        Amazon Nova chatbot utilizing streaming responses.
    """
    _common_name = 'nova-lite-v1'

    def __init__(self, remote_tls_port=443):
        """
            Initializes the chatbot instance.
        """
        
        # Initialize
        super().__init__(remote_tls_port)

        # Load environment variables from .env file
        load_dotenv()

        # Retrieve AWS credentials from environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')  # Optional

        # Check if any required credentials are missing
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError('AWS credentials are not set in the environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.')

        # Initialize the Boto3 client
        self._transport = LocalPortSaverTransport()
        self._client = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=str(aws_access_key_id),
            aws_secret_access_key=str(aws_secret_access_key)
        )
        self._model_id = 'us.amazon.nova-lite-v1:0'

    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt to the model and returns the response along with the local port.
        """

        # Build request
        system_list = [
            {
                'text': 'Act as a creative writing assistant. When the user provides you with a topic, write a short story about that topic.'
            }
        ]
        message_list = [{ 'role': 'user', 'content': [{ 'text': prompt }]}]
        inf_params = { 'maxTokens': 3000, 'temperature': temperature}
        request_body = {
            'schemaVersion': 'messages-v1',
            'messages': message_list,
            'system': system_list,
            'inferenceConfig': inf_params,
        }

        start_time = datetime.now()
        response = self._client.invoke_model_with_response_stream(
            modelId=self._model_id, body=json.dumps(request_body)
        )

        request_id = response.get('ResponseMetadata', {}).get('RequestId')
        print(f'Request ID: {request_id}')
        print('Awaiting first token...')

        chunk_count = 0
        time_to_first_token = None
        full_response = []

        stream = response.get('body')
        if stream:
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    chunk_json = json.loads(chunk.get('bytes').decode())
                    content_block_delta = chunk_json.get('contentBlockDelta')
                    if content_block_delta:
                        if time_to_first_token is None:
                            time_to_first_token = datetime.now() - start_time
                            print(f'Time to first token: {time_to_first_token}')

                        chunk_count += 1
                        text_chunk = content_block_delta.get('delta', {}).get('text', '')
                        full_response.append(text_chunk)
                        print(text_chunk, end='')

        return full_response, self._transport.get_local_port()

    def get_temperature(self):
        """
            Returns the default temperature setting for the model.
        """

        # Return 1.0 by default
        return 1.0
