from abc import ABC
from abc import abstractmethod
import importlib.util
import inspect
import os
import sys
import httpx

class ChatbotBase(ABC):
    """
        Base class for chatbots.
    """

    def __init__(self, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Save members
        self._remote_tls_port = remote_tls_port

    @abstractmethod
    def send_prompt(self, prompt, temperature):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
            Returns a tuple of (response, local_port) - if local port cannot be determined return (response, None).
        """
        pass

    @abstractmethod
    def get_temperature(self):
        """
            Gets the temperature of the model.
        """
        pass

    @abstractmethod
    def get_common_name(self):
        """
            Gets the common name of the model.
        """
        pass

    @abstractmethod
    def match_tls_server_name(self, server_name):
        """
            Matches the TLS server name.
        """
        pass

class LocalPortSaverTransport(httpx.HTTPTransport):
    """
        An HTTP transport that saves the local port as a member.
        Useful to be used as an HTTP client transport.
    """

    def handle_request(self, request):
        """
            Handles a request.
        """

        # Handle the request
        response = super().handle_request(request)

        # Get the port from the pool
        assert len(self._pool.connections) > 0, Exception('Expecting at least one connection')
        local_port = self._pool.connections[-1]._connection._network_stream._sock.getsockname()[1]
        if local_port > 0 and local_port <= 0xFFFF:
            if hasattr(self, 'local_port'):
                assert self.local_port == local_port, Exception(f'Local port already indicated previously: {self.local_port} vs. {local_port}')
            else:
                self.local_port = local_port

        # Return the response
        return response

    def get_local_port(self):
        """
            Gets the local port or None if not found.
        """

        # Either return the local port or None
        return getattr(self, 'local_port', None)

class ChatbotUtils(object):
    """
        Utilities for chatbot classes.
    """

    @staticmethod
    def load_chatbots(base_path):
        """
            Loads all chatbots from the given base path.
        """

        # Saves chatbot classes
        chatbot_classes = {}

        # Iterate all Python files in base path
        candidates = [ os.path.join(base_path, filename) for filename in os.listdir(base_path) if filename.endswith('.py') ]
        for file_path in candidates:

            # Load dynamically
            module_name = os.path.basename(file_path).split('.')[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Only discover relevant subclassses
            subclasses = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, ChatbotBase) and obj is not ChatbotBase:
                    subclasses.append(obj)
            assert len(subclasses) < 2, Exception(f'Multiple chatbot classes found in file "{os.path.basename(file_path)}"')
            assert len(subclasses) == 1, Exception(f'No chatbot classes found in file "{os.path.basename(file_path)}"')

            # Save subclass
            subclass_name = subclasses[0].__name__.lower()
            assert subclass_name not in chatbot_classes, Exception(f'Ambiguity for chatbot class "{subclass_name}"')
            chatbot_classes[subclass_name] = subclasses[0]

        # Return subclasses
        return chatbot_classes

