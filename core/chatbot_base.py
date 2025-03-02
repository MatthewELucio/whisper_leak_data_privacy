from abc import ABC
from abc import abstractmethod
import importlib.util
import inspect
import os
import sys

class ChatbotBase(ABC):
    """
        Base class for chatbots.
    """

    def __init__(self, api_key, remote_tls_port=443):
        """
            Creates an instance.
        """

        # Save members
        self._api_key = api_key
        self._remote_tls_port = remote_tls_port

    @abstractmethod
    def send_prompt(self, prompt):
        """
            Sends a prompt. Pulls data back as fast as possible (asynchronously) but waits.
        """
        pass

class ChatbotLoaderUtils(object):
    """
        Loader utilities for chatbot classes.
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

