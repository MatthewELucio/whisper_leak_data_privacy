#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.utils import PromptUtils
from core.chatbot_utils import ChatbotUtils
from core.model import TrainingSetCollector

import os
import json

def get_self_dir():
    """
        Get the self directory.
    """

    # Return the self directory
    return os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    """
        Parse arguments.
    """

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot', required=True)
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path', required=True)
    parser.add_argument('-t', '--tlsport', type=int, help='The remote TLS port', default=443)
    args = parser.parse_args()
    assert args.tlsport > 0 and args.tlsport <= 0xFFFF, Exception(f'Invalid remote TLS port: {args.tlsport}')
    PrintUtils.end_stage()

    # Return the parsed arguments
    return args

def get_chatbot_class(chatbot_name):
    """
        Get the chatbot class from the given chatbot name.
    """

    # Load all chatbots
    PrintUtils.start_stage('Loading chatbots')
    chatbots = ChatbotUtils.load_chatbots(os.path.join(get_self_dir(), 'chatbots'))
    assert len(chatbots) > 0, Exception('Could not load any chatbots')
    chatbot_names = '\n'.join([ f'\t*{chatbot.__name__}*' for chatbot in chatbots.values() ])
    PrintUtils.print_extra(f'Loaded chatbots:\n{chatbot_names}')
    PrintUtils.end_stage()

    # Validating chatbot class exists
    PrintUtils.start_stage('Initializing chatbot class')
    chatbot_class = chatbots.get(chatbot_name.lower(), None)
    assert chatbot_class is not None, Exception(f'Chatbot "{chatbot_name}" does not exist')
    PrintUtils.print_extra(f'Using chatbot *{chatbot_class.__name__}*')
    PrintUtils.end_stage()

    # Return the class
    return chatbot_class

def main():
    """
        Main routine.
    """

    # Catch-all
    is_user_cancelled = False
    last_error = None
    try:

        # Suppress STDERR
        OsUtils.suppress_stderr()

        # Print logo
        PrintUtils.print_logo()

        # Validate high privileges
        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
        PrintUtils.end_stage()

        # Parsing arguments
        args = parse_arguments()

        # Get the chatbot object
        chatbot_class = get_chatbot_class(args.chatbot)

        # Read prompts
        prompts = PromptUtils.read_prompts(args.prompts)
        
        # Build the training set
        training_set_path = os.path.join(get_self_dir(), 'data')
        collector = TrainingSetCollector(prompts['positive']['prompts'], prompts['positive']['repeat'], prompts['negative']['prompts'], prompts['negative']['repeat'], training_set_path, args.tlsport)
        training_set = collector.get_training_set(chatbot_class)

    # Handle exceptions
    except Exception as ex:

        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)

        # Save error and print it as an extra
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex

    # Handle cancel operations
    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra(f'Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True

    # Cleanups
    finally:

        # Cleanup any sniffing that may still be happening
        PrintUtils.start_stage('Running cleanup code')
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.end_stage()

        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
        else:
            PrintUtils.print_extra(f'Finished successfully\n')

if __name__ == '__main__':
    main()
