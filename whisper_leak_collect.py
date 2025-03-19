#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
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
    parser.add_argument('-e', '--empty', help='Whether to permit empty responses', action='store_true')
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
    chatbot_names = ', '.join([ f'*{chatbot.__name__}*' for chatbot in chatbots.values() ])
    PrintUtils.print_extra(f'Loaded chatbots: {chatbot_names}')
    PrintUtils.end_stage()

    # Validating chatbot class exists
    PrintUtils.start_stage('Initializing chatbot class')
    chatbot_class = chatbots.get(chatbot_name.lower(), None)
    assert chatbot_class is not None, Exception(f'Chatbot "{chatbot_name}" does not exist')
    PrintUtils.print_extra(f'Using chatbot *{chatbot_class.__name__}*')
    PrintUtils.end_stage()

    # Return the class
    return chatbot_class

def read_prompts(json_path):
    """
        Reads prompts file and validate its structure.
    """

    # Read prompts
    PrintUtils.start_stage(f'Reading prompts')
    with open(json_path, 'r') as fp:
        contents = json.load(fp)

    # Validate the file structure
    assert isinstance(contents, dict), Exception('Invalid format for prompts JSON file')
    prompt_types = [ 'positive', 'negative' ]
    for prompt_type in prompt_types:
        prompts_data = contents.get(prompt_type, None)
        assert prompts_data is not None, Exception(f'Missing {prompt_type} prompts')
        assert isinstance(prompts_data, dict), Exception(f'Invalid structure for {prompt_type} prompts')
        repeats = prompts_data.get('repeat', None)
        assert repeats is not None, Exception(f'Missing key "repeat" in {prompt_type} prompts')
        assert isinstance(repeats, int) and repeats > 0, Exception(f'Invalid repeat value in {prompt_type} prompts')
        prompts = prompts_data.get('prompts', None)
        assert prompts is not None, Exception(f'Missing key "prompts" in {prompt_type} prompts')
        assert isinstance(prompts, list), Exception(f'Invalid structure for prompts in {prompts_type} prompts')
        assert len(prompts) > 0, Exception(f'The prompt list for {prompt_type} prompts is empty')
        assert len([ elem for elem in prompts if not isinstance(elem, str) ]) == 0, Exception('Invalid prompt format in {prompts_type} prompts')
        PrintUtils.print_extra(f'Loaded *{len(prompts)}* {prompt_type} prompts with repetition of *{repeats}*')

    # Return result
    PrintUtils.end_stage()
    return contents

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
        prompts = read_prompts(args.prompts)
        
        # Build the training set
        training_set_path = os.path.join(get_self_dir(), 'training_set')
        collector = TrainingSetCollector(prompts['positive']['prompts'], prompts['positive']['repeat'], prompts['negative']['prompts'], prompts['negative']['repeat'], training_set_path, args.tlsport)
        training_set = collector.get_training_set(chatbot_class, args.empty)

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
