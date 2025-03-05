#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.chatbot_base import ChatbotLoaderUtils
from core.model import TrainingSetCollector

import os

def get_self_dir():
    """
        Get the self directory.
    """

    # Return the self directory
    return os.path.dirname(os.path.abspath(__file__))

def get_prompts(filepath, prompt_type):
    """
        Gets prompts.
    """

    # Read prompts 
    PrintUtils.start_stage(f'Reading {prompt_type} prompts')
    with open(filepath, 'r') as fp:
        prompts = [ line.strip() for line in fp.read().split('\n') if len(line.strip()) > 0 ]
    assert len(prompts) > 0, Exception('Could not load any prompts')
    PrintUtils.print_extra(f'Loaded *{len(prompts)}* prompts')
    PrintUtils.end_stage()
    return prompts

def parse_arguments():
    """
        Parse arguments.
    """

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot', required=True)
    parser.add_argument('-a', '--apikey', help='The API key for the chatbot', required=True)
    parser.add_argument('-p', '--pprompts', help='The positive prompts file path', required=True)
    parser.add_argument('-n', '--nprompts', help='The negative prompts file path', required=True)
    parser.add_argument('-r', '--repetition',type=int,  help='The repetition count per prompt', default=5)
    parser.add_argument('-t', '--tlsport', type=int, help='The remote TLS port', default=443)
    args = parser.parse_args()
    assert args.repetition > 0, Exception(f'Invalid repetition count: {args.repetition}')
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
    chatbots = ChatbotLoaderUtils.load_chatbots(os.path.join(get_self_dir(), 'chatbots'))
    assert len(chatbots) > 0, Exception('Could not load any chatbots')
    chatbot_names = ', '.join([ f'*{name}*' for name in chatbots.keys() ])
    PrintUtils.print_extra(f'Loaded chatbots: {chatbot_names}')
    PrintUtils.end_stage()

    # Validating chatbot class exists
    PrintUtils.start_stage('Initializing chatbot class')
    chatbot_class = chatbots.get(chatbot_name, None)
    assert chatbot_class is not None, Exception(f'Chatbot "{chatbot_name}" does not exist')
    PrintUtils.print_extra(f'Using chatbot *{chatbot_name}*')
    PrintUtils.end_stage()

    # Return the class
    return chatbot_class

def main():
    """
        Main routine.
    """

    # Catch-all
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
        
        # Read the API key
        PrintUtils.start_stage('Reading API key')
        api_key = args.apikey
        try:
            with open(args.apikey, 'r') as fp:
                api_key = fp.read().strip()
        except Exception:
            PrintUtils.print_extra('*WARNING*: Treating API key as a *literal string*')
            PrintUtils.print_extra('Consider using a path for the API key in the future')
        PrintUtils.end_stage()

        # Get the chatbot object
        chatbot_class = get_chatbot_class(args.chatbot)

        # Read prompts
        positive_prompts = get_prompts(args.pprompts, 'positive')
        negative_prompts = get_prompts(args.nprompts, 'negative')
        
        # Get the training set
        collector = TrainingSetCollector(positive_prompts, negative_prompts, args.repetition, os.path.join(get_self_dir(), 'training_set'), args.tlsport)
        training_set = collector.get_training_set(chatbot_class, api_key)

        # Prepare the classifier
        classifier = collector.prepare_classifier(training_set)

    # Handle exceptions
    except Exception as ex:

        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        
        # Print error
        PrintUtils.print_error(ex)

    # Cleanups
    finally:

        # Cleanup any sniffing that may still be happening
        NetworkUtils.stop_sniffing_tls(best_effort=True)

if __name__ == '__main__':
    main()
