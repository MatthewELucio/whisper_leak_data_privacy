#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.chatbot_base import ChatbotLoaderUtils

import os

def main():
    """
        Main routine.
    """

    # Catch-all
    try:

        # Print logo
        PrintUtils.print_logo()

        # Validate high privileges
        PrintUtils.start_stage('Validating high privileges')
        assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
        PrintUtils.end_stage()

        # Parsing arguments
        PrintUtils.start_stage('Parsing command-line arguments')
        parser = ThrowingArgparse()
        parser.add_argument('-c', '--chatbot', help='The chatbot', required=True)
        parser.add_argument('-a', '--apikey', help='The API key for the chatbot', required=True)
        parser.add_argument('-p', '--prompts', help='The prompts file path', required=True)
        parser.add_argument('-o', '--output', help='The output directory path', required=True)
        parser.add_argument('-r', '--remoteport', type=int, help='The remote TLS port', default=443)
        args = parser.parse_args()
        assert args.remoteport > 0 and args.remoteport <= 0xFFFF, Exception(f'Invalid remote TLS port {args.remoteport}')
        PrintUtils.end_stage()

        # Reading API key
        PrintUtils.start_stage('Reading API key')
        api_key = args.apikey
        try:
            with open(args.apikey, 'r') as fp:
                api_key = fp.read()
        except Exception:
            PrintUtils.print_extra('Treating API key as a *literal string*')
            PrintUtils.print_extra('Consider using a path for the API key in the future')
        PrintUtils.end_stage()

        # Load all chatbots
        PrintUtils.start_stage('Loading chatbots')
        chatbots = ChatbotLoaderUtils.load_chatbots(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbots'))
        assert len(chatbots) > 0, Exception('Could not load any chatbots')
        chatbot_names = ', '.join([ f'"{name}"' for name in chatbots.keys() ])
        PrintUtils.print_extra(f'Loaded chatbots: {chatbot_names}')
        PrintUtils.end_stage()

        # Validating chatbot class exists and initialize it
        PrintUtils.start_stage('Initializing chatbot class')
        chatbot_class = chatbots.get(args.chatbot, None)
        assert chatbot_class is not None, Exception(f'Chatbot "{args.chatbot}" does not exist')
        chatbot_obj = chatbot_class(args.apikey, args.remoteport)
        PrintUtils.end_stage()

        # Read prompts
        PrintUtils.start_stage('Reading prompts')
        with open(args.prompts, 'r') as fp:
            prompts = [ line.strip() for line in fp.read().split('\n') if len(line.strip()) > 0 ]
        assert len(prompts) > 0, Exception('Could not load any prompts')
        PrintUtils.print_extra(f'Loaded {len(prompts)} prompts')
        PrintUtils.end_stage()

    # Handle exceptions
    except Exception as ex:

        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        
        # Print error
        PrintUtils.print_error(ex)

if __name__ == '__main__':
    main()
