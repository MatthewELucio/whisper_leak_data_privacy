#!/usr/bin/env python3
import sys
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.utils import PromptUtils
from core.chatbot_utils import ChatbotUtils
from core.model import TrainingSetCollector
from core.data_sync import run_consolidation_task

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true', help='Run the data collection task (default behavior if no task specified).')
    group.add_argument('--consolidate', action='store_true', help='Run the data consolidation task instead of collection.')

    parser.add_argument('-c', '--chatbot', help='The chatbot (required for collection)', required=False, default="AzureGPT41")
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path (required for collection)', required=False, default="./prompts/standard/prompts.json")
    parser.add_argument('-t', '--tlsport', type=int, help='The remote TLS port (for collection)', default=443)
    parser.add_argument('-o', '--output', type=str, help='The output folder for the data gathering (for collection)', default="data/main")
    parser.add_argument('-T', '--temperature', type=float, help='Override temperature value to use for the chatbot (for collection).')

    args = parser.parse_args()
    # --- Validation based on task ---
    if args.collect:
        if not args.chatbot:
            parser.error("--chatbot is required when using --collect")
        if not args.prompts:
            parser.error("--prompts is required when using --collect")
        assert args.tlsport > 0 and args.tlsport <= 0xFFFF, Exception(f'Invalid remote TLS port: {args.tlsport}')
    elif args.consolidate:
        # No specific args needed for consolidate other than the flag itself
        pass
    else:
         # Default to collect if neither is explicitly chosen (though the group requires one)
         # This else block might not be strictly necessary due to mutually_exclusive_group requirement
         if not args.chatbot:
            parser.error("--chatbot is required for the default collection task")
         if not args.prompts:
            parser.error("--prompts is required for the default collection task")
         args.collect = True # Assume collect if somehow no flag is set

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
    is_user_cancelled = False
    last_error = None
    consolidation_success = None # Track consolidation result

    try:
        PrintUtils.print_logo()

        args = parse_arguments()

        # --- Task Dispatch ---
        if args.consolidate:
            PrintUtils.print_extra("Starting data consolidation task...")
            # Call the consolidation function from data_sync.py
            consolidation_success = run_consolidation_task()
            if not consolidation_success:
                raise Exception("Data consolidation failed.")
            PrintUtils.print_extra("Data consolidation task finished.")
            # Skip the rest of the collection logic

        elif args.collect:
            # Validate high privileges (needed for both tasks potentially, keep it)
            PrintUtils.start_stage('Validating high privileges')
            assert OsUtils.is_high_privileges(), Exception('User does not run in high privileges')
            PrintUtils.end_stage()

            PrintUtils.print_extra("Starting data collection task...")
            # Get the chatbot object
            chatbot_class = get_chatbot_class(args.chatbot)

            # Read prompts
            prompts = PromptUtils.read_prompts(args.prompts)

            # Build the training set
            training_set_path = os.path.join(get_self_dir(), args.output)
            collector = TrainingSetCollector(prompts['positive']['prompts'], prompts['positive']['repeat'], prompts['negative']['prompts'], prompts['negative']['repeat'], training_set_path, args.tlsport)
            training_set = collector.get_training_set(chatbot_class, args.temperature)
            PrintUtils.print_extra("Data collection task finished.")
        # --- End Task Dispatch ---

    except Exception as ex:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra(f'Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True
    finally:
        PrintUtils.start_stage('Running cleanup code')
        # Cleanup any sniffing that may still be happening (only relevant for collection)
        if 'args' in locals() and args.collect: # Only stop sniffing if collection ran
             NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.end_stage()

        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
            sys.exit(1) # Exit with error code
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
            sys.exit(1) # Exit with error code
        elif consolidation_success is not None: # Check consolidation specific result
             if consolidation_success:
                 PrintUtils.print_extra(f'Consolidation finished successfully\n')
                 sys.exit(0)
             else:
                 # Error message already printed by run_consolidation_task or the exception handler
                 # PrintUtils.print_error(f'Consolidation failed\n') # Redundant
                 sys.exit(1)
        else: # Assume collection finished successfully if no error/cancel/consolidation
            PrintUtils.print_extra(f'Collection finished successfully\n')
            sys.exit(0)

if __name__ == '__main__':
    main()
