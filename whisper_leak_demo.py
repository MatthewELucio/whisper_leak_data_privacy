#!/usr/bin/env python3
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import ThrowingArgparse
from core.utils import NetworkUtils
from core.chatbot_utils import ChatbotUtils
from core.model import TrainingSetCollector

import pyshark
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
    parser.add_argument('-i', '--interface', help='The network interface', required=True)
    args = parser.parse_args()
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

def packet_callback(packet):
    """
        Callback for all packets.
    """

    # Saves all streams
    stream_sequences = {}

    # Handle gracefully
    try:

        # Only handle TLS packets
        if hasattr(packet, 'tls'):
            
            # Handle handleshakes to identify important streams
            if getattr(packet.tls, 'handshake_type', None) == '1' and getattr(packet.tls, 'handshake_extensions_server_name') == 'whatever':
                stream_sequences[(packet.ip.src, packet.tcp.srcport, packet.ip.dst, packet.tcp.dstport)] = []
                PrintUtils.print_extra(f'{packet.sniff_time}: New stream: *{packet.ip.src}:{packet.tcp.srcport}* --> *{packet.ip.dst}:{packet.tcp.dstport}*')

    # Log exceptions
    except Exception as ex:
        PrintUtils.print_extra(f'*WARNING* - error processing packet: {ex}')

def main():
    """
        Main routine.
    """

    # Catch-all
    is_user_cancelled = False
    last_error = None
    capture = None
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

        # Start the capture
        PrintUtils.start_stage('Initializing sniffing')
        capture = pyshark.LiveCapture(interface=args.interface, display_filter='tcp')
        PrintUtils.end_stage()
        capture.apply_on_packets(packet_callback)

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
        if capture is not None:
            capture.close()
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
