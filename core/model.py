import random
import re
import time
from .utils import OsUtils
from .utils import PrintUtils
from .utils import NetworkUtils

import numpy
import hashlib
import os
import pyshark
import json

class Sequence(object):
    """
        Container for sequences.
    """

    def __init__(self, first_timestamp):
        """
            Creates an instance.
        """

        # Containers for the sequences
        self.time_seq = []
        self.size_seq = []

        # Save the timestamp
        self._last_timestamp = first_timestamp

    def add_pair(self, timestamp, packet_size):
        """
            Adds a new pair to the sequence.
        """

        # Append
        self.time_seq.append(timestamp - self._last_timestamp)
        self.size_seq.append(packet_size)

        # Save the timestamp
        self._last_timestamp = timestamp

class Datapoint(object):
    """
        Container for prompt datapoints.
    """

    def __init__(self, pcap_path, seq_path):
        """
            Creates an instance.
        """

        # Save members
        self.pcap_path = pcap_path
        self.seq_path = seq_path
        self.seq = None
        self.local_port = 0
        self.remote_port = 0

        # Cleanup (best-effort) if data does not exist in case of leftovers
        if not self.exists():
            assert OsUtils.del_file(pcap_path), Exception(f'Failed deleting existing file: {pcap_path}')
            assert OsUtils.del_file(seq_path), Exception(f'Failed deleting existing file: {seq_path}')

        # Load sequence preemptively
        else:
            self.load_seq()

    def exists(self, include_pcap=False):
        """
            Indicates if the datapoint actually exists (we only account for the sequence file).
        """

        # Indicate files exist
        if include_pcap:
            if not os.path.isfile(self.pcap_path):
                return False
        return os.path.isfile(self.seq_path)

    def _validate_seq(self, seq):
        """
            Validates a sequence.
        """

        # Validate all fields
        assert isinstance(seq, dict), Exception('Invalid sequence type')
        assert 'local_port' in seq and isinstance(seq['local_port'], int) and seq['local_port'] > 0 and seq['local_port'] <= 0xFFFF, Exception(f'Missing or invalid local port data in sequence file: {self.seq_path}')
        assert 'remote_port' in seq and isinstance(seq['remote_port'], int) and seq['remote_port'] > 0 and seq['remote_port'] <= 0xFFFF, Exception(f'Missing or invalid remote port data in sequence file: {self.seq_path}')
        assert 'temperature' in seq and isinstance(seq['temperature'], float) and seq['temperature'] >= 0, Exception(f'Missing or invalid temperature in sequence file: {self.seq_path}')
        assert 'prompt' in seq and isinstance(seq['prompt'], str) and len(seq['prompt']) > 0, Exception(f'Missing or invalid prompt in sequence file: {self.seq_path}')
        assert 'pertubated_prompt' in seq and isinstance(seq['pertubated_prompt'], str) and len(seq['pertubated_prompt']) > 0, Exception(f'Missing or invalid pertubated prompt in sequence file: {self.seq_path}')
        assert 'response' in seq and isinstance(seq['response'], str), Exception(f'Missing or invalid response in sequence file: {self.seq_path}')
        assert 'data_lengths' in seq and isinstance(seq['data_lengths'], list) and len([ val for val in seq['data_lengths'] if (not isinstance(val, int)) or val < 0 ]) == 0, Exception(f'Missing or invalid data lengths in sequence file: {self.seq_path}')
        assert 'time_diffs' in seq and isinstance(seq['time_diffs'], list) and len([ val for val in seq['time_diffs'] if (not isinstance(val, float)) or val < 0 ]) == 0, Exception(f'Missing or invalid time differences list in sequence file: {self.seq_path}')
        assert len(seq['data_lengths']) == len(seq['time_diffs']), Exception(f'Time differences and data lenghts size mismatch in sequence file: {self.seq_path}')
        assert len(seq['data_lengths']) > 0, Exception('No data found in sequence file: {self.seq_path}')

    def load_seq(self):
        """
            Load the sequence from the sequence file.
        """

        # Validate datapoint exists
        assert self.exists(), Exception('Datapoint does not exist')

        # Deserialize the sequence data
        with open(self.seq_path, 'r') as fp:
            
            # Load as JSON
            self.seq = json.load(fp)

        # Validate JSON
        self._validate_seq(self.seq)
    
    def save_seq(self):
        """
            Saves the sequence to the sequence file.
        """

        # Validate sequence and save it
        self._validate_seq(self.seq)
        with open(self.seq_path, 'w') as fp:
            json.dump(self.seq, fp, indent=2)

    def to_sequence_object(self, first_timestamp=0.0):
        """
            Returns a new Sequence object.
        """

        # Validate data is not empty
        data_lengths = self.seq.get('data_lengths', None)
        assert data_lengths is not None, Exception(f'Missing data lengths')
        time_diffs = self.seq.get('time_diffs', None)
        assert time_diffs is not None and len(time_diffs) > 0, Exception(f'Missing time differences')
        assert len(time_diffs) == len(data_lengths), Exception(f'Mismatching lengths for time differences and data lengths')
        
        # Create sequence
        seq = Sequence(first_timestamp)
        last_timestamp = first_timestamp
        for i in range(len(time_diffs)):
            seq.add_pair(last_timestamp + time_diffs[i], data_lengths[i])
            last_timestamp = time_diffs[i]

        # Return result
        return seq

    def generate_seq(self, local_port, remote_port, prompt, pertubated_prompt, response, temperature):
        """
            Runs the analysis on the PCAP path and writes the sequence file.
            Note this also automatically populates the sequence data in the datapoint.

            Returns: (number of data points collected, average data length)
        """

        # Validate PCAP file exists
        assert os.path.isfile(self.pcap_path), Exception(f'PCAP file does not exist: {self.pcap_path}')

        # Capture file will require cleanups
        cap = None
        try:

            # Start building the sequence
            self.seq = {}
            self.seq['timestamp'] = time.time()
            self.seq['local_port'] = local_port
            self.seq['remote_port'] = remote_port
            self.seq['prompt'] = prompt
            self.seq['pertubated_prompt'] = pertubated_prompt
            self.seq['response'] = ''.join(response)
            self.seq['response_tokens'] = response
            self.seq['response_token_count'] = len(response)
            self.seq['response_token_count_nonempty'] = len([ token for token in response if len(token) > 0 ])
            self.seq['response_token_count_empty'] = len([ token for token in response if len(token) == 0 ])
            self.seq['temperature'] = temperature
            self.seq['data_lengths'] = []
            self.seq['time_diffs'] = []

            # Run the analysis
            cap = pyshark.FileCapture(self.pcap_path, display_filter=f'tcp.port == {local_port} || tcp.port == {remote_port}')
            client_hello_found = False
            prev_sniff_time = None

            # Iterate all packets
            for packet in cap:
                
                # Only handle TLS
                if not hasattr(packet, 'tls'):
                    continue

                # Check for ClientHello packets
                if hasattr(packet.tls, 'handshake_type') and packet.tls.handshake_type == '1' and int(packet.tcp.dstport) == remote_port and int(packet.tcp.srcport) == local_port:
                    client_hello_found = True
                    prev_sniff_time = float(packet.sniff_time.timestamp())
                    continue
                
                # Check for ApplicationData only if we have seen ClientHello
                if not client_hello_found:
                    continue
                if hasattr(packet.tls, 'app_data') and int(packet.tcp.dstport) == local_port and int(packet.tcp.srcport) == remote_port:
                    timestamp = float(packet.sniff_time.timestamp())
                    data_length = int(packet.length)
                    self.seq['data_lengths'].append(data_length)
                    self.seq['time_diffs'].append(timestamp - prev_sniff_time)
                    prev_sniff_time = timestamp

            # Validate some data was acquired
            assert len(self.seq) > 0, Exception(f'PCAP file has no data: {self.pcap_path}')

            # Write the sequence file
            self.save_seq()

        # Cleanup
        finally:

            # Close capture
            if cap is not None:
                cap.close()
        
        # Return the number of data points collected, and average data length
        return len(self.seq['data_lengths']), numpy.mean(self.seq['data_lengths']) if len(self.seq['data_lengths']) > 0 else 0.0

class TrainingSetCollector(object):
    """
        The training set collector.
    """

    def __init__(self, positive_prompts, positive_repeats, negative_prompts, negative_repeats, out_directory_base, remote_tls_port):
        """
            Creates an instance.
        """

        # Save members
        self._positive_prompts = positive_prompts
        self._positive_repeats = positive_repeats
        self._negative_prompts = negative_prompts
        self._negative_repeats = negative_repeats
        self._remote_tls_port = remote_tls_port

        # Create and save the output directory
        self._out_dir = out_directory_base
        assert OsUtils.mkdir(self._out_dir), Exception(f'Could not get or make directory "{self._out_dir}"')

    def get_datapoint(self, prompt, index, chatbot_name):
        """
            Gets a datapoint for the given prompt, an index and the chatbot name.
        """

        # Get the file paths
        chatbot_name_normalized = chatbot_name.replace(' ', '_')
        base_path = os.path.join(self._out_dir, f'{hashlib.sha1(prompt.encode()).hexdigest()}_{index}_{chatbot_name_normalized}')
        pcap_path = f'{base_path}.pcap'
        seq_path = f'{base_path}.seq'

        # Return the datapoint
        return Datapoint(pcap_path, seq_path)

    def get_training_set(self, chatbot_class):
        """
            Gets or generates the training set for the given chatbot class.
        """

        # Start as a stage
        PrintUtils.start_stage('Generating training set')

        # Saves state
        skip_count = 0
        curr_count = 0
        training_set = {}
        last_local_port = 0

        # Save all prompts
        all_prompts = self._negative_prompts + self._positive_prompts

        # Calculate maximum repeat value
        max_repeats = max(self._positive_repeats, self._negative_repeats)

        # Built the task list, [(prompt, index), ...]
        task_list = []
        for prompt in all_prompts:
            repeats = self._negative_repeats if prompt in self._negative_prompts else self._positive_repeats
            pertubated_prompts = self._perturbate_prompt(prompt, max_repeats)
            numpy.random.shuffle(pertubated_prompts) # Shuffle

            if len(pertubated_prompts) < max_repeats:
                raise Exception(f'Not enough pertubated prompts for prompt: {prompt}')

            for index in range(repeats):
                pertubated_prompt = pertubated_prompts[index]

                task_list.append((prompt, pertubated_prompt, index))
        
        # Shuffle the task list
        numpy.random.shuffle(task_list)

        total_datapoints = len(task_list)

        # Iterate each prompt and either fetch existing data or truly generate data for it
        failed = 0
        data_length, avg_size, token_count = 0, 0.0, 0
        training_set = {}
        for (prompt, pertubated_prompt, index) in task_list:
            # Update progress
            percentage = (curr_count * 100) // total_datapoints
            PrintUtils.start_stage(f'Generating training set ({curr_count} / {total_datapoints} = {percentage}%), {failed} failed. Latest: {data_length} events, {avg_size:.1f} bytes per event, {token_count} tokens.', override_prev=True)
            curr_count += 1

            if prompt not in training_set:
                training_set[prompt] = []

            # Fetch the datapoint for the prompt
            datapoint = self.get_datapoint(prompt, index, chatbot_class.__name__)
            if datapoint.exists():
                skip_count += 1
                training_set[prompt].append(datapoint)
                continue

            # Start sniffing to collect data
            NetworkUtils.start_sniffing_tls(datapoint.pcap_path, self._remote_tls_port)

            # Create a chatbot object to make sure we get fresh connections
            chatbot_obj = chatbot_class(self._remote_tls_port)
            temperature = chatbot_obj.get_temperature()
            try:
                response, local_port = chatbot_obj.send_prompt(pertubated_prompt, temperature)
                assert isinstance(response, list), Exception('Got an invalid response from chatbot: {chatbot_class.__name__}')
                assert len(response) > 0 and len(''.join(response)) > 0, Exception(f'Got empty response for prompt: {pertubated_prompt}')

                # Discover new ports and stop sniffing (unless local port was provided by chatbot)
                if local_port is None:
                    new_local_ports = NetworkUtils.get_self_local_ports(self._remote_tls_port)
                    NetworkUtils.stop_sniffing_tls()
                    new_local_ports = [ port for port in new_local_ports if last_local_port != port ]
                    assert len(new_local_ports) < 2, Exception('Ambiguity in local TLS ports')
                    if len(new_local_ports) == 1:
                        last_local_port = new_local_ports[0]
                else:
                    assert local_port > 0 and local_port <= 0xFFFF, Exception(f'Invalid port indicated by chatbot: {local_port}')
                    last_local_port = local_port
                    NetworkUtils.stop_sniffing_tls()

                # Perform the analysis and set the data
                data_length, avg_size = datapoint.generate_seq(last_local_port, self._remote_tls_port, prompt, pertubated_prompt, response, temperature)
                token_count = len(response)
                training_set[prompt].append(datapoint)
            except Exception as e:
                PrintUtils.print_extra(f'Failed to generate training set for prompt: {prompt}')
                PrintUtils.print_extra(f'Exception: {str(e)}')
                NetworkUtils.stop_sniffing_tls()
                failed += 1
                continue

        # Finish stage and return the training set
        PrintUtils.start_stage('Generating training set', override_prev=True)
        if skip_count > 0:
            PrintUtils.print_extra(f'Datapoints pre-existed: *{skip_count}*')
        PrintUtils.end_stage()
        return training_set

    def _perturbate_prompt(self, prompt, N):
        """
        Generates N distinct variations of a given prompt by adding spaces at random positions.

        Args:
            prompt: The original string prompt.
            N: The number of distinct variations required.

        Returns:
            A list containing up to N distinct variations of the prompt.
        """
        if N <= 0:
            return []

        # Handle empty prompt case
        if not prompt.strip():
            return [" " * i for i in range(1, N + 1)][:N]

        # Tokenize prompt and identify insertion points
        words = prompt.split()
        num_points = len(words) + 1  # Before first word, between words, after last word
        
        variations = [prompt]  # Start with the original prompt
        unique_variations = {prompt}  # Set for fast duplicate checking
        
        # Generate remaining variations
        while len(variations) < N:
            # Start fresh insertion plan for this variation
            spaces = [0] * num_points
            
            # Add spaces until we get a unique variation
            while True:
                # Add a space at a random position
                position = random.randint(0, num_points - 1)
                spaces[position] += 1
                
                # Construct the new prompt
                parts = []
                
                # Add spaces before first word
                if spaces[0] > 0:
                    parts.append(" " * spaces[0])
                    
                # Add words with spaces between/after them
                for i, word in enumerate(words):
                    parts.append(word)
                    # Add spaces after word (standard + extra)
                    if i < len(words) - 1:
                        parts.append(" " + " " * spaces[i + 1])
                    elif spaces[-1] > 0:
                        parts.append(" " * spaces[-1])
                
                new_prompt = "".join(parts)
                
                # If unique, add to variations and move to next
                if new_prompt not in unique_variations:
                    variations.append(new_prompt)
                    unique_variations.add(new_prompt)
                    break
                    
                # Prevent infinite loops if we can't find more unique variations
                if sum(spaces) > 100:  # Arbitrary limit to prevent excessive space addition
                    return variations
        
        return variations