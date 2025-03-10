from core.utils import OsUtils
from core.utils import PrintUtils
from core.utils import NetworkUtils

import numpy
import hashlib
import os
import pyshark
import json

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

    @staticmethod
    def _validate_seq(seq):
        """
            Validates a sequence.
        """

        # Validate all fields
        assert isinstance(seq, dict), Exception('Invalid sequence type')
        assert 'local_port' in seq and isinstance(seq['local_port'], int) and seq['local_port'] > 0 and seq['local_port'] <= 0xFFFF, Exception(f'Missing or invalid local port data in sequence file: {self.seq_path}')
        assert 'remote_port' in seq and isinstance(seq['remote_port'], int) and seq['remote_port'] > 0 and seq['remote_port'] <= 0xFFFF, Exception(f'Missing or invalid remote port data in sequence file: {self.seq_path}')
        assert 'temperature' in seq and isinstance(seq['temperature'], float) and seq['temperature'] >= 0, Exception(f'Missing or invalid temperature in sequence file: {self.seq_path}')
        assert 'prompt' in seq and isinstance(seq['prompt'], str) and len(seq['prompt']) > 0, Exception(f'Missing or invalid prompt in sequence file: {self.seq_path}')
        assert 'response' in seq and isinstance(seq['response'], str) and len(seq['response']) > 0, Exception(f'Missing or invalid response in sequence file: {self.seq_path}')
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
        self.__class__._validate_seq(self.seq)
    
    def save_seq(self):
        """
            Saves the sequence to the sequence file.
        """

        # Validate sequence and save it
        self.__class__._validate_seq(self.seq)
        with open(self.seq_path, 'w') as fp:
            json.dump(self.seq, fp, indent=2)

    def generate_seq(self, local_port, remote_port, prompt, response, temperature):
        """
            Runs the analysis on the PCAP path and writes the sequence file.
            Note this also automatically populates the sequence data in the datapoint.
        """

        # Validate PCAP file exists
        assert os.path.isfile(self.pcap_path), Exception(f'PCAP file does not exist: {self.pcap_path}')

        # Capture file will require cleanups
        cap = None
        try:

            # Start building the sequence
            self.seq = {}
            self.seq['local_port'] = local_port
            self.seq['remote_port'] = remote_port
            self.seq['prompt'] = prompt
            self.seq['response'] = response
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

class TrainingSetCollector(object):
    """
        The training set collector.
    """

    def __init__(self, positive_prompts, negative_prompts, repeat_count, out_directory_base, remote_tls_port):
        """
            Creates an instance.
        """

        # Save members
        self._positive_prompts = positive_prompts
        self._negative_prompts = negative_prompts
        self._repeat_count = repeat_count
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

    def prepare_classifier(self, training_set):
        """
            Prepares the classifier for the given training set.
        """

        # Start a stage
        PrintUtils.start_stage('Preparing classifier')

        # Split training set to data (x) and labels (y)
        x = []
        y = []

        # Iterate the entire training set
        for label, samples in training_set.items():

            # Create feature vectors from all samples
            feature_vectors = []
            for sample in samples:

                # Extract statistical features
                times, sizes = zip(*sample.seq)
                feature_vector = [
                    numpy.mean(times), numpy.std(times), numpy.min(times), numpy.max(times),  # Time stats
                    numpy.mean(sizes), numpy.std(sizes), numpy.min(sizes), numpy.max(sizes),  # Size stats
                ]
                feature_vectors.append(feature_vector)
            
            # Aggregate all samples per label
            x.append(numpy.mean(feature_vectors, axis=0))  # Take mean across samples
            y.append(label)

        # Turn into numpy arrays
        x, y = numpy.array(x), numpy.array(y)

        # Normalize
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Train classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_scaled, y)

        # Return classifier
        PrintUtils.end_stage()
        return model

    def get_training_set(self, chatbot_class, api_key):
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
        all_prompts = self._positive_prompts + self._negative_prompts

        # Iterate each prompt and either fetch existing data or truly generate data for it
        for prompt in all_prompts:
           
            # Add prompt
            training_set[prompt] = []

            # Handle the repeat count
            for index in range(self._repeat_count):

                # Update progress
                percentage = (curr_count * 100) // (len(all_prompts) * self._repeat_count)
                PrintUtils.start_stage(f'Generating training set ({curr_count} / {len(all_prompts) * self._repeat_count} = {percentage}%)', override_prev=True)
                curr_count += 1

                # Fetch the datapoint for the prompt
                datapoint = self.get_datapoint(prompt, index, chatbot_class.__name__)
                if datapoint.exists():
                    skip_count += 1
                    training_set[prompt].append(datapoint)
                    continue

                # Start sniffing to collect data
                NetworkUtils.start_sniffing_tls(datapoint.pcap_path, self._remote_tls_port)

                # Create a chatbot object to make sure we get fresh connections
                chatbot_obj = chatbot_class(api_key, self._remote_tls_port)
                temperature = chatbot_obj.get_temperature()
                response, local_port = chatbot_obj.send_prompt(prompt, temperature)
                assert isinstance(response, str), Exception('Got an invalid response from chatbot: {chatbot_class.__name__}')
                assert len(response) > 0, Exception(f'Got empty response for prompt: {prompt}')

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
                datapoint.generate_seq(last_local_port, self._remote_tls_port, prompt, response, temperature)
                training_set[prompt].append(datapoint)

        # Finish stage and return the training set
        PrintUtils.start_stage('Generating training set', override_prev=True)
        if skip_count > 0:
            PrintUtils.print_extra(f'Datapoints pre-existed: *{skip_count}*')
        PrintUtils.end_stage()
        return training_set

