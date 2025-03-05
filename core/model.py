from core.utils import OsUtils
from core.utils import PrintUtils
from core.utils import NetworkUtils

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import numpy
import hashlib
import os
import pyshark
import struct

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

    def exists(self):
        """
            Indicates if the datapoint actually exists.
        """

        # Indicate
        return os.path.isfile(self.pcap_path) and os.path.isfile(self.seq_path)

    def load_seq(self):
        """
            Load the sequence from the sequence file.
        """

        # Validate datapoint exists
        assert self.exists(), Exception('Datapoint does not exist')

        # Deserialize the sequence data
        self.seq = []
        with open(self.seq_path, 'rb') as fp:

            # Read the local and remote ports
            self.local_port, self.remote_port = struct.unpack('<HH', fp.read(struct.calcsize('<HH')))

            # Read the sequence itself
            while True:
                chunk = fp.read(struct.calcsize('<dL'))
                if len(chunk) == 0:
                    break
                self.seq.append(struct.unpack('<dL', chunk))

        # Validate we have data
        assert len(self.seq) > 0, Exception(f'Empty data for sequence file: {self.seq_path}')

    def generate_seq(self, local_port, remote_port):
        """
            Runs the analysis on the PCAP path and writes the sequence file.
            Note this also automatically populates the sequence data in the datapoint.
        """

        # Validate PCAP file exists
        assert os.path.isfile(self.pcap_path), Exception(f'PCAP file does not exist: {self.pcap_path}')

        # Capture file will require cleanups
        cap = None
        try:

            # Run the analysis
            cap = pyshark.FileCapture(self.pcap_path, display_filter=f'tcp.port == {local_port} || tcp.port == {remote_port}')
            serialized_seq = b''
            self.seq = []
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
                    self.seq.append((timestamp - prev_sniff_time, data_length))
                    serialized_seq += struct.pack('<dL', timestamp - prev_sniff_time, data_length)
                    prev_sniff_time = timestamp

            # Validate some data was acquired
            assert len(self.seq) > 0, Exception(f'PCAP file has no data: {self.pcap_path}')

            # Commit data to sequence file
            with open(self.seq_path, 'wb') as fp:
                serialized_seq = struct.pack('<HH', local_port, remote_port) + serialized_seq
                fp.write(serialized_seq)

        # Cleanup
        finally:

            # Close capture
            if cap is not None:
                cap.close()

class TrainingSetCollector(object):
    """
        The training set collector.
    """

    def __init__(self, prompts, repeat_count, out_directory_base, remote_tls_port):
        """
            Creates an instance.
        """

        # Save members
        self._prompts = prompts
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

        # Iterate each prompt and either fetch existing data or truly generate data for it
        for prompt in self._prompts:
           
            # Add prompt
            training_set[prompt] = []

            # Handle the repeat count
            for index in range(self._repeat_count):

                # Update progress
                PrintUtils.start_stage(f'Generating training set ({curr_count} / {len(self._prompts) * self._repeat_count})', override_prev=True)
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
                chatbot_obj.send_prompt(prompt)

                # Discover new ports and stop sniffing
                new_local_ports = NetworkUtils.get_self_local_ports(self._remote_tls_port)
                NetworkUtils.stop_sniffing_tls()
                new_local_ports = [ port for port in new_local_ports if last_local_port != port ]
                assert len(new_local_ports) < 2, Exception('Ambiguity in local TLS ports')
                if len(new_local_ports) == 1:
                    last_local_port = new_local_ports[0]

                # Perform the analysis and set the data
                datapoint.generate_seq(last_local_port, self._remote_tls_port)
                training_set[prompt].append(datapoint)

        # Finish stage and return the training set
        PrintUtils.start_stage('Generating training set', override_prev=True)
        if skip_count > 0:
            PrintUtils.print_extra(f'Datapoints pre-existed: *{skip_count}*')
        PrintUtils.end_stage()
        return training_set

