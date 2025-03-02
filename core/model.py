from core.utils import OsUtils
from core.utils import PrintUtils
from core.utils import NetworkUtils

import hashlib
import os

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

        # Cleanup (best-effort) if data does not exist in case of leftovers
        if not self.exists():
            assert OsUtils.del_file(pcap_path), Exception(f'Failed deleting existing file *{pcap_path}*')
            assert OsUtils.del_file(seq_path), Exception(f'Failed deleting existing file *{seq_path}*')

    def exists(self):
        """
            Indicates if the datapoint actually exists.
        """

        # Indicate
        return os.path.isfile(self.pcap_path) and os.path.isfile(self.seq_path)

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

        # Create a hash of the prompts
        prompts_hash = hashlib.sha1() 
        for prompt in prompts:
            prompts_hash.update(prompt.encode())

        # Get the output directory for the prompts
        OsUtils.mkdir(out_directory_base)
        self._out_dir = os.path.join(out_directory_base, prompts_hash.hexdigest())
        assert OsUtils.mkdir(self._out_dir), Exception(f'Could not get or make directory "{self._out_dir}"')

    def get_datapoint(self, prompt, index):
        """
            Gets a datapoint for the given prompt and index.
        """

        # Get the file paths
        base_path = os.path.join(self._out_dir, f'{hashlib.sha1(prompt.encode()).hexdigest()}_{index}')
        pcap_path = f'{base_path}.pcap'
        seq_path = f'{base_path}.seq'

        # Return the datapoint
        return Datapoint(pcap_path, seq_path)
    
    def get_training_set(self, chatbot):
        """
            Gets or generates the training set for the given chatbot.
        """

        # Start as a stage
        PrintUtils.start_stage('Generating training set')

        # Saves state
        skip_count = 0
        curr_count = 0
        training_set = {}
        last_local_ports = set()

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
                datapoint = self.get_datapoint(prompt, index)
                if datapoint.exists():
                    skip_count += 1
                    training_set[prompt].append(datapoint)
                    continue

                # Perform sniffing to collect data
                NetworkUtils.start_sniffing_tls(self._remote_tls_port)
                chatbot.send_prompt(prompt)
                new_local_ports = NetworkUtils.get_self_local_ports(self._remote_tls_port)
                capture = NetworkUtils.stop_sniffing_tls()
                new_local_ports = new_local_ports - last_local_ports
                import pdb; pdb.set_trace()

        # Finish stage
        PrintUtils.start_stage('Generating training set', override_prev=True)
        PrintUtils.end_stage()


