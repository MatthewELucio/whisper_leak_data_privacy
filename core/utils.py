import os
import pyshark
import colorama
import sys
import ctypes
import re
import base64
import argparse
import time
import psutil

# Initialize colorama
colorama.init()

class ThrowingArgparse(argparse.ArgumentParser):
    """
        Custom argument parser that does not bail out or print usage on parsing errors, but throws an exception instead.
    """

    def error(self, message):
        """
            Handles errors gracefully.
        """

        # Throw
        raise Exception(message)

class OsUtils(object):
    """
        Generic OS utilities.
    """

    @staticmethod
    def del_file(path):
        """
            Tries to delete a file (best-effort).
            Indicates whether the file doesn't exist afterwards.
        """

        # Best-effort deletion
        try:
            os.unlink(path)
        except Exception:
            pass

        # Indicate result
        return not os.path.isfile(path)

    @staticmethod
    def mkdir(path):
        """
            Try to make a directory (best-effort).
            Indicates whether the directory exists afterwards.
        """

        # Best-effort creation
        try:
            os.mkdir(path)
        except Exception:
            pass

        # Indicate result
        return os.path.isdir(path)

    @staticmethod
    def is_high_privileges():
        """
            Indicates if we run in high privileges.
        """

        # Handle POSIX
        if sys.platform in ('linux', 'darwin'):
            return os.geteuid() == 0

        # Handle Windows
        if sys.platform == 'win32':
            return ctypes.windll.shell32.IsUserAnAdmin() != 0

        # Unsupported platform
        raise Exception(f'Unsupported platform "{sys.platform}"')

class PrintUtils(object):
    """
        Printing utilities.
    """

    # Define colors
    WHITE = colorama.Fore.WHITE + colorama.Style.BRIGHT
    GREEN = colorama.Fore.GREEN + colorama.Style.BRIGHT
    RED = colorama.Fore.RED + colorama.Style.BRIGHT
    GREY = colorama.Fore.WHITE+ colorama.Style.NORMAL
    YELLOW = colorama.Fore.YELLOW + colorama.Style.NORMAL
    DARKGREY = colorama.Fore.LIGHTBLACK_EX + colorama.Style.BRIGHT
    RESET_COLORS = colorama.Style.RESET_ALL

    # Pretty printing
    PP_LEN = 120

    # Saves stage and extra
    _in_stage = False
    _extra = []

    @classmethod
    def print_logo(cls):
        """
            Prints the logo.
        """

        # Print the logo
        logo = base64.b64decode(b'CiAgICDilojilojilZcgICAg4paI4paI4pWX4paI4paI4pWXICDilojilojilZfilojilojilZfilojilojilojilojilojilojilojilZfilojilojilojilojilojilojilZcg4paI4paI4paI4paI4paI4paI4paI4pWX4paI4paI4paI4paI4paI4paI4pWXICAgICDilojilojilZcgICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVlyDilojilojilojilojilojilZcg4paI4paI4pWXICDilojilojilZcKICAgIOKWiOKWiOKVkSAgICDilojilojilZHilojilojilZEgIOKWiOKWiOKVkeKWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVl+KWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVlyAgICDilojilojilZEgICAgIOKWiOKWiOKVlOKVkOKVkOKVkOKVkOKVneKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVl+KWiOKWiOKVkSDilojilojilZTilZ0KICAgIOKWiOKWiOKVkSDilojilZcg4paI4paI4pWR4paI4paI4paI4paI4paI4paI4paI4pWR4paI4paI4pWR4paI4paI4paI4paI4paI4paI4paI4pWX4paI4paI4paI4paI4paI4paI4pWU4pWd4paI4paI4paI4paI4paI4pWXICDilojilojilojilojilojilojilZTilZ0gICAg4paI4paI4pWRICAgICDilojilojilojilojilojilZcgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVkeKWiOKWiOKWiOKWiOKWiOKVlOKVnSAKICAgIOKWiOKWiOKVkeKWiOKWiOKWiOKVl+KWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVkeKWiOKWiOKVkeKVmuKVkOKVkOKVkOKVkOKWiOKWiOKVkeKWiOKWiOKVlOKVkOKVkOKVkOKVnSDilojilojilZTilZDilZDilZ0gIOKWiOKWiOKVlOKVkOKVkOKWiOKWiOKVlyAgICDilojilojilZEgICAgIOKWiOKWiOKVlOKVkOKVkOKVnSAg4paI4paI4pWU4pWQ4pWQ4paI4paI4pWR4paI4paI4pWU4pWQ4paI4paI4pWXIAogICAg4pWa4paI4paI4paI4pWU4paI4paI4paI4pWU4pWd4paI4paI4pWRICDilojilojilZHilojilojilZHilojilojilojilojilojilojilojilZHilojilojilZEgICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKVkSAg4paI4paI4pWRICAgIOKWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKWiOKWiOKWiOKWiOKWiOKVl+KWiOKWiOKVkSAg4paI4paI4pWR4paI4paI4pWRICDilojilojilZcKICAgICDilZrilZDilZDilZ3ilZrilZDilZDilZ0g4pWa4pWQ4pWdICDilZrilZDilZ3ilZrilZDilZ3ilZrilZDilZDilZDilZDilZDilZDilZ3ilZrilZDilZ0gICAgIOKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVnSAg4pWa4pWQ4pWdICAgIOKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVkOKVkOKVkOKVkOKVkOKVneKVmuKVkOKVnSAg4pWa4pWQ4pWd4pWa4pWQ4pWdICDilZrilZDilZ0KICAgICAgICAgICAgICAgIERhdGEgbGVha2FnZSBwcm9vZi1vZi1jb25jZXB0IGZvciBMYXJnZSBMYW5ndWFnZSBNb2RlbCBjaGF0Ym90cwogICAgICAgICAgICAgICAgICAgIEJ5IEpvbmF0aGFuIEJhciBPciAoIkpCTyIpLCBAeW9feW9feW9famJvCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=').decode()
        print(f'{cls.WHITE}{logo}{cls.RESET_COLORS}')

    @classmethod
    def is_in_stage(cls):
        """
            Indicates whether we are in stage.
        """

        # Return result
        return cls._in_stage

    @classmethod
    def start_stage(cls, message, override_prev=False):
        """
            Starts a stage.
        """

        # Validate we are not in stage unless we override a previous stage possible
        assert (not cls._in_stage) or override_prev, Exception('Entering a stage without finishing the previous one')

        # Potentially override
        if override_prev:
            print('\r' + (' ' * cls.PP_LEN) + '\r', end='')

        # Print title
        title = message[:cls.PP_LEN - 5]
        title += ' ...'
        title += (cls.PP_LEN - 1 - len(title)) * '.'
        title += ' '
        print(f'{cls.GREY}{title}{cls.RESET_COLORS}', end='')

        # Flush
        sys.stdout.flush()

        # Indicate we are in stage
        cls._in_stage = True

    @classmethod
    def end_stage(cls, fail_message=None, throw_on_fail=True):
        """
            Ends a stage.
        """

        # Print status
        status = f'{cls.GREY}[  {cls.GREEN}OK{cls.GREY}  ]{cls.RESET_COLORS}' if fail_message is None else f'{cls.GREY}[ {cls.RED}FAIL{cls.GREY} ]{cls.RESET_COLORS}'
        print(status)

        # Indicate we are not in a stage
        cls._in_stage = False

        # Prints extra contents
        cls._dump_extra()

        # Optionally throw
        if throw_on_fail and (fail_message is not None):
            raise Exception(fail_message)

    @classmethod
    def print_error(cls, message):
        """
            Prints a raw error message.
        """

        # Prints the error message
        print(f'{cls.RED}ERROR{cls.GREY}: {message}{cls.RESET_COLORS}')

    @classmethod
    def print_extra(cls, message):
        """
            Prints extra contents. Will be pending if stage is not complete.
        """

        # Adds message as an extra
        cls._extra.append(message)
        if not cls._in_stage:
            cls._dump_extra()

    @classmethod
    def _dump_extra(cls):
        """
            Dumps extra contents.
        """

        # Validate we are not in a stage
        assert not cls._in_stage, Exception('Cannot dump extra while in a stage')
        
        # Print messages and replace special strings with pretty colors
        for message in cls._extra:
            msg = re.sub(r'\*.+?\*', lambda m:f'{cls.YELLOW}{m.group(0)[1:-1]}{cls.DARKGREY}', message)
            print(f'{cls.DARKGREY}{msg}{cls.RESET_COLORS}')
        
        # Reset extra
        cls._extra = []

class NetworkUtils(object):
    """
        Network utilities.
    """

    # Sniffng handle and pending result
    _sniffer = None
    _sniffer_result = None

    @staticmethod
    def get_self_local_ports(remote_port):
        """
            Gets local TCP ports by self process connected to the given remote port.
        """

        # Get all TCP connections for remote port by self-PID
        return set([ conn.laddr.port for conn in psutil.net_connections(kind='inet') if hasattr(conn.laddr, 'port') and getattr(conn.raddr, 'port', None) == remote_port and conn.status == 'ESTABLISHED' and conn.pid == os.getpid() ])

    @classmethod
    def start_sniffing_tls(cls, remote_port=443):
        """
            Starts sniffing TLS.
        """

        # Validate sniffing is not being done
        assert (cls._sniffer is None) and (cls._sniffer_result is None), Exception('Active sniffing already performed')

        # Starts a live capture
        cls._sniffer = pyshark.LiveCapture(bpf_filter=f'tcp port {remote_port}', display_filter='tls')
        cls._sniffer_result = cls._sniffer.sniff_continuously()

        # Sleep for a while
        time.sleep(2)

    @classmethod
    def stop_sniffing_tls(cls, best_effort=False):
        """
            Stops sniffing TLS and returns the sniffing result back.
            Supply best_effort to try a best-effort stop, which will return None is no active sniffing is being done.
        """

        # Validate sniffing is being done
        if cls._sniffer is None or cls._sniffer_result is None:
            assert best_effort, Exception('Active sniffing has not started')
            return None

        # Sleep for a while
        time.sleep(2)

        # Stops the live capture
        cls._sniffer.close()
        
        # Nullify and return result
        result = cls._sniffer_result
        cls._sniffer = None
        cls._sniffer_result = None
        return result

