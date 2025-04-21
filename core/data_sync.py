#!/usr/bin/env python3
"""
Data synchronization and consolidation tool using rclone.

Consolidates sequence files per chatbot while preserving the directory
structure found within the source 'data' directory. For example,
files in 'data/subdir/hash_trial_Chatbot.seq' will be consolidated into
'consolidated/subdir/Chatbot.seq'.

Uses rclone copy for non-destructive downloads and uploads.
"""

import os
import json
import glob
import re
import shutil
from pathlib import Path
import sys
import subprocess
from datetime import datetime
from collections import defaultdict

# Ensure core.utils can be imported assuming this script is in core/
from core.utils import PrintUtils

# --- Configuration ---
# Rclone settings
DEFAULT_REMOTE = "b2"
DEFAULT_BUCKET = "whisper-leak" # Set your bucket name

# Paths for consolidation
REMOTE_DATA_PATH = "data"
REMOTE_CONSOLIDATED_PATH = "consolidated"
LOCAL_CONSOLIDATED_PATH = "data" # Local path for consolidated data
# CONSOLIDATED_FILENAME is derived dynamically

# Local temporary directories for consolidation
TEMP_BASE_DIR = "./_consolidation_temp"
TEMP_DATA_DIR = os.path.join(TEMP_BASE_DIR, "data")
TEMP_CONSOLIDATED_DIR = os.path.join(TEMP_BASE_DIR, "consolidated")
# --- End Configuration ---

# --- rclone Helper Functions (identical to previous version) ---

def _check_rclone_installed():
    """Checks if rclone is installed and accessible."""
    try:
        process = subprocess.Popen(["rclone", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=15)
        if process.returncode == 0 and "rclone" in stdout:
             PrintUtils.print_extra("rclone installation verified.")
             return True
        else:
             PrintUtils.print_error(f"rclone --version failed (code {process.returncode}): {stderr or stdout}")
             return False
    except FileNotFoundError:
        PrintUtils.print_error("rclone command not found. Please ensure rclone is installed and in your PATH.")
        return False
    except subprocess.TimeoutExpired:
        PrintUtils.print_error("rclone --version command timed out.")
        return False
    except Exception as e:
        PrintUtils.print_error(f"Error checking rclone version: {e}")
        return False

def _check_rclone_remote_config(remote_name):
    """Checks if the specified rclone remote is configured."""
    try:
        process = subprocess.Popen(["rclone", "listremotes"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode == 0:
             remotes = stdout.splitlines()
             if f"{remote_name}:" in remotes:
                 PrintUtils.print_extra(f"rclone remote '{remote_name}' configuration found.")
                 return True
             else:
                 PrintUtils.print_error(f"rclone remote '{remote_name}' is not configured. Found remotes: {remotes}. Please run 'rclone config' or 'rclone config create b2 b2 account=<> key=<>'.")
                 return False
        else:
             PrintUtils.print_error(f"rclone listremotes failed (code {process.returncode}): {stderr or stdout}")
             return False
    except FileNotFoundError:
        PrintUtils.print_error("rclone command not found while checking remotes.")
        return False
    except subprocess.TimeoutExpired:
         PrintUtils.print_error("rclone listremotes command timed out.")
         return False
    except Exception as e:
        PrintUtils.print_error(f"Error checking rclone remotes: {e}")
        return False

def _run_rclone_command(command_args, operation_desc):
    """Runs an rclone command and handles output/errors."""
    PrintUtils.print_extra(f"Running rclone command: {' '.join(command_args)}")
    try:
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.stdout.close()
        return_code = process.wait(timeout=3600) # 1 hour timeout
        if return_code == 0:
            PrintUtils.print_extra(f"rclone operation '{operation_desc}' completed successfully.")
            return True
        else:
            PrintUtils.print_error(f"rclone operation '{operation_desc}' failed with exit code {return_code}.")
            return False
    except FileNotFoundError:
        PrintUtils.print_error(f"rclone command failed. Ensure rclone is installed and in PATH.")
        return False
    except subprocess.TimeoutExpired:
        PrintUtils.print_error(f"rclone operation '{operation_desc}' timed out.")
        process.kill()
        return False
    except Exception as e:
        PrintUtils.print_error(f"Failed to execute rclone command for '{operation_desc}': {e}")
        return False

def download_folder_copy(remote_name, bucket_name, remote_folder, local_folder):
    """Downloads a folder using 'rclone copy'."""
    PrintUtils.print_extra(f"Downloading '{remote_folder}' from {remote_name}:{bucket_name} to '{local_folder}'")
    os.makedirs(local_folder, exist_ok=True)
    remote_full_path = f"{remote_name}:{bucket_name}/{remote_folder}"
    local_sync_path = os.path.join(local_folder, "")
    command = ["rclone", "copy", "--progress", "--update", remote_full_path, local_sync_path]
    success = _run_rclone_command(command, f"download {remote_folder}")
    return success

def upload_file_copy(remote_name, bucket_name, local_file, remote_path):
    """Uploads a single file using 'rclone copyto' (destination includes filename)."""
    remote_target = f"{remote_name}:{bucket_name}/{remote_path}"
    PrintUtils.print_extra(f"Uploading '{os.path.basename(local_file)}' to '{remote_target}'")
    if not os.path.exists(local_file):
        PrintUtils.print_extra(f"Local file not found: {local_file}")
        return False
    command = ["rclone", "copyto", "--progress", "--update", local_file, remote_target]
    success = _run_rclone_command(command, f"upload {os.path.basename(local_file)}")
    return success

# --- Metadata Extraction (identical to previous version) ---

def extract_chatbot_and_metadata(filename):
    """
    Extracts ChatbotName, hash, trial, and extra from the filename.
    Returns None if the filename doesn't match the expected pattern.
    """
    base_filename = os.path.basename(filename)
    pattern = r"^([a-f0-9]{40})_(\d+)_(\w+?)(?:_([^_]+))?\.seq$"
    match = re.match(pattern, base_filename)
    if not match:
        return None
    metadata = {
        "hash": match.group(1),
        "trial": int(match.group(2)),
        "chatbot_name": match.group(3),
    }
    if match.group(4):
        metadata["extra"] = match.group(4)
    return metadata

# --- Consolidation Task ---

def run_consolidation_task():
    """
    Performs data consolidation preserving directory structure.
    1. Downloads 'data' and 'consolidated' folders.
    2. Scans downloaded 'data' dir structure and identifies chatbot files within each subdir.
    3. For each subdir/chatbot combination:
        a. Loads existing consolidated data from corresponding path in downloaded 'consolidated'.
        b. Merges new data from 'data' files (based on hash).
        c. Writes the updated consolidated list to the corresponding path in 'temp_consolidated'.
    4. Uploads all updated consolidated files back to their respective remote paths.
    5. Cleans up temporary folders.
    """
    PrintUtils.print_extra(f"=== Starting Structured Data Consolidation Task at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    overall_success = False
    files_to_upload = {} # {local_path: remote_path_relative_to_bucket}

    # --- Pre-checks ---
    if not _check_rclone_installed(): return False
    if not _check_rclone_remote_config(DEFAULT_REMOTE): return False
    # --- End Pre-checks ---

    try:
        # --- Setup Temporary Directories ---
        PrintUtils.print_extra("Setting up temporary directories...")
        if os.path.exists(TEMP_BASE_DIR):
            PrintUtils.print_extra(f"Removing existing temporary directory '{TEMP_BASE_DIR}'...")
            try:
                #shutil.rmtree(TEMP_BASE_DIR)
                pass
            except Exception as e:
                 PrintUtils.print_error(f"Failed to remove existing temp dir '{TEMP_BASE_DIR}': {e}")
                 return False
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_CONSOLIDATED_DIR, exist_ok=True)
        PrintUtils.print_extra(f"Created temporary directories: '{TEMP_DATA_DIR}', '{TEMP_CONSOLIDATED_DIR}'")
        # --- End Setup ---

        # --- Step 1: Download Data ---
        PrintUtils.print_extra(f"--- Downloading source data ({REMOTE_DATA_PATH}) ---")
        if not download_folder_copy(DEFAULT_REMOTE, DEFAULT_BUCKET, REMOTE_DATA_PATH, TEMP_DATA_DIR):
            return False

        # --- Step 2: Download Existing Consolidated Data ---
        PrintUtils.print_extra(f"--- Downloading existing consolidated data ({REMOTE_CONSOLIDATED_PATH}) ---")
        if not download_folder_copy(DEFAULT_REMOTE, DEFAULT_BUCKET, REMOTE_CONSOLIDATED_PATH, TEMP_CONSOLIDATED_DIR):
            PrintUtils.print_extra(f"Warning: Could not download or '{REMOTE_CONSOLIDATED_PATH}' is empty remotely. Will create new consolidated files as needed.")
        # --- End Downloads ---

        # --- Step 3: Scan Data Files and Group by Structure + Chatbot ---
        PrintUtils.print_extra("--- Scanning downloaded data files and grouping by structure ---")
        # Structure: { relative_subdir: { chatbot_name: [list_of_seq_files] } }
        grouped_files = defaultdict(lambda: defaultdict(list))
        total_seq_files_found = 0
        parsed_file_count = 0

        for root, _, files in os.walk(TEMP_DATA_DIR):
            total_seq_files_found += len([f for f in files if f.endswith(".seq")])
            relative_subdir = os.path.relpath(root, TEMP_DATA_DIR)
            if relative_subdir == ".":
                relative_subdir = "" # Represent root level as empty string

            for filename in files:
                if filename.endswith(".seq"):
                    metadata = extract_chatbot_and_metadata(filename)
                    if metadata:
                        full_path = os.path.join(root, filename)
                        grouped_files[relative_subdir][metadata['chatbot_name']].append(full_path)
                        parsed_file_count += 1
                    # else: # Optional: log files that don't match pattern
                        # PrintUtils.print_extra(f"Skipping non-standard file: {os.path.join(relative_subdir, filename)}")


        if not grouped_files:
            PrintUtils.print_error(f"No valid chatbot sequence files found matching the pattern in '{TEMP_DATA_DIR}'. Nothing to consolidate.")
            # Consider this success if no *valid* files, but maybe warn?
            overall_success = True # No work to do, technically not a failure.
            return overall_success # Exit early

        PrintUtils.print_extra(f"Scanned {total_seq_files_found} total .seq files.")
        PrintUtils.print_extra(f"Grouped {parsed_file_count} valid files into {len(grouped_files)} subdirectories.")
        # --- End Scanning ---

        # --- Step 4: Process Each Subdirectory / Chatbot Group ---
        total_new_entries_added = 0
        total_consolidated_files_updated = 0

        for relative_subdir, chatbot_groups in grouped_files.items():
            subdir_prefix = f"[{relative_subdir}]" if relative_subdir else "[root]"
            PrintUtils.print_extra(f"\n--- Processing Subdirectory: {subdir_prefix} ({len(chatbot_groups)} chatbots) ---")

            for chatbot_name, data_files in chatbot_groups.items():
                PrintUtils.print_extra(f"  Processing Chatbot: {chatbot_name} ({len(data_files)} files)...")

                consolidated_entries = []
                existing_hashes = set()

                # Define paths for this specific group
                consolidated_filename = f"{chatbot_name}.seq"
                # Path within the *local temporary* consolidated dir
                local_consolidated_target_path = os.path.join(TEMP_CONSOLIDATED_DIR, relative_subdir, consolidated_filename)
                 # Path within the *remote* consolidated dir (relative to bucket)
                remote_consolidated_target_path = os.path.join(REMOTE_CONSOLIDATED_PATH, relative_subdir, consolidated_filename).replace("\\", "/") # Ensure forward slashes for remote


                # 4a: Load Existing Consolidated Data (from local temp copy)
                try:
                    if os.path.exists(local_consolidated_target_path) and os.path.getsize(local_consolidated_target_path) > 0:
                        PrintUtils.print_extra(f"    Loading existing data from '{os.path.relpath(local_consolidated_target_path, TEMP_BASE_DIR)}'...")
                        with open(local_consolidated_target_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            consolidated_entries = existing_data
                            for entry in consolidated_entries:
                                if isinstance(entry, dict) and 'hash' in entry and 'trial' in entry:
                                    existing_hashes.add(entry['hash'] + str(entry['trial']))
                            PrintUtils.print_extra(f"    Loaded {len(consolidated_entries)} existing entries ({len(existing_hashes)} unique hashes).")
                        else:
                            PrintUtils.print_extra(f"    Warning: Existing file is not a list. Starting fresh.")
                    else:
                        PrintUtils.print_extra(f"    No existing consolidated file found or file is empty. Starting fresh.")
                except json.JSONDecodeError as e:
                    PrintUtils.print_extra(f"    Warning: Error decoding JSON: {e}. Starting fresh.")
                except Exception as e:
                    PrintUtils.print_extra(f"    Warning: Error reading existing file: {e}. Starting fresh.")

                # 4b: Merge New Data
                new_entries_for_group = 0
                PrintUtils.print_extra(f"    Merging {len(data_files)} source files...")
                for seq_file_path in data_files:
                    try:
                        metadata = extract_chatbot_and_metadata(os.path.basename(seq_file_path))
                        if not metadata or metadata['chatbot_name'] != chatbot_name:
                            continue # Should not happen based on grouping

                        file_hash = metadata['hash']
                        trial = metadata['trial']
                        if file_hash + str(trial) in existing_hashes:
                            continue # Already consolidated

                        with open(seq_file_path, 'r', encoding='utf-8') as f:
                            entry_data = json.load(f)

                        if not isinstance(entry_data, dict):
                            PrintUtils.print_extra(f"    Warning: Data in {os.path.basename(seq_file_path)} is not a JSON object. Skipping.")
                            continue

                        # Add metadata keys
                        entry_data['hash'] = file_hash
                        entry_data['trial'] = metadata['trial']
                        if 'extra' in metadata:
                            entry_data['extra'] = metadata['extra']

                        consolidated_entries.append(entry_data)
                        existing_hashes.add(file_hash + str(trial))
                        new_entries_for_group += 1

                    except json.JSONDecodeError as e:
                        PrintUtils.print_extra(f"    Warning: Invalid JSON in {os.path.basename(seq_file_path)}: {e}. Skipping.")
                    except Exception as e:
                        PrintUtils.print_extra(f"    Warning: Error processing file {os.path.basename(seq_file_path)}: {e}. Skipping.")

                if new_entries_for_group == 0 and not os.path.exists(local_consolidated_target_path):
                     PrintUtils.print_extra(f"    No new entries added and no existing file. Skipping write/upload for this group.")
                     continue # Nothing to save or upload

                PrintUtils.print_extra(f"    Added {new_entries_for_group} new entries.")
                PrintUtils.print_extra(f"    Total entries for {chatbot_name} in {subdir_prefix}: {len(consolidated_entries)}")
                total_new_entries_added += new_entries_for_group

                # 4c: Write Updated Consolidated File Locally (in temp dir)
                PrintUtils.print_extra(f"    Writing updated file to '{os.path.relpath(local_consolidated_target_path, TEMP_BASE_DIR)}'...")
                try:
                    # Ensure the target directory exists locally
                    os.makedirs(os.path.dirname(local_consolidated_target_path), exist_ok=True)
                    with open(local_consolidated_target_path, 'w', encoding='utf-8') as f:
                        json.dump(consolidated_entries, f, ensure_ascii=False, indent=2)
                    # Add to upload list
                    files_to_upload[local_consolidated_target_path] = remote_consolidated_target_path
                    total_consolidated_files_updated += 1
                except Exception as e:
                    PrintUtils.print_error(f"    Error writing consolidated file: {e}")
                    PrintUtils.print_error(f"    Skipping upload for {chatbot_name} in {subdir_prefix} due to write error.")
                    # Potentially remove from upload list if already added? No, just don't add it.

        PrintUtils.print_extra(f"\n--- Finished processing all groups. Total new entries added: {total_new_entries_added} ---")
        PrintUtils.print_extra(f"--- Total consolidated files written/updated: {total_consolidated_files_updated} ---")
        # --- End Group Processing ---


        # --- Step 5: Upload All Updated Consolidated Files ---
        PrintUtils.print_extra("--- Uploading updated consolidated files ---")
        successful_uploads = 0
        if not files_to_upload:
             PrintUtils.print_extra("No consolidated files were updated or created. Nothing to upload.")
             overall_success = True # Mark as success because no work needed/failed
        else:
            PrintUtils.print_extra(f"Attempting to upload {len(files_to_upload)} files...")
            upload_count = 0
            for local_path, remote_path in files_to_upload.items():
                upload_count += 1
                PrintUtils.print_extra(f"  Uploading ({upload_count}/{len(files_to_upload)}): {os.path.basename(local_path)} -> {remote_path}")
                if upload_file_copy(DEFAULT_REMOTE, DEFAULT_BUCKET, local_path, remote_path):
                    successful_uploads += 1
                else:
                    PrintUtils.print_error(f"  Failed to upload {os.path.basename(local_path)}. See rclone output above.")
                    # Continue trying to upload others

            PrintUtils.print_extra(f"Successfully uploaded {successful_uploads}/{len(files_to_upload)} files.")
            if successful_uploads == len(files_to_upload):
                overall_success = True # All steps completed and all uploads succeeded
            else:
                PrintUtils.print_error("Some uploads failed. Check logs.")
                overall_success = False # Mark failure if any upload failed

    except Exception as e:
        PrintUtils.print_error(f"An unexpected error occurred during consolidation: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
    finally:
        # --- Step 6: Cleanup ---
        PrintUtils.print_extra("\n--- Cleaning up temporary directories ---")
        try:
            if os.path.exists(TEMP_BASE_DIR):
                #shutil.rmtree(TEMP_BASE_DIR)
                PrintUtils.print_extra(f"Removed temporary directory: {TEMP_BASE_DIR}")
            else:
                PrintUtils.print_extra("Temporary directory already removed or not created.")
        except Exception as e:
            PrintUtils.print_extra(f"Warning: Error during cleanup: {e}")
        # --- End Cleanup ---

        end_time = datetime.now()
        status = "Completed Successfully" if overall_success else "Failed"
        PrintUtils.print_extra(f"\n=== Structured Data Consolidation Task {status} at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        if overall_success and files_to_upload:
             PrintUtils.print_extra(f"Consolidated data uploaded to various paths within: {DEFAULT_REMOTE}:{DEFAULT_BUCKET}/{REMOTE_CONSOLIDATED_PATH}/")
        elif overall_success and not files_to_upload:
             PrintUtils.print_extra("No new data found or processed, or no consolidated files needed updating.")
        elif not overall_success:
             PrintUtils.print_error("Consolidation task failed. Check logs for details.")

    return overall_success

def download_folder(local_folder, remote_folder, remote=DEFAULT_REMOTE, bucket=DEFAULT_BUCKET):
    return download_folder_copy(remote, bucket, remote_folder, local_folder)

def upload_folder(local_folder, remote_folder, remote=DEFAULT_REMOTE, bucket=DEFAULT_BUCKET):
    PrintUtils.print_extra(f"(Using simplified upload_folder wrapper for {local_folder})")
    PrintUtils.print_extra("Uploading folder '{local_folder}' to {remote}:{bucket}/{remote_folder}")
    if not os.path.isdir(local_folder):
         PrintUtils.print_extra(f"Local folder not found: {local_folder}")
         return False
    remote_full_path = f"{remote}:{bucket}/{remote_folder}"
    local_sync_path = os.path.join(local_folder, "")
    command = ["rclone", "copy", "--progress", local_sync_path, remote_full_path]
    success = _run_rclone_command(command, f"upload folder {local_folder}")
    return success

def download_training_data(remote=DEFAULT_REMOTE, bucket=DEFAULT_BUCKET):
    """
    Downloads the training data from the remote storage.
    """
    # Check if rclone is installed and configured
    if not _check_rclone_installed():
        return False
    
    if not _check_rclone_remote_config(remote):
        return False

    return download_folder(LOCAL_CONSOLIDATED_PATH, REMOTE_CONSOLIDATED_PATH, remote, bucket)
