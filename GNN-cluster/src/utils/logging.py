import os
import json

def log_metrics(epoch, accuracy, precision, recall, f1, hits_at_k, full_accuracy, log_dir, log_file='training_log.txt'):
    log_entry = f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Hits_at_k: {hits_at_k:.4f}, full_accuracy: {full_accuracy:.4f} \n"
    # print(log_entry)  # Print to console for quick access

    # Ensure directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Full path to the log file
    log_file_path = os.path.join(log_dir, log_file)

    # Write the log entry to the file
    with open(log_file_path, 'a') as f:
        f.write(log_entry)


def save_config(config, dir, name='config.json'):
    # Ensure directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Full path to the log file
    log_file_path = os.path.join(dir, name)

    # Save the config to a JSON file
    with open(log_file_path, 'w') as f:
        json.dump(config, f)
