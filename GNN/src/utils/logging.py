def log_metrics(epoch, accuracy, precision, recall, f1, log_file='training_log.txt'):
    log_entry = f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n"
    print(log_entry)  # Print to console for quick access

    with open(log_file, 'a') as f:
        f.write(log_entry)
