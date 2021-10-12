def log_accuracies(text_data):
    file = open("log.txt", "a")
    text_to_log = '\n' + text_data
    file.write(text_to_log)
    file.close()
    print("Logged")