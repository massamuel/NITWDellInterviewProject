def log_accuracies(text_data):
    testfile =  open("log.txt", "a")
    text_to_log = text_data + '\n'
    testfile.write(text_to_log)
    testfile.close()