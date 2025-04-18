import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import processor
import numpy as np
import os
import model
import messenger
import torch

class file_monitor(FileSystemEventHandler):

    def on_created(self, event):
        #if a file was created, call backend for classification and notification 
        predict_and_send(event.src_path)

def monitor_folder(folder_path):
    #create subject and observer 
    subject = file_monitor()
    observer = Observer()

    #register subject with observer
    observer.schedule(subject, folder_path, recursive= False)
    observer.start()

    #keep running until keyboard interrupt 
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def predict_and_send(path):

    input = wav_processor.process_data(path)
    predict = mod(input)
    print(predict)
    
    #found the prediction that appears the most within the batch 
    res = np.zeros((5,1))
    for outcome in predict:
        res[outcome.argmax()] += 1

    if res.argmax() != 3:
        #my_messenger.send(res.argmax())
        print(res.argmax())

    #delete file after process
    os.remove(path)


mod = model.model(1,32,3,5)
mod.load_state_dict(torch.load("model2.pth", weights_only= True, map_location= torch.device('cpu')))
wav_processor = processor.processor()
mail_messenger = messenger.messenger()

#get folder path 
folder_path = os.path.join(os.getcwd(), "wav_files")
monitor_folder(folder_path)