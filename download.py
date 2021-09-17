import requests
import re
import os
import sys
from queue import Queue
from threading import Thread


url = r'http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/'
file_path = r'./data/ISIA_Food500/'
os.makedirs(file_path, exist_ok=True)

content = requests.get(url).content
content = str(content)

paths = re.findall(r'"[^=/]+"', content)
paths = [path[1:-1] for path in paths]


class Worker(Thread):
    def __init__(self, queue):
        super(Worker, self).__init__()
        self.queue = queue

    def run(self):
        i, path = self.queue.get()
        message = requests.get(url+path, stream=True)
        length = message.headers['Content-Length']
        length = int(length)
        print(f'task{i} start, downloading {path}: total {length} bytes')
        current_length = 0
        with open(file_path+path ,'wb') as f:
            for buffer in message.iter_content(1048576):
                if buffer:
                    f.write(buffer)
                    current_length += len(buffer)
                elif current_length < length:
                    print('download failed: {.2f} finished'.format(current_length/length))
                percent = current_length / length
                done = int(50 * percent)
                sys.stdout.write("\r[%s%s] %d%% task%d" % ('█' * done, ' ' * (50 - done), 100 * percent, i))
                sys.stdout.flush()
        print(f'task{i} finished')

    
def start():
    print('download started')
    queue = Queue()
    for i, path in enumerate(paths):
        queue.put((i, path))
    for _ in range(10):
        worker = Worker(queue)
        worker.daemon = True
        worker.start()

    queue.join()

    print('download finished')


start()