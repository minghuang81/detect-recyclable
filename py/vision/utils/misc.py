import time
import torch


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}
        self.cumTime = {}
        self.cumCount = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(key+" is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval

    def cumul(self, key):
        if key not in self.clock:
            raise Exception(key+" need to run start("+key+") first")
        interval = time.time() - self.clock[key]
        self.cumTime[key] = self.cumTime.get(key,0) + interval
        self.cumCount[key] = self.cumCount.get(key,0) + 1
   
    def getCumul(self, key):
        return self.cumTime.get(key,0)
   
    def getAvg(self, key):
        return self.getCumul(key)/self.cumCount.get(key,1)
   
def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))
