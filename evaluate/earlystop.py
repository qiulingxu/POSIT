import torch as T
import torch.nn as nn
class ToyLearner(nn.Module):
    def __init__(self):
        super().__init__()
        
class early_stop():
    def __init__(self, model, optimizer, max_tolerate = 10):
        # The loss should actually be gain here
        self.max_tolerate = max_tolerate
        self.gains = [-1e9]
        self.epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.best_epoch = 0
        self.failed = 0
        self.save_snap_shot()

    def save_snap_shot(self):
        self.snapshot = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gain': self.gains[self.best_epoch],
            }

    def get_best(self):
        return self.snapshot
    
    def should_stop(self,):
        return (self.failed>=self.max_tolerate)

    def add_loss(self, gain):
        self.epoch += 1
        self.gains.append(gain)
        if gain > self.gains[self.best_epoch]:
            self.best_epoch = self.epoch
            self.save_snap_shot()
            self.failed = 0
        else:
            self.failed += 1


if __name__=="__main__":
    model = ToyLearner()
    losses = list(range(100)) + [0,] * 100
    opt = model
    es = early_stop(model, opt, max_tolerate=10)
    for i, loss in enumerate(losses):
        es.add_loss(loss)
        print(i, es. failed, es.should_stop())
        if es.should_stop():
            break