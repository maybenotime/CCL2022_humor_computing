import torch 
from fastNLP import Callback

class FGM_callback(Callback):       #对抗训练
    def on_train_begin(self):
        self.fgm = FGM(self.model)
    
    def on_batch_begin(self,batch_x,batch_y,indices):       #保存一下x,y
        self.batch_x = batch_x
        self.batch_y = batch_y
    def on_backward_end(self):         
        self.fgm.attack()           #embedding层加扰动
        fgm_prediction = self.trainer._data_forward(self.model,self.batch_x)        #再次前向传播
        with self.trainer.auto_cast():
            loss = self.trainer._compute_loss(fgm_prediction,self.batch_y).mean()
        loss = loss / self.trainer.update_every         
        self.trainer.grad_scaler.scale(loss).backward()
        self.fgm.restore()

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='word_embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}