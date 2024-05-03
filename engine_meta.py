import gc

import learn2learn as l2l
import os
import time
import torch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util import misc
from timm.utils import accuracy


class maml_learner(object):
    def __init__(self, model, dataset, ways, args, optimizer, loss_scaler):
        self.model = model
        self.dataset = dataset
        self.ways = ways
        self.device = args.device
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.args = args

    # num_tasks  num of task to generate
    # create a set of task from given dataset
    def build_tasks(self, ways=1, shots=1, num_tasks=1000):
        dataset = l2l.data.MetaDataset(self.dataset)
        train_tasks = l2l.data.TaskDataset(dataset, task_transforms=[l2l.data.transforms.NWays(dataset, n=ways),
                                                                     l2l.data.transforms.KShots(dataset, k=shots),
                                                                     l2l.data.transforms.LoadData(dataset),
                                                                     l2l.data.transforms.RemapLabels(dataset,
                                                                                                     shuffle=True),
                                                                     l2l.data.transforms.ConsecutiveLabels(
                                                                         dataset)], num_tasks=num_tasks)
        return train_tasks

    # re-order samples and make their original labels as (0 ,..., n-1). ConsecutiveLabels
    # do not keep the original labels, use (0 ,..., n-1); RemapLabels
    # def model_save(self, path):
    #     filename = path + '(1)' if os.path.exists(path) else path
    #     torch.save(self.model.state_dict(), filename)
    #     print(f'Save model at: {filename}')

    def accuracy(predictions, targets):
        # predictions: (n, nc), targets: (n,)
        # predictions = predictions.argmax(dim=-1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.shape[0]

    def inner(self, data, labels, maml):
        gc.collect()
        torch.cuda.empty_cache()
        learner = maml.clone()  # copy from model
        learner.train()
        loss_scaler = NativeScaler()
        optimizer = torch.optim.Adam(maml.parameters(), 0.005)
        # print(learner)
        data, labels = data.to(self.device), labels.to(self.device)  # support & query

        # inner
        loss_dict, _, _, _ = learner(data, None, mask_ratio=0.75)  # ssl
        loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
        loss_value = loss.item()
        # learner.adapt(loss_value) # learner update
        #
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=learner.parameters())

        # outer
        loss_d, _, _, pred = learner(data, labels, mask_ratio=0, reconstruct=False)  # classifier
        loss_s = torch.stack([loss_d[l] for l in loss_d]).sum()
        loss_value_s = loss_s.item()
        print(loss_s, loss_value_s)
        accuracy_out, _ = accuracy(pred, labels, topk=(1, 2))
        gc.collect()
        torch.cuda.empty_cache()
        #return learner, loss_s, accuracy_out
        return learner, loss_value_s, accuracy_out

    def outer(self, shots=1):
        print("---------------")
        gc.collect()
        torch.cuda.empty_cache()
        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks(train_ways, shots, 1000)  # support & query
        loss_scaler_out = NativeScaler()
        epochs = 10  # 1000
        meta_batch_size = 16
        for epoch in range(epochs):
            maml = l2l.algorithms.MAML(self.model, lr=0.001)  # lr canbe anything here since we dont do adapt()
            # print(maml)
            start_time = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_train_loss = 0.0
            #meta_train_loss = torch.tensor(0.0).to(self.device)

            self.optimizer.zero_grad()
            for _ in range(meta_batch_size):
                torch.cuda.empty_cache()
                task = train_tasks.sample()
                data, labels = task
                self.optimizer.zero_grad()
                _, loss, acc = self.inner(data, labels, maml)
                #meta_train_loss = meta_train_loss + loss
                meta_train_loss = meta_train_loss + loss
                meta_train_accuracy = meta_train_accuracy + acc

            meta_train_loss = meta_train_loss / meta_batch_size
            meta_train_accuracy = meta_train_accuracy / meta_batch_size
            # update maml
            self.model.train()
            self.optimizer.zero_grad()
            # loss_tensor = torch.tensor(meta_train_loss)
            # loss_tensor.to(self.device)
            #torch.tensor(meta_train_loss).to(self.device)
            #loss_scaler_out(meta_train_loss, self.optimizer, parameters=self.model.parameters())
            loss_scaler_out(torch.tensor(meta_train_loss).to(self.device), self.optimizer, parameters=self.model.parameters())
            # Print some metrics
            end_time = time.time()
            print(f'Time /epoch: {end_time - start_time:.4f} s')
            print('\n')
            print('Iteration', epoch + 1)
            #print(f'Meta Train Loss: {meta_train_loss.item(): .4f}')
            print(f'Meta Train Loss: {meta_train_loss: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy: .4f}')
            if epoch % 20 == 0 or epoch == epochs:
                print("save_model")
                print(self.args.output_dir)
                print('Iteration:', epoch, 'Meta Train Loss', meta_train_loss)
                misc.save_model(
                    args=self.args, model=self.model, model_without_ddp=maml, optimizer=self.optimizer,
                    loss_scaler=loss_scaler_out, epoch=epoch)

    def train(self):
        self.outer()
        return None

    def test(self, shots=1):
        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for testing...")
        train_tasks = self.build_tasks(train_ways, shots, 1000)  # support & query
        loss_scaler_out = NativeScaler()
        start_time = time.time()
        maml = l2l.algorithms.MAML(self.model, lr=0.001)

        meta_batch_size = 100
        meta_test_accuracy = 0.0
        meta_test_loss = torch.tensor(0.0).to(self.device)
        # meta_test_loss = 0.0
        for _ in range(meta_batch_size):
            task = train_tasks.sample()
            data, labels = task
            _, loss, acc = self.inner(data, labels, maml)
            meta_test_loss = meta_test_loss + loss
            meta_test_accuracy = meta_test_accuracy + acc

        meta_test_loss = meta_test_loss / meta_batch_size
        meta_test_accuracy = meta_test_accuracy / meta_batch_size
        print(f'Meta Test Loss: {meta_test_loss.item(): .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy: .4f}')
        end_time = time.time()
        print(f'Time /epoch: {end_time - start_time:.4f} s')
