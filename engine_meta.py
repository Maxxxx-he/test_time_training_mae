import learn2learn as l2l
import os
import time
import torch

from util import misc


class maml_learner(object):
    def __init__(self, model, dataset, ways, args):
        self.model = model
        self.dataset = dataset
        self.ways = ways
        self.device = args.device

    # num_tasks  num of task to generate
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

    def train(self, save_path, shots=1):
        # inner_lr = 0.0025
        # outer_lr = 0.0025
        meta_lr = 0.005
        maml = l2l.algorithms.MAML(self.model, lr=0.01)
        #print(maml)
        print("---------------")
        opt = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')
        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks(train_ways, shots, 1000)  # support & query

        epochs = 1000
        meta_batch_size = 16
        for epoch in range(epochs):
            start_time = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            opt.zero_grad()
            for _ in range(meta_batch_size):
                # 1) Compute meta-training loss
                learner = maml.clone() #copy from model
                print(learner)
                task = train_tasks.sample()  # inner & outer in fast adapt

                data, labels = task
                data, labels = data.to(self.device), labels.to(self.device)  # support & query

                # inner
                loss_dict, _, _, _ = learner(data, None, mask_ratio=0.75)  # ssl
                loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
                loss_value = loss.item()
                #learner.adapt(loss_value) # learner update
                loss_value.backward()





                # outer
                loss_d, _, _, pred = learner(data, labels, mask_ratio=0, reconstruct=False)  # classifier
                meta_train_loss += loss_d

            meta_train_loss = meta_train_loss / meta_batch_size

            # Print some metrics
            end_time = time.time()
            print(f'Time /epoch: {end_time - start_time:.4f} s')
            print('\n')
            print('Iteration', epoch + 1)
            print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')

            if epoch % 20 == 0:
                print('Iteration:', epoch, 'Meta Train Loss', meta_train_loss.item())
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            #update maml
            opt.zero_grad()
            meta_train_loss.backward()
            opt.step()

    # def test(self):
    #     print("test")
