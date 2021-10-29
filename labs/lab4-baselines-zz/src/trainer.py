import torch
from torch import nn, optim
import copy


class Trainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

    def train(self, model, train_dataset, dev_dataset=None, lr=5e-4, betas=(0.9, 0.999), batch_size=32):
        num_workers = 2
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
        num_epoch = 10
        min_loss = float('inf')
        best_model = None
        dev_loss = list()
        train_loss = list()
        model = model.to(self.device)
        print(f"total number of batches: {len(trainloader)}")
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            model = model.train()
            nrows = 0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                nrows += labels.shape[0]
                if i % 10 == 0:    # print every 2000 mini-batches
                    print(f"epoch {epoch + 1} - ({i}/{len(trainloader)}) train loss: {loss.item()}")
                    running_loss = 0.0
            train_loss.append(running_loss)
            if dev_dataset:
                dev_l = self.evaluate(model, dev_dataset, batch_size)
                dev_loss.append(dev_l)
                if dev_l < min_loss:
                    min_loss = dev_l
                    best_model = copy.deepcopy(model)
        return model, (min_loss, best_model), train_loss, dev_loss

    def evaluate(self, model, dataset, batch_size=32, num_workers=2):
        devloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()
        model = model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(devloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # print statistics
                running_loss += loss.item()
        print(f"dev loss: {running_loss}")
        return running_loss


    # def infer(self, x):
    #     pass
    
    # def evaluate(self, model, dataloader):
    #     pass

# model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
# print(model.classifier)

# model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
# print(model.classifier)
