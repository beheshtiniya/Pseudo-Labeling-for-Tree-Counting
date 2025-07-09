import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.dataset import make_Dataset
from utils.eval import eval_net
from utils.ssim_loss import SSIM_Loss
from utils import metics
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

def train_model(model, criterion, optimizer,scheduler ,dataload, valid_dataload, device, checkpoint_path, result_path, num_epochs,
                factor, arg):
    # print ("data loade:::",(dataload))
    best_loss = None
    best_val_loss = None
    train_loss_array=[]
    val_loss_array =[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        epoch_loss = 0
        mse1 = 0
        mae1 = 0
        r2_1 = 0
        step = 0

        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device).float()

            # print("x::",inputs.shape)
            # print("labels::", labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            if arg.model == 'TreeCountNet' or 'TENet':
                if arg.deepsupervision:
                    out1, out2, out3, outputs = model(inputs)

                    if arg.loss_function == 'SSIM':
                        los1 = criterion(outputs, labels)
                        los2 = criterion(outputs, labels)
                        los3 = criterion(outputs, labels)
                        loss_final = criterion(outputs, labels)
                        #loss = 0.1 * los1 + 0.3 * los2 + 0.6 * los3 + loss_final
                        loss = los1 + los2 + los3 + loss_final

                    elif arg.loss_function == 'SSIM_L2':
                        # print("arg.loss_function == 'SSIM_L2'")
                        # print("ssssssssssssssssssssssssssssssssssssssssssssssssssssss")
                        los1 = criterion(out1, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)

                        # print("los1:%0.15f" %(los1))
                        los2 = criterion(
                            out2, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        # print("los2:%0.15f" %(los2))
                        los3 = criterion(
                            out3, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        # print("los3: %0.15f" %(los3))
                        loss_final = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                        # print("loss_final:%0.15f" %(loss_final))
                        loss = los1 + los2 + los3 + loss_final
                        # print("loss:%0.15f" %(loss))
                    else:
                        los1 = criterion(out1, labels)
                        los2 = criterion(out2, labels)
                        los3 = criterion(out3, labels)
                        loss_final = criterion(outputs, labels)
                        loss = los1 + los2 + los3 + loss_final

                else:
                    outputs = model(inputs)

                    if arg.loss_function == 'SSIM':
                        loss = criterion(outputs, labels)
                    elif arg.loss_function == 'SSIM_L2':

                        loss = criterion(
                            outputs, labels) * arg.ssim_weight + torch.nn.MSELoss()(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)

                if arg.loss_function == 'SSIM':
                    loss = criterion(outputs, labels)
                elif arg.loss_function == 'SSIM_L2':
                    ssim_loss = criterion(outputs, labels)
                    l2_loss = torch.nn.MSELoss()(outputs, labels)
                    loss = l2_loss + ssim_loss*arg.ssim_weight
                else:
                    loss = criterion(outputs, labels)
            epoch_loss += float(loss)
            # print("epoch_loss" , epoch_loss)
            # print("step", step)
            loss.backward()
            optimizer.step()
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: ADAM lr %.7f -> %.7f" % (epoch, before_lr, after_lr))
        # print("final step: ", step)
        print("spoch loss:",epoch_loss)
        print('step:',step)
        train_loss = epoch_loss / step
        train_loss_array.append(train_loss)

        print("epoch %d loss:%0.6f " % (
            epoch, epoch_loss / step))

        # print("call loss fun")

        val_loss = eval_net(
            model, valid_dataload, factor, arg)
        val_loss_array.append(val_loss)
        print('Valid_Loss:', val_loss)
        if epoch % 10 == 0:
            print("epoch : ", epoch)
            print("---------------------------------------------------")

            print(" train_loss_array: ", train_loss_array)
            print(" val_loss_array: ", val_loss_array)
        if best_loss == None:
            best_loss = train_loss
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_loss.pth')
            print('Checkpoint 0 saved !'.format(epoch))
        elif train_loss < best_loss:
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_loss.pth')
            if epoch % 5 == 0 :
                torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_loss {} .pth'.format(epoch))
            best_loss = train_loss
            print('Checkpoint {} train_loss improved and saved !'.format(epoch))
        elif epoch == num_epochs - 1:
            torch.save(model.state_dict(),
                       checkpoint_path + 'weights_last.pth')
            print('Checkpoint {} improved and saved !'.format(epoch))
        else:
            print('Best_loss not improved !')

        if best_val_loss == None:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_val_loss.pth')
        elif val_loss < best_val_loss:
            torch.save(model.state_dict(), checkpoint_path +
                       'weights_best_val_loss.pth')
            best_val_loss = val_loss
            print('Checkpoint {} val_loss improved and saved !'.format(epoch))
        else:
            print('Best_val_loss not improved !')
    train_over = np.array(train_loss_array)
    validation = np.array(val_loss_array)
    z = np.arange(0, num_epochs, 1)
    X_axis = np.arange(len(z))
    fig, ax = plt.subplots()
    ax.plot(z, validation, color="green", label="loss_validation")
    ax.plot(z, train_over, color="red", label="loss_train")
    plt.xticks(X_axis, z)
    plt.xlabel("number of epoch")
    plt.ylabel("magnitude")
    plt.title("over fit")
    plt.legend()
    plt.show()
    return model


# 训练模型
def train(model, device, data_path, checkpoint_path, result_path, arg):
    model = model.to(device)
    model = nn.DataParallel(model)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("MODEL::",model)
    # print("device::",device)
    # print("datase_path",data_path)
    # print("chekpoint_path",checkpoint_path)
    # print("resut_path",result_path)
    # print("arg",arg)
    # print("arg.loss_function", arg.loss_function)
    batch_size = arg.batch_size
    if arg.loss_function == 'L2':
        criterion = torch.nn.MSELoss()

    if arg.loss_function == 'SSIM':
        criterion = SSIM_Loss()

    if arg.loss_function == 'SSIM_L2':
        criterion = SSIM_Loss()

    def lr_lambda(epoch):
        # LR to be 0.1 * (1/1+0.01*epoch)
        base_lr = 0.01
        factor = 0.01
        return base_lr / (1 + factor * epoch)

    lr = 0.01
    # optimizer = optim.Adam(model.parameters(),weight_decay=0.1)
    optimizer = optim.Adam(model.parameters() , lr=lr)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    y_transforms = transforms.ToTensor()
    dataset_path = data_path
    train_path = dataset_path + '/train'
    # print("11111111111111111")
    train_dataset = make_Dataset(
        root=train_path, arg=arg, transform=x_transforms, target_transform=y_transforms)

    len_data = int(train_dataset.__len__())
    print("len_data",len_data)
    train, valid = torch.utils.data.random_split(
        train_dataset, [int(len_data*70/100), int(len_data *30/100)])
    print("train",len(train))
    print ("valid",len(valid))

    # valid_dataset = make_Dataset(
    #    valid, arg=arg, transform=x_transforms, target_transform=y_transforms)
    train_dataloader = DataLoader(
        train, batch_size=batch_size, num_workers=4)
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, num_workers=4)

    print(" valid_dataloader:::", len(valid_dataloader))
    print("criterion:",criterion)
    train_model(model, criterion, optimizer,scheduler ,train_dataloader, valid_dataloader, device=device,

                checkpoint_path=checkpoint_path, result_path=result_path, num_epochs=arg.num_epochs, factor=arg.factor,
                arg=arg)
