import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import dataset
import models
import utils
from  Visualizer import denorm
from opts import opts

def main(opt):
    print(opt)

    torch.manual_seed(opt.seed)

    print('loading datasets')
    #TODO calculating mand and std of MPV dataset
    transformRGB = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    transformGrey = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5,), std = (0.5,))])

    train_data = dataset.MPVdataset(transformRGB = transformRGB, transformGrey = transformGrey)
    print('number of data point: ', len(train_data))

    trainloader = DataLoader(dataset = train_data, batch_size = opt.batch_size, shuffle = True, pin_memory = False, drop_last = True)#pin_memory can be True if gpu is available
    print('number of iter:', len(trainloader))
    print('datasets loading finished!')

    #load models
    if opt.resume:
        print('relaoding old model')
        try:
            model_path = opt.load_model
            Unet, Unet_optimizer, current_epoch = utils.load_checkpoint(model_path, opt)
        except:
            print('try to reload the model but either model path is incorrect or model architecture does not fit')
    else:
        print('creating new model')
        Unet = models.Unet(n_channels = 4, n_classes = 3)
        Unet_optimizer = torch.optim.Adam(Unet.parameters(), lr = opt.lr, betas = [0.5, 0.999])
        current_epoch = 1

    print('model created')

    #TODO multi-gpus
    if opt.gpu and torch.cuda.is_available():
        print('using gpu')
        Unet = Unet.cuda()
    else:
        print('using cpu')

    if opt.mode == 'train':
        Unet.train()
    elif opt.mode == 'test':
        Unet.eval()
    else:
        print('incorrect mode setting')
        
    #loss function settings
    criterionL1 = nn.L1Loss()

    if current_epoch >= opt.n_epochs:
        raise Exception('training already finished')
    else:
        print('start training')
    
    #start training
    for epoch in range(current_epoch, opt.n_epochs+1):
        for i, (cloth_front, warped_cloth_front_mask, human_parse, warped_cloth_front) in enumerate(trainloader, 1):
            '''
            print('cloth_front.shape:', cloth_front.shape) #[batch-size, 3, 256, 192]
            print('warped_cloth_front.shape:', warped_cloth_front.shape) #[batch-size, 3, 256, 192]
            print('warped_cloth_front_mask.shape:', warped_cloth_front_mask.shape) #[batch-size, 1, 256, 192]
            print('human_parse.shape:', human_parse.shape) #[batch-size, 3, 256, 192]
            '''

            if opt.gpu and torch.cuda.is_available():
                cloth_front = cloth_front.cuda()
                warped_cloth_front = warped_cloth_front.cuda()
                warped_cloth_front_mask = warped_cloth_front_mask.cuda()
                human_parse = human_parse.cuda()

            if opt.show_input:
                print(opt.show_input)
                plt.imshow(cloth_front[0].permute(1, 2, 0))
                plt.show()
                plt.imshow(warped_cloth_front[0].permute(1, 2, 0))
                plt.show()
                plt.imshow(human_parse[0].permute(1, 2, 0))
                plt.show()
                plt.imshow(warped_cloth_front_mask[0].permute(1, 2, 0).squeeze(2), cmap = 'gray', vmin = 0, vmax = 255)
                plt.show()
            
            inp = torch.cat((cloth_front, warped_cloth_front_mask), 1)
            out = Unet(inp)
            loss = criterionL1(warped_cloth_front, out)

            Unet_optimizer.zero_grad()
            loss.backward()
            Unet_optimizer.step()
            
            if (i % 250 == 0):#TODO
                print('saving images')
                original_img = denorm(label_img.cpu().data)
                save_image(original_img, './img/original_img-{}.png'.format((i+1)*opt.batch_size))

                target_img = denorm(output_img.cpu().data)
                save_image(target_img, './img/target_img-{}.png'.format((i+1)*opt.batch_size))

            if (i + 1) % 100 == 0:
                print("iteration: {} / {}, Epoch: {} / {}, loss: {:.6f}".format(str(i), str(len(trainloader)), epoch, opt.n_epochs), loss.data)

        torch.save({'epoch': epoch, 'model_state_dict': Unet.state_dict(), 'optimizer_state_dict': Unet_optimizer.state_dict()}, './checkpoints/Unet_checkpoint.pth')

if __name__ == '__main__':
    opt = opts()
    opt = opt.parse()
    main(opt)
        





        
