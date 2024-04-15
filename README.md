# ToySatCLIP

##Goal
My implementation of a ToySatCLIP model trained on the S2-100k dataset.

## Model Architecture

SatCLIP(
  (image_encoder): Resnet(
    (visual): ResNet(
      (conv1): Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act1): ReLU(inplace=True)
          (aa): Identity()
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act2): ReLU(inplace=True)
        )
      )
      (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (fc): Identity()
    )
  )
  (location_encoder): LE(
    (encoder): LocationEncoder(
      (LocEnc0): LocationEncoderCapsule(
        (capsule): Sequential(
          (0): GaussianEncoding()
          (1): Linear(in_features=512, out_features=1024, bias=True)
          (2): ReLU()
          (3): Linear(in_features=1024, out_features=1024, bias=True)
          (4): ReLU()
          (5): Linear(in_features=1024, out_features=1024, bias=True)
          (6): ReLU()
        )
        (head): Sequential(
          (0): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
      (LocEnc1): LocationEncoderCapsule(
        (capsule): Sequential(
          (0): GaussianEncoding()
          (1): Linear(in_features=512, out_features=1024, bias=True)
          (2): ReLU()
          (3): Linear(in_features=1024, out_features=1024, bias=True)
          (4): ReLU()
          (5): Linear(in_features=1024, out_features=1024, bias=True)
          (6): ReLU()
        )
        (head): Sequential(
          (0): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
      (LocEnc2): LocationEncoderCapsule(
        (capsule): Sequential(
          (0): GaussianEncoding()
          (1): Linear(in_features=512, out_features=1024, bias=True)
          (2): ReLU()
          (3): Linear(in_features=1024, out_features=1024, bias=True)
          (4): ReLU()
          (5): Linear(in_features=1024, out_features=1024, bias=True)
          (6): ReLU()
        )
        (head): Sequential(
          (0): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
    )
  )
  (image_projection): Project(
    (projection_layer): Linear(in_features=512, out_features=1024, bias=True)
    (fc_layer): Linear(in_features=1024, out_features=1024, bias=True)
    (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (gelu_activation): GELU(approximate='none')
    (dropout_layer): Dropout(p=0.1, inplace=False)
  )
  (loc_projection): Project(
    (projection_layer): Linear(in_features=512, out_features=1024, bias=True)
    (fc_layer): Linear(in_features=1024, out_features=1024, bias=True)
    (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (gelu_activation): GELU(approximate='none')
    (dropout_layer): Dropout(p=0.1, inplace=False)
  )
)

## Latest Training Logs

```

Training
Epoch: 1
100%
 59/59 [07:02<00:00,  6.18s/it, lr=1e-5, train_loss=29]
100%
 7/7 [00:46<00:00,  6.00s/it, valid_loss=6.55]
Validation Loss: tensor(6.7682, device='cuda:0')
Validation Loss: tensor(6.5403, device='cuda:0')
Validation Loss: tensor(6.4647, device='cuda:0')
Validation Loss: tensor(6.5110, device='cuda:0')
Validation Loss: tensor(6.5306, device='cuda:0')
Validation Loss: tensor(6.5088, device='cuda:0')
Validation Loss: tensor(6.5220, device='cuda:0')
Best Model is saved!
Training
Epoch: 2
100%
 59/59 [06:55<00:00,  6.06s/it, lr=1e-5, train_loss=7.27]
100%
 7/7 [00:46<00:00,  6.01s/it, valid_loss=4.56]
Validation Loss: tensor(4.6274, device='cuda:0')
Validation Loss: tensor(4.5563, device='cuda:0')
Validation Loss: tensor(4.6537, device='cuda:0')
Validation Loss: tensor(4.5609, device='cuda:0')
Validation Loss: tensor(4.5758, device='cuda:0')
Validation Loss: tensor(4.6773, device='cuda:0')
Validation Loss: tensor(3.9688, device='cuda:0')
Best Model is saved!
Training
Epoch: 3
100%
 59/59 [06:58<00:00,  5.98s/it, lr=1e-5, train_loss=4.68]
100%
 7/7 [00:46<00:00,  6.01s/it, valid_loss=4.25]
Validation Loss: tensor(4.2912, device='cuda:0')
Validation Loss: tensor(4.2538, device='cuda:0')
Validation Loss: tensor(4.3463, device='cuda:0')
Validation Loss: tensor(4.2913, device='cuda:0')
Validation Loss: tensor(4.2727, device='cuda:0')
Validation Loss: tensor(4.3638, device='cuda:0')
Validation Loss: tensor(3.5934, device='cuda:0')
Best Model is saved!
Training
Epoch: 4
100%
 59/59 [06:57<00:00,  5.95s/it, lr=1e-5, train_loss=4.31]
100%
 7/7 [00:46<00:00,  5.98s/it, valid_loss=4.11]
Validation Loss: tensor(4.1388, device='cuda:0')
Validation Loss: tensor(4.1191, device='cuda:0')
Validation Loss: tensor(4.1999, device='cuda:0')
Validation Loss: tensor(4.1717, device='cuda:0')
Validation Loss: tensor(4.1177, device='cuda:0')
Validation Loss: tensor(4.2375, device='cuda:0')
Validation Loss: tensor(3.4598, device='cuda:0')
Best Model is saved!
Training
Epoch: 5
100%
 59/59 [06:57<00:00,  5.96s/it, lr=1e-5, train_loss=4.13]
100%
 7/7 [00:46<00:00,  5.94s/it, valid_loss=3.97]
Validation Loss: tensor(4.0136, device='cuda:0')
Validation Loss: tensor(3.9844, device='cuda:0')
Validation Loss: tensor(4.0429, device='cuda:0')
Validation Loss: tensor(4.0379, device='cuda:0')
Validation Loss: tensor(3.9732, device='cuda:0')
Validation Loss: tensor(4.1057, device='cuda:0')
Validation Loss: tensor(3.3484, device='cuda:0')
Best Model is saved!

```

## Implementation Details

## SatCLIP Class:

- This class is the main model architecture.
- It initializes with parameters like temperature, ie (image embedding size), and le (location embedding size).
- It contains components for image and location encoding, as well as projections.
- The `forward` method computes the loss by comparing image and location embeddings.

## Image and Location Encoders:

- The `image_encoder` and `location_encoder` are responsible for encoding image and location data, respectively.
- The `image_encoder` uses a pre-trained ResNet18 model (`Resnet`) for image feature extraction.
- The `location_encoder` uses a pre-trained location encoder.

## Projection Layers:

- `image_projection` and `loc_projection` are projection layers that transform the encoded features into a common embedding space.
- These layers might help in aligning the representations of images and locations.

## Loss Calculation:

- The model computes a contrastive loss based on the similarity between image and location embeddings.
- It calculates individual cross-entropy losses for images and locations and combines them to form the final loss.

## Helper Functions:

- `calculate_cross_entropy`: Computes the cross-entropy loss between predictions and targets.

## LE Class (Location Encoder):

- This class is responsible for encoding location data.
- It initializes with the custom geoclip `LocationEncoder`.
- The `forward` method converts latitude and longitude coordinates into embeddings using the `LocationEncoder`.

## Resnet Class:

- This class defines the image encoder using a pre-trained MOCO-ResNet18 model.
- The ResNet18 model's fully connected layer (`fc`) is made trainable while freezing the rest of the model.

