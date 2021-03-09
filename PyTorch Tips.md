# Torch Tips

## Tensor Operations

```python
torch
├── (Tensor)
│   ├── view(*shape)      # e.g. x.view(-1, 3, 12) 
│   │                     ##  -1 automatically filled
│   └── item()            # get if Tensor is a scalar
│
├── empty(*size)          # e.g. x = torch.empty(2, 3)
├── stack(tensors, dim=0)
└── cat(tensors, dim=0)
```



## Data Preparation

```python
torch
└── utils
    └── data
        ├── Dataset     # A class to override 
        │               ##  `__len__` & `__getitem__`
        ├── TensorDataset(data_tensor, target_tensor)
        ├── DataLoader(dataset, batch_size=1,
        │              shuffle=False,
        │              collate_fn=\
        │                  <function default_collate>)
        │               # define `collate_fn` yourself
        └── sampler
            ├── SequentialSampler(data_source)
            └── RandomSampler(data_source)
```



## Neural Network Model Construction

这是 PyTorch 最主要的 module，docs 比较复杂，分成

- `torch.nn`
- `torch.nn.functional`
- `torch.nn.init`
- `torch.optim`
- `torch.autograd`

### Training

```python
torch
├── (Tensor)
│   ├── backward()
│   │
│   ├── cpu()
│   ├── cuda()
│   └── to(torch.device)            # x = x.to(device)
├── cuda
│   └── is_available()
│       #  if torch.cuda.is_available():
│       ##     device = "cuda"
│       ## else: device = "cpu"
│
├── nn as nn
│   │### Models ###
│   ├── Module
│   │   ├── load_state_dict(torch.load(PATH))
│   │   ├── train()
│   │   └── eval()
│   ├── Sequential(layers)
│   │
│   │### Initializations ###
│   ├── init
│   │   └── uniform_(w)     # In-place, 
│   │                       ##  w is a `torch.Tensor`.
│   │
│   │### Layers ###
│   ├── Linear(in_feat, out_feat)
│   ├── Dropout(rate)
│   │
│   │### Activations ###
│   ├── Softmax(dim=None)
│   ├── Sigmoid()
│   ├── ReLU()
│   ├── LeakyReLU(negative_slope=0.01)
│   ├── Tanh()
│   ├── GELU()
│   ├── ReLU6() # Model Compression
│   │ # --> Corresponding functions
│   ├── functional as F  ────────────────────────────┐
│   │   ├── softmax(input, dim=None)                 │
│   │   ├── sigmoid(input)                           │
│   │   ├── relu(input)                              │
│   │   ├── leaky_relu(input,                        │
│   │   │              negative_slope=0.01)          │
│   │   ├── tanh(input)                              │
│   │   ├── gelu(input)                              │
│   │   └── relu6(input)                             │
│   │                                                │
│   │### Losses ###                                  │
│   ├── MSELoss()                                    │
│   ├── CrossEntropyLoss()                           │
│   ├── BCELoss()                                    │
│   ├── NLLLoss()                                    │
│   │ # --> Corresponding functions                  │
│   └──<functional as F> <───────────────────────────┘
│       ├── mse_loss(input, target)
│       ├── cross_entropy(input, 
│       │                 target: torch.LongTensor)
│       ├── binary_cross_entropy(input, target)
│       ├── log_softmax(input)
│       └── nll_loss(log_softmax_output, target)
│           # F.nll_loss(F.log_softmax(input), target)
│
│    ### Optimizers ###
├── optim
│   ├── (Optimizer)
│   │       ├── zero_grad()
│   │       ├── step()
│   │       └── state_dict()
│   │  
│   ├── SGD(model.parameters(), lr=0.1, momentum=0.9)
│   ├── Adagrad(model.parameters(), lr=0.01, 
│   │           lr_decay=0, weight_decay=0, 
│   │           initial_accumulator_value=0,eps=1e-10)
│   ├── RMSProp(model.parameters(), lr=0.01, 
│   │           alpha=0.99, eps=1e-08, weight_decay=0,
│   │           momentum=0)
│   ├── Adam(model.parameters(), lr=0.001, 
│   │        betas=(0.9, 0.999), eps=1e-08,
│   │        weight_decay=0)
│   │   
│   └── lr_scheduler
│       └── ReduceLROnPlateau(optimizer)
│
│── load(PATH)
│── save(model, PATH)
│
└── autograd
    └── backward(tensors)
```



### Testing

```python
torch
├── nn
│   └── Module
│       ├── load_state_dict(torch.load(PATH))
│       └── eval()
├── optim
│   └── (Optimizer)
│       └── state_dict()
└── no_grad()              # with torch.no_grad(): ...
```



## CNN

- Convolutional Layers
- Pooling Layers
- `torchvision` 

```python
torch
├── (Tensor)
│   └── view(*shape)
├── nn
│   │### Layers ###
│   ├── Conv2d(in_channels, out_channels, 
│   │          kernel_size, stride=1, padding=0)
│   ├── ConvTranspose2d(in_channels, out_channels, 
│   │          kernel_size, stride=1, padding=0, 
│   │          output_padding=0)
│   ├── MaxPool2d(kernel_size, stride=None, 
│   │             padding=0, dilation=1)
│   │             # stride default: kernel_size
│   ├── BatchNorm2d(num_feat)
│   └── BatchNorm1d(num_feat)
├── stack(tensors, dim=0)
└── cat(tensors, dim=0)

torchvision
├── models as models # Useful pretrained
├── transforms as transforms
│   ├── Compose(transforms) # Wrapper
│   ├── ToPILImage(mode=None)
│   ├── RandomHorizontalFlip(p=0.5)
│   ├── RandomRotation(degrees)
│   ├── ToTensor()
│   └── Resize(size)
└── utils
    ├── make_grid(tensor, nrow=8, padding=2)
    └── save_image(tensor, filename, nrow=8,padding=2)
```



## RNN

- Recurrent Layers
- Gensim Word2Vec

```python
torch
├── nn
│   ├── Embedding(num_embed, embed_dim)
│   │   # embedding = nn.Embedding(
│   │   ##               *(w2vmodel.wv.vectors.shape))
│   ├── Parameter(params: torch.FloatTensor)
│   │   # embedding.weight = nn.Parameter(
│   │   ##  torch.FloatTensor(w2vmodel.wv.vectors))
│   ├── LongTensor          # Feeding Indices of words
│   │
│   ├── LSTM(inp_size, hid_size, num_layers)
│   │   # input: input, (h_0, c_0)
│   └── GRU(inp_size, hid_size, num_layers)
├── stack(tensors, dim=0)
└── cat(tensors, dim=0)
    
gensim
└── models
    └── word2Vec
        └── Word2Vec(sentences) # list or words/tokens
```



## 说明

1. 如果遇到中间出现 `as` 语法的，表示常使用 `import ... as ...` 语法 import，例如：

   ```python
   torch
   ├── nn as nn
   ```

   这边表示常常在 coding 时以

   ```python
   import torch.nn as nn
   ```

   语法 import，其下的 modules、classes 和 functions 也以 `nn.SOMETHING` 形式出现，如同

   ```python
   import numpy as np   # np.array...
   import pandas as pd  # pd.read_csv...
   ```

2. 如果看到的是有加 `()` 括号的，代表是一个 object 的 method。例如：

   ```python
   torch
   └── (Tensor)
       ├── view
       └── item
   ```

   这代表一个 `torch.Tensor` object 的 `view` 跟 `item` method

3. 如果看到有加 `<>` 角括号，代表与前面出现过的是同一个 module，后面会有线条连接到在这个段落中第一次出现的地方。例如：

   ```python
   │   ├── functional as F  ────────────────────────────┐
   │   │   ├── relu                                     │
   │   │  ...(Some lines)...                            │
   │   └──<functional as F> <───────────────────────────┘
   │       └── nll_loss
   ```

   这边两个 `functional` 指的是同一个，不过因为包含两种不同用途而分开写。
