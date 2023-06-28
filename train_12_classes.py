# ----------------------------------------------------------------------------------------------------------------------
# Código que treina um modelo de segmentação semântica para áreas urbanas considerando 12 classes diferentes
# ----------------------------------------------------------------------------------------------------------------------

# Bibliotecas básicas
import os
import cv2
import time
import shutil
import logging
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# Bibliotecas do modelo e de leitura dos dados
import torch
import torchvision
import torchmetrics
import multiprocessing
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as seg_models
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Biblioteca de Aumento de Dados
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ----------------------------------------------------------------------------------------------------------------------

# Classe que ajusta as classes da máscara (de 35 para 20 classes mais relevantes)
class MaskGenerator:
    ignore_index = 255  # Índice das classes que serão ignoradas
    valid_classes = None  # Lista dos índices das classes selecionadas
    class_names = None  # Nomes das classes consideradas
    ignored_classes = None  # Lista dos índices das classes ignoradas
    class_map = None  # Mapeamento das classes antigas para as classes selecionadas
    class_colours = None  # Mapeamento das cores para as classes selecionadas
    # ------------------------------------------------------------------------------------------------------------------

    # Função que escolhe as 20 classes mais relevantes
    def __init__(self):

        # Define os índices das classes que serão ignoradas
        self.ignored_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 29, 30, 31, -1]

        # Define os índices das classes que serão consideradas
        self.valid_classes = [self.ignore_index, 7, 8, 11, 20, 21, 22, 23, 24, 26, 32, 33]

        # Define os nomes das classes consideradas
        self.class_names = ['unlabelled',
                            'road',
                            'sidewalk',
                            'building',
                            'traffic_sign',
                            'vegetation',
                            'terrain',
                            'sky',
                            'person',
                            'car',
                            'motorcycle',
                            'bicycle']

        # Mapeia os índices das classes originais para novos índices (0 a 19)
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        # Define a cor de cada classe (para colorir os objetos)
        colors = [
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [0, 0, 142],
            [0, 0, 230],
            [119, 11, 32]
        ]

        # Mapeia as cores para os novos índices definidos (0 a 19)
        self.class_colours = dict(zip(range(len(self.class_map)), colors))
    # ------------------------------------------------------------------------------------------------------------------

    # Função que remove as classes indesejadas (ignoradas) da máscara e ajusta os índices das classes consideradas
    ## Máscara está em escala de cinza
    def encode_segmap(self, mask):

        # Para todas as classes ignoradas
        for ignored_class in self.ignored_classes:
            # Valores da máscara que pertencem a classes ignoradas recebem o valor de "ignore_index"
            mask[mask == ignored_class] = self.ignore_index

        # Para todas as classes consideradas
        for valid_class in self.valid_classes:
            # Valores da máscara que pertencem a classes válidas recebem o índice do mapeamento de classes
            mask[mask == valid_class] = self.class_map[valid_class]

        return (mask)
    # ------------------------------------------------------------------------------------------------------------------

    # Função que converte a máscara de escala de cinza para RGB
    def decode_segmap(self, gray_mask):

        # Transforma a máscara em numpy
        temp = gray_mask.numpy()

        # Define cada canal RGB
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()

        # Para todas as classes válidas
        for l in range(0, len(self.class_map)):
            # Associa as cores pré-definidas de cada classe
            r[temp == l] = self.class_colours[l][0]
            g[temp == l] = self.class_colours[l][1]
            b[temp == l] = self.class_colours[l][2]

        # Combina os canais em uma imagem RGB
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return (rgb)
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# Classe para realizar a leitura dos dados
class DataReader(Cityscapes):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        # Define as transformações que serão aplicadas nas images (Resize, Flip e Normalização)
        transform = A.Compose([
            #A.Resize(256, 512),
            A.Resize(128, 256),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Lê as imagens e transforma para RGB
        image = Image.open(self.images[index]).convert("RGB")

        # Percorre as anotações
        targets: Any = []
        for target_index, target_type in enumerate(self.target_type):

            # Se a anotação indica um polígono
            if (target_type == "polygon"):

                # Lê os dados do polígono
                target = self._load_json(self.targets[index][target_index])

            # Caso contrário, lê a imagem anotada
            else:
                target = Image.open(self.targets[index][target_index])

            targets.append(target)

        # Confere o tamanho da list de targets
        if (len(targets) > 1):
            target = tuple(targets)
        else:
            target = targets[0]

        # Checa as transformações
        if (transform is not None):
            transformed = transform(image=np.array(image), mask=np.array(target))

        # Se não houver transformações informa erro
        else:
            return (None, None)

        return (transformed["image"], transformed["mask"])
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# Função para plotar as correções da máscara para as classes selecionadas
def plot_mask_correction(image, mask):

    # Define as configurações do plot
    subplot_config = (2, 2)
    plt.rcParams['figure.figsize'] = list((20, 10))
    figure, axes = plt.subplots(subplot_config[0], subplot_config[1])

    # Corrige as classes
    new_mask = mask_gen.encode_segmap(mask.clone())

    # Tranforma para RGB
    new_mask_rgb = mask_gen.decode_segmap(new_mask.clone())

    # Ignora os avisos da biblioteca matplotlib
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    # Ajusta as informações de cada subplot
    axes[0, 0].imshow(image.permute(1, 2, 0))
    axes[0, 0].axis('off')
    axes[0, 0].set_title("Imagem Original Normalizada")

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title("Máscara Original")

    axes[1, 0].imshow(new_mask, cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Mácara Corrigida (Escala de Cinza)")

    axes[1, 1].imshow(new_mask_rgb)
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Máscara Corrigida (RGB)")

    plt.suptitle("Exemplo de Imagem e Máscara do Conjunto de Dados")
    plt.savefig('correcao_mascara.png')

    # Volta a considerar os avisos de log
    logger.setLevel(old_level)
# ----------------------------------------------------------------------------------------------------------------------

# Define a classe do modelo de segmentação
class Model(LightningModule):

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, n_classes):
        super(Model, self).__init__()

        # Define a arquitetura (UNet)
        ## O encoder pode ser 'resnet34', 'mobilenet_v2', 'efficientnet-b7', etc
        ## São utilizados os pesos pré-treinados da ImageNet
        ## A entrada é composta por imagens coloridas (3 canais)
        ## Os canais da saída do modelo correspondem ao número de classes
        self.layer = seg_models.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes,
        )

        # Variável para controlar as épocas
        self.epoch = -1

        # Define os parâmetros
        ## Learning Rate
        self.lr = 1e-3

        # Tamanho do batch
        self.batch_size = 8

        # Número de threads
        self.numworker = multiprocessing.cpu_count() // 4

        # Função de custo (Coeficiente Dice)
        self.criterion = seg_models.losses.DiceLoss(mode='multiclass')

        # Jacard como métrica
        self.metrics = torchmetrics.JaccardIndex(task="multiclass", num_classes=n_classes)

        # Leitura do conjunto de treinamento
        self.train_class = DataReader('./', split='train', mode='fine', target_type='semantic')

        # Leitura do conjunto de validação
        self.val_class = DataReader('./', split='val', mode='fine', target_type='semantic')
    # ------------------------------------------------------------------------------------------------------------------

    # Cálculo da perda e da métrica (IoU)
    def process(self, image, segment):
        # Saída da rede
        out = self(image)

        # Define a classe de ajuste das classes
        mask_gen = MaskGenerator()

        # Ajusta as classes
        segment = mask_gen.encode_segmap(segment)

        # Calcula a perda
        loss = self.criterion(out, segment.long())

        # Calcula o Jacard
        jacard = self.metrics(out, segment)

        return (loss, jacard)
    # ------------------------------------------------------------------------------------------------------------------

    # Cálculo do forward da rede (entrada com pesos, bias e funções de ativação de todas as camadas)
    def forward(self, x):
        return (self.layer(x))
    # ------------------------------------------------------------------------------------------------------------------

    # Definição do otimizador para o backpropagation (Adam)
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return (opt)
    # ------------------------------------------------------------------------------------------------------------------

    # Definição do dataloader para realizar o treinamento (classes, batch e threads)
    def train_dataloader(self):
        return DataLoader(self.train_class,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworker,
                          pin_memory=True)
    # ------------------------------------------------------------------------------------------------------------------

    # Definição de cada passo de trainamento
    def training_step(self, batch, batch_idx):
        # Separa as imagens e máscaras do batch
        image, segment = batch

        # Calcula a perda e o Jacard do batch
        loss, jacard = self.process(image, segment)

        # Printa
        # https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        self.log('train_epoch', float(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_jacard', jacard, on_step=False, on_epoch=True, prog_bar=False)
        return (loss)
    # ------------------------------------------------------------------------------------------------------------------

    # Definição do dataloader para realizar a validação
    def val_dataloader(self):
        return DataLoader(self.val_class,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.numworker,
                          pin_memory=True)
    # ------------------------------------------------------------------------------------------------------------------

    # Definição de cada passo da validação
    def validation_step(self, batch, batch_idx):

        # Separa as imagens e máscaras do batch
        image, segment = batch

        # Calcula a perda ou IoU do batch
        loss, jacard = self.process(image, segment)

        # Printa
        self.log('val_epoch', float(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_jacard', jacard, on_step=False, on_epoch=True, prog_bar=False)

        # Salva o modelo
        if(self.epoch != self.current_epoch):
            self.epoch += 1
            torch.save(self.state_dict(), 'model_epoch_{}.pth'.format(self.epoch))

        return (loss)
    # ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # DADOS SOBRE OS CONJUNTOS DE TREINAMENTO, VALIDAÇÃO E TESTE -------------------------------------------------------

    # Leitura dos dados
    train_dataset = Cityscapes('./', split='train', mode='fine', target_type='semantic')
    val_dataset = Cityscapes('./', split='val', mode='fine', target_type='semantic')
    test_dataset = Cityscapes('./', split='test', mode='fine', target_type='semantic')

    print("Conjunto de Treinamento:")
    # Imprime a quantidade de exemplos dentro do conjunto de dados
    print("  - {} exemplos".format(len(train_dataset)))

    # Imprime o tamanho de cada imagem do conjunto de dados
    print("  - {} de largura (cada imagem)".format(train_dataset[0][0].size[0]))
    print("  - {} de altura (cada imagem)".format(train_dataset[0][0].size[1]))

    print("\nConjunto de Validação:")
    # Imprime a quantidade de exemplos dentro do conjunto de dados
    print("  - {} exemplos".format(len(val_dataset)))

    # Imprime o tamanho de cada imagem do conjunto de dados
    print("  - {} de largura (cada imagem)".format(val_dataset[0][0].size[0]))
    print("  - {} de altura (cada imagem)".format(val_dataset[0][0].size[1]))

    print("\nConjunto de Teste:")
    # Imprime a quantidade de exemplos dentro do conjunto de dados
    print("  - {} exemplos".format(len(test_dataset)))

    # Imprime o tamanho de cada imagem do conjunto de dados
    print("  - {} de largura (cada imagem)".format(test_dataset[0][0].size[0]))
    print("  - {} de altura (cada imagem)".format(test_dataset[0][0].size[1]))
    # ------------------------------------------------------------------------------------------------------------------

    # APRESENTAÇÃO DAS ALTERAÇÕES NA MÁSCARA ---------------------------------------------------------------------------

    # Escolhe as classes
    mask_gen = MaskGenerator()

    print("Mapeamento das Classes:")
    print(mask_gen.class_map)

    print("\nMapeamento das Cores:")
    print(mask_gen.class_colours)

    print("\nClasses Ignoradas:")
    print(mask_gen.ignored_classes)

    # Lê os dados
    train_dataset = DataReader('./', split='val', mode='fine', target_type='semantic')

    # Extrai uma imagem e sua máscara correspondente
    image, mask = train_dataset[1]

    # Plota as correções da máscara
    plot_mask_correction(image, mask)
    # ------------------------------------------------------------------------------------------------------------------

    # TREINAMENTO DO MODELO --------------------------------------------------------------------------------------------

    # Descobre o número de classes consideradas nas máscaras
    n_classes = len(mask_gen.class_map)

    # Cria o modelo e inicializa
    model = Model(n_classes)

    # Define uma callback para imprimir os valores de perda (validação)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='file', save_last=True)

    # Define os hiperparâmetros de treino e os callbacks
    trainer = Trainer(max_epochs=2,
                      accelerator="auto",
                      precision='16',
                      log_every_n_steps=1,
                      callbacks=[checkpoint_callback])

    # Treina o modelo
    trainer.fit(model)

    # Salva o modelo treinado
    torch.save(model.state_dict(), 'model.pth')
    # ------------------------------------------------------------------------------------------------------------------

    # CRIA UM DIRETÓRIO E SALVA OS RESULTADOS --------------------------------------------------------------------------

    # Cria o diretório
    exp_dir = "Resultados_{}".format(time.time())
    os.mkdir(exp_dir)

    # Move os plots
    shutil.move("correcao_mascara.png", os.path.join(exp_dir, "correcao_mascara.png"))

    # Move as métricas do Lightning Module
    shutil.move("lightning_logs", os.path.join(exp_dir, "lightning_logs"))
    shutil.move("checkpoints", os.path.join(exp_dir, "checkpoints"))

    # Move os modelos
    models_path = glob("model*.pth")
    for model_name in models_path:
        shutil.move(model_name, os.path.join(exp_dir, model_name))
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------