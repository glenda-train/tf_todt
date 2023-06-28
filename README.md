# Segmentação Semântica de Áreas Urbanas

* **Objetivo:** realizar a segmentação semântica de regiões urbanas, realizando testes tanto em cenários europeus como brasileiros.
* **Conjunto de Dados:** Cityscapes Dataset, que pode ser acessado [aqui](https://www.cityscapes-dataset.com/).
* **Modelo:** Arquitetura U-Net, utilizando uma ResNet34 como Encoder e Decoder.
* **Imagens de Teste:** Vídeo de demonstração do conjunto de dados Cityscapes, vídeo no campus da UFPR Centro Politécnico e vídeos de perspectiva aérea obtidos com um drone.

* As mídias consideradas neste trabalho podem ser obtidas no site do conjunto de dados ou no arquivo .zip presente neste [link](https://drive.google.com/drive/folders/1SbYEKd2K213LZqZbn9oVgvHEIABIsAYd?usp=sharing).
* O link acima também contém os resultados alcançados neste trabalho assim como um relatório detalhado.

## Organização do Repositório:

| Arquivo             | Descrição                                                                                              |
|:-------------------:|:------------------------------------------------------------------------------------------------------:|
| train_20_classes.py | Arquivo contendo os códigos utilizados para treinar o modelo de segmentação semântica do experimento 1 |
| test_20_classes.py  | Arquivo contendo os códigos utilizados para testar o modelo de segmentação semântica do experimento 1  |
| train_12_classes.py | Arquivo contendo os códigos utilizados para treinar o modelo de segmentação semântica do experimento 2 |
| test_12_classes.py  | Arquivo contendo os códigos utilizados para testar o modelo de segmentação semântica do experimento 2  |
