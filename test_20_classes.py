# ----------------------------------------------------------------------------------------------------------------------
# Código que testa o modelo treinado em várias imagens e vídeos
# ----------------------------------------------------------------------------------------------------------------------
import glob
from train_20_classes import *
# ----------------------------------------------------------------------------------------------------------------------

# Função que transforma uma imagem lida pela OpenCV no formato PIL
def cv2_to_pil(image):
    cv2_image = image.copy()
    pil_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(pil_image)

    return(pil_image)
# ----------------------------------------------------------------------------------------------------------------------

# Função que converte uma imagem do tipo PIL em OpenCV
def pil_to_cv2(image):
    pil_image = image.copy()
    cv2_image = np.asarray(pil_image, np.uint8)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    return (cv2_image)
# ----------------------------------------------------------------------------------------------------------------------

# Função que aplica borramento em uma imagem do tipo OpenCV
def apply_blur(image, kernel=(4, 4)):
    blur_image = cv2.blur(image, kernel)

    return(blur_image)
# ----------------------------------------------------------------------------------------------------------------------

# Classe que contém funções para realizar a avaliação das imagens e vídeos
class EvalUtils:

    def __init__(self, model_name):
        self.model_name = model_name
    # ------------------------------------------------------------------------------------------------------------------

    # Função que calcula a previsão para uma imagem (PIL image)
    def eval_sample(self, original_image):

        # Define a classe para ajustar a máscara
        mask_gen = MaskGenerator()
        n_classes = len(mask_gen.class_map)

        # Carrega o modelo treinado
        model = Model(n_classes)
        model.load_state_dict(torch.load(self.model_name))
        model = model.cuda()
        model.eval()

        # Define as transformações que serão aplicadas na image (Resize, Flip e Normalização)
        transform = A.Compose([
            A.Resize(256, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # Aplica as transformações
        image_dict = transform(image=np.array(original_image))

        # Transforma para tensor
        image = image_dict["image"].unsqueeze(0)

        # Faz a predição e ajusta as dimensões
        prediction = model(image.cuda())
        prediction = prediction.detach().cpu()
        prediction = torch.squeeze(prediction, 0)
        prediction = torch.argmax(prediction, 0)

        # Ajusta a predição
        decoded_prediction = mask_gen.decode_segmap(prediction)

        return (decoded_prediction)
    # ------------------------------------------------------------------------------------------------------------------

    # Função que prediz o conjunto de teste
    def eval_test_set(self, size=10):
        output = {"images": [], "masks": [], "predictions": []}

        # Descobre os caminhos de todas as imagens do conjunto de teste
        filenames = sorted(glob("leftImg8bit/test/*/*.png", recursive=True))[:size]

        print("\nAvaliando Amostras do Conjunto de Teste...")
        for filename in tqdm(filenames):

            # Lê a imagem
            image = Image.open(filename).convert("RGB")

            # Lê a máscara
            mask_name = filename.split("/")[-1].replace("leftImg8bit", "gtFine_color")
            mask_path = os.path.join("gtFine", "test", filename.split("/")[-2], mask_name)
            mask = Image.open(mask_path).convert("RGB")

            # Faz a previsão (segmentação
            prediction = self.eval_sample(image)

            output["images"].append(image)
            output["masks"].append(mask)
            output["predictions"].append(prediction)

        return(output)
    # ------------------------------------------------------------------------------------------------------------------

    # Função para plotar os resultados
    def plot_result(self, image, prediction, transparency):

        # Ignora os avisos da biblioteca matplotlib
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)

        # Define as configurações do plot
        subplot_config = (1, 3)
        plt.rcParams['figure.figsize'] = list((15, 7))
        figure, axes = plt.subplots(subplot_config[0], subplot_config[1])

        # Ajusta as informações de cada subplot
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title("Imagem de Teste")

        axes[1].imshow(prediction)
        axes[1].axis('off')
        axes[1].set_title("Previsão")

        axes[2].imshow(transparency)
        axes[2].axis('off')
        axes[2].set_title("Transparência")

        plt.suptitle("Exemplos dos Resultados Obtidos com a Segmentação")
        plt.show()

        # Volta a considerar os avisos de log
        logger.setLevel(old_level)
    # ------------------------------------------------------------------------------------------------------------------

    # Função que sobrepõe a imagem e o resultado com transparência
    def get_transparency(self, image, mask, shape=(512, 256)):

        # Transforma para OpenCV
        cv2_image = pil_to_cv2(image)

        # Ajusta as entradas
        cv2_image = cv2.resize(cv2_image, (shape))
        cv2_mask = cv2.resize(mask, (shape))
        cv2_mask = np.asarray(cv2_mask * 255, np.uint8)
        cv2_mask = cv2.cvtColor(cv2_mask, cv2.COLOR_BGR2RGB)

        # Aplica a transparência
        alpha = 0.5
        overlay = cv2_image.copy()
        overlay = np.asarray(overlay, np.uint8)
        output = cv2_mask
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        return(output)
    # ----------------------------------------------------------------------------------------------------------------------

    # Function that plots the results (as mask or as transparency)
    def plot_all_results(self, size=10):
        output = self.eval_test_set(size=size)

        # Plota os resultados
        index = 0
        for index in range(size):

            # Lê a imagem (PIL)
            image = output["images"][index]

            # Converte para OpenCV
            cv2_image = pil_to_cv2(image)

            # Aplica o borramento
            blur_image = apply_blur(cv2_image)

            # Converte para PIL
            image = cv2_to_pil(blur_image)

            # Lê a máscara
            mask = output["masks"][index]

            # Lê a predição
            prediction = output["predictions"][index]

            # Plota
            transparency = self.get_transparency(image, prediction)
            self.plot_result(image, prediction, transparency)
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------

# Função que faz a previsão de cada frame em um vídeo e mostra o resultado com sobreposição de transparência
def eval_video(model_name="",
               video_filename="20230531_134822.mp4", crops=(800, -1000, 0, -200), video_name=""):

    # Usa algumas funções da classe de avaliação de imagens
    data_eval = EvalUtils(model_name)

    # Lê o vídeo
    cap = cv2.VideoCapture(video_filename)

    # Define o nome do vídeo que será salvo
    if(video_name == ""):
        video_name = str(time.time()).replace(".", "") + ".mp4"

    # Um writer para salvar o vídeo
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 20, (512, 256))

    # Verifica se abriu corretamente
    if (cap.isOpened() == False):
        print("Erro ao abrir o arquivo do vídeo")

    while (cap.isOpened()):

        # Captura um frame
        ret, frame = cap.read()
        if (ret == True):

            # Imagem original sem borramento
            frame_r = cv2.resize(frame[crops[0]:crops[1], crops[2]:crops[3]], (512, 256))
            image = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)

            # Faz o crop e o resize no frame
            frame_blur = cv2.blur(frame, (4, 4))
            frame_blur_r = cv2.resize(frame_blur[crops[0]:crops[1], crops[2]:crops[3]], (512, 256))

            # Transforme de OpenCV para PIL
            image_blur = cv2.cvtColor(frame_blur_r, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_blur)

            # Faz a previsão da imagem
            prediction = data_eval.eval_sample(image_pil)

            # Sobrepõe a imagem e o resultado com transparência
            alpha = 0.5
            overlay = image.copy()
            overlay = np.asarray(overlay, np.uint8)
            output = np.asarray(prediction * 255, np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

            writer.write(output)

            # Mostra o frame atual
            # cv2.imshow("Frame", output)

            # Verifica se o usuário não informou que deseja fechar a visualização
            # if (cv2.waitKey(25) & 0xFF == ord("q")):
            #     break

        else:
            break

    # Fecha o vídeo
    cap.release()
    writer.release()

    # Remove as janelas
    cv2.destroyAllWindows()
# ----------------------------------------------------------------------------------------------------------------------

# Função que faz a previsão de cada frame em um vídeo e mostra o resultado com sobreposição de transparência
## Adaptada para o caso do vídeo ser passado como uma sequência de imagens (png)
def eval_video_database(model_name="",
               video_dir="demoVideo/stuttgart_00", crops=(800, -1000, 0, -200), video_name=""):

    # Usa algumas funções da classe de avaliação de imagens
    data_eval = EvalUtils(model_name)

    # Descobre o caminho dos imagens que formam o vídeo
    filenames = sorted(glob(video_dir + "/*", recursive=True))

    # Define o nome do vídeo que será salvo
    if(video_name == ""):
        video_name = str(time.time()).replace(".", "") + ".mp4"

    # Um writer para salvar o vídeo
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 20, (512, 256))

    # Lê imagem por imagem
    for filename in filenames:

        # Captura um frame
        frame = cv2.imread(filename)

        # Imagem original sem borramento
        frame_r = cv2.resize(frame[crops[0]:crops[1], crops[2]:crops[3]], (512, 256))
        image = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)

        # Faz o crop e o resize no frame
        frame_blur = cv2.blur(frame, (4, 4))
        frame_blur_r = cv2.resize(frame_blur[crops[0]:crops[1], crops[2]:crops[3]], (512, 256))

        # Transforme de OpenCV para PIL
        image_blur = cv2.cvtColor(frame_blur_r, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_blur)

        # Faz a previsão da imagem
        prediction = data_eval.eval_sample(image_pil)

        # Sobrepõe a imagem e o resultado com transparência
        alpha = 0.5
        overlay = image.copy()
        overlay = np.asarray(overlay, np.uint8)
        output = np.asarray(prediction * 255, np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        writer.write(output)

        # # Mostra o frame atual
        # cv2.imshow("Frame", output)
        #
        # # Verifica se o usuário não informou que deseja fechar a visualização
        # if (cv2.waitKey(25) & 0xFF == ord("q")):
        #     break

    # Fecha o vídeo
    writer.release()

    # Remove as janelas
    cv2.destroyAllWindows()
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Escolhe as classes
    mask_gen = MaskGenerator()

    # Descobre o número de classes consideradas nas máscaras
    n_classes = len(mask_gen.class_map)

    # Cria o modelo e inicializa
    model = Model(n_classes)
    # ------------------------------------------------------------------------------------------------------------------

    # AVALIAÇÃO DO MODELO NO CONJUNTO DE IMAGENS DE TESTE --------------------------------------------------------------

    # Avalia o modelo no conjunto de teste
    MODEL_NAME = "model_20_classes.pth"
    images_eval = EvalUtils(MODEL_NAME)
    images_eval.plot_all_results()
    # ------------------------------------------------------------------------------------------------------------------

    # AVALIAÇÃO DO MODELO NO VÍDEO DA UFPR -----------------------------------------------------------------------------

    # Avalia o model em um vídeo
    print("\nTestando o vídeo da UFPR...")
    eval_video(model_name=MODEL_NAME, video_filename="ufpr.mp4", crops=(800, -1000, 0, -200),
               video_name="ufpr_res_20_classes.mp4")
    # ------------------------------------------------------------------------------------------------------------------

    # AVALIAÇÃO DOS VÍDEOS DO DRONE ------------------------------------------------------------------------------------

    # Vídeo 1
    print("\nTestando o vídeo 1 do drone...")
    eval_video(model_name=MODEL_NAME, video_filename="drone_1.mp4", crops=(0, -1, 0, -1),
               video_name="drone_1_res_20_classes.mp4")

    # Vídeo 2
    print("\nTestando o vídeo 2 do drone...")
    eval_video(model_name=MODEL_NAME, video_filename="drone_2.mp4", crops=(0, -1, 0, -1),
               video_name="drone_2_res_20_classes.mp4")

    # Vídeo 3
    print("\nTestando o vídeo 3 do drone...")
    eval_video(model_name=MODEL_NAME, video_filename="drone_3.mp4", crops=(0, -1, 0, -1),
               video_name="drone_3_res_20_classes.mp4")
    # ------------------------------------------------------------------------------------------------------------------

    # AVALIAÇÃO DOS VÍDEOS DISPONIBILIZADOS JUNTO COM A BASE DE DADOS --------------------------------------------------

    # Vídeo 1
    print("\nTestando o vídeo 1 do conjunto de dados...")
    eval_video_database(model_name=MODEL_NAME, video_dir="demoVideo/stuttgart_00", crops=(0, -1, 0, -1),
                        video_name="database_0_res_20_classes.mp4")

    # Vídeo 2
    print("\nTestando o vídeo 2 do conjunto de dados...")
    eval_video_database(model_name=MODEL_NAME, video_dir="demoVideo/stuttgart_01", crops=(0, -1, 0, -1),
                        video_name="database_1_res_20_classes.mp4")

    # Vídeo 3
    print("\nTestando o vídeo 3 do conjunto de dados...")
    eval_video_database(model_name=MODEL_NAME, video_dir="demoVideo/stuttgart_02", crops=(0, -1, 0, -1),
                        video_name="database_2_res_20_classes.mp4")
    # ------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------