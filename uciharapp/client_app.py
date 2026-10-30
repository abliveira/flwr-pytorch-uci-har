"""Aplicação do cliente Flower para o conjunto de dados UCI HAR."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from uciharapp.task import Net, load_data
from uciharapp.task import test_metrics as test_metrics_fn
from uciharapp.task import train as train_fn

# Cria a aplicação do cliente Flower.
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Treina o modelo utilizando os dados locais do participante."""

    # Cria o modelo e carrega os parâmetros globais recebidos do servidor.
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Seleciona a GPU, caso esteja disponível; caso contrário, utiliza a CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Recupera as configurações de particionamento e treinamento do cliente.
    # Cada cliente é associado a um participante específico do conjunto UCI HAR.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    dataset_dir = context.run_config["dataset-dir"]

    # Carrega os dados do participante atribuído ao cliente.
    # O conjunto local é dividido em dados de treinamento e validação.
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size, dataset_dir
    )

    # Executa o treinamento local utilizando a quantidade de épocas e a taxa
    # de aprendizado definidas nas configurações da execução.
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Converte os parâmetros atualizados do modelo para o formato esperado pelo Flower.
    model_record = ArrayRecord(model.state_dict())

    # Registra a perda média do treinamento local e a quantidade de exemplos utilizados.
    # O número de exemplos permite que o servidor considere o tamanho dos conjuntos
    # locais durante a agregação dos parâmetros pela estratégia FedAvg.
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)

    # Organiza os parâmetros e as métricas na mensagem de resposta ao servidor.
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Avalia o modelo recebido utilizando os dados locais de validação."""

    # Cria o modelo e carrega os parâmetros globais recebidos do servidor.
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Seleciona a GPU, caso esteja disponível; caso contrário, utiliza a CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Recupera as configurações de particionamento e o tamanho dos lotes.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    dataset_dir = context.run_config["dataset-dir"]

    # Carrega os dados do participante atribuído ao cliente.
    # Nesta etapa, utiliza-se apenas o conjunto de validação local.
    _, valloader = load_data(
        partition_id, num_partitions, batch_size, dataset_dir
    )

    # Avalia o modelo global recebido do servidor com os dados de validação do cliente.
    # Diferentemente da etapa de treinamento, esta etapa não atualiza os parâmetros do modelo.
    metrics = test_metrics_fn(model, valloader, device)

    # Adiciona a quantidade de exemplos utilizados na validação.
    # Essa informação permite considerar o tamanho de cada conjunto local
    # durante a agregação ponderada das métricas pelo servidor.
    metrics["num-examples"] = len(valloader.dataset)

    # Organiza as métricas de avaliação na mensagem de resposta ao servidor.
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
