"""Funções utilitárias do conjunto de dados UCI HAR para a aplicação Flower com PyTorch."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Número de características de entrada e de classes de atividade física.
NUM_FEATURES = 561
NUM_CLASSES = 6

# Diretório raiz do projeto e arquivos necessários para os conjuntos de treino e teste.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_FILES = tuple(
    f"{split}/{filename}_{split}.txt"
    for split in ("train", "test")
    for filename in ("X", "y", "subject")
)


class Net(nn.Module):
    """Rede neural MLP que classifica as 561 características do conjunto UCI HAR."""

    def __init__(self):
        super().__init__()
        # Camadas totalmente conectadas que transformam as características em seis classes.
        self.fc1 = nn.Linear(NUM_FEATURES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        """Executa a passagem direta e retorna as pontuações das seis classes."""
        # ReLU introduz não linearidade nas camadas ocultas.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Retorna as pontuações brutas (logits), utilizadas pela função CrossEntropyLoss.
        return self.fc3(x)


# Armazena os conjuntos já carregados para evitar leituras repetidas dos arquivos.
_dataset_cache: dict[tuple[str, str], TensorDataset] = {}


def _dataset_files_exist(dataset_dir: Path) -> bool:
    """Verifica se todos os arquivos necessários estão presentes no diretório."""
    return all((dataset_dir / filename).is_file() for filename in DATASET_FILES)


def resolve_dataset_dir(configured_dir: str | Path | None = None) -> Path:
    """Localiza e valida o diretório que contém os arquivos locais do UCI HAR."""
    # Prioriza o caminho recebido como argumento e, em seguida, a variável de ambiente.
    configured_dir = configured_dir or os.environ.get("UCI_HAR_DATASET_DIR")
    if not configured_dir:
        raise ValueError(
            "O caminho do conjunto UCI HAR não foi configurado. Defina `dataset-dir` "
            "no pyproject.toml ou a variável de ambiente `UCI_HAR_DATASET_DIR`."
        )

    # Converte o caminho em um objeto Path e resolve caminhos relativos a partir da raiz do projeto.
    dataset_dir = Path(configured_dir).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_DIR / dataset_dir
    if not _dataset_files_exist(dataset_dir):
        raise FileNotFoundError(
            f"Os arquivos do UCI HAR não foram encontrados em {dataset_dir}. O Flower "
            "executa a aplicação a partir de um pacote que não inclui os arquivos do conjunto. "
            "Defina `dataset-dir` com o caminho absoluto do conjunto usando `--run-config`, "
            "por exemplo: "
            '`flwr run . --run-config "dataset-dir=\'<caminho-absoluto-do-conjunto>\'"`. '
            "O diretório deve conter train/X_train.txt, train/y_train.txt, "
            "train/subject_train.txt e os arquivos correspondentes do conjunto de teste."
        )
    return dataset_dir


def _load_dataset(
    split: str, dataset_dir: str | Path | None = None
) -> TensorDataset:
    """Carrega e armazena em cache uma divisão oficial do UCI HAR."""
    # Aceita somente as divisões oficiais de treino e teste.
    if split not in ("train", "test"):
        raise ValueError(f"Divisão desconhecida do UCI HAR: {split}")
    resolved_dir = resolve_dataset_dir(dataset_dir)
    cache_key = (split, str(resolved_dir.resolve()))
    if cache_key not in _dataset_cache:
        # Lê as características, os rótulos e os identificadores dos participantes.
        features = np.loadtxt(
            resolved_dir / split / f"X_{split}.txt", dtype=np.float32
        )
        labels = np.loadtxt(
            resolved_dir / split / f"y_{split}.txt", dtype=np.int64
        )
        subjects = np.loadtxt(
            resolved_dir / split / f"subject_{split}.txt", dtype=np.int64
        )

        # Converte os rótulos de 1–6 para 0–5, conforme o padrão esperado pelo PyTorch.
        labels = np.asarray(labels).reshape(-1) - 1
        subjects = np.asarray(subjects).reshape(-1)

        # Confere se cada amostra possui exatamente 561 características.
        if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Eram esperadas {NUM_FEATURES} características na divisão "
                f"{split} do UCI HAR, mas o formato encontrado foi {features.shape}"
            )

        # Garante que características, rótulos e participantes tenham o mesmo número de amostras.
        if (
            features.shape[0] != labels.shape[0]
            or features.shape[0] != subjects.shape[0]
        ):
            raise ValueError(
                f"As características, os rótulos e os participantes da divisão "
                f"{split} do UCI HAR possuem quantidades diferentes de linhas"
            )

        # Verifica se os rótulos estão dentro do intervalo das seis classes.
        if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
            raise ValueError(f"Os rótulos do UCI HAR {split} devem estar no intervalo 1–6")

        # Converte os arrays NumPy em tensores e mantém os dados disponíveis em cache.
        _dataset_cache[cache_key] = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels.copy()),
            torch.from_numpy(subjects.copy()),
        )
    return _dataset_cache[cache_key]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    dataset_dir: str | Path | None = None,
):
    """Carrega os dados de um participante por cliente e separa treino e validação locais."""
    dataset = _load_dataset("train", dataset_dir)
    features, labels, subjects = dataset.tensors
    participant_ids = torch.unique(subjects, sorted=True)

    # Permite variar o número de clientes, mantendo exatamente um participante por cliente.
    if not 1 <= num_partitions <= len(participant_ids):
        raise ValueError(
            f"O UCI HAR possui {len(participant_ids)} participantes de treino; configure "
            f"entre 1 e {len(participant_ids)} clientes Flower, mas foram recebidos {num_partitions}"
        )
    if not 0 <= partition_id < num_partitions:
        raise ValueError(
            f"partition_id deve estar no intervalo [0, {num_partitions}), mas foi recebido {partition_id}"
        )

    # Associa cada cliente a um participante distinto, seguindo a ordem dos identificadores.
    participant_id = participant_ids[partition_id]
    partition_indices = torch.where(subjects == participant_id)[0]
    if len(partition_indices) < 2:
        raise ValueError(
            f"O participante {participant_id.item()} do UCI HAR precisa ter pelo menos "
            "duas amostras para criar os conjuntos de treino e validação"
        )

    # Divide as amostras do participante de forma reproduzível, usando uma semente fixa.
    split_generator = torch.Generator().manual_seed(42 + partition_id)
    local_indices = partition_indices[
        torch.randperm(len(partition_indices), generator=split_generator)
    ]

    # Reserva 20% das amostras para validação local e utiliza as restantes para treino.
    validation_size = max(1, int(len(local_indices) * 0.2))
    train_indices = local_indices[validation_size:]
    validation_indices = local_indices[:validation_size]
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    validation_dataset = TensorDataset(
        features[validation_indices], labels[validation_indices]
    )

    # Cria os carregadores de dados que fornecem lotes de amostras ao modelo.
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(validation_dataset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset_dir: str | Path | None = None):
    """Carrega o conjunto oficial de teste do UCI HAR para avaliação centralizada."""
    # Utiliza os dados oficiais de teste, separados dos participantes usados no treino.
    dataset = _load_dataset("test", dataset_dir)
    return DataLoader(TensorDataset(*dataset.tensors[:2]), batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Treina o modelo com os dados locais de um cliente."""
    # Move o modelo para o dispositivo de processamento escolhido.
    net.to(device)

    # CrossEntropyLoss calcula a diferença entre os logits e os rótulos verdadeiros.
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # SGD atualiza os parâmetros do modelo; o momentum ajuda a orientar as atualizações.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0

    # Percorre os dados locais durante a quantidade de épocas configurada.
    for _ in range(epochs):
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)

            # Zera os gradientes, calcula a perda e atualiza os parâmetros da rede.
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Retorna a perda média observada nos lotes processados durante o treinamento.
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test_metrics(net, testloader, device):
    """Avalia o modelo e retorna métricas de classificação e confiança."""
    # Coloca o modelo em modo de avaliação e desativa a atualização dos parâmetros.
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total_examples = 0

    # Acumula a confiança das previsões para calcular médias gerais e separadas por acerto.
    confidence_sum = 0.0
    correct_confidence_sum = 0.0
    wrong_confidence_sum = 0.0

    # Inicializa contadores por classe para calcular precisão, recall e F1 macro.
    true_positive = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    predicted_counts = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    actual_counts = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)

    # Não calcula gradientes durante a avaliação, reduzindo o consumo de memória.
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)

            # Converte os logits em probabilidades e obtém a classe com maior probabilidade.
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            matches = predictions == labels

            # Acumula a perda, os acertos e a confiança das previsões.
            loss += criterion(outputs, labels).item()
            correct += matches.sum().item()
            total_examples += labels.size(0)
            confidence_sum += confidence.sum().item()
            correct_confidence_sum += confidence[matches].sum().item()
            wrong_confidence_sum += confidence[~matches].sum().item()

            # Conta os verdadeiros positivos, as previsões e os exemplos reais de cada classe.
            true_positive += torch.bincount(labels[matches], minlength=NUM_CLASSES)
            predicted_counts += torch.bincount(predictions, minlength=NUM_CLASSES)
            actual_counts += torch.bincount(labels, minlength=NUM_CLASSES)

    # Calcula a proporção de acertos e a perda média por lote.
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    # Calcula precisão e recall por classe, evitando divisões por zero.
    precision_per_class = true_positive.float() / predicted_counts.clamp(min=1)
    recall_per_class = true_positive.float() / actual_counts.clamp(min=1)

    # Calcula o F1 de cada classe e depois a média não ponderada entre as seis classes.
    f1_per_class = (
        2 * precision_per_class * recall_per_class
        / (precision_per_class + recall_per_class).clamp(min=1e-12)
    )

    # Retorna as métricas com os nomes esperados pelo restante da aplicação Flower.
    return {
        "eval_loss": loss,
        "eval_acc": accuracy,
        "eval_precision": precision_per_class.mean().item(),
        "eval_recall": recall_per_class.mean().item(),
        "eval_f1": f1_per_class.mean().item(),
        "eval_confidence": confidence_sum / max(total_examples, 1),
        "eval_correct_confidence": correct_confidence_sum / max(correct, 1),
        "eval_wrong_confidence": wrong_confidence_sum / max(total_examples - correct, 1),
    }


def test(net, testloader, device):
    """Valida o modelo e retorna a perda e a acurácia."""
    # Reutiliza a função de avaliação para manter os cálculos consistentes.
    metrics = test_metrics(net, testloader, device)
    return metrics["eval_loss"], metrics["eval_acc"]
