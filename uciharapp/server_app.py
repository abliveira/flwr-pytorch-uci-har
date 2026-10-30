"""Aplicação do servidor Flower para o conjunto de dados UCI HAR."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from uciharapp.logging_utils import LoggingFedAvg

from uciharapp.task import (
    Net,
    load_centralized_dataset,
    resolve_dataset_dir,
    test_metrics,
)

# Cria a aplicação do servidor Flower.
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Ponto de entrada principal da aplicação do servidor."""

    # Lê as configurações definidas para a execução no pyproject.toml.
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Resolve o caminho do conjunto de dados e verifica se os arquivos necessários existem.
    dataset_dir = str(resolve_dataset_dir(context.run_config["dataset-dir"]))
    print(f"Using UCI HAR dataset at {dataset_dir}")

    # Inicializa o modelo global e converte seus parâmetros para o formato esperado pelo Flower.
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Inicializa a estratégia FedAvg com registros personalizados e resumidos.
    # A fração de avaliação define a proporção de clientes selecionados para avaliação em cada rodada.
    strategy = LoggingFedAvg(
        fraction_evaluate=fraction_evaluate,
        total_rounds=num_rounds,
    )

    print("\nStarting federated training")
    print(f"Rounds: {num_rounds}  |  Evaluation fraction: {fraction_evaluate:.0%}")

    # Inicia o treinamento federado utilizando a estratégia FedAvg.
    # O modelo global inicial é distribuído aos clientes, que realizam o treinamento local.
    # Ao final de cada rodada, os parâmetros recebidos são agregados pelo servidor.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        # Executa a avaliação centralizada do modelo global ao final de cada rodada.
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, dataset_dir
        ),
    )

    # Salva os parâmetros finais do modelo, caso essa opção esteja habilitada na configuração.
    if context.run_config["save-model"]:
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
        print("\nFinal model saved: final_model.pt")

    print("\nFederated training finished.")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, dataset_dir: str | None = None
) -> MetricRecord:
    """Avalia o modelo global e exibe um resumo das métricas de desempenho."""

    # Cria uma instância do modelo e carrega os parâmetros globais recebidos do servidor.
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())

    # Seleciona a GPU, caso esteja disponível; caso contrário, utiliza a CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Carrega o conjunto de teste oficial do UCI HAR para a avaliação centralizada.
    # Esse conjunto é separado dos dados utilizados pelos clientes durante o treinamento.
    test_dataloader = load_centralized_dataset(dataset_dir)

    # Avalia o modelo global utilizando os dados de teste centralizados.
    # As métricas retornadas incluem acurácia, perda, precisão, revocação, F1 e confiança.
    metrics = test_metrics(model, test_dataloader, device)

    # Identifica se a avaliação corresponde ao modelo inicial ou a uma rodada de treinamento.
    phase = "BASELINE (BEFORE TRAINING)" if server_round == 0 else "GLOBAL EVALUATION"

    # Exibe um cabeçalho para facilitar a identificação dos resultados no terminal.
    print(f"\n{'-' * 78}")
    if server_round == 0:
        print(f"{phase}  |  INITIAL MODEL")
    else:
        print(f"ROUND {server_round:02d}  |  {phase}")

    print("-" * 78)

    # Exibe as principais métricas de classificação.
    # Precisão, revocação e F1 são calculados como médias macro entre as seis classes.
    print(
        f"Loss: {metrics['eval_loss']:.4f}  |  "
        f"Accuracy: {metrics['eval_acc']:.2%}  |  "
        f"Macro precision: {metrics['eval_precision']:.4f}  |  "
        f"Macro recall: {metrics['eval_recall']:.4f}  |  "
        f"Macro-F1: {metrics['eval_f1']:.4f}"
    )

    # Armazena todas as métricas no MetricRecord para que fiquem disponíveis
    # no objeto de resultados retornado pelo Flower ao final da execução.
    return MetricRecord(
        {
            "accuracy": metrics["eval_acc"],
            "loss": metrics["eval_loss"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
            "f1": metrics["eval_f1"],
            "confidence": metrics["eval_confidence"],
            "correct_confidence": metrics["eval_correct_confidence"],
            "wrong_confidence": metrics["eval_wrong_confidence"],
        }
    )
